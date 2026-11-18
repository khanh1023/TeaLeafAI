"""Re-evaluate the unchanged Keras and browser ONNX models on the frozen test split."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'training/.cache/matplotlib'))
import argparse
import collections
import csv
import datetime
import hashlib
import json
import time

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import onnxruntime as ort


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(truth, probabilities, classes):
    if probabilities.shape != (len(truth), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid model output')
    pred = probabilities.argmax(1)
    report = classification_report(truth, pred, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    correct = int(np.count_nonzero(truth == pred))
    return dict(accuracy=float(accuracy_score(truth, pred)), correct=correct,
                incorrect=len(truth)-correct, test_images=len(truth),
                balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                macro_precision=report['macro avg']['precision'],
                macro_recall=report['macro avg']['recall'],
                macro_f1=report['macro avg']['f1-score'],
                weighted_precision=report['weighted avg']['precision'],
                weighted_recall=report['weighted avg']['recall'],
                weighted_f1=report['weighted avg']['f1-score'],
                classification_report=report,
                confusion_matrix=confusion_matrix(truth, pred, labels=list(range(len(classes)))).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'training/reports/reevaluation')
    args = parser.parse_args()
    started = time.monotonic()
    manifest = ROOT/'training/reports/expanded-dataset-manifest.json'
    records = json.loads(manifest.read_text())
    classes = json.loads((ROOT/'backend/models/classes.json').read_text())
    browser = json.loads((ROOT/'frontend/public/model/manifest.json').read_text())
    if classes != browser['classes'] or digest(manifest) != browser['dataset_manifest_sha256']:
        raise ValueError('Class order or frozen dataset manifest mismatch')
    if len({r['path'] for r in records}) != len(records):
        raise ValueError('Repeated dataset path')
    for field in ['group', 'raw_sha256', 'pixel_sha256', 'sha256']:
        seen = collections.defaultdict(set)
        for record in records:
            seen[record[field]].add(record['split'])
        if any(len(splits) > 1 for splits in seen.values()):
            raise ValueError(f'Data crosses partitions: {field}')
    entries = [r for r in records if r['split'] == 'test']
    data = ROOT/'training/dataset'
    images = []
    for record in entries:
        path = (data/record['path']).resolve()
        if not path.is_relative_to(data.resolve()) or digest(path) != record['sha256']:
            raise ValueError(f'Unsafe or changed test image: {record["path"]}')
        with Image.open(path) as image:
            images.append(np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32))
    truth = np.array([classes.index(r['label']) for r in entries])
    keras_path = ROOT/'backend/models/tea_leaf_mobilenetv2.keras'
    onnx_path = ROOT/'frontend/public/model/tea-leaf.onnx'
    if digest(onnx_path) != browser['sha256']:
        raise ValueError('ONNX checksum differs from the published manifest')
    print(f'Verified {len(entries)} frozen test files, class order and split integrity.', flush=True)
    model = tf.keras.models.load_model(keras_path, compile=False)
    keras_blocks = []
    for offset in range(0, len(images), 16):
        keras_blocks.append(model(np.stack(images[offset:offset+16]), training=False).numpy())
    keras_probs = np.concatenate(keras_blocks)
    print('Keras predictions complete.', flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
    onnx_probs = np.concatenate([session.run(['probabilities'], {'images': im[None]})[0] for im in images])
    result = dict(evaluated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  model_version=browser['model_version'], classes=classes,
                  keras_sha256=digest(keras_path), onnx_sha256=digest(onnx_path),
                  dataset_manifest_sha256=digest(manifest),
                  checks=dict(test_images_sha256_verified=True, groups_disjoint=True,
                              exact_hashes_disjoint=True, class_order_verified=True,
                              retrained=False, split_changed=False),
                  preprocessing='Curated RGB JPEG -> Pillow bilinear resize 224x224 -> float32 raw 0–255; embedded Rescaling.',
                  test_counts=dict(collections.Counter(r['label'] for r in entries)),
                  keras=metrics(truth, keras_probs, classes),
                  onnx_native=metrics(truth, onnx_probs, classes),
                  parity=dict(top1_disagreements=int(np.count_nonzero(keras_probs.argmax(1) != onnx_probs.argmax(1))),
                              max_absolute_probability_error=float(np.max(np.abs(keras_probs-onnx_probs)))),
                  software_versions=dict(tensorflow=tf.__version__, onnxruntime=ort.__version__),
                  limitations=[
                      'Same internal held-out source split used in development; repeated evaluation is not an independent field study.',
                      'Groups approximate image similarity; not all original plant or session identities are known.',
                      'Labels are inherited from source datasets, not independently diagnosed for every image.',
                      'Canvas/browser resizing is evaluated separately; different preprocessing can change predictions.'
                  ])
    original = np.load(ROOT/'training/reports/test_predictions.npz')
    if not np.array_equal(original['truth'], truth):
        raise ValueError('Test ordering differs from the original evaluation')
    result['agreement_with_original_evaluation'] = dict(
        top1_disagreements=int(np.count_nonzero(original['probabilities'].argmax(1) != keras_probs.argmax(1))),
        max_absolute_probability_error=float(np.max(np.abs(original['probabilities']-keras_probs))))
    result['elapsed_seconds'] = time.monotonic()-started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'native.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(args.output/'native-predictions.npz', truth=truth, keras=keras_probs, onnx=onnx_probs)
    with (args.output/'native-predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'path', 'source', 'actual', 'keras_prediction', 'onnx_prediction', 'correct', 'sha256'])
        for i, record in enumerate(entries):
            k, o = int(keras_probs[i].argmax()), int(onnx_probs[i].argmax())
            writer.writerow([record['id'], record['path'], record['source'], record['label'], classes[k], classes[o], k == truth[i], record['sha256']])
    print(json.dumps({k:result[k] for k in ['keras', 'parity', 'agreement_with_original_evaluation', 'elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
