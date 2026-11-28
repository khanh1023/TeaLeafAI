"""Choose single-view vs four-flip averaging on validation, then evaluate test.

The alternatives and minimum-loss selection rule are fixed before this run.
Flips are inference views of the same photo, never additional dataset samples.
The saved four-view Keras model reuses one backbone; the browser can execute
the exported graph without a separate preprocessing implementation.
"""
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache/matplotlib'))

import argparse
import datetime
import hashlib
import json
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'training/dataset-v3')
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new directory; no results are overwritten.')
    report = json.loads((args.initial / 'evaluation.json').read_text())
    records = json.loads((args.data / 'manifest.json').read_text())
    classes = json.loads((args.data / 'classes.json').read_text())
    assert report['dataset_manifest_sha256'] == digest(args.data / 'manifest.json')
    assert report['model_sha256'] == digest(args.initial / 'tea_leaf_mobilenetv2.keras')
    assert report['classes'] == classes
    args.output.mkdir(parents=True); started = time.time()
    model = tf.keras.models.load_model(args.initial / 'tea_leaf_mobilenetv2.keras', compile=False)
    forward = tf.function(lambda x: model(x, training=False), input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])

    def arrays(entries):
        block = []
        for r in entries:
            with Image.open(args.data / r['path']) as image:
                block.append(np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32))
        return np.stack(block)

    validation = [r for r in records if r['split'] == 'val']
    truth_val = np.array([classes.index(r['label']) for r in validation])
    probabilities = [[], [], [], []]
    for offset in range(0, len(validation), 32):
        original = arrays(validation[offset:offset+32])
        views = [original, original[:, :, ::-1, :], original[:, ::-1, :, :], original[:, ::-1, ::-1, :]]
        for i, view in enumerate(views):
            probabilities[i].append(forward(np.ascontiguousarray(view)).numpy())
    views = [np.concatenate(blocks) for blocks in probabilities]
    single, average = views[0], np.mean(views, axis=0)
    for candidate in [single, average]:
        assert candidate.shape == (len(validation), len(classes))
        assert np.isfinite(candidate).all() and (candidate >= 0).all()
        assert np.max(np.abs(candidate.sum(1) - 1)) < 1e-4

    def val_metrics(probs):
        return dict(loss=float(-np.log(np.maximum(probs[np.arange(len(truth_val)), truth_val], 1e-7)).mean()),
                    accuracy=float(accuracy_score(truth_val, probs.argmax(1))),
                    macro_f1=float(f1_score(truth_val, probs.argmax(1), average='macro')))

    candidates = {'single_view': val_metrics(single), 'four_flip_mean': val_metrics(average)}
    selected = 'four_flip_mean' if candidates['four_flip_mean']['loss'] < candidates['single_view']['loss'] - 1e-4 else 'single_view'
    protocol = dict(dataset_manifest_sha256=report['dataset_manifest_sha256'],
                    base_model_sha256=report['model_sha256'], selected=selected, validation_images=len(validation),
                    candidates=candidates, views=['original', 'horizontal_flip', 'vertical_flip', 'both_flips'],
                    selection='Choose four-flip arithmetic probability mean only if validation cross-entropy improves by >1e-4; otherwise single view. No test input used for this selection.',
                    dataset_sample_count_unchanged=True, script_sha256=digest(Path(__file__)),
                    selected_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    # Record the choice before any held-out test inference.
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    np.savez_compressed(args.output / 'validation-candidates.npz', truth=truth_val, single_view=single, four_flip_mean=average,
                        image_ids=np.array([r['id'] for r in validation]))
    if selected == 'four_flip_mean':
        raw = tf.keras.Input((224, 224, 3), name='images')
        outputs = [model(raw, training=False)]
        for axes in [(2,), (1,), (1, 2)]:
            outputs.append(model(tf.keras.ops.flip(raw, axis=axes), training=False))
        model = tf.keras.Model(raw, tf.keras.layers.Average(name='flip_probability_mean')(outputs), name='tea_leaf_mobilenetv2_flip_mean')
    best = args.output / 'tea_leaf_mobilenetv2.keras'; model.save(best)
    # Reload to verify that no custom-object registration is needed by export/backend.
    model = tf.keras.models.load_model(best, compile=False)
    forward = tf.function(lambda x: model(x, training=False), input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    test = [r for r in records if r['split'] == 'test']
    truth = np.array([classes.index(r['label']) for r in test]); blocks = []
    for offset in range(0, len(test), 32):
        blocks.append(forward(arrays(test[offset:offset+32])).numpy())
    probs = np.concatenate(blocks); pred = probs.argmax(1)
    old_metrics = {k: report[k] for k in ['accuracy', 'balanced_accuracy', 'macro_f1', 'test_images']}
    report = dict(report)
    report['inference_protocol'] = protocol
    report['base_checkpoint_test_metrics'] = old_metrics
    report.update(model_sha256=digest(best), accuracy=float(accuracy_score(truth, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                  macro_f1=float(f1_score(truth, pred, average='macro')), weighted_f1=float(f1_score(truth, pred, average='weighted')),
                  classification_report=classification_report(truth, pred, target_names=classes, labels=list(range(len(classes))), output_dict=True, zero_division=0),
                  confusion_matrix=confusion_matrix(truth, pred, labels=list(range(len(classes)))).tolist(),
                  target_met=bool(accuracy_score(truth, pred) > .90),
                  elapsed_seconds=report['elapsed_seconds']+time.time()-started,
                  trained_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['limitations'] = report['limitations'] + [
        'Single-view/four-flip inference was selected using validation only. All flip views reuse the same source photo and are not independent test observations.',
        'The development test has already been reported for preceding checkpoints. These are internal development results, not a fresh blind test.']
    (args.output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output / 'classes.json').write_text(json.dumps(classes, indent=2))
    np.savez_compressed(args.output / 'test_predictions.npz', truth=truth, probabilities=probs, image_ids=np.array([r['id'] for r in test]))
    print(json.dumps(dict(selected=selected, candidates=candidates, accuracy=report['accuracy'], macro_f1=report['macro_f1'], target_met=report['target_met']), indent=2), flush=True)


if __name__ == '__main__':
    main()
