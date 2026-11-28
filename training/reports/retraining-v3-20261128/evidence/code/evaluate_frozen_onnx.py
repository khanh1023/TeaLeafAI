"""Evaluate a fixed ONNX artifact on a versioned test split, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--metadata', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new output directory.')
    metadata = json.loads(args.metadata.read_text())
    classes = json.loads((args.data / 'classes.json').read_text())
    records = json.loads((args.data / 'manifest.json').read_text())
    assert classes == metadata['classes']
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == metadata['sha256']
    entries = [r for r in records if r['split'] == 'test']
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
    probabilities = []
    for r in entries:
        content = (args.data / r['path']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == r['sha256']
        with Image.open(args.data / r['path']) as image:
            x = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)[None]
        probabilities.append(session.run(None, {metadata['input_name']:x})[0][0])
    probs = np.array(probabilities); truth = np.array([classes.index(r['label']) for r in entries]); pred = probs.argmax(1)
    assert np.isfinite(probs).all() and (probs >= 0).all()
    assert np.max(np.abs(probs.sum(1) - 1)) < 1e-4
    report = dict(model_version=metadata['model_version'], onnx_sha256=metadata['sha256'],
                  training_dataset_manifest_sha256=metadata['dataset_manifest_sha256'],
                  dataset_manifest_sha256=hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
                  evaluation_split='test', test_images=len(entries), correct=int((truth == pred).sum()),
                  accuracy=float(accuracy_score(truth,pred)), balanced_accuracy=float(balanced_accuracy_score(truth,pred)),
                  macro_f1=float(f1_score(truth,pred,average='macro')),
                  classification_report=classification_report(truth,pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0),
                  confusion_matrix=confusion_matrix(truth,pred,labels=list(range(len(classes)))).tolist(),
                  preprocessing='Pillow RGB bilinear 224x224, raw 0-255, native ONNX Runtime CPU',
                  scope='Fixed-checkpoint internal test evaluation; no training, checkpoint or inference-protocol selection.')
    args.output.mkdir(parents=True)
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output/'predictions.npz',truth=truth,probabilities=probs,image_ids=np.array([r['id'] for r in entries]))
    print(json.dumps({k:report[k] for k in ['correct','test_images','accuracy','macro_f1']},indent=2))


if __name__ == '__main__':
    main()
