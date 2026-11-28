"""Export TeaLeafAI Keras to ONNX with versioned training evidence and parity checks."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache/matplotlib'))
import argparse
import json
import sys
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.export_metadata import (sha_file, read_json, load_dataset, evidence_metadata,
                                      bind_checkpoint_evaluation, select_test_records, parity_check, validate_evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'backend/models/tea_leaf_mobilenetv2.keras')
    parser.add_argument('--classes', type=Path, default=ROOT / 'backend/models/classes.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'frontend/public/model')
    parser.add_argument('--evaluation', type=Path, help='Evaluation bound to this checkpoint and dataset.')
    parser.add_argument('--data', type=Path, help='Versioned dataset used to train/evaluate this checkpoint.')
    parser.add_argument('--manifest', type=Path, help='Manifest override; defaults to DATA/manifest.json.')
    parser.add_argument('--report', type=Path, help='Directory for parity JSON and final-test predictions.')
    parser.add_argument('--model-version', help='Explicit deployment version, otherwise derived from evidence/hash.')
    parser.add_argument('--browser-resize', choices=['canvas_legacy', 'pillow_bilinear_v1'], default='canvas_legacy',
                        help='Declare the browser resize contract; does not change native evaluation or ONNX input tensors.')
    parser.add_argument('--evaluate-test', action='store_true', help='Verify Keras/ONNX on every held-out test image.')
    args = parser.parse_args()
    if args.evaluate_test and (not args.data or not args.evaluation or not args.report):
        parser.error('--evaluate-test requires --data, --evaluation and --report.')
    if args.manifest and not args.data:
        parser.error('--manifest requires --data.')
    classes = read_json(args.classes)
    if not isinstance(classes, list) or len(set(classes)) != len(classes):
        raise ValueError('Class labels must be an ordered list without duplicates.')
    checkpoint_sha = sha_file(args.model)
    evaluation = (bind_checkpoint_evaluation(read_json(args.evaluation), checkpoint_sha)
                  if args.evaluation else None)
    evidence = None
    metadata = {}
    if args.data:
        evidence = load_dataset(args.data, classes, args.manifest,
                                evaluation.get('dataset_manifest_sha256') if evaluation is not None else None)
        metadata = evidence_metadata(evidence, evaluation)
    elif evaluation is not None:
        metrics = validate_evaluation(evaluation, classes)
        metadata.update(test_accuracy=metrics['accuracy'], test_macro_f1=metrics['macro_f1'],
                        test_images=metrics['test_images'],
                        trained_at=evaluation.get('trained_at', evaluation.get('completed_at')),
                        dataset_manifest_sha256=evaluation.get('dataset_manifest_sha256'))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report:
        args.report.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(args.model, compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3) or model.output_shape[-1] != len(classes):
        raise ValueError('Checkpoint input/output shapes differ from the browser contract.')

    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name='images')])
    def inference(images):
        return {'probabilities': model(images, training=False)}

    onnx_path = args.output / 'tea-leaf.onnx'
    tf2onnx.convert.from_function(inference, input_signature=inference.input_signature,
                                 opset=17, output_path=str(onnx_path))
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    if evidence:
        selection = ([r for r in evidence['records'] if r['split'] == 'test'] if args.evaluate_test
                     else select_test_records(evidence, classes))
        parity, predictions = parity_check(model, session, evidence, selection, classes)
        if args.evaluate_test:
            expected = validate_evaluation(evaluation, classes, len(selection))
            if (parity['keras_test']['correct'] != expected['correct'] or
                    parity['keras_test']['confusion_matrix'] != evaluation['confusion_matrix']):
                raise ValueError('Keras final-test predictions differ from the recorded checkpoint evaluation.')
            metadata.update(onnx_test_accuracy=parity['onnx_test']['accuracy'],
                            onnx_test_macro_f1=parity['onnx_test']['macro_f1'])
            np.savez_compressed(args.report / 'onnx-test-predictions.npz', **predictions)
        if args.report:
            (args.report / 'onnx-export-parity.json').write_text(json.dumps({
                'keras_sha256': checkpoint_sha, 'onnx_sha256': sha_file(onnx_path),
                'dataset_manifest_sha256': evidence['dataset_manifest_sha256'],
                'full_test': args.evaluate_test, **parity}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        # Retain the original lightweight no-dataset export workflow.
        inputs = [np.random.default_rng(42).uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)]
        source = ROOT / 'DuLieuGoc'
        for folder in sorted(source.iterdir()) if source.exists() else []:
            if folder.is_dir():
                sample = next((p for p in folder.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), None)
                if sample:
                    with Image.open(sample) as image:
                        inputs.append(np.asarray(image.convert('RGB').resize((224, 224)), dtype=np.float32)[None])
        maximum = 0.; disagreements = 0
        for array in inputs:
            expected = model(array, training=False).numpy()
            actual = session.run(None, {session.get_inputs()[0].name: array})[0]
            maximum = max(maximum, float(np.max(np.abs(expected - actual))))
            disagreements += int(expected.argmax() != actual.argmax())
        if maximum >= 1e-4 or disagreements:
            raise ValueError('ONNX parity failed.')
        parity = dict(parity_samples=len(inputs), parity_max_absolute_error=maximum,
                      parity_top1_disagreements=disagreements)
    version = args.model_version
    if not version and evaluation is not None:
        date = str(evaluation.get('trained_at', evaluation.get('completed_at', 'undated'))).split('T')[0].replace('-', '')
        version = f"mobilenetv2-{metadata.get('dataset_version', 'export')}-{date}-{checkpoint_sha[:10]}"
    if version and args.browser_resize == 'pillow_bilinear_v1':
        version += '-bilinear-v1'
    manifest = dict(format='onnx', file=onnx_path.name, input_name=session.get_inputs()[0].name,
                    output_name=session.get_outputs()[0].name, input_shape=[1, 224, 224, 3],
                    pixel_range=[0, 255], normalization='Embedded in model; do not normalize input twice.',
                    classes=classes, sha256=sha_file(onnx_path), keras_sha256=checkpoint_sha,
                    size_bytes=onnx_path.stat().st_size,
                    browser_resize=args.browser_resize,
                    **{k: parity[k] for k in ['parity_samples', 'parity_max_absolute_error', 'parity_top1_disagreements']},
                    **metadata)
    if version:
        manifest['model_version'] = version
    if evaluation and evaluation.get('inference_protocol'):
        method = evaluation['inference_protocol']['selected']
        manifest.update(inference_method=method, inference_views=4 if method == 'four_flip_mean' else 1,
                        base_keras_sha256=evaluation['inference_protocol']['base_model_sha256'])
    (args.output / 'classes.json').write_text(json.dumps(classes, indent=2) + '\n', encoding='utf-8')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
