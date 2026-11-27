"""Export versioned comparison checkpoints to ONNX with held-out parity evidence."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache/matplotlib'))
import argparse
import hashlib
import json
import sys
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.compare_models import CaffePreprocessing, CLASSES, MODELS  # Registers serializable preprocessing.
from training.export_metadata import (sha_file, read_json, load_dataset, evidence_metadata,
                                      bind_checkpoint_evaluation, select_test_records, parity_check, validate_evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'frontend/public/model/comparison')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'training/dataset')
    parser.add_argument('--manifest', type=Path, help='Defaults to DATA/manifest.json; v1 remains the default DATA.')
    parser.add_argument('--version-label', help='Version label, e.g. frozen-dataset-v2-20261126.')
    parser.add_argument('--evaluate-test', action='store_true', help='Verify every final held-out image, not only representative samples.')
    args = parser.parse_args()
    protocol = read_json(args.run / 'protocol.json')
    config = protocol['config']
    evidence = load_dataset(args.data, CLASSES, args.manifest, config['dataset_manifest_sha256'])
    selection = ([r for r in evidence['records'] if r['split'] == 'test'] if args.evaluate_test
                 else select_test_records(evidence, CLASSES))
    args.output.mkdir(parents=True, exist_ok=True); args.report.mkdir(parents=True, exist_ok=True)
    exported = []
    date = None
    for name in MODELS:
        tf.keras.backend.clear_session()
        checkpoint = args.run / name / 'model.keras'
        evaluation = bind_checkpoint_evaluation(read_json(args.run / name / 'evaluation.json'), sha_file(checkpoint))
        native = validate_evaluation(evaluation, CLASSES, evidence['split_totals']['test'])
        if evaluation.get('dataset_manifest_sha256', config['dataset_manifest_sha256']) != config['dataset_manifest_sha256']:
            raise ValueError('Checkpoint evaluation uses a different dataset version.')
        model = tf.keras.models.load_model(checkpoint, compile=False)
        if tuple(model.input_shape[1:]) != (224, 224, 3) or model.output_shape[-1] != len(CLASSES):
            raise ValueError('Checkpoint input/output shapes differ from the browser contract.')
        key = name.lower(); target = args.output / (key + '.onnx')

        @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name='images')])
        def inference(images):
            return {'probabilities': model(images, training=False)}

        print('Exporting', name, flush=True)
        tf2onnx.convert.from_function(inference, input_signature=inference.input_signature,
                                     opset=17, output_path=str(target))
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
        parity, predictions = parity_check(model, session, evidence, selection, CLASSES, threshold=1e-4)
        if args.evaluate_test:
            if (parity['keras_test']['correct'] != native['correct'] or
                    parity['keras_test']['confusion_matrix'] != evaluation['confusion_matrix']):
                raise ValueError('Keras final-test predictions differ from the recorded checkpoint evaluation.')
        if target.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError('Model exceeds GitHub single-file limit.')
        date = str(evaluation.get('completed_at', evaluation.get('trained_at', 'undated'))).split('T')[0].replace('-', '')
        label = args.version_label or f'frozen-{date}'
        item = dict(id=key, name=name, file=target.name, format='onnx',
                    model_version=f'{key}-comparison-{label}-{evaluation["model_sha256"][:10]}',
                    input_name=session.get_inputs()[0].name, output_name=session.get_outputs()[0].name,
                    input_shape=[1, 224, 224, 3], size_bytes=target.stat().st_size,
                    sha256=sha_file(target), keras_sha256=evaluation['model_sha256'],
                    total_parameters=evaluation['total_parameters'],
                    **{k: parity[k] for k in ['parity_samples', 'parity_max_absolute_error', 'parity_top1_disagreements']},
                    native_test=native)
        if args.evaluate_test:
            item['onnx_test'] = {k: parity['onnx_test'][k] for k in ['accuracy', 'macro_f1', 'correct', 'test_images']}
            np.savez_compressed(args.report / (key + '-onnx-test-predictions.npz'), **predictions)
        exported.append(item)
        (args.report / (key + '-export-parity.json')).write_text(json.dumps({
            'model': item, 'dataset_manifest_sha256': evidence['dataset_manifest_sha256'],
            'full_test': args.evaluate_test, **parity}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(name, 'bytes:', item['size_bytes'], 'max parity error:', parity['parity_max_absolute_error'], flush=True)
        del session, model
    signature = hashlib.sha256(''.join(item['sha256'] for item in exported).encode()).hexdigest()[:16]
    label = args.version_label or f'frozen-{date}'
    catalog = dict(protocol_id=f'comparison-{label}-seed{config["seed"]}-{signature}', classes=CLASSES,
                   **evidence_metadata(evidence),
                   training='Official ImageNet backbone frozen; identical head and budget; same group70/15/15 split.',
                   training_protocol=config,
                   preprocessing='Raw RGB float32 0–255,224x224; architecture-specific normalization embedded in each model.',
                   models=exported)
    for path in [args.output / 'manifest.json', args.report / 'browser-model-catalog.json']:
        path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('All three models exported and verified.', flush=True)


if __name__ == '__main__':
    main()
