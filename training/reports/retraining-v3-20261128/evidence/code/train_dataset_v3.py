"""Continue MobileNetV2 only when all previous held-out assignments are preserved.

Select checkpoints by validation loss, retaining the initial checkpoint unless
validation improves. Test inference is deferred until both training stages end.
The 90% user goal is not a stopping/selection criterion measured on test.
"""
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import argparse
import collections
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'training/dataset-v3')
    parser.add_argument('--previous-data', type=Path, default=ROOT / 'training/dataset-v2')
    parser.add_argument('--initial', type=Path, default=ROOT / 'backend/models/tea_leaf_mobilenetv2.keras')
    parser.add_argument('--initial-report', type=Path, default=ROOT / 'training/evaluation.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head-epochs', type=int, default=4)
    parser.add_argument('--fine-epochs', type=int, default=18)
    parser.add_argument('--fine-layers', type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new run directory; existing runs are never overwritten.')
    args.output.mkdir(parents=True)
    started = time.time(); tf.keras.utils.set_random_seed(42)
    records = json.loads((args.data / 'manifest.json').read_text())
    previous = json.loads((args.previous_data / 'manifest.json').read_text())
    classes = json.loads((args.data / 'classes.json').read_text())
    summary = json.loads((args.data / 'split_summary.json').read_text())
    initial_report = json.loads(args.initial_report.read_text())
    assert digest(args.previous_data / 'manifest.json') == initial_report['dataset_manifest_sha256']
    assert summary['previous_manifest_sha256'] == initial_report['dataset_manifest_sha256']
    assert digest(args.initial) == initial_report['model_sha256']
    assert initial_report['classes'] == classes
    lookup = {r['id']: r for r in records}
    for old in previous:
        current = lookup[old['id']]
        for field in ['sha256', 'raw_sha256', 'label', 'split']:
            assert current[field] == old[field], (old['id'], field)
    assert len(records) == summary['total_images']
    # This permits warm starts only because no old validation/test image has
    # moved into train. It is still internal development evaluation.
    model = tf.keras.models.load_model(args.initial, compile=False)
    backbone = next(layer for layer in model.layers if layer.name.startswith('mobilenetv2'))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical', seed=42),
        tf.keras.layers.RandomRotation(.045, seed=43, fill_mode='reflect'),
        tf.keras.layers.RandomContrast(.12, seed=44),
        tf.keras.layers.RandomBrightness(.10, value_range=(0, 255), seed=45),
    ])

    def read_pillow(path):
        value = path.item() if isinstance(path, np.ndarray) else path
        with Image.open(value.decode()) as image:
            return np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)

    def load(path, label):
        image = tf.numpy_function(read_pillow, [path], tf.float32)
        image.set_shape((224, 224, 3))
        return image, label

    def split(name, training=False):
        entries = [r for r in records if r['split'] == name]
        paths = [str(args.data / r['path']) for r in entries]
        labels = [classes.index(r['label']) for r in entries]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load, num_parallel_calls=2).with_options(options)
        if training:
            dataset = dataset.shuffle(len(entries), seed=42)
        dataset = dataset.batch(16)
        if training:
            dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=1)
        return dataset.prefetch(1)

    train = split('train', True); val = split('val')
    train_counts = collections.Counter(r['label'] for r in records if r['split'] == 'train')
    total = sum(train_counts.values())
    weights = {i: (total / (len(classes) * train_counts[c])) ** .5 for i, c in enumerate(classes)}
    normalizer = sum(train_counts[c] * weights[i] for i, c in enumerate(classes)) / total
    weights = {i: value / normalizer for i, value in weights.items()}
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    base_loss, base_accuracy = [float(v) for v in model.evaluate(val, verbose=0)]
    best = args.output / 'tea_leaf_mobilenetv2.keras'
    model.save(best)
    protocol = dict(dataset_manifest_sha256=digest(args.data / 'manifest.json'),
                    previous_manifest_sha256=digest(args.previous_data / 'manifest.json'),
                    initial_model_sha256=digest(args.initial), previous_assignments_verified=len(previous),
                    classes=classes, seed=42, head_maximum_epochs=args.head_epochs,
                    fine_maximum_epochs=args.fine_epochs, fine_last_layers=args.fine_layers,
                    class_weight_exponent=.5, class_weights={classes[i]: v for i, v in weights.items()},
                    selection='Minimum unweighted validation cross-entropy across initial, head adaptation and fine-tuning; test unused for selection.',
                    initial_validation_loss=base_loss, initial_validation_accuracy=base_accuracy,
                    training_script_sha256=digest(Path(__file__)), created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('Initial validation:', base_loss, base_accuracy, 'Class weights:', weights, flush=True)
    histories = []; global_best_loss = base_loss; selected = 'initial_v2'
    for stage, maximum, learning_rate in [('head_adaptation', args.head_epochs, 1e-4), ('fine_tuning', args.fine_epochs, 1e-5)]:
        model = tf.keras.models.load_model(best, compile=False)
        backbone = next(layer for layer in model.layers if layer.name.startswith('mobilenetv2'))
        backbone.trainable = stage == 'fine_tuning'
        if backbone.trainable:
            for layer in backbone.layers:
                layer.trainable = False
            for layer in backbone.layers[-args.fine_layers:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(best, monitor='val_loss', save_best_only=True, initial_value_threshold=global_best_loss),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4 if stage == 'fine_tuning' else 2, min_delta=1e-4),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=2, min_lr=2e-7),
            tf.keras.callbacks.CSVLogger(str(args.output / f'{stage}-epochs.csv')),
        ]
        print('Training stage', stage, 'trainable parameters', trainable, flush=True)
        stage_started = time.time()
        history = model.fit(train, validation_data=val, epochs=maximum, class_weight=weights, callbacks=callbacks, verbose=2)
        stage_loss = min(history.history['val_loss'])
        if stage_loss < global_best_loss:
            global_best_loss = float(stage_loss); selected = stage
        histories.append(dict(stage=stage, maximum_epochs=maximum, epochs_completed=len(history.history['loss']),
                              learning_rate_initial=learning_rate, trainable_parameters=trainable,
                              elapsed_seconds=time.time()-stage_started,
                              history={k: [float(v) for v in values] for k, values in history.history.items()}))
        (args.output / 'training-progress.json').write_text(json.dumps(dict(selected=selected, selected_validation_loss=global_best_loss, stages=histories), indent=2))
    # No further training or choice based on the following held-out predictions.
    model = tf.keras.models.load_model(best, compile=False)
    entries = [r for r in records if r['split'] == 'test']
    forward = tf.function(lambda x: model(x, training=False), input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    probabilities = []
    for offset in range(0, len(entries), 32):
        arrays = []
        for r in entries[offset:offset+32]:
            with Image.open(args.data / r['path']) as image:
                arrays.append(np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32))
        probabilities.append(forward(np.stack(arrays)).numpy())
    probabilities = np.concatenate(probabilities)
    truth = np.array([classes.index(r['label']) for r in entries]); pred = probabilities.argmax(1)
    report = dict(model='MobileNetV2', num_classes=len(classes), classes=classes,
                  model_sha256=digest(best), dataset_manifest_sha256=protocol['dataset_manifest_sha256'], dataset=summary,
                  accuracy=float(accuracy_score(truth, pred)), balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                  macro_f1=float(f1_score(truth, pred, average='macro')), weighted_f1=float(f1_score(truth, pred, average='weighted')),
                  classification_report=classification_report(truth, pred, target_names=classes, labels=list(range(len(classes))), output_dict=True, zero_division=0),
                  confusion_matrix=confusion_matrix(truth, pred, labels=list(range(len(classes)))).tolist(),
                  test_images=len(truth), training=dict(protocol, selected=selected, selected_validation_loss=global_best_loss, stages=histories),
                  elapsed_seconds=time.time()-started, trained_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  target_accuracy=.90, target_met=bool(accuracy_score(truth, pred) > .90),
                  limitations=summary['limitations']+['The warm-start v2 model was previously selected on the preserved v2 validation partition; expanded test results are internal development evidence.',
                                                       'More images and a 90% goal do not imply 90% measured accuracy. Use recorded predictions and per-class metrics.'])
    (args.output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output / 'classes.json').write_text(json.dumps(classes, indent=2))
    np.savez_compressed(args.output / 'test_predictions.npz', truth=truth, probabilities=probabilities,
                        image_ids=np.array([r['id'] for r in entries]))
    print(json.dumps({k: report[k] for k in ['accuracy', 'balanced_accuracy', 'macro_f1', 'test_images', 'target_met', 'elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
