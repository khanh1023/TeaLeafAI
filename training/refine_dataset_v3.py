"""Continue a validation-selected v3 checkpoint on exactly the same frozen split.

The preliminary v3 internal test was already reported. This additional run keeps
it unchanged and uses only validation loss for checkpoint/epoch selection.
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
    p.add_argument('--initial', type=Path, required=True, help='Directory with checkpoint and evaluation.json.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--layers', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=5e-6)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new output directory; existing runs are preserved.')
    initial = json.loads((args.initial / 'evaluation.json').read_text())
    records = json.loads((args.data / 'manifest.json').read_text())
    classes = json.loads((args.data / 'classes.json').read_text())
    assert initial['dataset_manifest_sha256'] == digest(args.data / 'manifest.json')
    assert initial['model_sha256'] == digest(args.initial / 'tea_leaf_mobilenetv2.keras')
    assert classes == initial['classes']
    args.output.mkdir(parents=True); started = time.time()
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.models.load_model(args.initial / 'tea_leaf_mobilenetv2.keras', compile=False)
    backbone = next(layer for layer in model.layers if layer.name.startswith('mobilenetv2'))
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-args.layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical', seed=42),
        tf.keras.layers.RandomRotation(.045, seed=43, fill_mode='reflect'),
        tf.keras.layers.RandomContrast(.12, seed=44),
        tf.keras.layers.RandomBrightness(.10, value_range=(0, 255), seed=45),
    ])
    options = tf.data.Options(); options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1

    def read_pillow(path):
        value = path.item() if isinstance(path, np.ndarray) else path
        with Image.open(value.decode()) as image:
            return np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)

    def load(path, label):
        image = tf.numpy_function(read_pillow, [path], tf.float32)
        image.set_shape((224, 224, 3)); return image, label

    def split(name, train=False):
        subset = [r for r in records if r['split'] == name]
        paths = [str(args.data / r['path']) for r in subset]
        labels = [classes.index(r['label']) for r in subset]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(options)
        if train:
            ds = ds.shuffle(len(subset), seed=42)
        ds = ds.map(load, num_parallel_calls=2).batch(16)
        if train:
            ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=1)
        return ds.prefetch(1)

    train = split('train', True); val = split('val')
    weights = {classes.index(c): float(v) for c, v in initial['training']['class_weights'].items()}
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    initial_loss, initial_accuracy = [float(v) for v in model.evaluate(val, verbose=0)]
    best = args.output / 'tea_leaf_mobilenetv2.keras'; model.save(best)
    protocol = dict(dataset_manifest_sha256=digest(args.data / 'manifest.json'),
                    initial_model_sha256=initial['model_sha256'], classes=classes, seed=42,
                    maximum_epochs=args.epochs, last_layers=args.layers, batch_size=16,
                    learning_rate=args.learning_rate, class_weight_exponent=.5,
                    class_weights=initial['training']['class_weights'],
                    initial_validation_loss=initial_loss, initial_validation_accuracy=initial_accuracy,
                    selection='Minimum unweighted validation cross-entropy; initial checkpoint retained unless validation improves. Test not used for checkpoint selection.',
                    previous_internal_test_already_evaluated=True,
                    motivation='Validation learning curve was still improving near the original epoch budget; increase fine-tuning depth and budget while preserving all partitions.',
                    training_script_sha256=digest(Path(__file__)),
                    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best, monitor='val_loss', save_best_only=True, initial_value_threshold=initial_loss),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=1e-4),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(str(args.output / 'refinement-epochs.csv')),
    ]
    trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print('Initial validation:', initial_loss, initial_accuracy, 'trainable parameters:', trainable, flush=True)
    history = model.fit(train, validation_data=val, epochs=args.epochs, class_weight=weights,
                        callbacks=callbacks, shuffle=False, verbose=2)
    model = tf.keras.models.load_model(best, compile=False)
    selected_loss = min(initial_loss, min(history.history['val_loss']))
    selected = 'refinement' if selected_loss < initial_loss else 'initial_v3'
    # The checkpoint is now fixed. No more fitting or choice after test inference.
    entries = [r for r in records if r['split'] == 'test']
    forward = tf.function(lambda x: model(x, training=False), input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    probabilities = []
    for offset in range(0, len(entries), 32):
        arrays = []
        for r in entries[offset:offset+32]:
            with Image.open(args.data / r['path']) as image:
                arrays.append(np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32))
        probabilities.append(forward(np.stack(arrays)).numpy())
    probabilities = np.concatenate(probabilities); truth = np.array([classes.index(r['label']) for r in entries])
    pred = probabilities.argmax(1)
    report = dict(initial)
    report['training'] = dict(initial['training'])
    report['training'].update(selected=selected, selected_validation_loss=selected_loss,
                              selection=protocol['selection'], refinement_protocol=protocol,
                              stages=initial['training']['stages'] + [dict(stage='refinement', maximum_epochs=args.epochs,
                                  epochs_completed=len(history.history['loss']), trainable_parameters=trainable,
                                  learning_rate_initial=args.learning_rate, elapsed_seconds=time.time()-started,
                                  history={k: [float(v) for v in values] for k, values in history.history.items()})])
    report.update(model_sha256=digest(best), accuracy=float(accuracy_score(truth, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                  macro_f1=float(f1_score(truth, pred, average='macro')), weighted_f1=float(f1_score(truth, pred, average='weighted')),
                  classification_report=classification_report(truth, pred, target_names=classes, labels=list(range(len(classes))), output_dict=True, zero_division=0),
                  confusion_matrix=confusion_matrix(truth, pred, labels=list(range(len(classes)))).tolist(),
                  trained_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  elapsed_seconds=initial['elapsed_seconds']+time.time()-started,
                  previous_internal_test_metrics={k: initial[k] for k in ['accuracy', 'balanced_accuracy', 'macro_f1', 'test_images']},
                  target_met=bool(accuracy_score(truth, pred) > .90))
    report['limitations'] = initial['limitations'] + [
        'A preliminary v3 internal test was evaluated before this continuation. The frozen test is reused for development reporting; it is not a new blind test. Further checkpoint selection used validation loss only.']
    (args.output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output / 'classes.json').write_text(json.dumps(classes, indent=2))
    np.savez_compressed(args.output / 'test_predictions.npz', truth=truth, probabilities=probabilities,
                        image_ids=np.array([r['id'] for r in entries]))
    print(json.dumps({k: report[k] for k in ['accuracy', 'balanced_accuracy', 'macro_f1', 'test_images', 'target_met']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
