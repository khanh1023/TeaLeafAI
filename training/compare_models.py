"""Controlled frozen-backbone comparison; use the companion Colab notebook for a free GPU.

This is a NEW experiment, not an evaluation of the deployed fine-tuned checkpoint.
All models use the same fixed split, augmentation, head, budget and validation rule.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.cache/matplotlib'))
os.environ.setdefault('KERAS_HOME', str(Path(__file__).resolve().parent/'.cache/keras'))
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, __version__ as pillow_version
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = '585cabd7be1741675fa5bed31c14116b9f72f7eca11d40610313ddb15cd48707'
CLASSES = ['algal_leaf', 'anthracnose', 'bird_eye_spot', 'brown_blight',
           'gray_light', 'healthy', 'red_leaf_spot', 'white_spot']
MODELS = ['MobileNetV2', 'ResNet50', 'VGG16']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_dataset(data, manifest_path, expected_sha=MANIFEST_SHA):
    if expected_sha is not None and sha(manifest_path) != expected_sha:
        raise ValueError('Manifest differs from the frozen 70/15/15 split.')
    records = json.loads(Path(manifest_path).read_text())
    seen_paths = set()
    for field in ['group', 'sha256', 'raw_sha256', 'pixel_sha256']:
        groups = collections.defaultdict(set)
        for r in records:
            groups[r[field]].add(r['split'])
        if any(len(v) != 1 for v in groups.values()):
            raise ValueError(f'Cross-split duplication: {field}')
    for r in records:
        path = (data / r['path']).resolve()
        if not path.is_relative_to(data.resolve()) or r['path'] in seen_paths:
            raise ValueError('Unsafe or duplicate dataset path.')
        seen_paths.add(r['path'])
        if r['label'] not in CLASSES or r['split'] not in ['train', 'val', 'test']:
            raise ValueError('Unknown label/split.')
        if sha(path) != r['sha256']:
            raise ValueError(f'Image checksum mismatch: {r["path"]}')
    return records


@tf.keras.utils.register_keras_serializable(package='TeaLeafAI')
class CaffePreprocessing(tf.keras.layers.Layer):
    """Official ResNet50/VGG16 ImageNet convention: RGB -> BGR -> mean subtraction."""
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        return tf.reverse(x, axis=[-1]) - tf.constant([103.939, 116.779, 123.68])


def build_extractor(name, weights='imagenet'):
    constructors = {'MobileNetV2': tf.keras.applications.MobileNetV2,
                    'ResNet50': tf.keras.applications.ResNet50,
                    'VGG16': tf.keras.applications.VGG16}
    base = constructors[name](include_top=False, weights=weights, input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input((224, 224, 3), name='images', dtype='float32')
    if name == 'MobileNetV2':
        x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name='preprocessing')(inputs)
    else:
        x = CaffePreprocessing(name='preprocessing')(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='features')(x)
    return tf.keras.Model(inputs, x, name=name + '_extractor')


def image_array(data, record, index, view, seed):
    with Image.open(data / record['path']) as source:
        image = source.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    if view:
        rng = random.Random(seed + index + view * 100000)
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        image = image.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR,
                             fillcolor=image.getpixel((0, 0)))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(.85, 1.15))
    return np.asarray(image, dtype=np.float32)


def extract(extractor, records, data, split, batch, views, seed):
    entries = [(i, r, view) for i, r in enumerate(records) if r['split'] == split
               for view in range(views + 1 if split == 'train' else 1)]
    blocks, targets = [], []
    for start in range(0, len(entries), batch):
        chunk = entries[start:start + batch]
        images = np.stack([image_array(data, r, i, v, seed) for i, r, v in chunk])
        blocks.append(extractor(images, training=False).numpy())
        targets.extend(CLASSES.index(r['label']) for _, r, _ in chunk)
        if start % (batch * 20) == 0:
            print(f'  {split}: {min(start + batch, len(entries))}/{len(entries)}', flush=True)
    return np.concatenate(blocks), np.asarray(targets, dtype=np.int32)


def build_head(dimension):
    inputs = tf.keras.Input((dimension,))
    x = tf.keras.layers.Dropout(.3)(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(.2)(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(8, activation='softmax')(x))


def metrics(truth, probabilities):
    pred = probabilities.argmax(1)
    report = classification_report(truth, pred, labels=list(range(8)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(truth, pred)),
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'test_images': len(truth), 'correct': int((pred == truth).sum()),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(truth, pred, labels=list(range(8))).tolist()}


def benchmark(model, data, records, seed, warmups=10, count=30, repeats=2):
    test = [(i, r) for i, r in enumerate(records) if r['split'] == 'test']
    indices = np.linspace(0, len(test) - 1, min(count, len(test)), dtype=int).tolist()
    tensors = [tf.convert_to_tensor(image_array(data, test[j][1], test[j][0], 0, seed)[None])
               for j in indices]
    # Same execution mode and dtype for all models; .numpy() synchronizes GPU output.
    for i in range(warmups):
        model(tensors[i % len(tensors)], training=False).numpy()
    times = []
    for _ in range(repeats):
        for tensor in tensors:
            start = time.perf_counter()
            model(tensor, training=False).numpy()
            times.append((time.perf_counter() - start) * 1000)
    return {'count': len(times), 'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95, method='higher')),
            'sample_ids': [test[j][1]['id'] for j in indices],
            'definition': 'Warm batch-1 TensorFlow eager full-model call, float32 raw RGB tensor, '
                          'with output .numpy() synchronization; excludes file reading, '
                          'Pillow resize, initial loading and web/browser/network operations.',
            'warmup_calls': warmups, 'repeats': repeats}


def run_one(name, records, data, dest, config):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config['seed'])
    # Official ImageNet weights download/build excluded from feature+head training timing.
    extractor = build_extractor(name)
    start = time.perf_counter()
    features, targets = {}, {}
    for split in ['train', 'val']:
        features[split], targets[split] = extract(extractor, records, data, split,
            config['feature_batch'], config['augmented_views'], config['seed'])
    extraction_seconds = time.perf_counter() - start
    head = build_head(features['train'].shape[1])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    counts = collections.Counter(targets['train'].tolist())
    weights = {k: len(targets['train']) / (8 * v) for k, v in counts.items()}
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    train = tf.data.Dataset.from_tensor_slices((features['train'], targets['train']))
    train = train.shuffle(len(targets['train']), seed=config['seed']).batch(32).with_options(options)
    val = tf.data.Dataset.from_tensor_slices((features['val'], targets['val'])).batch(64).with_options(options)
    checkpoint = dest / 'best_head.keras'
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-4),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_loss', save_best_only=True)]
    start = time.perf_counter()
    history = head.fit(train, validation_data=val, epochs=config['epochs'], class_weight=weights,
                       callbacks=callbacks, verbose=2)
    head_seconds = time.perf_counter() - start
    head = tf.keras.models.load_model(checkpoint, compile=False)
    model = tf.keras.Model(extractor.input, head(extractor.output), name='tea_' + name)
    model_path = dest / 'model.keras'
    model.save(model_path)  # complete inference model, no optimizer state
    # Test inference is performed only AFTER model/epoch selection is finished.
    test_features, truth = extract(extractor, records, data, 'test', config['feature_batch'], 0, config['seed'])
    probabilities = head(test_features, training=False).numpy()
    result = {'name': name, **metrics(truth, probabilities), 'model_bytes': model_path.stat().st_size,
        'model_sha256': sha(model_path), 'total_parameters': model.count_params(),
        'trainable_parameters': sum(int(np.prod(v.shape)) for v in model.trainable_weights),
        'selected_epoch': int(np.argmin(history.history['val_loss'])) + 1,
        'epochs_completed': len(history.history['loss']),
        'history': {k: [float(v) for v in values] for k, values in history.history.items()},
        'feature_extraction_train_val_seconds': extraction_seconds,
        'head_training_seconds': head_seconds,
        'feature_plus_head_seconds': extraction_seconds + head_seconds,
        'latency': benchmark(model, data, records, config['seed']),
        'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    test = [r for r in records if r['split'] == 'test']
    with (dest / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['id', 'actual', 'predicted', *CLASSES])
        for r, probs in zip(test, probabilities):
            writer.writerow([r['id'], r['label'], CLASSES[int(probs.argmax())], *probs.tolist()])
    np.savez_compressed(dest / 'predictions.npz', truth=truth, probabilities=probabilities,
                        ids=np.asarray([r['id'] for r in test]))
    (dest / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def export_report(output, results, metadata):
    """Export measured results only; incomplete comparisons cannot be presented as complete."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from docx import Document
    from docx.shared import Cm, Pt
    headers = ['Mô hình', 'Accuracy (%)', 'Precision macro (%)', 'Recall macro (%)',
               'F1 macro (%)', 'File Keras (MB)', 'Tham số', 'Warm median (ms/ảnh)']
    rows = [[r['name'], f'{r["accuracy"]*100:.2f}', f'{r["macro_precision"]*100:.2f}',
             f'{r["macro_recall"]*100:.2f}', f'{r["macro_f1"]*100:.2f}',
             f'{r["model_bytes"]/1e6:.2f}', str(r['total_parameters']),
             f'{r["latency"]["median_ms"]:.2f}'] for r in results]
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(headers); w.writerows(rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, key, title, scale in [(axes[0, 0], 'accuracy', 'Accuracy test (%)', 100),
            (axes[0, 1], 'macro_f1', 'F1 macro test (%)', 100),
            (axes[1, 0], 'model_bytes', 'Dung lượng Keras không gồm optimizer (MB)', 1e-6),
            (axes[1, 1], None, 'Suy luận warm batch 1 — trung vị (ms/ảnh)', 1)]:
        values = [r[key]*scale if key else r['latency']['median_ms'] for r in results]
        bars = ax.bar([r['name'] for r in results], values, color=['#226b50', '#537f9c', '#bb8d48'])
        ax.set_title(title, fontsize=11); ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, 110 if scale == 100 else max(values)*1.2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=11)
    fig.suptitle('SO SÁNH CHUYỂN GIAO — BACKBONE ĐÓNG BĂNG', fontsize=15, weight='bold')
    fig.text(.5, .025, 'Cùng bộ chia và quy trình; một seed. Đây là thí nghiệm mới, không phải điểm của website.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(.02, .07, .98, .94))
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(output / ('comparison.'+ext), dpi=300)
    plt.close(fig)
    doc = Document(); normal = doc.styles['Normal']; normal.font.name='Times New Roman'; normal.font.size=Pt(12)
    doc.add_heading('3.3.4. So sánh hiệu quả giữa các mô hình phân loại', level=1)
    doc.add_paragraph('Thí nghiệm so sánh MobileNetV2, ResNet50 và VGG16 sử dụng trọng số ImageNet, '
        'đóng băng toàn bộ backbone và huấn luyện bộ phân loại có cùng cấu trúc. Bộ chia gồm '
        '1.541 ảnh train, 327 ảnh validation và 329 ảnh test; ảnh tăng cường chỉ áp dụng cho train. '
        'Mỗi mô hình dùng cùng ngân sách tối đa 20 epoch, Adam, trọng số lớp và quy tắc chọn checkpoint '
        'theo validation loss. Chuẩn hóa được thực hiện theo yêu cầu riêng của từng backbone.')
    table = doc.add_table(rows=1, cols=4); table.style='Table Grid'
    for cell, text in zip(table.rows[0].cells, ['Tiêu chí', *[r['name'] for r in results]]): cell.text=text
    for index, title in enumerate(headers[1:], 1):
        for cell, text in zip(table.add_row().cells, [title, *[row[index] for row in rows]]): cell.text=text
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs: run.font.size=Pt(11)
    best = max(results, key=lambda r: r['macro_f1'])
    doc.add_paragraph(f'Trong lần chạy này, {best["name"]} đạt F1 macro cao nhất '
        f'({best["macro_f1"]*100:.2f}%). Việc lựa chọn mô hình cần cân nhắc đồng thời Accuracy, '
        'F1 macro, dung lượng và thời gian suy luận; không kết luận ưu thế tổng quát từ một seed.')
    doc.add_picture(str(output / 'comparison.png'), width=Cm(16))
    doc.add_paragraph('Dung lượng đo trên các file Keras suy luận cùng định dạng, không lưu trạng thái optimizer. '
        'Các backbone đều bỏ bộ phân loại ImageNet gốc (include_top=False). Thời gian là phép gọi TensorFlow '
        'eager batch1 sau warm-up trên cùng máy; không gồm đọc/resize ảnh và không đại diện tốc độ trình duyệt. '
        'Đây là thí nghiệm transfer learning đóng băng backbone, chưa đánh giá fine-tuning toàn mạng. '
        'Kết quả có thể khác checkpoint MobileNetV2 đã tinh chỉnh đang dùng trên website. '
        'Test là bộ nội bộ đã dùng trong phát triển, chưa phải kiểm định thực địa độc lập.')
    doc.add_paragraph('Nguồn: chạy training/compare_models.py của TeaLeafAI. '
        'Checksum dữ liệu, cấu hình, môi trường, lịch sử và dự đoán từng ảnh được lưu trong JSON/CSV/NPZ đi kèm.')
    doc.save(output / 'So-sanh-mo-hinh-muc-3-3-4.docx')
    (output / 'comparison.json').write_text(json.dumps({'metadata': metadata, 'results': results},
                                                       ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'training/dataset')
    parser.add_argument('--manifest', type=Path, default=ROOT/'training/reports/expanded-dataset-manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--feature-batch', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augmented-views', type=int, default=2)
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs != 20 or args.augmented_views != 2:
        parser.error('The report protocol is fixed at 20 maximum epochs and two augmented training views.')
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus and not args.allow_cpu:
        raise RuntimeError('No GPU. In Colab select Runtime > Change runtime type > T4 GPU. '
                           'CPU execution is possible with --allow-cpu but can be slow.')
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    records = validate_dataset(args.data, args.manifest)
    assert collections.Counter(r['split'] for r in records) == {'train':1541, 'val':327, 'test':329}
    config = {'protocol': 'frozen-imagenet-common-head-v1', 'seed': args.seed, 'epochs': args.epochs,
        'augmented_views': args.augmented_views, 'feature_batch': args.feature_batch,
        'dataset_manifest_sha256': sha(args.manifest), 'script_sha256': sha(__file__),
        'dtype': 'float32', 'models': MODELS, 'head_batch': 32,
        'head': 'GAP -> Dropout .3 -> Dense128 ReLU -> Dropout .2 -> Dense8 softmax',
        'optimizer': 'Adam 1e-3, ReduceLROnPlateau factor .3 patience2 min_lr1e-6',
        'checkpoint': 'minimum validation loss; EarlyStopping patience5 min_delta1e-4',
        'augmentation': 'original + deterministic mirror/rotation ±15 degrees/brightness/contrast .85–1.15 views; train only'}
    environment = {'python':platform.python_version(), 'tensorflow':tf.__version__,
        'keras':getattr(tf.keras, '__version__', 'unknown'), 'numpy':np.__version__, 'pillow':pillow_version,
        'platform':platform.platform(), 'gpus':[tf.config.experimental.get_device_details(g) for g in gpus],
        'logical_devices':[d.name for d in tf.config.list_logical_devices()],
        'build':tf.sysconfig.get_build_info()}
    environment = json.loads(json.dumps(environment))
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output/'protocol.json'
    metadata = {'config': config, 'environment': environment}
    if lock.exists():
        old = json.loads(lock.read_text())
        if not args.resume or old != metadata:
            raise ValueError('Output already exists or protocol/environment changed; select a fresh output directory.')
    lock.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    results = []
    for name in MODELS:
        dest = args.output/name; dest.mkdir(exist_ok=True)
        result_path = dest/'evaluation.json'
        if args.resume and result_path.exists():
            if sha(dest/'model.keras') != json.loads(result_path.read_text())['model_sha256']:
                raise ValueError(f'Saved model changed: {name}')
            results.append(json.loads(result_path.read_text()))
            continue
        print(f'\n=== {name}: NEW controlled experiment ===', flush=True)
        results.append(run_one(name, records, args.data, dest, config))
    assert len(results) == 3
    export_report(args.output, results, metadata)
    print('Completed three REAL experiments. Word/PNG/CSV/JSON saved in', args.output, flush=True)


if __name__ == '__main__':
    main()
