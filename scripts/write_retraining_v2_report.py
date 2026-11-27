"""Publish measured v2 training evidence as an editable Word report and figures.

Requires the completed validation-selected default checkpoint and all three
controlled comparison results. Never trains, fabricates metrics, changes models,
or promotes a deployment. Optional browser evidence must cover the whole test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
V2_SHA = '47a24012c4f67957c9ec98b997e37404575738d068ec004c415ab7bd03ecf001'
MODELS = ['MobileNetV2', 'ResNet50', 'VGG16']
CLASSES = ['algal_leaf', 'anthracnose', 'bird_eye_spot', 'brown_blight',
           'gray_light', 'healthy', 'red_leaf_spot', 'white_spot']
SHORT_LABELS = ['Đốm tảo', 'Thán thư', 'Mắt chim', 'Đốm nâu',
                'Đốm xám', 'Lá khỏe', 'Đốm đỏ', 'Đốm trắng']
sys.path.insert(0, str(ROOT / 'training'))
from export_metadata import bind_checkpoint_evaluation, load_dataset, read_json, sha_file, validate_evaluation


def need(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f'Required completed artifact is missing: {path}')
    return path


def close(a, b, field):
    if not math.isclose(float(a), float(b), rel_tol=0., abs_tol=1e-9):
        raise ValueError(f'Recorded metric differs from saved predictions: {field}')


def probability_metrics(truth, probabilities):
    import numpy as np
    truth = np.asarray(truth)
    probabilities = np.asarray(probabilities)
    if truth.ndim != 1 or probabilities.shape != (len(truth), len(CLASSES)) or not len(truth):
        raise ValueError('Predictions have the wrong shape.')
    if not np.issubdtype(truth.dtype, np.integer) or np.any(truth < 0) or np.any(truth >= len(CLASSES)):
        raise ValueError('Unknown prediction ground-truth class.')
    if (not np.isfinite(probabilities).all() or np.any(probabilities < -1e-6)
            or np.any(probabilities > 1 + 1e-6)
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5, rtol=0.)):
        raise ValueError('Predictions must be finite normalized probabilities.')
    predicted = probabilities.argmax(axis=1)
    cm = np.bincount(truth * len(CLASSES) + predicted,
                     minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))
    support = cm.sum(axis=1)
    precision = np.divide(cm.diagonal(), cm.sum(axis=0), out=np.zeros(len(CLASSES)), where=cm.sum(axis=0) > 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(len(CLASSES)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(CLASSES)), where=precision + recall > 0)
    correct = int(cm.trace())
    return dict(test_images=len(truth), correct=correct, accuracy=correct / len(truth),
                macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                macro_f1=float(f1.mean()), balanced_accuracy=float(recall[support > 0].mean()),
                weighted_f1=float((f1 * support).sum() / len(truth)), confusion_matrix=cm.tolist(),
                per_class={c: dict(precision=float(precision[i]), recall=float(recall[i]),
                                   f1=float(f1[i]), support=int(support[i])) for i, c in enumerate(CLASSES)})


def check_saved_predictions(path, report, test):
    import numpy as np
    with np.load(need(path), allow_pickle=False) as saved:
        truth = saved['truth']
        expected = np.array([CLASSES.index(r['label']) for r in test], dtype=np.int32)
        if not np.array_equal(truth, expected):
            raise ValueError(f'Prediction labels/order differ from the frozen manifest: {path}')
        if 'ids' in saved and saved['ids'].tolist() != [r['id'] for r in test]:
            raise ValueError(f'Prediction IDs/order differ from the frozen manifest: {path}')
        result = probability_metrics(truth, saved['probabilities'])
    validate_evaluation(report, CLASSES, len(test))
    if report['confusion_matrix'] != result['confusion_matrix']:
        raise ValueError(f'Confusion matrix differs from saved predictions: {path}')
    for key in ['accuracy', 'macro_f1', 'weighted_f1', 'balanced_accuracy', 'macro_precision', 'macro_recall']:
        if key in report:
            close(report[key], result[key], key)
    if 'correct' in report and report['correct'] != result['correct']:
        raise ValueError('Recorded correct count differs from predictions.')
    return result


def verify_export(manifest, folder, checkpoint_sha, dataset_sha):
    if manifest.get('keras_sha256') != checkpoint_sha or manifest.get('dataset_manifest_sha256', dataset_sha) != dataset_sha:
        raise ValueError('ONNX export is bound to a different checkpoint/dataset.')
    model = need(folder / manifest['file'])
    if sha_file(model) != manifest['sha256'] or model.stat().st_size != manifest['size_bytes']:
        raise ValueError('ONNX file differs from its manifest.')
    if manifest.get('parity_top1_disagreements') != 0 or manifest.get('parity_samples') != 1026:
        raise ValueError('Final report requires successful full-test ONNX parity.')
    return manifest


def collect(args):
    # Check completion before opening output files or importing plotting/Word libraries.
    run = args.run.resolve()
    default_dir = run / 'mobile-final'
    default_file = need(default_dir / 'evaluation.json')
    default_checkpoint = need(default_dir / 'tea_leaf_mobilenetv2.keras')
    comparison_path = args.comparison
    if comparison_path is None:
        comparison_path = next((p for p in [run / 'comparison/comparison.json', run / 'comparison/final.json',
                                            run / 'comparison.json', run / 'final.json'] if p.is_file()), None)
    if comparison_path is None:
        raise ValueError('Completed three-model comparison.json/final.json is missing; partial results cannot be published as complete.')
    comparison_path = need(comparison_path.resolve())
    comparison_dir = comparison_path.parent
    comparison = read_json(comparison_path)
    config = comparison['metadata']['config']
    if config.get('models') != MODELS or config.get('dataset_manifest_sha256') != V2_SHA:
        raise ValueError('Comparison does not identify the frozen three-model v2 protocol.')
    if config.get('split_counts') != {'train': 4788, 'val': 1026, 'test': 1026}:
        raise ValueError('Comparison split differs from the v2 split.')
    raw_results = comparison['results']
    if not isinstance(raw_results, list) or [r.get('name') for r in raw_results] != MODELS:
        raise ValueError('Comparison requires all three models, once each, in protocol order.')
    evidence = load_dataset(args.data, CLASSES, expected_sha=V2_SHA)
    test = [r for r in evidence['records'] if r['split'] == 'test']
    default = bind_checkpoint_evaluation(read_json(default_file), sha_file(default_checkpoint))
    if default.get('classes') != CLASSES or default.get('dataset_manifest_sha256') != V2_SHA:
        raise ValueError('Default checkpoint evaluation uses a different dataset or class order.')
    selection = default.get('training', {}).get('fine_tuning', {}).get('selected')
    if selection not in ['initial_head', 'finetuned']:
        raise ValueError('Default checkpoint has not recorded its final validation-based selection.')
    training = default['training']; fine = training['fine_tuning']
    for history in [training['history'], fine['history']]:
        if (not history.get('loss') or len(history['loss']) != len(history.get('val_loss', []))
                or any(not math.isfinite(n) or n < 0 for key in ['loss', 'val_loss'] for n in history[key])):
            raise ValueError('Training history is missing, nonfinite or inconsistent.')
    minimum_fine_loss = min(fine['history']['val_loss'])
    expected_selection = 'finetuned' if minimum_fine_loss < fine['initial_validation_loss'] else 'initial_head'
    if selection != expected_selection:
        raise ValueError('Selected default model differs from its validation-loss decision.')
    close(fine['selected_validation_loss'], min(fine['initial_validation_loss'], minimum_fine_loss), 'selected validation loss')
    default_metrics = check_saved_predictions(default_dir / 'test_predictions.npz', default, test)
    results = []
    files = {'default/evaluation.json': default_file,
             'default/test_predictions.npz': default_dir / 'test_predictions.npz',
             'comparison/comparison.json': comparison_path,
             'comparison/protocol.json': need(comparison_dir / 'protocol.json')}
    if read_json(files['comparison/protocol.json']) != comparison['metadata']:
        raise ValueError('Comparison aggregate and protocol metadata differ.')
    for raw in raw_results:
        name = raw['name']; folder = comparison_dir / name
        checkpoint = need(folder / 'model.keras')
        recorded = bind_checkpoint_evaluation(read_json(need(folder / 'evaluation.json')), sha_file(checkpoint))
        if recorded != raw or raw.get('classes') != CLASSES or raw.get('dataset_manifest_sha256') != V2_SHA:
            raise ValueError(f'Aggregate/per-model evidence differs: {name}')
        if checkpoint.stat().st_size != raw['model_bytes']:
            raise ValueError(f'Keras file size differs from its report: {name}')
        measured = check_saved_predictions(folder / 'predictions.npz', raw, test)
        latency = raw['latency']
        if (not isinstance(latency['count'], int) or latency['count'] <= 0
                or latency['count'] != len(latency['sample_ids']) * latency['repeats']
                or not all(r in {t['id'] for t in test} for r in latency['sample_ids'])
                or not all(math.isfinite(latency[k]) and latency[k] > 0 for k in ['median_ms', 'mean_ms', 'p95_ms'])):
            raise ValueError(f'Latency evidence is incomplete: {name}')
        results.append(dict(name=name, model_sha256=raw['model_sha256'], model_bytes=raw['model_bytes'],
                            total_parameters=raw['total_parameters'], metrics=measured, latency=latency))
        for filename in ['evaluation.json', 'predictions.npz', 'predictions.csv']:
            files[f'comparison/{name}/{filename}'] = need(folder / filename)

    default_export_dir = run / 'exports/default'
    catalog_export_dir = run / 'exports/comparison'
    export_reports = run / 'export-reports'
    exports = {}
    default_manifest = default_export_dir / 'manifest.json'
    catalog_path = catalog_export_dir / 'manifest.json'
    if args.require_exports and not (default_manifest.is_file() and catalog_path.is_file()):
        raise ValueError('All default/comparison ONNX exports must be complete before publication.')
    if default_manifest.is_file():
        manifest = verify_export(read_json(default_manifest), default_export_dir, default['model_sha256'], V2_SHA)
        if manifest['classes'] != CLASSES:
            raise ValueError('Default ONNX class order differs.')
        parity_path = need(export_reports / 'default/onnx-export-parity.json')
        parity = read_json(parity_path)
        if (parity.get('full_test') is not True or parity['keras_sha256'] != default['model_sha256']
                or parity['onnx_sha256'] != manifest['sha256'] or parity['dataset_manifest_sha256'] != V2_SHA
                or parity['keras_test']['confusion_matrix'] != default_metrics['confusion_matrix']):
            raise ValueError('Default ONNX parity evidence is stale or incomplete.')
        validate_evaluation(parity['onnx_test'], CLASSES, len(test))
        exports['default'] = manifest
        exports['default_onnx_metrics'] = parity['onnx_test']
        files['exports/default/manifest.json'] = default_manifest
        files['exports/default/onnx-export-parity.json'] = parity_path
    if catalog_path.is_file():
        catalog = read_json(catalog_path)
        if catalog['classes'] != CLASSES or catalog['dataset_manifest_sha256'] != V2_SHA:
            raise ValueError('Comparison ONNX catalog differs from v2.')
        if [m['name'] for m in catalog['models']] != MODELS:
            raise ValueError('Comparison ONNX catalog is incomplete.')
        for item, native in zip(catalog['models'], results):
            verify_export(item, catalog_export_dir, native['model_sha256'], V2_SHA)
            parity_path = need(export_reports / 'comparison' / (item['id'] + '-export-parity.json'))
            parity = read_json(parity_path)
            if (parity.get('full_test') is not True or parity['dataset_manifest_sha256'] != V2_SHA
                    or parity['model']['sha256'] != item['sha256']
                    or parity['keras_test']['confusion_matrix'] != native['metrics']['confusion_matrix']):
                raise ValueError(f'Comparison export parity evidence differs: {item["name"]}')
            validate_evaluation(parity['onnx_test'], CLASSES, len(test))
            files[f'exports/comparison/{item["id"]}-export-parity.json'] = parity_path
        exports['comparison'] = catalog
        files['exports/comparison/manifest.json'] = catalog_path

    browser = None
    if args.browser_evaluation:
        if 'default' not in exports:
            raise ValueError('Browser evidence requires the verified default ONNX manifest.')
        browser_file = args.browser_evaluation
        if browser_file.is_dir():
            browser_file = browser_file / 'browser-predictions.json'
        raw = read_json(need(browser_file))
        if (raw.get('dataset_manifest_sha256') != V2_SHA or raw.get('test_images') != len(test)
                or raw.get('onnx_sha256') != exports['default']['sha256']
                or raw.get('model_version') != exports['default']['model_version']):
            raise ValueError('Browser evidence is not the whole frozen test for the selected default model.')
        rows = raw['predictions']
        if len(rows) != len(test) or [r['id'] for r in rows] != [r['id'] for r in test]:
            raise ValueError('Browser evidence has missing/repeated/reordered test IDs.')
        if any(r['actual'] != t['label'] for r, t in zip(rows, test)):
            raise ValueError('Browser ground truth differs from the frozen manifest.')
        metrics = probability_metrics([CLASSES.index(r['actual']) for r in rows],
                                      [[r['probabilities'][c] for c in CLASSES] for r in rows])
        for row in rows:
            predicted = max(CLASSES, key=lambda c: row['probabilities'][c])
            if row['predicted'] != predicted or row['correct'] != (row['predicted'] == row['actual']):
                raise ValueError('Browser predicted class/correct flag differs from probabilities.')
        close(raw['accuracy'], metrics['accuracy'], 'browser accuracy')
        if raw['correct'] != metrics['correct'] or raw.get('uncaught_browser_errors') or raw.get('image_uploads') != 0:
            raise ValueError('Browser evaluation includes errors, uploads or inconsistent counts.')
        browser = dict(metrics=metrics, model_version=raw['model_version'], onnx_sha256=raw['onnx_sha256'],
                       preprocessing=raw['preprocessing'], browser_version=raw['browser_version'],
                       evaluated_at=raw['evaluated_at'], warm_timing=raw.get('warm_timing'),
                       timing_definitions=raw.get('timing_definitions'), device=raw.get('device'))
        files['browser/browser-predictions.json'] = browser_file

    train_labels = Counter(r['label'] for r in evidence['records'] if r['split'] == 'train')
    majority = train_labels.most_common(1)[0][0]
    baseline_correct = sum(r['label'] == majority for r in test)
    if majority != 'healthy' or baseline_correct != 467:
        raise ValueError('The baseline does not match the pinned v2 split.')
    return dict(dataset_manifest_sha256=V2_SHA, split_totals=evidence['split_totals'],
                dataset_images=evidence['dataset_images'], classes=CLASSES,
                default=dict(report=default, metrics=default_metrics, selection=selection),
                comparison=dict(metadata=comparison['metadata'], models=results), exports=exports,
                browser=browser, baseline=dict(label=majority, correct=baseline_correct,
                                              test_images=len(test), accuracy=baseline_correct / len(test)),
                source_files={name: dict(sha256=sha_file(path), bytes=path.stat().st_size)
                              for name, path in sorted(files.items())}), files


def pct(value):
    return f'{100 * value:.2f}'.replace('.', ',') + '%'


def default_description(content):
    if content['default']['selection'] == 'initial_head':
        return ('MobileNetV2 giữ checkpoint head đã huấn luyện với backbone đóng băng. '
                'Fine-tuning đã được thử nhưng không được chọn vì không cải thiện validation loss.')
    return ('MobileNetV2 dùng checkpoint sau fine-tuning các lớp cuối, '
            'được chọn theo validation loss trước khi đánh giá test.')


def measured_text(content):
    default = content['default']['metrics']; baseline = content['baseline']
    lines = [
        'Bộ v2 có 6.840 ảnh: 4.788 train, 1.026 validation và 1.026 test, đúng tỷ lệ tổng 70/15/15. '
        'Ảnh tăng cường chỉ áp dụng cho train; bốn ảnh dự phòng không tham gia đánh giá.',
        default_description(content),
        f'Với Python/Pillow và Keras, checkpoint mặc định phân loại đúng {default["correct"]}/1.026 ảnh test, '
        f'Accuracy {pct(default["accuracy"])}, Precision macro {pct(default["macro_precision"])}, '
        f'Recall macro {pct(default["macro_recall"])} và F1 macro {pct(default["macro_f1"])}.',
        f'Quy tắc luôn đoán lớp nhiều nhất trong train, healthy, đúng {baseline["correct"]}/1.026 ảnh '
        f'({pct(baseline["accuracy"])}). Đây là đường cơ sở trên cùng v2, không phải một mạng CNN được huấn luyện.',
        'Ba mô hình so sánh bên dưới đều dùng backbone ImageNet đóng băng và cùng quy trình head/augmentation/budget. '
        'MobileNetV2 trong bảng là checkpoint thí nghiệm so sánh, được phân biệt với checkpoint mặc định. '
        'Kết quả của hai phiên bản không được ghép để xếp hạng.',
        'Thời gian đối chiếu là lời gọi toàn mô hình TensorFlow eager sau warm-up, batch 1, '
        'không gồm đọc ảnh, resize, tải mô hình hoặc mạng. Không dùng thời gian này làm tốc độ trình duyệt.',
        'Kết quả thuộc bộ test nội bộ theo nhóm ảnh, chưa phải kiểm định thực địa độc lập. '
        'Thông tin cây/phiên chụp nguồn còn thiếu. Không so sánh trực tiếp số Accuracy v1 và v2 để kết luận '
        'mức cải thiện, vì ảnh và bộ chia đã thay đổi.'
    ]
    if content['exports'].get('default'):
        onnx = content['exports']['default_onnx_metrics']
        lines.append(f'ONNX/Python của checkpoint mặc định được kiểm tra trên đủ 1.026 tensor test; '
                     f'đúng {onnx["correct"]}/1.026, Accuracy {pct(onnx["accuracy"])}. '
                     'Đối chiếu Keras–ONNX là kiểm tra chuyển đổi cùng mô hình, không phải hai kiến trúc.')
    if content['browser']:
        browser = content['browser']['metrics']
        lines.append(f'Đường xử lý Canvas và ONNX Runtime Web trong Chromium đạt '
                     f'{browser["correct"]}/1.026, Accuracy {pct(browser["accuracy"])}, '
                     f'F1 macro {pct(browser["macro_f1"])}. Đây là phép đo riêng trên đủ test; '
                     'khác biệt với Python có thể đến từ đọc/đổi kích thước ảnh.')
    return lines


def comparison_rows(content):
    onnx = content['exports'].get('comparison')
    lookup = {m['name']: m for m in onnx['models']} if onnx else {}
    headers = ['Mô hình', 'Đúng/test', 'Accuracy', 'Precision macro', 'Recall macro', 'F1 macro',
               'Keras (MB)', 'TF warm median (ms)']
    if onnx:
        headers.append('ONNX (MB)')
    rows = []
    for item in content['comparison']['models']:
        m = item['metrics']
        row = [item['name'], f'{m["correct"]}/{m["test_images"]}', pct(m['accuracy']),
               pct(m['macro_precision']), pct(m['macro_recall']), pct(m['macro_f1']),
               f'{item["model_bytes"]/1e6:.2f}', f'{item["latency"]["median_ms"]:.2f}']
        if onnx:
            row.append(f'{lookup[item["name"]]["size_bytes"]/1e6:.2f}')
        rows.append(row)
    return headers, rows


def figures(output, content):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'training/.cache/matplotlib'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'TeaLeafAI-retraining-v2'})
    generated = []

    def save(fig, stem):
        for ext in ['png', 'svg']:
            filename = stem + '.' + ext
            metadata = {'Software': 'TeaLeafAI'} if ext == 'png' else {'Date': '2026-11-27', 'Creator': 'TeaLeafAI'}
            fig.savefig(output / filename, dpi=300, bbox_inches='tight', metadata=metadata)
            generated.append(filename)
        plt.close(fig)

    def confusion(matrix, stem, title):
        cm = np.asarray(matrix)
        fig, ax = plt.subplots(figsize=(9, 7.3))
        im = ax.imshow(cm, cmap='Greens')
        ax.set_xticks(range(8), SHORT_LABELS, rotation=35, ha='right')
        ax.set_yticks(range(8), SHORT_LABELS)
        ax.set_xlabel('Nhãn dự đoán'); ax.set_ylabel('Nhãn thực tế'); ax.set_title(title, pad=15)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=11,
                        color='white' if cm[i, j] > cm.max() * .55 else '#183e31')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        save(fig, stem)

    confusion(content['default']['metrics']['confusion_matrix'], '01-default-confusion',
              'MobileNetV2 mặc định — Python/Keras — test 1.026 ảnh')
    history = content['default']['report']['training']['history']
    fine = content['default']['report']['training']['fine_tuning']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, values, title in [(axes[0], history, 'Huấn luyện head'),
                              (axes[1], fine['history'], 'Thử fine-tuning')]:
        epochs = np.arange(1, len(values['loss']) + 1)
        ax.plot(epochs, values['loss'], label='Train loss', color='#21694f')
        ax.plot(epochs, values['val_loss'], label='Validation loss', color='#bd7741')
        ax.set(title=title, xlabel='Epoch', ylabel='Loss'); ax.legend(); ax.grid(alpha=.15)
    fig.suptitle('Chọn theo validation loss; train có trọng số lớp', y=1.02)
    fig.text(.5, -.04, 'Checkpoint được giữ: ' + ('head ban đầu' if fine['selected'] == 'initial_head' else 'sau fine-tuning'), ha='center')
    fig.tight_layout(); save(fig, '02-training-history')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    names = [r['name'] for r in content['comparison']['models']]
    for ax, key, title in [(axes[0], 'accuracy', 'Accuracy test (%)'),
                            (axes[1], 'macro_f1', 'F1 macro test (%)')]:
        values = [r['metrics'][key] * 100 for r in content['comparison']['models']]
        bars = ax.bar(names, values, color=['#226b50', '#537f9c', '#bb8d48'])
        ax.set_title(title); ax.set_ylim(0, 105)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 1, f'{value:.2f}', ha='center')
    fig.suptitle('Thí nghiệm so sánh — backbone ImageNet đóng băng', fontsize=14)
    fig.tight_layout(); save(fig, '03-controlled-comparison')
    if content['browser']:
        confusion(content['browser']['metrics']['confusion_matrix'], '04-browser-confusion',
                  'MobileNetV2 mặc định — Canvas/WASM — test 1.026 ảnh')
    return generated


def deterministic_docx(path):
    with zipfile.ZipFile(path) as archive:
        members = [(name, archive.read(name)) for name in sorted(archive.namelist())]
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


def word_report(output, content, paragraphs):
    from docx import Document
    from docx.shared import Cm, Pt
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21); section.page_height = Cm(29.7)
    section.left_margin = section.right_margin = Cm(2)
    section.top_margin = section.bottom_margin = Cm(1.8)
    style = doc.styles['Normal']; style.font.name = 'Times New Roman'; style.font.size = Pt(12)
    style.paragraph_format.line_spacing = 1.12
    doc.core_properties.title = 'Kết quả huấn luyện và đánh giá TeaLeafAI — dữ liệu v2'
    doc.core_properties.created = doc.core_properties.modified = dt.datetime(2026, 11, 27, tzinfo=dt.timezone.utc)
    doc.add_heading('KẾT QUẢ HUẤN LUYỆN VÀ ĐÁNH GIÁ — BỘ DỮ LIỆU V2', 0)
    for paragraph in paragraphs[:4]:
        doc.add_paragraph(paragraph)
    doc.add_heading('Kết quả mô hình mặc định trên từng lớp', 1)
    table = doc.add_table(rows=1, cols=5); table.style = 'Table Grid'
    for cell, text in zip(table.rows[0].cells, ['Lớp', 'Precision', 'Recall', 'F1-score', 'Số ảnh test']): cell.text = text
    for label in CLASSES:
        row = content['default']['metrics']['per_class'][label]
        for cell, text in zip(table.add_row().cells, [label, pct(row['precision']), pct(row['recall']), pct(row['f1']), str(row['support'])]): cell.text = text
    doc.add_picture(str(output / '01-default-confusion.png'), width=Cm(16.6))
    doc.add_paragraph('Hình 1. Ma trận nhầm lẫn của checkpoint mặc định trên 1.026 ảnh test, Python/Keras.')
    doc.add_picture(str(output / '02-training-history.png'), width=Cm(16.6))
    doc.add_paragraph('Hình 2. Lịch sử train/validation và quyết định giữ checkpoint; không chọn epoch theo test.')
    doc.add_heading('3.3.4. So sánh hiệu quả giữa các mô hình phân loại', 1)
    doc.add_paragraph(paragraphs[4])
    # Transposed table is readable in portrait Word and remains fully editable.
    headers, rows = comparison_rows(content)
    table = doc.add_table(rows=1, cols=4); table.style = 'Table Grid'
    for cell, text in zip(table.rows[0].cells, ['Tiêu chí', *MODELS]): cell.text = text
    for i, header in enumerate(headers[1:], 1):
        for cell, text in zip(table.add_row().cells, [header, *[row[i] for row in rows]]): cell.text = text
    doc.add_paragraph('MB = 1.000.000 byte. File Keras đối chiếu không lưu optimizer; các backbone đều bỏ classifier ImageNet gốc (include_top=False).')
    doc.add_picture(str(output / '03-controlled-comparison.png'), width=Cm(16.6))
    doc.add_paragraph('Hình 3. Kết quả trên cùng bộ test và cùng giao thức backbone đóng băng.')
    for paragraph in paragraphs[5:]: doc.add_paragraph(paragraph)
    if content['browser']:
        doc.add_picture(str(output / '04-browser-confusion.png'), width=Cm(16.6))
        doc.add_paragraph('Hình 4. Kết quả theo đường đọc ảnh Canvas/WASM của ứng dụng; không ghép với Python để tăng số quan sát.')
    doc.add_heading('Định danh và phạm vi bằng chứng', 1)
    doc.add_paragraph('Dataset manifest SHA-256: ' + content['dataset_manifest_sha256'])
    doc.add_paragraph('Checkpoint mặc định SHA-256: ' + content['default']['report']['model_sha256'])
    for model in content['comparison']['models']:
        doc.add_paragraph(model['name'] + ' đối chiếu — Keras SHA-256: ' + model['model_sha256'])
    if content['exports'].get('default'):
        doc.add_paragraph('Default ONNX SHA-256: ' + content['exports']['default']['sha256'])
    doc.add_paragraph('Nguồn: checkpoint, lịch sử và dự đoán thật của chương trình TeaLeafAI. Dữ liệu nguồn và giấy phép trong docs/DATASET_V2.md. Báo cáo này không tự xác nhận website đã triển khai đúng model; việc đó cần kiểm tra manifest và luồng UI trên tên miền thực tế.')
    filename = 'TeaLeafAI-ket-qua-huan-luyen-v2.docx'
    doc.save(output / filename); deterministic_docx(output / filename)
    return filename


def publish(args, content, source_files):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise ValueError('Output is not empty. Choose a new directory or explicitly pass --overwrite for this v2 report.')
        if not (output / 'summary.json').is_file() or read_json(output / 'summary.json').get('dataset_manifest_sha256') != V2_SHA:
            raise ValueError('--overwrite is limited to a recognized v2 report, never an unrelated directory.')
    if output == args.run.resolve() or output == args.data.resolve() or output in args.run.resolve().parents:
        raise ValueError('Output cannot replace a source/checkpoint/dataset directory.')
    if (output / 'summary.json').exists() and read_json(output / 'summary.json').get('browser') and not content['browser']:
        raise ValueError('Refusing to replace a browser-complete report with native-only evidence.')
    output.mkdir(parents=True, exist_ok=True)
    for name, source in source_files.items():
        target = output / 'evidence' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    (output / 'summary.json').write_text(json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    made = figures(output, content)
    paragraphs = measured_text(content)
    word = word_report(output, content, paragraphs)
    headers, rows = comparison_rows(content)
    import csv
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, lineterminator='\n'); writer.writerow(headers); writer.writerows(rows)
    lines = ['# Huấn luyện và đánh giá TeaLeafAI — dữ liệu v2', '',
             f'[Tải Word có bảng chỉnh sửa được]({word}) · [Số liệu JSON](summary.json) · [Bảng CSV](comparison.csv)', '']
    for paragraph in paragraphs: lines.extend([paragraph, ''])
    lines += ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(headers)-1)) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    lines += ['', '## Hình và bằng chứng', '',
              '- [Ma trận mặc định](01-default-confusion.png) · [Lịch sử huấn luyện](02-training-history.png) · [So sánh có kiểm soát](03-controlled-comparison.png).',
              '- [Báo cáo mặc định gốc](evidence/default/evaluation.json) và [kết quả so sánh gốc](evidence/comparison/comparison.json).',
              '- Thư mục evidence giữ lịch sử, dự đoán và thông tin xuất mô hình thật; không nhân đôi checkpoint lớn.',
              '- Không có kết luận cải thiện v1–v2 trên hai bộ test khác nhau; không khẳng định MobileNetV2 luôn vượt trội.',
              '- Báo cáo không tự xác nhận trạng thái triển khai. Cần đối chiếu manifest trên tên miền thật với bản xuất đã kiểm tra.', '',
              'Dataset manifest SHA-256: `' + V2_SHA + '`.',
              'Checkpoint mặc định SHA-256: `' + content['default']['report']['model_sha256'] + '`.', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'noi-dung-bao-cao.txt').write_text('\n\n'.join(paragraphs) + '\n', encoding='utf-8')
    if args.update_guide:
        target = ROOT / 'docs/RETRAINING_V2.md'
        relative = os.path.relpath(output / 'README.md', target.parent).replace(os.sep, '/')
        guide = ['# Huấn luyện và đánh giá với dữ liệu v2', '',
                 'Kết quả dưới đây được tạo từ checkpoint, ma trận và dự đoán thật. Trạng thái triển khai được xác nhận riêng, không suy ra từ việc sinh báo cáo.', '']
        for paragraph in paragraphs: guide.extend([paragraph, ''])
        guide += [f'[Báo cáo, Word, hình và toàn bộ bằng chứng]({relative}) · [Nguồn và bộ chia dữ liệu](DATASET_V2.md).', '',
                  'Manifest SHA-256: `' + V2_SHA + '`.', '',
                  'Báo cáo v1 trên 2.197 ảnh/test 329 ảnh tiếp tục được giữ làm lịch sử. Tài liệu và ZIP cũ không mặc nhiên là phiên bản v2.', '']
        target.write_text('\n'.join(guide), encoding='utf-8')
    included = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{sha_file(p)}  {p.relative_to(output).as_posix()}\n' for p in included), encoding='utf-8')
    print(json.dumps({'output': str(output), 'default_selection': content['default']['selection'],
                      'default_correct': content['default']['metrics']['correct'], 'test_images': 1026,
                      'comparison_models': MODELS, 'browser_complete': content['browser'] is not None,
                      'word': word, 'figures': made}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'training/reports/retraining-v2-20261127')
    parser.add_argument('--data', type=Path, default=ROOT / 'training/dataset-v2')
    parser.add_argument('--comparison', type=Path, help='Explicit complete comparison.json/final.json (metadata + all three results).')
    parser.add_argument('--browser-evaluation', type=Path, help='Complete browser-predictions.json, or its parent directory.')
    parser.add_argument('--require-exports', action='store_true', help='Require all four complete ONNX exports and full-test parity reports.')
    parser.add_argument('--update-guide', action='store_true', help='Replace only docs/RETRAINING_V2.md with measured summary and report links.')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate this report after adding complete evidence; preserves unrelated files.')
    args = parser.parse_args()
    try:
        content, source_files = collect(args)
        publish(args, content, source_files)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit('Refusing to publish an incomplete or inconsistent report: ' + str(error)) from error


if __name__ == '__main__':
    main()
