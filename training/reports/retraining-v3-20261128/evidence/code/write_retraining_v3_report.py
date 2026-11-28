"""Write measured v3 results, editable Word tables and figures from bound evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'training/.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.shared import Inches, Pt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ['Đốm tảo', 'Thán thư', 'Đốm mắt chim', 'Đốm nâu', 'Đốm xám', 'Lá khỏe', 'Đốm đỏ', 'Đốm trắng']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(truth, probs, classes):
    pred = probs.argmax(1)
    return dict(accuracy=float(accuracy_score(truth, pred)), balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                macro_f1=float(f1_score(truth, pred, average='macro')), weighted_f1=float(f1_score(truth, pred, average='weighted')),
                correct=int((truth == pred).sum()), test_images=len(truth),
                confusion_matrix=confusion_matrix(truth, pred, labels=list(range(8))).tolist(),
                classification_report=classification_report(truth, pred, labels=list(range(8)), target_names=classes, output_dict=True, zero_division=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'training/dataset-v3')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--browser', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'training/reports/retraining-v3-20261128')
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new report directory; existing reports are never replaced.')
    records = json.loads((args.data / 'manifest.json').read_text())
    data = json.loads((args.data / 'split_summary.json').read_text())
    classes = json.loads((args.data / 'classes.json').read_text())
    evaluation = json.loads((args.run / 'mobile-final/evaluation.json').read_text())
    manifest = json.loads((args.run / 'export/manifest.json').read_text())
    parity = json.loads((args.run / 'export-reports/onnx-export-parity.json').read_text())
    browser = json.loads(args.browser.read_text())
    dataset_sha = sha(args.data / 'manifest.json')
    for evidence in [data, evaluation, manifest, parity, browser]:
        assert evidence.get('dataset_manifest_sha256', evidence.get('manifest_sha256')) == dataset_sha
    assert evaluation['model_sha256'] == manifest['keras_sha256'] == sha(args.run / 'mobile-final/tea_leaf_mobilenetv2.keras')
    assert browser['onnx_sha256'] == manifest['sha256'] == sha(args.run / 'export/tea-leaf.onnx')
    assert parity['full_test'] and parity['parity_top1_disagreements'] == 0
    test = [r for r in records if r['split'] == 'test']
    assert len(test) == data['split_totals']['test'] == parity['parity_samples'] == browser['test_images']
    truth = np.array([classes.index(r['label']) for r in test])
    native_npz = np.load(args.run / 'mobile-final/test_predictions.npz')
    assert np.array_equal(native_npz['truth'], truth)
    assert list(native_npz['image_ids']) == [r['id'] for r in test]
    native = scores(truth, native_npz['probabilities'], classes)
    assert native['confusion_matrix'] == evaluation['confusion_matrix']
    assert abs(native['accuracy'] - evaluation['accuracy']) < 1e-12
    assert [r['id'] for r in browser['predictions']] == [r['id'] for r in test]
    assert [r['actual'] for r in browser['predictions']] == [r['label'] for r in test]
    web_probs = np.array([[r['probabilities'][c] for c in classes] for r in browser['predictions']])
    assert np.isfinite(web_probs).all() and (web_probs >= 0).all() and (web_probs <= 1).all()
    assert np.max(np.abs(web_probs.sum(1) - 1)) < 1e-3
    web = scores(truth, web_probs, classes)
    assert web['correct'] == browser['correct'] and abs(web['accuracy'] - browser['accuracy']) < 1e-12
    args.output.mkdir(parents=True)
    evidence_dir = args.output / 'evidence'; evidence_dir.mkdir()
    for source, name in [(args.run / 'mobile-final/evaluation.json', 'evaluation.json'),
                         (args.run / 'mobile-final/protocol.json', 'protocol.json'),
                         (args.run / 'mobile-final/test_predictions.npz', 'native-predictions.npz'),
                         (args.run / 'export/manifest.json', 'model-manifest.json'),
                         (args.run / 'export-reports/onnx-export-parity.json', 'onnx-parity.json'),
                         (args.browser, 'browser-predictions.json')]:
        shutil.copyfile(source, evidence_dir / name)
    summary = dict(dataset_manifest_sha256=dataset_sha, dataset_version=data['version'],
                   total_images=data['total_images'], added_images=data['added_images'], split_totals=data['split_totals'],
                   native=native, browser=web, model_version=manifest['model_version'], onnx_sha256=manifest['sha256'],
                   target_accuracy=.90, native_target_met=native['accuracy'] > .90, browser_target_met=web['accuracy'] > .90,
                   selection=evaluation['training']['selection'], selected_validation_loss=evaluation['training']['selected_validation_loss'],
                   limitations=evaluation['limitations'])
    if evaluation.get('calibration_protocol'):
        summary['calibration_protocol'] = evaluation['calibration_protocol']
    baseline_path = args.run / 'baseline-evaluation/evaluation.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    if baseline:
        assert baseline['dataset_manifest_sha256'] == dataset_sha and baseline['test_images'] == len(test)
        assert baseline['training_dataset_manifest_sha256'] == data['previous_manifest_sha256']
        source = args.run / 'baseline-evaluation/predictions.npz'
        predictions = np.load(source)
        assert list(predictions['image_ids']) == [r['id'] for r in test]
        assert scores(truth, predictions['probabilities'], classes)['confusion_matrix'] == baseline['confusion_matrix']
        summary['fixed_v2_baseline'] = baseline
        shutil.copyfile(baseline_path, evidence_dir / 'v2-baseline-on-v3-test.json')
        shutil.copyfile(source, evidence_dir / 'v2-baseline-on-v3-test.npz')
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    def save(fig, name):
        fig.tight_layout(); fig.savefig(args.output / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(args.output / f'{name}.svg', bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8)); pos = np.arange(8)
    ax.bar(pos, [data['counts'][c] for c in classes], color=['#458c70' if c != 'healthy' else '#c5a748' for c in classes])
    for i, c in enumerate(classes): ax.text(i, data['counts'][c]+30, str(data['counts'][c]), ha='center')
    ax.set_xticks(pos, LABELS, rotation=20); ax.set_ylabel('Số ảnh'); ax.set_ylim(0, max(data['counts'].values())*1.15)
    ax.set_title(f'Dữ liệu v3: {data["total_images"]:,} ảnh — train / validation / test = 70 / 15 / 15')
    save(fig, '01-so-luong-anh')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); history = evaluation['training']['stages']
    for key, val_key, ax, title in [('loss', 'val_loss', axes[0], 'Hàm mất mát'), ('accuracy', 'val_accuracy', axes[1], 'Accuracy')]:
        training = [value for stage in history for value in stage['history'][key]]
        validation = [value for stage in history for value in stage['history'][val_key]]
        ax.plot(range(1, len(training)+1), training, label='Train (trọng số, tăng cường)' if key == 'loss' else 'Train (tăng cường)')
        ax.plot(range(1, len(validation)+1), validation, label='Validation')
        completed = 0
        for i, stage in enumerate(history[:-1]):
            completed += stage['epochs_completed']
            ax.axvline(completed+.5, color='gray', linestyle=':' if i == 0 else '--', label='Bắt đầu fine-tuning' if i == 0 else 'Tinh chỉnh sâu hơn')
        ax.set_xlabel('Epoch cộng dồn'); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, '02-qua-trinh-huan-luyen')
    fig, ax = plt.subplots(figsize=(8.5, 7)); cm = np.array(web['confusion_matrix'])
    image = ax.imshow(cm, cmap='Greens'); fig.colorbar(image, ax=ax, fraction=.046)
    for i in range(8):
        for j in range(8): ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()*.55 else 'black', fontsize=9)
    ax.set_xticks(range(8), LABELS, rotation=40, ha='right'); ax.set_yticks(range(8), LABELS)
    ax.set_xlabel('Nhãn dự đoán'); ax.set_ylabel('Nhãn thực tế')
    ax.set_title(f'MobileNetV2 trên trình duyệt: {web["correct"]}/{len(test)} đúng ({web["accuracy"]:.2%})')
    save(fig, '03-ma-tran-nham-lan-web')
    paragraphs = [
        'BỔ SUNG DỮ LIỆU VÀ ĐÁNH GIÁ MOBILENETV2 — PHIÊN BẢN V3',
        f'Bộ dữ liệu được mở rộng thêm {data["added_images"]:,} ảnh sau lọc, đạt {data["total_images"]:,} ảnh. '
        f'Chia {data["split_totals"]["train"]:,} ảnh huấn luyện, {data["split_totals"]["val"]:,} ảnh validation và {len(test):,} ảnh test, đúng tỷ lệ 70/15/15. '
        f'Lớp healthy có {data["counts"]["healthy"]:,} ảnh, nhiều hơn mỗi lớp bệnh.',
        'Ảnh trùng được loại trước khi chia; các nhóm ảnh tương tự và loạt chụp cùng phút được giữ trong một tập. '
        'Toàn bộ 6.840 ảnh v2 giữ nguyên nhãn, dữ liệu và tập cũ, không chuyển ảnh validation/test vào huấn luyện. '
        'Tăng cường dữ liệu chỉ áp dụng cho train. Ba lớp thán thư, đốm mắt chim và đốm trắng chưa có ảnh độc lập bổ sung.',
        'Mô hình tiếp tục từ checkpoint v2 với kiểm tra toàn bộ phân chia cũ. Phần phân loại được thích nghi trước, '
        'sau đó tinh chỉnh các lớp cuối MobileNetV2, giữ BatchNormalization đóng băng. Checkpoint được chọn bằng validation loss; '
        'loss train có trọng số lớp và tăng cường ảnh, validation dùng ảnh giữ nguyên và không trọng số. '
        'tập test dùng tính kết quả sau mỗi lượt lựa chọn. Lượt v3 đầu đã được báo cáo trước khi tinh chỉnh tiếp, '
        'nên kết quả cuối vẫn là đánh giá phát triển nội bộ, không phải phép thử mù mới.',
        f'Keras/ONNX Python đạt Accuracy {native["accuracy"]:.2%} ({native["correct"]}/{len(test)}), '
        f'balanced accuracy {native["balanced_accuracy"]:.2%}, macro-F1 {native["macro_f1"]:.4f}. '
        f'Luồng website thực tế trên Chromium đạt Accuracy {web["accuracy"]:.2%} ({web["correct"]}/{len(test)}), '
        f'balanced accuracy {web["balanced_accuracy"]:.2%}, macro-F1 {web["macro_f1"]:.4f}. '
        + ('Hai kết quả được báo riêng. Website đã dùng cùng quy tắc bilinear chống răng cưa như Pillow; '
         'bộ giải mã JPEG và quản lý màu của trình duyệt vẫn có thể tạo chênh lệch nhỏ.'
         if browser.get('browser_resize') == 'pillow_bilinear_v1' else
         'Hai kết quả được tách riêng do Canvas và Pillow xử lý ảnh khác nhau.'),
        ('Mô hình trên website đã vượt mục tiêu Accuracy 90% trên tập test nội bộ.' if web['accuracy'] > .90 else
         f'Kết quả website hiện chưa đạt mục tiêu trên 90%; giá trị đo được là {web["accuracy"]:.2%}. Không điều chỉnh tập test hoặc thay số liệu để đạt mốc đặt trước.'),
        'Đây là đánh giá nội bộ trên các nguồn đã thu thập, chưa phải kiểm định thực địa độc lập tại Việt Nam. '
        'Validation/test cũ đã được xem trong quá trình phát triển. Kết quả 84,60%/84,11% v2 được đo trên bộ test khác về số mẫu, '
        'không dùng chênh lệch trực tiếp giữa hai phiên bản để khẳng định mức cải thiện. '
        'Ba mô hình ở trang So sánh tiếp tục thuộc thí nghiệm v2 đóng băng và không đại diện cho một phép so sánh kiến trúc mới trên v3.',
        'Nguồn mới: Datta tea-leaf-disease qua Project-AgML (CC BY-NC 4.0, học thuật/phi thương mại, chỉ chọn tên ảnh dạng ngày giờ); '
        'Hu và cộng sự (2019), DOI 10.17632/dbjyfkn6jr.1 (CC BY 4.0, chỉ lớp Tea red leaf spot). '
        'Nhãn được kế thừa từ nhà cung cấp, chưa có xác minh mới bởi chuyên gia.'
    ]
    inference = evaluation.get('inference_protocol')
    if inference:
        paragraph = ('Kết quả cuối là trung bình xác suất của ảnh gốc và ba hướng lật của cùng ảnh. '
                     if inference['selected'] == 'four_flip_mean' else
                     'Giữ dự đoán một hướng sau khi kiểm tra phương án trung bình bốn hướng lật. ')
        paragraph += (f"Lựa chọn dựa trên validation loss: một hướng {inference['candidates']['single_view']['loss']:.4f}; "
                      f"bốn hướng {inference['candidates']['four_flip_mean']['loss']:.4f}. "
                      'Không chọn bằng Accuracy test và không tính các hướng lật thành ảnh dữ liệu độc lập.')
        paragraphs.insert(4, paragraph)
    if baseline:
        paragraphs.insert(len(paragraphs)-1,
            f'Đối chứng trên cùng {len(test):,} ảnh test v3 và cùng tiền xử lý Pillow: checkpoint MobileNetV2 v2 cũ '
            f'đạt {baseline["accuracy"]:.2%} ({baseline["correct"]}/{len(test)}) với ONNX/Python; '
            f'checkpoint v3 mới đạt {native["accuracy"]:.2%} ({native["correct"]}/{len(test)}). '
            'Đây là so sánh trước/sau mở rộng và tinh chỉnh của cùng kiến trúc, không phải so sánh MobileNetV2 với ResNet50 hoặc VGG16.')
    calibration = evaluation.get('calibration_protocol')
    if calibration:
        original = calibration['candidates'][0]
        selected_calibration = calibration['selected_validation']
        paragraphs.insert(len(paragraphs)-1,
            f"Hiệu chỉnh trọng số lớp: xác suất đầu ra được chia cho trọng số huấn luyện của từng lớp lũy thừa β rồi chuẩn hóa lại. "
            f"β = {calibration['selected_beta']} được chọn trong lưới 0; 0,25; 0,5; 0,75; 1 trên validation, "
            f"yêu cầu Accuracy và macro-F1 không giảm. Validation loss giảm từ {original['loss']:.4f} xuống {selected_calibration['loss']:.4f}; "
            f"macro-F1 tăng từ {original['macro_f1']:.4f} lên {selected_calibration['macro_f1']:.4f}. "
            'Bước này không thay đổi ảnh hay cách chia dữ liệu. Các kết quả test trước hiệu chỉnh được lưu riêng; đây vẫn là đánh giá phát triển nội bộ.')
    (args.output / 'noi-dung-bao-cao.txt').write_text('\n\n'.join(paragraphs), encoding='utf-8')
    doc = Document(); style = doc.styles['Normal']; style.font.name = 'Times New Roman'; style.font.size = Pt(12)
    doc.add_heading(paragraphs[0], 0)
    # Keep the prose complete even when inference-selection details are present.
    for text in paragraphs[1:4]: doc.add_paragraph(text)
    doc.add_paragraph('Bảng: Số lượng ảnh của từng lớp sau mở rộng.')
    table = doc.add_table(rows=1, cols=5); table.style = 'Table Grid'
    for cell, text in zip(table.rows[0].cells, ['Nhãn', 'Train', 'Validation', 'Test', 'Tổng']): cell.text = text
    for c in classes:
        values = [c, *[str(data['split_counts'][s][c]) for s in ['train','val','test']], str(data['counts'][c])]
        for cell, text in zip(table.add_row().cells, values): cell.text = text
    doc.add_picture(str(args.output / '01-so-luong-anh.png'), width=Inches(6))
    evaluation_start = 5 if inference else 4
    if inference: doc.add_paragraph(paragraphs[4])
    for text in paragraphs[evaluation_start:evaluation_start+2]: doc.add_paragraph(text)
    doc.add_picture(str(args.output / '02-qua-trinh-huan-luyen.png'), width=Inches(6))
    doc.add_paragraph('Bảng: Precision, Recall và F1-score trên luồng website thực tế.')
    table = doc.add_table(rows=1, cols=5); table.style = 'Table Grid'
    for cell, text in zip(table.rows[0].cells, ['Nhãn', 'Precision', 'Recall', 'F1-score', 'Số ảnh test']): cell.text = text
    for c in classes:
        r = web['classification_report'][c]
        for cell, text in zip(table.add_row().cells, [c, f'{r["precision"]:.4f}', f'{r["recall"]:.4f}', f'{r["f1-score"]:.4f}', str(int(r['support']))]): cell.text = text
    doc.add_picture(str(args.output / '03-ma-tran-nham-lan-web.png'), width=Inches(6))
    for text in paragraphs[evaluation_start+2:]: doc.add_paragraph(text)
    path = args.output / 'TeaLeafAI-ket-qua-huan-luyen-v3.docx'; doc.save(path)
    with zipfile.ZipFile(path) as archive: assert archive.testzip() is None
    readback = Document(path); assert len(readback.tables) == 2 and len(readback.inline_shapes) == 3
    (args.output / 'README.md').write_text('# Kết quả MobileNetV2 v3\n\n'+'\n\n'.join(paragraphs[1:])+'\n\n[Word có bảng chỉnh sửa và 3 hình](TeaLeafAI-ket-qua-huan-luyen-v3.docx) · [Số liệu và giới hạn](summary.json) · [Văn bản để dán](noi-dung-bao-cao.txt)\n', encoding='utf-8')
    files = sorted(p for p in args.output.rglob('*') if p.is_file())
    (args.output / 'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(args.output)}' for p in files)+'\n')
    print(json.dumps({k: summary[k] for k in ['total_images', 'native_target_met', 'browser_target_met']}, indent=2))


if __name__ == '__main__':
    main()
