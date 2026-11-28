"""Promote verified v3 staging artifacts locally; never push/deploy from this script."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--browser', type=Path, required=True)
    p.add_argument('--reports', type=Path, default=ROOT / 'training/reports/retraining-v3-20261128')
    args = p.parse_args()
    model_dir = args.run / 'mobile-final'; export = args.run / 'export'
    evaluation = json.loads((model_dir / 'evaluation.json').read_text())
    manifest = json.loads((export / 'manifest.json').read_text())
    browser = json.loads(args.browser.read_text())
    records = json.loads((ROOT / 'training/dataset-v3/manifest.json').read_text())
    dataset_sha = sha(ROOT / 'training/dataset-v3/manifest.json')
    assert dataset_sha == evaluation['dataset_manifest_sha256'] == manifest['dataset_manifest_sha256'] == browser['dataset_manifest_sha256']
    assert evaluation['model_sha256'] == manifest['keras_sha256'] == sha(model_dir / 'tea_leaf_mobilenetv2.keras')
    assert manifest['sha256'] == browser['onnx_sha256'] == sha(export / 'tea-leaf.onnx')
    assert browser.get('evaluation_split', 'test') == 'test'
    assert browser['browser_resize'] == manifest['browser_resize'] == 'pillow_bilinear_v1'
    assert browser['model_version'] == manifest['model_version']
    test = [r for r in records if r['split'] == 'test']
    assert [r['id'] for r in test] == [r['id'] for r in browser['predictions']]
    assert len(test) == manifest['parity_samples'] == browser['test_images']
    assert manifest['parity_top1_disagreements'] == 0 and browser['image_uploads'] == 0
    assert not browser['uncaught_browser_errors']
    manifest.update(browser_test_accuracy=browser['accuracy'], browser_test_correct=browser['correct'],
                    browser_test_images=browser['test_images'], browser_evaluated_at=browser['evaluated_at'],
                    browser_evaluation_scope='Internal development test, Chromium/WASM; not independent field validation.')
    (export / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/write_retraining_v3_report.py'), '--run', str(args.run),
                    '--browser', str(args.browser), '--output', str(args.reports)], check=True)
    summary = json.loads((args.reports / 'summary.json').read_text())
    assert summary['native']['accuracy'] == evaluation['accuracy'] and summary['browser']['accuracy'] == browser['accuracy']
    for name in ['tea-leaf.onnx', 'manifest.json', 'classes.json']:
        shutil.copyfile(export / name, ROOT / 'frontend/public/model' / name)
    shutil.copyfile(model_dir / 'tea_leaf_mobilenetv2.keras', ROOT / 'backend/models/tea_leaf_mobilenetv2.keras')
    shutil.copyfile(model_dir / 'classes.json', ROOT / 'backend/models/classes.json')
    shutil.copyfile(model_dir / 'evaluation.json', ROOT / 'training/evaluation.json')
    shutil.copyfile(args.reports / 'TeaLeafAI-ket-qua-huan-luyen-v3.docx', ROOT / 'downloads/TeaLeafAI-ket-qua-huan-luyen-v3.docx')
    readme_path = ROOT / 'README.md'; text = readme_path.read_text()
    begin = text.index('## 8. Huấn luyện, dữ liệu và giới hạn')
    end = text.index('### Các kết quả v1 được giữ để đối chiếu lịch sử', begin)
    method = 'trung bình xác suất từ ảnh gốc và ba hướng lật của cùng ảnh' if manifest.get('inference_views') == 4 else 'dự đoán một hướng'
    if evaluation.get('calibration_protocol'):
        method += f", sau đó hiệu chỉnh theo trọng số lớp với β={evaluation['calibration_protocol']['selected_beta']} chọn trên validation"
    goal = ('Kết quả trên trình duyệt đã vượt mốc 90% trên tập test nội bộ.' if browser['accuracy'] > .90 else
            f"**Kết quả trên trình duyệt hiện chưa đạt mục tiêu trên 90%**; Accuracy đo được là **{browser['accuracy']:.2%}**.")
    block = f'''## 8. Huấn luyện, dữ liệu và giới hạn

**Mô hình mặc định hiện tại dùng dữ liệu v3:** **8.280 ảnh**, bổ sung **1.440 ảnh** sau lọc so với v2;
gồm **3.435 ảnh lá khỏe**, chia chính xác **5.796 train / 1.242 validation / 1.242 test (70/15/15)**.
Mọi ảnh v2 giữ nguyên nhãn, nội dung và tập đã chia. [Nguồn ảnh, giấy phép và bảng 8 lớp](docs/DATASET_V3.md).

Keras/ONNX Python với tiền xử lý Pillow đạt **{evaluation['accuracy']:.2%} ({summary['native']['correct']}/1.242 ảnh đúng)**,
macro-F1 **{evaluation['macro_f1']:.4f}**. Luồng thực tế trên Chromium đạt
**{browser['accuracy']:.2%} ({browser['correct']}/1.242 ảnh đúng)**, macro-F1 **{summary['browser']['macro_f1']:.4f}**.
{goal}

Checkpoint và phương án suy luận được chọn bằng validation loss. Mô hình dùng **{method}**;
các hướng lật không được tính thành mẫu dữ liệu độc lập. Website dùng resize bilinear chống răng cưa theo quy tắc Pillow;
JPEG decoder và quản lý màu của trình duyệt vẫn có thể tạo chênh lệch nhỏ.
[Báo cáo, ảnh và bằng chứng dự đoán](training/reports/retraining-v3-20261128/README.md) ·
[Tải báo cáo Word](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-ket-qua-huan-luyen-v3.docx).

Ba mô hình ở trang **Thử ba mô hình** tiếp tục thuộc thí nghiệm **v2 với 6.840 ảnh**, cùng backbone đóng băng:
MobileNetV2 **82,16%**, ResNet50 **85,58%**, VGG16 **83,14%** trên test v2.
Không dùng điểm MobileNetV2 v3 đã tinh chỉnh để thay vào bảng so sánh kiến trúc v2.
[Quy trình v2](docs/RETRAINING_V2.md) · [Lượt v3 đầu được lưu đầy đủ](training/reports/retraining-v3-preliminary-20261128/README.md).

[Tải gói khôi phục đầy đủ dữ liệu v3](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-khoi-phuc-du-lieu-v3.zip),
giải nén rồi chạy `KHOI-PHUC-DU-LIEU.cmd`, hoặc từ mã nguồn chạy `python training/restore_dataset_v3.py`.
Gói nhỏ chứa ảnh bổ sung và tải phần v2 đã công bố qua Internet; cần Python chuẩn, không cần TensorFlow để khôi phục.
Xem [lệnh tái lập huấn luyện và lựa chọn suy luận](docs/DATASET_V3.md).

Đây là đánh giá phát triển nội bộ: test đã được báo cáo ở lượt trước, không phải kiểm thử mù mới hay kiểm định thực địa
độc lập tại Việt Nam. Ba lớp thán thư, đốm mắt chim và đốm trắng chưa có ảnh độc lập mới được bổ sung.

'''
    text = text[:begin] + block + text[end:]
    text = text.replace('Trang Phân tích dùng MobileNetV2 đã được fine-tune', 'Trang Phân tích dùng MobileNetV2 v3 đã được fine-tune')
    text = text.replace('không chứa mô hình v2 hiện tại.', 'không chứa mô hình mặc định hiện tại.')
    old = 'Cả bốn bản ONNX được đối chiếu với Keras trên toàn bộ **1.026 ảnh test v2**, không có nhãn top-1 bất đồng; sai số xác suất lớn nhất dưới `2.1e-5`.'
    new = 'Mô hình chính v3 được đối chiếu Keras/ONNX trên **1.242 ảnh test v3**; ba mô hình so sánh được đối chiếu trên **1.026 ảnh test v2**. Không có nhãn top-1 bất đồng khi chuyển định dạng; sai số xác suất nhỏ hơn `1e-4`.'
    assert old in text; text = text.replace(old, new)
    readme_path.write_text(text)
    print(json.dumps(dict(promoted_locally=True, model_version=manifest['model_version'], browser_accuracy=browser['accuracy'], target_met=browser['accuracy']>.90), indent=2))


if __name__ == '__main__':
    main()
