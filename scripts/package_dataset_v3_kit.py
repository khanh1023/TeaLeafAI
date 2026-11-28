"""Build a small downloadable restore kit; no TensorFlow/Node is required."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'downloads/TeaLeafAI-khoi-phuc-du-lieu-v3.zip')
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new file; existing download kits are never overwritten.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = ['training/restore_dataset_v3.py', 'training/download_dataset_v2.py', 'training/download_dataset_v3.py',
             'training/data/dataset-v3-additions.zip', 'training/reports/dataset-v2-20261125/package.json',
             'training/reports/dataset-v3-20261128/package.json', 'docs/DATASET_V3.md']
    command = '@echo off\r\ncd /d "%~dp0"\r\nwhere py >nul 2>&1\r\nif %errorlevel% equ 0 (\r\n  py -3 training\\restore_dataset_v3.py\r\n) else (\r\n  python training\\restore_dataset_v3.py\r\n)\r\npause\r\n'
    readme = ('BỘ KHÔI PHỤC DỮ LIỆU TEALEAFAI V3\n\n'
              '1. Cài Python 3.10+ (chọn Add Python to PATH nếu trình cài đặt đề nghị).\n'
              '2. Giải nén toàn bộ gói, giữ nguyên cấu trúc thư mục.\n'
              '3. Windows: nhấp đúp KHOI-PHUC-DU-LIEU.cmd. Linux/macOS: python3 training/restore_dataset_v3.py\n\n'
              'Cần Internet để tải bộ v2 khoảng 255 MB từ GitHub; gói này chứa thêm 1.440 ảnh v3 và 10 ảnh dự phòng.\n'
              'Sau khôi phục có đầy đủ 8.280 ảnh: 5.796 train / 1.242 validation / 1.242 test, đúng 70/15/15.\n'
              'Chỉ dùng thư viện chuẩn Python; chưa cần cài TensorFlow. Mỗi tệp được kiểm tra SHA-256.\n'
              'Thư mục đích: training/dataset-v3. Thư mục đã có được xác minh, không tự ghi đè.\n'
              'Đây là gói dữ liệu, không phải chương trình website hoặc mô hình đạt Accuracy 90%.\n'
              'Nguồn ảnh và điều kiện sử dụng (gồm CC BY-NC 4.0 cho nguồn Datta) được ghi trong docs/DATASET_V3.md.\n')
    with zipfile.ZipFile(args.output, 'w') as archive:
        def write(name, content, stored=False):
            info = zipfile.ZipInfo('TeaLeafAI-data-v3/' + name, date_time=(2026, 11, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16; archive.writestr(info, content)
        for name in names:
            write(name, (ROOT / name).read_bytes(), stored=name.endswith('.zip'))
        write('KHOI-PHUC-DU-LIEU.cmd', command.encode('ascii'))
        write('HUONG-DAN.txt', readme.encode('utf-8-sig'))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
    receipt = dict(filename=args.output.name, size_bytes=args.output.stat().st_size,
                   sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   archive_type='Restore kit with bundled v3 additions and pinned v2 download dependency',
                   full_dataset_images=8280, full_split_counts={'train':5796, 'val':1242, 'test':1242})
    (ROOT / 'training/reports/dataset-v3-20261128/restore-kit.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
