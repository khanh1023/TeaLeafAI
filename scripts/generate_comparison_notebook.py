"""Generate a Colab notebook pinned to the committed comparison script and dataset."""
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
script_bytes = (ROOT/'training/compare_models.py').read_bytes()
assert subprocess.check_output(['git', 'show', f'{revision}:training/compare_models.py'], cwd=ROOT) == script_bytes
script_sha = hashlib.sha256(script_bytes).hexdigest()
cells = []


def markdown(text):
    cells.append({'cell_type':'markdown', 'metadata':{}, 'source':textwrap.dedent(text).strip()+'\n'})


def code(text):
    cells.append({'cell_type':'code', 'execution_count':None, 'metadata':{}, 'outputs':[],
                  'source':textwrap.dedent(text).strip()+'\n'})


markdown('''
    # TeaLeafAI — So sánh MobileNetV2, ResNet50 và VGG16

    **Mục tiêu:** chạy một thí nghiệm mới có thể kiểm tra, xuất bảng Word và hình cho mục3.3.4.
    Notebook chưa chứa kết quả huấn luyện. Hãy chạy toàn bộ các ô để tạo số liệu thật.

    1. Chọn **Runtime (Thời gian chạy) → Change runtime type → T4 GPU → Save**.
    2. Chọn **Runtime → Run all** và chờ cả ba mô hình hoàn tất.
    3. Tải ZIP ở ô cuối; mở `So-sanh-mo-hinh-muc-3-3-4.docx` để dán vào báo cáo.

    Dùng gói Colab miễn phí khi có GPU khả dụng. Google quyết định hạn mức và thời lượng phiên;
    notebook không bảo đảm luôn được cấp GPU, không yêu cầu mua Colab Pro hay nhập API key.
    Không cần kết nối Google Drive. Hãy tải ZIP trước khi phiên Colab kết thúc.
''')
markdown('''
    ## Quy trình so sánh

    - Cùng2.197 ảnh, chia cố định1.541 train /327 validation /329 test, xấp xỉ70/15/15.
    - Cùng ảnh RGB224×224 và biến đổi tăng cường xác định trước, chỉ trên train.
    - Trọng số ImageNet, **đóng băng toàn bộ backbone**, GAP + Dropout + Dense128 + Dense8.
    - Adam, class weights, tối đa20 epoch, chọn checkpoint bằng validation loss.
    - Chuẩn hóa đúng từng kiến trúc: MobileNetV2 về[-1,1]; ResNet50/VGG16 dùng BGR trừ trung bình ImageNet.
    - Chỉ chạy test sau khi chọn checkpoint. Một seed42; chưa phải so sánh nhiều seed hoặc fine-tuning toàn mạng.
    - File Keras cùng định dạng, không gồm optimizer; dung lượng/parameters đo từ từng mô hình đã huấn luyện.
    - Latency: batch1 TensorFlow eager, sau warm-up, cùng máy; không gồm đọc/resize ảnh hay tốc độ trình duyệt.

    Kết quả MobileNetV2 ở đây có thể khác69,30% Python/66,87% website vì đây là quy trình huấn luyện mới.
    Không trộn kết quả mô hình mới với checkpoint đã fine-tune của website để xếp hạng.
    Test là bộ nội bộ đã sử dụng trong phát triển, chưa phải kiểm định thực địa độc lập.
''')
code('''
    # Giữ TensorFlow/CUDA do Colab cung cấp, không cài lại GPU runtime.
    %pip -q install python-docx scikit-learn pillow matplotlib
''')
code('''
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('Chưa có GPU. Chọn Runtime → Change runtime type → T4 GPU, rồi chạy lại.')
    print('TensorFlow:', tf.__version__)
    print('GPU:', [tf.config.experimental.get_device_details(d) for d in devices])
''')
setup = '''
import hashlib, json, os, subprocess, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path('/content/TeaLeafAI-comparison')
ROOT.mkdir(parents=True, exist_ok=True)
REVISION = __REVISION__
BASE = f'https://raw.githubusercontent.com/khanh1023/TeaLeafAI/{REVISION}/'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def download(relative, expected_sha):
    dest = ROOT / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and digest(dest) == expected_sha:
        print('Đã có:', relative)
        return dest
    partial = dest.with_suffix(dest.suffix+'.part')
    with urllib.request.urlopen(BASE+relative, timeout=120) as response, open(partial, 'wb') as out:
        for chunk in iter(lambda: response.read(1024*1024), b''):
            out.write(chunk)
    assert digest(partial) == expected_sha, 'Checksum không khớp: '+relative
    partial.replace(dest)
    print('Đã tải và xác minh:', relative)
    return dest

SCRIPT = download('training/compare_models.py', __SCRIPT_SHA__)
MANIFEST = download('training/reports/expanded-dataset-manifest.json',
    '585cabd7be1741675fa5bed31c14116b9f72f7eca11d40610313ddb15cd48707')
ARCHIVE = download('downloads/TeaLeafAI-training-data.zip',
    '5a72681192bf3223ec85ccef12d66f1789eb457b088805dca999837c403bc134')
with zipfile.ZipFile(ARCHIVE) as archive:
    for name in archive.namelist():
        assert (ROOT/name).resolve().is_relative_to(ROOT.resolve()), 'Unsafe ZIP member'
    archive.extractall(ROOT)
DATA = ROOT/'TeaLeafAI-training/dataset'
assert digest(DATA/'manifest.json') == digest(MANIFEST)
OUTPUT = ROOT/'comparison-results'
print('Đã chuẩn bị 2.197 ảnh. Mã nguồn và bộ chia được ghim tại commit:', REVISION)
'''
code(setup.replace('__REVISION__',repr(revision)).replace('__SCRIPT_SHA__',repr(script_sha)))
code('''
    # Không đổi seed/epoch sau khi xem test để lựa chọn một kết quả đẹp hơn.
    # --resume chỉ dùng lại các kết quả có cùng code, cấu hình, dữ liệu và môi trường.
    command = [sys.executable, str(SCRIPT), '--data', str(DATA), '--manifest', str(MANIFEST),
               '--output', str(OUTPUT), '--resume']
    print('Bắt đầu ba mô hình; thời gian phụ thuộc GPU được Colab cấp.')
    subprocess.run(command, check=True)
''')
code('''
    import pandas as pd
    from IPython.display import display, Image
    assert (OUTPUT/'comparison.json').exists(), 'Cần chạy xong cả ba mô hình trước.'
    display(pd.read_csv(OUTPUT/'comparison.csv'))
    display(Image(filename=str(OUTPUT/'comparison.png'), width=950))
    print('Word:', OUTPUT/'So-sanh-mo-hinh-muc-3-3-4.docx')
''')
code('''
    # ZIP giữ cả3 mô hình thật để kiểm tra lại, có thể lớn khoảng vài trăm MB.
    # Để False nếu chỉ muốn tải bảng Word, hình và số liệu; trọng số còn trong phiên Colab.
    INCLUDE_TRAINED_MODELS = True
    from google.colab import files
    package = ROOT/'TeaLeafAI-so-sanh-ba-mo-hinh-ket-qua.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file() and (INCLUDE_TRAINED_MODELS or path.suffix != '.keras'):
                archive.write(path, path.relative_to(OUTPUT).as_posix())
        archive.write(SCRIPT, 'source/compare_models.py')
        archive.write(MANIFEST, 'source/dataset-manifest.json')
    print('ZIP:', package, f'({package.stat().st_size/1e6:.1f} MB)')
    files.download(str(package))
''')
markdown('''
    ## Đọc kết quả và sử dụng trong báo cáo

    - `So-sanh-mo-hinh-muc-3-3-4.docx`: văn bản và bảng chỉnh sửa được, kèm hình so sánh.
    - `comparison.png` (300 dpi), `comparison.svg`, `comparison.pdf`: hình chèn vào Word.
    - `comparison.csv`, `comparison.json`: bảng tổng hợp kết quả thật.
    - `protocol.json`: seed, checksum, cấu hình và phiên bản môi trường.
    - Mỗi thư mục mô hình chứa trọng số, lịch sử, ma trận nhầm lẫn, chỉ số từng lớp và dự đoán từng ảnh.

    Chỉ dùng kết quả sinh sau khi chạy đủ ba mô hình. Không dùng số94,5%/95,1%/94,8% từ bản nháp.
    Nếu GPU bị ngắt, không có bảo đảm tiến trình được giữ; hãy tải file và giữ lại ZIP.
    Dữ liệu và giấy phép: https://github.com/khanh1023/TeaLeafAI/blob/main/docs/TRAINING_DATA.md
''')
notebook = {'cells':cells,'metadata':{'colab':{'name':'TeaLeafAI — So sánh ba mô hình.ipynb'},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':4}
# Add spaces in Vietnamese prose while preserving code/URLs.
for cell in cells:
    if cell['cell_type']=='markdown':
        for old, new in [('mục3.3.4','mục 3.3.4'),('Cùng2.197','Cùng 2.197'),('định1.541','định 1.541'),
            ('/327','/327'),('RGB224','RGB 224'),('tối đa20','tối đa 20'),('về[-1','về [-1'),
            ('seed42','seed 42'),('batch1','batch 1'),('khác69','khác 69'),('số94','số 94'),('xỉ70','xỉ 70')]:
            cell['source']=cell['source'].replace(old,new)
text = json.dumps(notebook, ensure_ascii=False, indent=2)+'\n'
for path in [ROOT/'training/Compare_Models_Colab.ipynb', ROOT/'downloads/TeaLeafAI-so-sanh-mo-hinh.ipynb']:
    path.write_text(text)
print('Created empty-output notebook pinned to', revision)
