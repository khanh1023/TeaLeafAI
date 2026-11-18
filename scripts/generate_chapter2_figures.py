"""Create reproducible publication figures and an editable-caption Word handout."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'training/.cache/matplotlib'))
import hashlib
import json
import random
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = ROOT / 'docs/figures/chapter2'
OUT.mkdir(parents=True, exist_ok=True)
DOWNLOADS = ROOT / 'downloads'
RECORDS = json.loads((ROOT / 'training/reports/expanded-dataset-manifest.json').read_text())
SUMMARY = json.loads((ROOT / 'training/reports/expanded-dataset-summary.json').read_text())
CLASSES = json.loads((ROOT / 'backend/models/classes.json').read_text())
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'text.color': '#183c32', 'axes.labelcolor': '#183c32',
    'savefig.facecolor': 'white',
})
GREEN = '#216b50'
INK = '#183c32'
PALE = '#eef5ef'
GRAY = '#56665e'
FIGURES = []


def canvas(height=5.1):
    fig, ax = plt.subplots(figsize=(9, height))
    fig.subplots_adjust(left=.025, right=.975, top=.98, bottom=.025)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, title, body='', color=PALE, fontsize=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.008,rounding_size=.014',
                              facecolor=color, edgecolor=GREEN, linewidth=1.5))
    ax.text(x+w/2, y+h*(.67 if body else .5), title, ha='center', va='center',
            fontsize=fontsize, weight='bold', linespacing=1.4)
    if body:
        ax.text(x+w/2, y+h*.28, body, ha='center', va='center', fontsize=12.2,
                color=GRAY, linespacing=1.4)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=17,
                               linewidth=1.6, color=GREEN))


def save(fig, stem, caption, section, source, note):
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{stem}.{suffix}', dpi=300)
    plt.close(fig)
    FIGURES.append(dict(stem=stem, caption=caption, section=section, source=source, note=note))


# 2.1.3: real default browser workflow, including optional local history/export.
fig, ax = canvas()
ax.text(.5, .94, 'PHÂN TÍCH ẢNH LÁ CHÈ TRÊN TRÌNH DUYỆT', ha='center', weight='bold', fontsize=16)
xs = [.025, .37, .715]
w, h = .26, .235
box(ax, xs[0], .59, w, h, '1. Nhận ảnh', 'Tải ảnh hoặc\nchụp bằng camera')
box(ax, xs[1], .59, w, h, '2. Tiền xử lý', 'RGB · 224 × 224\nPixel đầu vào 0–255')
box(ax, xs[2], .59, w, h, '3. Suy luận', 'MobileNetV2\nONNX Runtime Web')
box(ax, xs[2], .19, w, h, '4. Kết quả', 'Nhãn dự đoán và\nđiểm của 8 nhóm')
box(ax, xs[1], .19, w, h, '5. Hướng xử lý', 'Dấu hiệu nhận biết\nvà chăm sóc ban đầu')
box(ax, xs[0], .19, w, h, '6. Lưu kết quả', 'Lịch sử trên thiết bị\nXuất JSON / CSV')
for a, b in [(0, 1), (1, 2)]:
    arrow(ax, (xs[a]+w+.006, .707), (xs[b]-.012, .707))
arrow(ax, (.845, .58), (.845, .435))
for a, b in [(2, 1), (1, 0)]:
    arrow(ax, (xs[a]-.012, .307), (xs[b]+w+.006, .307))
ax.text(.5, .055, 'Chế độ mặc định: xử lý trên thiết bị, không gửi ảnh lên máy chủ.',
        ha='center', fontsize=12.2, color=GRAY)
save(fig, 'hinh-2-1-quy-trinh-he-thong', 'Hình 2.1. Quy trình phân tích ảnh lá chè của hệ thống TeaLeafAI',
     '2.1.3. Đầu vào, đầu ra và yêu cầu hệ thống', 'Tác giả xây dựng từ mã nguồn TeaLeafAI.',
     'Đặt sau đoạn giới thiệu quy trình xử lý. Lịch sử chỉ lưu thông tin kết quả, không lưu ảnh.')


# Illustrative source labels, not externally verified diagnoses.
selected_ids = {
    'algal_leaf': '60d832a018767c20',
    'anthracnose': 'b73bdf0d6702dbde',
    'bird_eye_spot': '9c720cfd5295db67',
    'brown_blight': '4a17b9dfa361ab74',
    'gray_light': '04a66015923dc9dd',
    'healthy': 'd0f5daa525912fc9',
    'red_leaf_spot': '10ae37dcffeb3a70',
    'white_spot': '043994eb3437a80e',
}
labels = ['Đốm tảo', 'Thán thư', 'Đốm mắt chim', 'Đốm nâu', 'Đốm xám', 'Lá khỏe mạnh', 'Đốm đỏ', 'Đốm trắng']
samples = []
for key in CLASSES:
    matches = [r for r in RECORDS if r['label'] == key and r['id'].endswith(selected_ids[key])]
    if not matches:
        raise ValueError(f'Selected sample missing: {key}')
    samples.append(matches[0])
fig, axes = plt.subplots(2, 4, figsize=(9, 6.4))
fig.subplots_adjust(left=.025, right=.975, top=.94, bottom=.20, hspace=.46, wspace=.13)
for ax, key, label, record in zip(axes.flat, CLASSES, labels, samples):
    with Image.open(ROOT / 'training/dataset' / record['path']) as im:
        image = ImageOps.contain(im.convert('RGB'), (512, 512), Image.Resampling.LANCZOS)
        frame = Image.new('RGB', (512, 512), '#f3f4f1')
        frame.paste(image, ((512-image.width)//2, (512-image.height)//2))
    ax.imshow(frame)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#cbd7cd')
    ax.text(.5, -.075, label, transform=ax.transAxes, ha='center', va='top', fontsize=15, weight='bold')
    ax.text(.5, -.21, key, transform=ax.transAxes, ha='center', va='top', fontsize=10.3, color=GRAY)
fig.text(.5, .065, 'Ảnh minh họa nhãn dữ liệu; không thay thế chẩn đoán thực địa.',
         ha='center', fontsize=10.5, color=GRAY)
fig.text(.5, .025, 'Nguồn: TeaLeafAI; Billah và cộng sự (2025), Mendeley Data V2, CC BY 4.0.',
         ha='center', fontsize=9.8, color=GRAY)
save(fig, 'hinh-2-2-tam-nhom-la-che', 'Hình 2.2. Ảnh minh họa tám nhóm phân loại trong bộ dữ liệu',
     '2.2.3. Phân loại và gán nhãn dữ liệu',
     'Bộ dữ liệu gốc TeaLeafAI và Billah cùng cộng sự (2025), Mendeley Data V2, DOI: 10.17632/rrff4vb35m.2, CC BY 4.0.',
     'Đây là minh họa nhãn dữ liệu. Không sử dụng hình này như tài liệu xác minh tác nhân bệnh.')


# Actual raw-image and preprocessing example; no artificial symptom generation.
healthy = next(r for r in RECORDS if r['id'] == 'd0f5daa525912fc9')
with Image.open(ROOT / healthy['original']) as original:
    raw = original.copy()
    rgb = ImageOps.exif_transpose(original).convert('RGB')
prepared = rgb.resize((224, 224), Image.Resampling.BILINEAR)
fig, ax = canvas(5.7)
positions = [.025, .37, .715]
images = [raw, rgb, prepared]
headings = ['Ảnh gốc', 'Chuyển sang RGB', 'Đổi kích thước']
bodies = [f'{raw.width} × {raw.height} pixel', '3 kênh màu R, G, B', '224 × 224 × 3']
for x, image, title, body in zip(positions, images, headings, bodies):
    photo = ax.inset_axes([x, .27, .26, .52])
    photo.imshow(image); photo.set_xticks([]); photo.set_yticks([])
    for spine in photo.spines.values():
        spine.set_color('#cbd7cd')
    ax.text(x+.13, .87, title, ha='center', fontsize=14, weight='bold')
    ax.text(x+.13, .18, body, ha='center', fontsize=12)
arrow(ax, (.29, .53), (.355, .53)); arrow(ax, (.635, .53), (.7, .53))
ax.text(.5, .055, 'Pixel đưa vào mô hình: 0–255. Lớp Rescaling trong mô hình chuyển về [−1, 1].',
        ha='center', fontsize=11.5, color=GRAY)
save(fig, 'hinh-2-3-tien-xu-ly-anh', 'Hình 2.3. Các bước tiền xử lý ảnh đầu vào',
     '2.2.4. Tiền xử lý ảnh và chuẩn hóa kích thước',
     'Tác giả xây dựng từ ảnh UNADJUSTEDNONRAW_thumb_20d.jpg trong bộ dữ liệu gốc TeaLeafAI.',
     'Ảnh gốc có kích thước 768 × 1024 pixel. Việc thay đổi kích thước không phải phép cắt ảnh; không chuẩn hóa đầu vào hai lần.')


# Use the exact deterministic augmentation recipe used in train_expanded.py.
index = RECORDS.index(healthy)
with Image.open(ROOT / 'training/dataset' / healthy['path']) as image:
    standard = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
variants = [standard]
parameters = []
for view in [1, 2]:
    rng = random.Random(42+index+view*100000)
    image = standard.copy()
    flipped = rng.random() < .5
    if flipped:
        image = ImageOps.mirror(image)
    angle = rng.uniform(-15, 15)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR,
                         fillcolor=tuple(int(v) for v in np.asarray(image)[0, 0]))
    brightness, contrast = rng.uniform(.85, 1.15), rng.uniform(.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    variants.append(image)
    parameters.append(dict(view=view, horizontal_flip=flipped, angle=angle, brightness=brightness, contrast=contrast))
fig, axes = plt.subplots(1, 3, figsize=(9, 5.4))
fig.subplots_adjust(left=.035, right=.965, top=.91, bottom=.32, wspace=.13)
for ax, image, title in zip(axes, variants, ['Ảnh chuẩn', 'Biến thể 1', 'Biến thể 2']):
    ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, weight='bold', fontsize=14, pad=14)
    for spine in ax.spines.values():
        spine.set_color('#cbd7cd')
fig.text(.5, .23, 'Lật ngang (p = 0,5) · Xoay ±15° · Độ sáng/tương phản: 0,85–1,15', ha='center', fontsize=12)
fig.text(.5, .145, 'Giai đoạn huấn luyện bộ phân loại: 1 ảnh chuẩn + 2 biến thể cho mỗi ảnh train.', ha='center', fontsize=11.5)
fig.text(.5, .065, 'Chỉ tăng cường tập huấn luyện; giữ nguyên tập kiểm định và tập kiểm thử.', ha='center', fontsize=11.5, color=GRAY)
save(fig, 'hinh-2-4-tang-cuong-du-lieu', 'Hình 2.4. Minh họa tăng cường dữ liệu trên ảnh lá chè',
     '2.2.5. Tăng cường dữ liệu (Data Augmentation)',
     'Tác giả tạo bằng đúng phép biến đổi trong training/train_expanded.py từ ảnh healthy của bộ dữ liệu gốc.',
     'Hai biến thể minh họa giai đoạn huấn luyện bộ phân loại trên backbone đóng băng. Giai đoạn tinh chỉnh có cấu hình tăng cường riêng trong finetune_expanded.py.')


# Numbers come from the published split manifest, not rounded hypothetical values.
counts = [sum(SUMMARY['split_counts'][s].values()) for s in ['train', 'val', 'test']]
fig, ax = plt.subplots(figsize=(9, 5.6))
fig.subplots_adjust(left=.12, right=.96, top=.81, bottom=.22)
colors = [GREEN, '#6995ac', '#bf8c43']
bars = ax.bar(['Huấn luyện\n(Train)', 'Kiểm định\n(Validation)', 'Kiểm thử\n(Test)'], counts, width=.58, color=colors)
ax.set_ylim(0, 1900)
ax.set_ylabel('Số ảnh', fontsize=13)
ax.set_yticks([0, 500, 1000, 1500])
ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
for bar, count in zip(bars, counts):
    number = f'{count:,}'.replace(',', '.')
    percent = f'{count/SUMMARY["total_images"]*100:.2f}'.replace('.', ',')
    ax.text(bar.get_x()+bar.get_width()/2, count+65, f'{number} ảnh\n{percent}%',
            ha='center', va='bottom', fontsize=13, weight='bold', linespacing=1.5)
fig.text(.5, .94, 'TỔNG SỐ: 2.197 ẢNH · MỤC TIÊU CHIA: 70 / 15 / 15', ha='center', fontsize=15, weight='bold')
fig.text(.5, .08, 'Seed = 42. Nhóm ảnh gần trùng chỉ thuộc một tập; tỷ lệ thực tế xấp xỉ tỷ lệ mục tiêu.', ha='center', fontsize=11, color=GRAY)
save(fig, 'hinh-2-5-chia-tap-du-lieu', 'Hình 2.5. Phân chia bộ dữ liệu huấn luyện, kiểm định và kiểm thử',
     '2.2.6. Chia tập dữ liệu huấn luyện, kiểm định và kiểm thử',
     'Tác giả xây dựng từ training/reports/expanded-dataset-summary.json của TeaLeafAI.',
     'Tỷ lệ thực tế là 70,14% / 14,88% / 14,97%; sai lệch nhỏ do số ảnh nguyên và phải giữ nhóm ảnh gần trùng cùng một tập.')


# The exact outer model/head from train_expanded.py, not a generic CNN sketch.
fig, ax = canvas(6.0)
ax.text(.5, .95, 'KIẾN TRÚC MÔ HÌNH MOBILENETV2 TRONG TEALEAFAI', ha='center', fontsize=15, weight='bold')
xs = [.018, .272, .526, .78]
w, h = .2, .235
top = [('Đầu vào', 'RGB\n224 × 224 × 3'), ('Rescaling', 'x / 127,5 − 1\nKhoảng [−1, 1]'),
       ('MobileNetV2', 'Trọng số ImageNet\n7 × 7 × 1280'), ('GAP', 'Global Average\nPooling → 1280')]
bottom = [('Dense 8', 'Softmax\n8 điểm đầu ra'), ('Dropout', 'Tỷ lệ 0,2'), ('Dense 128', 'Hàm kích hoạt\nReLU'), ('Dropout', 'Tỷ lệ 0,3')]
for x, (title, body) in zip(xs, top):
    box(ax, x, .6, w, h, title, body, fontsize=13.2)
for x, (title, body) in zip(xs, bottom):
    box(ax, x, .225, w, h, title, body, fontsize=13.2)
for i in range(3):
    arrow(ax, (xs[i]+w+.004, .717), (xs[i+1]-.008, .717))
    arrow(ax, (xs[i+1]-.008, .342), (xs[i]+w+.004, .342))
arrow(ax, (.88, .59), (.88, .47))
ax.text(.5, .11, 'Đầu ra: 7 nhóm bệnh và 1 nhóm lá khỏe mạnh.', ha='center', fontsize=12.5)
ax.text(.5, .035, 'Huấn luyện bộ phân loại, sau đó tinh chỉnh 30 lớp cuối; giữ cố định BatchNormalization.', ha='center', fontsize=10.7, color=GRAY)
save(fig, 'hinh-2-6-kien-truc-mobilenetv2', 'Hình 2.6. Kiến trúc mô hình MobileNetV2 được sử dụng trong đề tài',
     '2.3.3. Lựa chọn mô hình MobileNetV2 cho bài toán phân loại bệnh lá chè',
     'Tác giả xây dựng từ training/train_expanded.py và training/finetune_expanded.py của TeaLeafAI.',
     'Đọc sơ đồ từ trái sang phải ở hàng trên, xuống hàng dưới rồi từ phải sang trái. GAP là phép lấy trung bình toàn cục trên hai chiều không gian.')


metadata = dict(figures=FIGURES, selected_samples=samples, augmentation_parameters=parameters,
                dataset_manifest_sha256=hashlib.sha256((ROOT/'training/reports/expanded-dataset-manifest.json').read_bytes()).hexdigest())
(OUT/'figure-metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

readme = ['# Hình minh họa chương 2 — TeaLeafAI', '',
          'PNG: 300 dpi, phù hợp chèn Word. SVG/PDF: bản vector cho sơ đồ và biểu đồ; phần ảnh mẫu vẫn là ảnh raster.',
          'File Word có sáu hình và chú thích dạng văn bản có thể sửa. Không có ảnh AI tạo ra thay cho mẫu bệnh.', '']
for f in FIGURES:
    readme.extend([f'## {f["caption"]}', '', f'- Vị trí chèn: **{f["section"]}**.',
                   f'- PNG: [{f["stem"]}.png]({f["stem"]}.png).', f'- Nguồn: {f["source"]}',
                   f'- Ghi chú: {f["note"]}', ''])
readme.extend(['## Sử dụng trong Word', '',
               'Chọn Insert → Pictures → This Device và chọn PNG. Giữ đúng tỷ lệ, đặt chiều rộng khoảng 15–16 cm.',
               'Chú thích hình đặt ngoài ảnh bằng References → Insert Caption để Word đánh số tự động.',
               'Các hình dùng nhãn kế thừa bộ dữ liệu, không khẳng định mọi ảnh đã được giám định tác nhân bệnh.',
               'Ảnh gốc thuộc bộ dữ liệu đã có của dự án; ảnh bổ sung Mendeley được ghi nguồn theo CC BY 4.0.',
               'Xem thêm [nguồn dữ liệu](../../TRAINING_DATA.md).', '',
               '## Tạo lại hình', '',
               'Cần bộ dữ liệu đã chia tại training/dataset và ảnh gốc trong DuLieuGoc.',
               'Cài matplotlib, pillow, numpy, python-docx rồi chạy `python scripts/generate_chapter2_figures.py` từ thư mục gốc.', ''])
(OUT/'README.md').write_text('\n'.join(readme))

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin, sec.bottom_margin = Cm(2), Cm(2)
sec.left_margin, sec.right_margin = Cm(2.5), Cm(2.5)
normal = doc.styles['Normal']
normal.font.name = 'Times New Roman'; normal.font.size = Pt(12)
normal.paragraph_format.space_after = Pt(8)
for i, f in enumerate(FIGURES):
    if i:
        doc.add_page_break()
    p = doc.add_paragraph('HÌNH MINH HỌA CHƯƠNG 2 — TEALEAFAI')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].bold = True
    p = doc.add_paragraph('Vị trí chèn: ' + f['section'])
    p.runs[0].bold = True
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(OUT/f'{f["stem"]}.png'), width=Cm(16))
    p = doc.add_paragraph(f['caption'])
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].bold = True
    p = doc.add_paragraph('Nguồn: ' + f['source'])
    p.runs[0].italic = True
    doc.add_paragraph(f['note'])
word = DOWNLOADS/'TeaLeafAI-hinh-chuong-2.docx'
doc.save(word)
package = DOWNLOADS/'TeaLeafAI-hinh-chuong-2.zip'
with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            archive.write(p, 'TeaLeafAI-hinh-chuong-2/'+p.name)
    archive.write(word, 'TeaLeafAI-hinh-chuong-2/'+word.name)
print(f'Created {len(FIGURES)} figures (PNG/SVG/PDF), Word and ZIP: {package.stat().st_size/1024**2:.1f} MiB.')
