"""Draw a handcrafted-feature/rule-based image-classification diagram for section 1.4.1."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'training/.cache/matplotlib'))
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from PIL import Image
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT=ROOT/'docs/figures/chapter1';OUT.mkdir(parents=True,exist_ok=True)
STEM='phan-loai-anh-phuong-phap-truyen-thong'
records=json.loads((ROOT/'training/reports/expanded-dataset-manifest.json').read_text())
record=next(r for r in records if r['id']=='4a17b9dfa361ab74')
photo_path=ROOT/'training/dataset'/record['path']
assert hashlib.sha256(photo_path.read_bytes()).hexdigest()==record['sha256']
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#183c32','font.size':12,
                     'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})
fig,ax=plt.subplots(figsize=(9,7))
fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.025)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
GREEN='#216b50';GRAY='#56665e'


def box(x,y,w,h,title,body,fill='#eef5ef',title_size=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007,rounding_size=.014',
                               facecolor=fill,edgecolor=GREEN,linewidth=1.5))
    ax.text(x+w/2,y+h*.69,title,ha='center',va='center',fontsize=title_size,weight='bold')
    ax.text(x+w/2,y+h*.28,body,ha='center',va='center',fontsize=11.5,color=GRAY,linespacing=1.5)


def arrow(start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,color=GREEN,linewidth=1.6))


ax.text(.5,.972,'PHÂN LOẠI ẢNH BẰNG PHƯƠNG PHÁP TRUYỀN THỐNG',ha='center',fontsize=14.8,weight='bold')
ax.text(.125,.89,'Ảnh đầu vào',ha='center',fontsize=13.7,weight='bold')
photo=ax.inset_axes([.025,.57,.20,.27])
with Image.open(photo_path) as image:photo.imshow(image.convert('RGB'))
photo.set_xticks([]);photo.set_yticks([])
for spine in photo.spines.values():spine.set_color('#cad7cd')
box(.025,.27,.20,.17,'Tiền xử lý','Đổi kích thước\nTách nền\nChuyển màu',title_size=12.5)
arrow((.125,.56),(.125,.451))

ax.text(.485,.915,'TRÍCH CHỌN ĐẶC TRƯNG',ha='center',fontsize=12.5,weight='bold')
box(.335,.70,.30,.16,'Màu sắc','RGB / HSV\nGiá trị và phân bố màu',fill='#edf5ef')
box(.335,.46,.30,.16,'Hình dạng','Diện tích · Chu vi\nĐộ tròn · Dài/rộng',fill='#eff4f8')
box(.335,.22,.30,.16,'Kết cấu','Phân bố mức xám\nĐộ thô, mịn, tương phản',fill='#f8f4e9')
ax.plot([.233,.28],[.355,.355],color=GREEN,linewidth=1.5)
ax.plot([.28,.28],[.30,.78],color=GREEN,linewidth=1.5)
for y in [.78,.54,.30]:arrow((.28,y),(.325,y))
for y in [.78,.54,.30]:arrow((.644,y),(.679,y))
ax.plot([.685,.685],[.30,.78],color=GREEN,linewidth=1.5)
arrow((.685,.68),(.735,.68))

box(.745,.60,.23,.16,'Tập đặc trưng','Lựa chọn thủ công\nTính bằng thuật toán',title_size=12.5)
box(.745,.35,.23,.16,'Ngưỡng / quy tắc','Giá trị ngưỡng\nĐiều kiện định trước',title_size=12.5)
box(.745,.10,.23,.16,'Kết quả phân loại','Kết quả theo\nquy tắc đã đặt',title_size=12.4)
arrow((.86,.59),(.86,.52));arrow((.86,.34),(.86,.27))
ax.text(.5,.055,'Người thiết kế xác định đặc trưng và quy tắc phân loại. Sơ đồ minh họa nguyên lý.',
        ha='center',fontsize=10.8,color=GRAY)
ax.text(.5,.012,'Ảnh nguồn: Billah và cộng sự (2025), Mendeley V2 · DOI 10.17632/rrff4vb35m.2 · CC BY 4.0.',
        ha='center',fontsize=9.2,color=GRAY)
for suffix in ['png','svg','pdf']:
    path=OUT/f'{STEM}.{suffix}';fig.savefig(path,dpi=300)
    if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
plt.close(fig)

caption='Quy trình phân loại ảnh lá chè bằng đặc trưng thủ công và quy tắc định trước.'
source=('Nguồn: Tác giả xây dựng sơ đồ. Ảnh lá chè trích từ Billah, Sagor, Sultana và Uddin (2025), '
        'Real-Life Dataset of Tea Plant Leaf Disease Classification, Mendeley Data V2, '
        'DOI: 10.17632/rrff4vb35m.2, CC BY 4.0.')
note=('Chèn vào mục 1.4.1. Phân loại ảnh bằng phương pháp truyền thống. '
      'Sơ đồ mô tả nguyên lý, không biểu diễn kết quả chẩn đoán hoặc độ chính xác đã đo. '
      'Ngưỡng màu chỉ xác định vùng thỏa điều kiện màu; không tự xác nhận tác nhân gây bệnh.')
(OUT/f'{STEM}-source.json').write_text(json.dumps(dict(
    section='1.4.1. Phân loại ảnh bằng phương pháp truyền thống',source_record=record,
    figure_type='Conceptual handcrafted-feature and deterministic-rule pipeline; no measured classification output',
    source_attribution=source),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/f'{STEM}.md').write_text(
    '# Phân loại ảnh bằng phương pháp truyền thống\n\n'
    f'![{caption}]({STEM}.png)\n\n**Vị trí chèn:** mục 1.4.1.\n\n'
    f'**Chú thích:** {caption}\n\n{source}\n\n{note}\n\nPNG 300 dpi; có bản SVG/PDF.\n',encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin,sec.bottom_margin=Cm(2),Cm(2)
sec.left_margin,sec.right_margin=Cm(2.5),Cm(2.5)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
p=doc.add_paragraph('PHÂN LOẠI ẢNH BẰNG PHƯƠNG PHÁP TRUYỀN THỐNG');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph('Vị trí chèn: 1.4.1. Phân loại ảnh bằng phương pháp truyền thống.').runs[0].bold=True
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT/f'{STEM}.png'),width=Cm(16))
p=doc.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph(source).runs[0].italic=True
doc.add_paragraph(note)
doc.save(ROOT/'downloads/TeaLeafAI-phan-loai-anh-truyen-thong.docx')
print('Created traditional classification diagram: PNG 300 dpi, SVG/PDF and Word.')
