"""Illustrate common deterministic augmentation operations on the same real tea leaf."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'training/.cache/matplotlib'))
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance,ImageOps
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT=ROOT/'docs/figures/chapter2';OUT.mkdir(parents=True,exist_ok=True)
STEM='minh-hoa-cac-phep-tang-cuong-du-lieu'
records=json.loads((ROOT/'training/reports/expanded-dataset-manifest.json').read_text())
record=next(r for r in records if r['id']=='d0f5daa525912fc9')
source=ROOT/'training/dataset'/record['path']
assert hashlib.sha256(source.read_bytes()).hexdigest()==record['sha256']
with Image.open(source) as image:
    baseline=ImageOps.exif_transpose(image).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
crop_box=(17,17,207,207)
variants=[
    ('(a) Ảnh đầu vào','RGB · 224 × 224 pixel',baseline),
    ('(b) Xoay ảnh','Xoay +15° ngược chiều kim đồng hồ',baseline.rotate(15,resample=Image.Resampling.BILINEAR,fillcolor=baseline.getpixel((0,0)))),
    ('(c) Lật ảnh','Lật ngang',ImageOps.mirror(baseline)),
    ('(d) Thay đổi độ sáng','Độ sáng ×1,25',ImageEnhance.Brightness(baseline).enhance(1.25)),
    ('(e) Phóng to và cắt','Cắt giữa 190 × 190\nĐưa về 224 × 224 pixel',baseline.crop(crop_box).resize((224,224),Image.Resampling.BILINEAR)),
    ('(f) Đổi độ tương phản','Độ tương phản ×1,40',ImageEnhance.Contrast(baseline).enhance(1.4)),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#183c32',
                     'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})
fig,axes=plt.subplots(2,3,figsize=(9,8.5))
fig.subplots_adjust(left=.03,right=.97,top=.88,bottom=.22,hspace=.42,wspace=.14)
fig.text(.5,.955,'MINH HỌA CÁC PHÉP TĂNG CƯỜNG DỮ LIỆU ẢNH',ha='center',fontsize=15.2,weight='bold')
for ax,(title,detail,image) in zip(axes.flat,variants):
    ax.imshow(image);ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('#cad7cd')
    ax.set_title(title,fontsize=12.6,weight='bold',pad=10)
    ax.text(.5,-.065,detail,transform=ax.transAxes,ha='center',va='top',fontsize=10.7,color='#56665e',linespacing=1.5)
fig.text(.5,.125,'Các biến thể được tạo riêng từ cùng một ảnh đầu vào; chỉ tăng cường tập huấn luyện.',
         ha='center',fontsize=10.5,color='#56665e')
fig.text(.5,.075,'Thông số trong hình dùng để minh họa các phương pháp phổ biến.',
         ha='center',fontsize=10.5,color='#56665e')
fig.text(.5,.03,'Nguồn ảnh: UNADJUSTEDNONRAW_thumb_20d.jpg, bộ dữ liệu gốc TeaLeafAI.',
         ha='center',fontsize=10.2,color='#56665e')
for suffix in ['png','svg','pdf']:
    p=OUT/f'{STEM}.{suffix}';fig.savefig(p,dpi=300)
    if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
plt.close(fig)

caption='Minh họa các phép tăng cường dữ liệu ảnh lá chè: xoay, lật, thay đổi độ sáng, phóng to/cắt và điều chỉnh độ tương phản.'
attribution='Nguồn: Tác giả tạo các biến thể bằng Pillow từ ảnh UNADJUSTEDNONRAW_thumb_20d.jpg trong bộ dữ liệu gốc TeaLeafAI.'
note=('Các phép biến đổi được áp dụng độc lập lên cùng ảnh RGB đầu vào 224 × 224 pixel. '
      'Cắt giữa lấy vùng 190 × 190 pixel, tương đương khoảng 85% mỗi chiều, rồi đưa về 224 × 224 pixel. '
      'Hình minh họa các phương pháp phổ biến; phép cắt/phóng to và các hệ số minh họa không phải toàn bộ cấu hình huấn luyện thực tế. '
      'Trong lần huấn luyện hiện tại, dự án áp dụng xoay, lật, thay đổi độ sáng và tương phản cho train; không tăng cường validation/test.')
metadata=dict(source_record=record,baseline='RGB JPEG -> Pillow bilinear resize 224x224',
              parameters={'rotation_degrees_counterclockwise':15,'rotation_fill':'original top-left pixel',
                          'horizontal_mirror':True,'brightness_factor':1.25,
                          'center_crop_box_xyxy':crop_box,'crop_output_size':[224,224],
                          'contrast_factor':1.4},
              scope='Common-method illustration, not the executed training augmentation configuration',
              variants_independent=True,source_attribution=attribution)
(OUT/f'{STEM}-source.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/f'{STEM}.md').write_text(
    '# Minh họa các phép tăng cường dữ liệu\n\n'
    f'![{caption}]({STEM}.png)\n\n**Vị trí chèn:** mục 2.2.5. Tăng cường dữ liệu ảnh.\n\n'
    f'**Chú thích:** {caption}\n\n{attribution}\n\n{note}\n\nPNG 300 dpi; có SVG/PDF.\n',encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin,sec.bottom_margin=Cm(2),Cm(2)
sec.left_margin,sec.right_margin=Cm(2.5),Cm(2.5)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
p=doc.add_paragraph('MINH HỌA CÁC PHÉP TĂNG CƯỜNG DỮ LIỆU ẢNH');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph('Vị trí chèn: 2.2.5. Tăng cường dữ liệu ảnh.').runs[0].bold=True
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT/f'{STEM}.png'),width=Cm(16))
p=doc.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph(attribution).runs[0].italic=True
doc.add_paragraph(note)
doc.save(ROOT/'downloads/TeaLeafAI-tang-cuong-du-lieu.docx')
print('Created six-panel augmentation plate: PNG 300 dpi, SVG/PDF, provenance and Word.')
