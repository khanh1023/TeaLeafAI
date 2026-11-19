"""Compose an academic photo plate of visible leaf features without inventing symptoms."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'training/.cache/matplotlib'))
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = ROOT/'docs/figures/chapter1'
OUT.mkdir(parents=True, exist_ok=True)
STEM = 'dac-diem-hinh-anh-la-che-bi-benh'
records = json.loads((ROOT/'training/reports/expanded-dataset-manifest.json').read_text())
panels = [
    dict(id='b73bdf0d6702dbde', title='(a) Đốm trên phiến lá',
         description='Các đốm có màu khác\nvùng mô xung quanh',
         circle=(246,235,32), arrow_start=(395,365), arrow_end=(271,264)),
    dict(id='60d832a018767c20', title='(b) Vùng lá biến màu',
         description='Phiến lá có vùng vàng,\nxanh nhạt xen kẽ',
         circle=(274,276,43), arrow_start=(435,155), arrow_end=(307,247)),
    dict(id='4a17b9dfa361ab74', title='(c) Mảng mô nâu, khô',
         description='Mảng nâu xám, bề mặt\ncó biểu hiện khô',
         circle=(265,290,43), arrow_start=(95,389), arrow_end=(233,320)),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#183c32',
                     'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})
fig, axes = plt.subplots(1,3,figsize=(9,5.8))
fig.subplots_adjust(left=.025,right=.975,top=.83,bottom=.29,wspace=.10)
fig.text(.5,.95,'ĐẶC ĐIỂM HÌNH ẢNH CỦA LÁ CHÈ BỊ BỆNH',ha='center',fontsize=15.3,weight='bold')
provenance=[]
for ax,panel in zip(axes,panels):
    record=next(r for r in records if r['id']==panel['id'])
    path=ROOT/'training/dataset'/record['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError(f'Source image changed: {path}')
    with Image.open(path) as image:
        image=image.convert('RGB')
        # Preserve the photo pixels and aspect ratio; only add letterboxing for layout.
        frame=Image.new('RGB',(512,512),'#f2f3f0')
        frame.paste(image,((512-image.width)//2,(512-image.height)//2))
    ax.imshow(frame)
    x,y,radius=panel['circle']
    ax.add_patch(Circle((x,y),radius,fill=False,edgecolor='#efb74a',linewidth=1.8))
    ax.annotate('',xy=panel['arrow_end'],xytext=panel['arrow_start'],
                arrowprops=dict(arrowstyle='-|>',color='#efb74a',lw=1.8,mutation_scale=15))
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('#cad7cd')
    ax.set_title(panel['title'],fontsize=12.8,weight='bold',pad=12)
    ax.text(.5,-.075,panel['description'],transform=ax.transAxes,ha='center',va='top',
            fontsize=11.8,color='#56665e',linespacing=1.5)
    provenance.append(dict(panel=panel['title'],visible_feature=panel['description'].replace('\n',' '),
                           source_record=record,annotations=panel))
fig.text(.5,.145,'Các dấu hiệu được mô tả theo ảnh; chỉ hình thái chưa đủ để xác định tác nhân gây bệnh.',
         ha='center',fontsize=10,color='#56665e')
fig.text(.5,.085,'Nguồn (a): bộ dữ liệu gốc TeaLeafAI. Nguồn (b, c): Billah và cộng sự (2025),',
         ha='center',fontsize=9.8,color='#56665e')
fig.text(.5,.04,'Mendeley Data V2 · DOI: 10.17632/rrff4vb35m.2 · CC BY 4.0.',
         ha='center',fontsize=9.8,color='#56665e')
for suffix in ['png','svg','pdf']:
    p=OUT/f'{STEM}.{suffix}';fig.savefig(p,dpi=300)
    if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
plt.close(fig)

source=('Nguồn: (a) bộ dữ liệu gốc TeaLeafAI; (b, c) Billah, Sagor, Sultana và Uddin (2025), '
        'Real-Life Dataset of Tea Plant Leaf Disease Classification, Mendeley Data V2, '
        'DOI: 10.17632/rrff4vb35m.2, giấy phép CC BY 4.0.')
note=('Màu ảnh và nội dung lá được giữ theo ảnh nguồn; vòng tròn và mũi tên chỉ vùng minh họa. '
      'Các biểu hiện được mô tả qua quan sát ảnh, không khẳng định tác nhân gây bệnh. '
      'Không dùng phần mép lá bị khuyết để kết luận bệnh vì tổn thương đó cũng có thể do nguyên nhân khác.')
(OUT/f'{STEM}-sources.json').write_text(json.dumps(dict(
    section='1.3.3. Đặc điểm hình ảnh của lá chè bị bệnh',
    display_operations=['RGB source photos, unchanged colors','aspect ratio preserved with letterboxing',
                        'circles and arrows mark visible features; no synthetic lesions'],
    source_attribution=source,panels=provenance),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/f'{STEM}.md').write_text(
    '# Đặc điểm hình ảnh của lá chè bị bệnh\n\n'
    f'![Đặc điểm hình ảnh của lá chè bị bệnh]({STEM}.png)\n\n'
    '**Vị trí chèn:** mục 1.3.3.\n\n'
    '**Chú thích:** Một số đặc điểm hình ảnh quan sát trên lá chè trong bộ dữ liệu bệnh.\n\n'
    f'{source}\n\n{note}\n\nPNG 300 dpi; có bản SVG/PDF.\n',encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin,sec.bottom_margin=Cm(2),Cm(2)
sec.left_margin,sec.right_margin=Cm(2.5),Cm(2.5)
style=doc.styles['Normal'];style.font.name='Times New Roman';style.font.size=Pt(12)
p=doc.add_paragraph('ĐẶC ĐIỂM HÌNH ẢNH CỦA LÁ CHÈ BỊ BỆNH');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph('Vị trí chèn: 1.3.3. Đặc điểm hình ảnh của lá chè bị bệnh.').runs[0].bold=True
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT/f'{STEM}.png'),width=Cm(16))
p=doc.add_paragraph('Một số đặc điểm hình ảnh quan sát trên lá chè trong bộ dữ liệu bệnh.')
p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].bold=True
doc.add_paragraph(source).runs[0].italic=True
for panel in panels:doc.add_paragraph(panel['title']+': '+panel['description'].replace('\n',' ')+'.')
doc.add_paragraph(note)
doc.save(ROOT/'downloads/TeaLeafAI-dac-diem-la-che-bi-benh.docx')
print('Created a three-photo scientific plate (PNG 300 dpi, SVG/PDF), provenance and Word.')
