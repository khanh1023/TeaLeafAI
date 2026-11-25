"""Build the requested short section3.3.4: verified-model scope and ONNX size."""
from pathlib import Path
import os
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'frontend/public/model/comparison/manifest.json').exists():
    raise SystemExit('This is the historical23/11 report. Three models now exist; use docs/TRY_THREE_MODELS.md and the25/11 evidence instead.')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'training/.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT=ROOT/'training/reports/model-comparison-20261123'
OUT.mkdir(parents=True,exist_ok=True)
STEM='3-3-4-pham-vi-kiem-chung'
onnx=ROOT/'frontend/public/model/tea-leaf.onnx'
manifest=json.loads((ROOT/'frontend/public/model/manifest.json').read_text())
size=onnx.stat().st_size
assert hashlib.sha256(onnx.read_bytes()).hexdigest()==manifest['sha256']
assert size==9580347
title='3.3.4. So sánh hiệu quả giữa các mô hình phân loại'
paragraphs=[
    'Trong chương trình hiện có, MobileNetV2 là kiến trúc có trọng số và kết quả đánh giá có thể kiểm chứng. '
    'Chưa có thực nghiệm ResNet50 và VGG16 trên cùng bộ chia dữ liệu nên chưa đủ cơ sở kết luận kiến trúc nào tốt hơn.',
    'File MobileNetV2 định dạng ONNX có dung lượng khoảng 9,58 MB. Keras và ONNX là hai định dạng thực thi '
    'của cùng mô hình; chúng không đại diện cho hai kiến trúc khác nhau.']
caption='Hình minh họa. Phạm vi kiểm chứng các kiến trúc và dung lượng mô hình ONNX hiện tại.'
headers=['Kiến trúc','Trọng số đã huấn luyện','Kết quả kiểm thử']
rows=[['MobileNetV2','Có','Có'],['ResNet50','Chưa có','Chưa có'],['VGG16','Chưa có','Chưa có']]
content={'title':title,'paragraphs':paragraphs,'caption':caption,'headers':headers,'rows':rows,
    'onnx_bytes':size,'onnx_decimal_mb':size/1e6,'onnx_sha256':manifest['sha256'],
    'model_version':manifest['model_version'],'mb_definition':'MB = 1,000,000 bytes',
    'scope':'Existing verified checkpoint and recorded test only; no new training, predictions or invented comparisons.',
    'sources':['training/reports/evaluation-20261122/data/native-results.json',
               'training/reports/evaluation-20261122/data/browser-metrics.json',
               'frontend/public/model/manifest.json','frontend/public/model/tea-leaf.onnx']}
(OUT/'report-content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
(OUT/'noi-dung-muc-3-3-4.txt').write_text(title+'\n\n'+'\n\n'.join(paragraphs)+'\n\n'+caption+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(10,6.2),facecolor='white')
fig.text(.06,.925,'PHẠM VI KIỂM CHỨNG CÁC MÔ HÌNH',fontsize=21,weight='bold',color='#193e32')
fig.text(.06,.865,'Tình trạng hồ sơ mô hình và kết quả đánh giá hiện có',fontsize=14,color='#617468')
ax=fig.add_axes([.06,.39,.88,.39]);ax.axis('off')
table=ax.table(cellText=rows,colLabels=headers,cellLoc='center',colWidths=[.25,.4,.35],bbox=[0,0,1,1])
table.auto_set_font_size(False);table.set_fontsize(14)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('#dbe5dc');cell.set_linewidth(.8)
    if r==0:
        cell.set_facecolor('#21694f');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
    elif r==1:
        cell.set_facecolor('#edf4e9');cell.get_text().set_color('#21694f');cell.get_text().set_weight('bold')
    else:
        cell.set_facecolor('#f8faf7');cell.get_text().set_color('#69766d')
fig.add_artist(FancyBboxPatch((.06,.145),.88,.18,boxstyle='round,pad=0.008',
    transform=fig.transFigure,facecolor='#edf4e9',edgecolor='none'))
fig.text(.085,.266,'MobileNetV2 — file ONNX: 9,58 MB',fontsize=19,weight='bold',color='#21694f')
fig.text(.085,.206,'Keras và ONNX: hai định dạng của cùng mô hình.',fontsize=14,color='#325647')
fig.text(.06,.065,'Chưa đủ thực nghiệm để xếp hạng độ chính xác giữa ba kiến trúc.',fontsize=13,color='#617468')
for ext in ['png','svg','pdf']:
    path=OUT/(STEM+'.'+ext);fig.savefig(path,dpi=300)
    if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
plt.close(fig)

doc=Document();section=doc.sections[0]
section.page_width=Cm(21);section.page_height=Cm(29.7)
section.left_margin=section.right_margin=Cm(2.25);section.top_margin=section.bottom_margin=Cm(2)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.keep_together=True
p=doc.add_paragraph(title);p.runs[0].bold=True;p.runs[0].font.size=Pt(14)
for text in paragraphs:doc.add_paragraph(text)
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
p.add_run().add_picture(str(OUT/(STEM+'.png')),width=Cm(16.5))
p=doc.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].italic=True;p.runs[0].font.size=Pt(11)
p=doc.add_paragraph('Nguồn: hồ sơ huấn luyện, đánh giá và file mô hình của TeaLeafAI. '
    'Dung lượng ONNX: 9.580.347 byte; MB được tính theo 1.000.000 byte.')
p.runs[0].font.size=Pt(10.5)
word=ROOT/'downloads/TeaLeafAI-muc-3-3-4-so-sanh-mo-hinh.docx';doc.save(word)
(OUT/'README.md').write_text('''# Mục 3.3.4 — bản ngắn, ảnh và nội dung

- [Tải Word](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-muc-3-3-4-so-sanh-mo-hinh.docx).
- [PNG 300 dpi](3-3-4-pham-vi-kiem-chung.png), [SVG](3-3-4-pham-vi-kiem-chung.svg), [PDF](3-3-4-pham-vi-kiem-chung.pdf).
- [Văn bản](noi-dung-muc-3-3-4.txt), [nguồn và số liệu](report-content.json).

MobileNetV2 có trọng số và kết quả kiểm thử được lưu trong hồ sơ. ResNet50/VGG16 chưa có thực nghiệm so sánh trên cùng bộ chia. Keras và ONNX là hai định dạng của cùng mô hình. File ONNX có 9.580.347 byte, tương đương khoảng 9,58 MB thập phân.

Sinh lại bằng `python scripts/generate_model_comparison_report.py`. Không huấn luyện mới hoặc dựng số liệu Accuracy cho các mạng chưa thực nghiệm. [Quy trình thực nghiệm bổ sung trên Colab](../../../docs/MODEL_COMPARISON.md).
''')
package=ROOT/'downloads/TeaLeafAI-muc-3-3-4-anh-va-noi-dung.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(OUT.iterdir()):
        if path.is_file():archive.write(path,'TeaLeafAI-muc-3-3-4/'+path.name)
    archive.write(word,'TeaLeafAI-muc-3-3-4/'+word.name)
print('Built short section3.3.4: two paragraphs, one figure, Word and ZIP.')
