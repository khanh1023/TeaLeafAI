"""Build an editable Vietnamese section3.3.4 from the completed, pinned experiments."""
from pathlib import Path
import os
import json
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'training/.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement

SOURCE=ROOT/'training/reports/model-comparison-20261125'
OUT=SOURCE/'section-3-3-4';OUT.mkdir(exist_ok=True)
NAMES=['ResNet50','VGG16','MobileNetV2']
records={name:json.loads((SOURCE/name/'evaluation.json').read_text()) for name in NAMES}
catalog=json.loads((ROOT/'frontend/public/model/comparison/manifest.json').read_text())
onnx={m['name']:m for m in catalog['models']}
for name in NAMES:
    result=records[name]
    data=np.load(SOURCE/name/'predictions.npz',allow_pickle=False)
    assert len(data['truth'])==result['test_images']==329
    assert int((data['truth']==data['probabilities'].argmax(1)).sum())==result['correct']
    assert result['model_sha256']==onnx[name]['keras_sha256']
    assert result['accuracy']==result['correct']/329

def fmt(value,digits=2):return f'{value:.{digits}f}'.replace('.',',')
def percent(value):return fmt(value*100)+'%'

quality_headers=['Tiêu chí','ResNet50','VGG16','MobileNetV2']
quality_rows=[
    ['Số ảnh phân loại đúng / tổng',[f'{records[n]["correct"]}/329' for n in NAMES]],
    ['Accuracy',[percent(records[n]['accuracy']) for n in NAMES]],
    ['Precision macro',[percent(records[n]['macro_precision']) for n in NAMES]],
    ['Recall macro',[percent(records[n]['macro_recall']) for n in NAMES]],
    ['F1-score macro',[percent(records[n]['macro_f1']) for n in NAMES]]]
quality_rows=[[title,*values] for title,values in quality_rows]
resource_rows=[
    ['Tổng tham số (triệu)',*[fmt(records[n]['total_parameters']/1e6) for n in NAMES]],
    ['File Keras suy luận (MB)',*[fmt(records[n]['model_bytes']/1e6) for n in NAMES]],
    ['File ONNX triển khai (MB)',*[fmt(onnx[n]['size_bytes']/1e6) for n in NAMES]],
    ['Trung vị suy luận CPU (ms/ảnh)',*[fmt(records[n]['latency']['median_ms']) for n in NAMES]]]
intro='Để đánh giá lựa chọn kiến trúc, đề tài thực nghiệm MobileNetV2, ResNet50 và VGG16 trên cùng bộ dữ liệu gồm 1.541 ảnh huấn luyện, 327 ảnh validation và 329 ảnh kiểm thử, xấp xỉ tỷ lệ 70/15/15. Cả ba sử dụng trọng số ImageNet, bỏ bộ phân loại gốc và bổ sung GAP, Dropout, Dense128, Dropout và Dense8 Softmax. Backbone được giữ cố định khi huấn luyện. Ảnh được đưa về kích thước224 × 224; tăng cường dữ liệu chỉ áp dụng cho train. Mỗi mạng có ngân sách tối đa20 epoch, sử dụng trọng số lớp và chọn checkpoint theo validation loss.'
intro=intro.replace('thước224','thước 224').replace('đa20','đa 20')
quality=(f'ResNet50 đạt Accuracy {percent(records["ResNet50"]["accuracy"])} và F1-score macro '
         f'{percent(records["ResNet50"]["macro_f1"])}, cao nhất trong lần thực nghiệm này. '
         f'VGG16 đạt Accuracy {percent(records["VGG16"]["accuracy"])}, còn MobileNetV2 đạt '
         f'{percent(records["MobileNetV2"]["accuracy"])}. ResNet50 thể hiện chất lượng phân loại cao hơn '
         'trên bộ test đã xét; ưu thế về quy mô và thời gian xử lý được phân tích tiếp ở Bảng3.10.').replace('Bảng3.10','Bảng 3.10')
measurement='Các chỉ số Accuracy, Precision, Recall và F1-score trong bảng được đo bằng TensorFlow/Keras trên cùng329 ảnh test; các chỉ số macro lấy trung bình đều của tám lớp. Dung lượng tính theo MB = 10⁶ byte; file Keras không chứa trạng thái optimizer. Thời gian đo trên cùng CPU Intel Xeon Platinum8573C với mức cấp phát2 vCPU, batch1; làm nóng10 lần rồi đo60 lần gọi, bảng ghi trung vị, không gồm đọc/resize ảnh, tải hay khởi tạo mô hình.'
for a,b in [('cùng329','cùng 329'),('Platinum8573C','Platinum 8573C'),('phát2','phát 2'),('batch1','batch 1'),('nóng10','nóng 10'),('đo60','đo 60')]:measurement=measurement.replace(a,b)
efficiency=(f'MobileNetV2 có quy mô nhỏ nhất, khoảng {fmt(records["MobileNetV2"]["total_parameters"]/1e6)} triệu tham số và '
            f'file ONNX {fmt(onnx["MobileNetV2"]["size_bytes"]/1e6)} MB; tương ứng ở ResNet50 là '
            f'{fmt(onnx["ResNet50"]["size_bytes"]/1e6)} MB và VGG16 là {fmt(onnx["VGG16"]["size_bytes"]/1e6)} MB. '
            f'Trung vị suy luận CPU của MobileNetV2 đạt {fmt(records["MobileNetV2"]["latency"]["median_ms"])} ms/ảnh, '
            'thấp nhất trong điều kiện đo. Các số đo này cho thấy ưu thế về kích thước tệp và thời gian xử lý của MobileNetV2 trong cấu hình khảo sát.')
conclusion='MobileNetV2 được lựa chọn cho hướng triển khai website gọn nhẹ, nơi dung lượng tải và tốc độ phản hồi là các tiêu chí cần cân nhắc bên cạnh độ chính xác. ResNet50 phù hợp hơn nếu ưu tiên chất lượng phân loại trong thí nghiệm này. Việc lựa chọn mô hình vì vậy phụ thuộc vào yêu cầu ứng dụng và sự cân bằng giữa chất lượng phân loại, dung lượng và thời gian phản hồi.'
limits='Đây là thí nghiệm đối chiếu mới ngày25/11/2026, dùng backbone đóng băng và một seed42, chưa phải tìm cấu hình tối ưu cho từng kiến trúc. Các số liệu trên không phải Accuracy hay thời gian đo trên Canvas/WASM của website; chúng cũng khác đợt đánh giá checkpoint MobileNetV2 đã fine-tune ở trang Phân tích. Tập test là dữ liệu nội bộ đã sử dụng trong phát triển, chưa phải kiểm định thực địa độc lập.'
limits=limits.replace('ngày25','ngày 25').replace('seed42','seed 42')
title='3.3.4. So sánh hiệu quả giữa các mô hình phân loại'
caption='Hình minh họa. Độ chính xác, F1-score macro và thời gian suy luận CPU của ba mô hình.'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})
fig,axes=plt.subplots(2,1,figsize=(9,8.4))
fig.subplots_adjust(left=.11,right=.97,top=.89,bottom=.13,hspace=.45)
x=np.arange(3);width=.32
for values,offset,label,color in [([records[n]['accuracy']*100 for n in NAMES],-width/2,'Accuracy','#4c7897'),
                                 ([records[n]['macro_f1']*100 for n in NAMES],width/2,'F1 macro','#216b50')]:
    bars=axes[0].bar(x+offset,values,width,color=color,label=label)
    for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+1.5,fmt(value),ha='center',fontsize=13)
axes[0].set_xticks(x,NAMES);axes[0].set_ylim(0,105);axes[0].set_ylabel('Tỷ lệ (%)');axes[0].set_title('Chất lượng phân loại trên 329 ảnh test',fontsize=15,pad=13)
axes[0].legend(loc='upper right',frameon=False,ncol=2,fontsize=12)
times=[records[n]['latency']['median_ms'] for n in NAMES]
bars=axes[1].bar(NAMES,times,color=['#4c7897','#b18a4b','#216b50'],width=.53)
for bar,value in zip(bars,times):axes[1].text(bar.get_x()+bar.get_width()/2,value+8,fmt(value),ha='center',fontsize=14)
axes[1].set_ylim(0,max(times)*1.22);axes[1].set_ylabel('Trung vị (ms/ảnh)');axes[1].set_title('Suy luận TensorFlow/Keras CPU sau làm nóng',fontsize=15,pad=13)
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('SO SÁNH BA MÔ HÌNH TRONG CÙNG THÍ NGHIỆM',fontsize=17,weight='bold',y=.97)
fig.text(.5,.055,'Backbone đóng băng · seed 42 · CPU 2 vCPU · số đo TensorFlow/Keras',ha='center',fontsize=12)
stem='hinh-so-sanh-3-mo-hinh-muc-3-3-4'
for ext in ['png','svg','pdf']:
    file=OUT/f'{stem}.{ext}';fig.savefig(file,dpi=300)
    if ext=='svg':file.write_text('\n'.join(s.rstrip() for s in file.read_text().splitlines())+'\n')
plt.close(fig)

doc=Document();section=doc.sections[0]
section.page_width=Cm(21);section.page_height=Cm(29.7)
section.left_margin=Cm(3);section.right_margin=Cm(2);section.top_margin=section.bottom_margin=Cm(2)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.space_after=Pt(7)
normal.paragraph_format.keep_together=True
p=doc.add_paragraph(title);p.runs[0].bold=True;p.runs[0].font.size=Pt(14)
p.paragraph_format.keep_with_next=True
doc.add_paragraph(intro)

def add_table(caption,headers,rows):
    p=doc.add_paragraph(caption);p.runs[0].italic=True;p.paragraph_format.keep_with_next=True
    t=doc.add_table(rows=1,cols=4);t.style='Table Grid';t.autofit=False
    widths=[4.4,3.85,3.85,3.9]
    for cell,text in zip(t.rows[0].cells,headers):cell.text=text
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    for row in rows:
        for cell,text in zip(t.add_row().cells,row):cell.text=text
    for index,row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for cell,width_cm in zip(row.cells,widths):
            cell.width=Cm(width_cm)
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(3)
                for run in p.runs:run.font.size=Pt(11);run.bold=index==0

add_table('Bảng 3.9. So sánh chất lượng phân loại trên cùng tập kiểm thử.',quality_headers,quality_rows)
doc.add_paragraph(quality)
add_table('Bảng 3.10. So sánh quy mô, dung lượng và thời gian suy luận.',quality_headers,resource_rows)
p=doc.add_paragraph(measurement);p.runs[0].font.size=Pt(10.5)
p=doc.add_paragraph();p.paragraph_format.page_break_before=True;p.paragraph_format.keep_with_next=True;p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(OUT/(stem+'.png')),width=Cm(16))
p=doc.add_paragraph(caption);p.runs[0].italic=True;p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].font.size=Pt(11)
doc.add_paragraph(efficiency);doc.add_paragraph(conclusion)
p=doc.add_paragraph(limits);p.runs[0].font.size=Pt(10.5)
p=doc.add_paragraph('Nguồn: kết quả thực nghiệm và dự đoán từng ảnh trong TeaLeafAI/training/reports/model-comparison-20261125; catalog mô hình ONNX triển khai trên website.');p.runs[0].font.size=Pt(10.5)
word=ROOT/'downloads/TeaLeafAI-muc-3-3-4-bao-cao-hoan-chinh.docx';doc.save(word)
content={'title':title,'intro':intro,'quality_analysis':quality,'quality_headers':quality_headers,'quality_rows':quality_rows,
         'resource_rows':resource_rows,'measurement':measurement,'efficiency':efficiency,'conclusion':conclusion,
         'limitations':limits,'caption':caption,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'comparison.json',ROOT/'frontend/public/model/comparison/manifest.json']}}
(OUT/'report-content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
lines=[title,'',intro,'','Bảng 3.9. So sánh chất lượng phân loại trên cùng tập kiểm thử.',' | '.join(quality_headers)]
lines.extend(' | '.join(row) for row in quality_rows)
lines.extend(['',quality,'','Bảng 3.10. So sánh quy mô, dung lượng và thời gian suy luận.',' | '.join(quality_headers)])
lines.extend(' | '.join(row) for row in resource_rows)
lines.extend(['',measurement,'',caption,'',efficiency,'',conclusion,'',limits,''])
(OUT/'noi-dung-3-3-4.txt').write_text('\n'.join(lines))
with zipfile.ZipFile(ROOT/'downloads/TeaLeafAI-muc-3-3-4-bao-cao-va-hinh.zip','w',zipfile.ZIP_DEFLATED) as z:
    z.write(word,word.name)
    for p in sorted(OUT.iterdir()):z.write(p,p.name)
print('Built section3.3.4 with two editable tables, measured results, figure and scope notes.')
