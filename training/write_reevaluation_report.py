"""Write a Vietnamese Word/Markdown report from measured native and browser predictions."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'training/.cache/matplotlib'))
import csv
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = ROOT/'training/reports/reevaluation'
native = json.loads((OUT/'native.json').read_text())
browser = json.loads((OUT/'browser-predictions.json').read_text())
parity = json.loads((OUT/'pixel-parity.json').read_text())
classes = native['classes']
records = browser['predictions']
frozen = [r for r in json.loads((ROOT/'training/reports/expanded-dataset-manifest.json').read_text()) if r['split']=='test']
assert [r['id'] for r in frozen] == [r['id'] for r in records]
truth = np.array([classes.index(r['actual']) for r in records])
pred = np.array([classes.index(r['predicted']) for r in records])
report = classification_report(truth, pred, labels=list(range(8)), target_names=classes, output_dict=True, zero_division=0)
b = dict(accuracy=float(accuracy_score(truth, pred)), correct=int((truth==pred).sum()), incorrect=int((truth!=pred).sum()),
         test_images=len(truth), macro_precision=report['macro avg']['precision'], macro_recall=report['macro avg']['recall'],
         macro_f1=report['macro avg']['f1-score'], weighted_f1=report['weighted avg']['f1-score'],
         classification_report=report, confusion_matrix=confusion_matrix(truth,pred,labels=list(range(8))).tolist())
(OUT/'browser-metrics.json').write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf-8')
n = native['keras']
old = np.load(OUT/'native-predictions.npz')
disagree = old['keras'].argmax(1) != pred
transition = dict(top1_disagreements=int(disagree.sum()),
                  native_correct_browser_wrong=int(((old['keras'].argmax(1)==truth)&(pred!=truth)).sum()),
                  native_wrong_browser_correct=int(((old['keras'].argmax(1)!=truth)&(pred==truth)).sum()))
(OUT/'pipeline-comparison.json').write_text(json.dumps(transition,indent=2))
with (OUT/'browser-predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['id','path','source','actual','predicted','correct'])
    for r in records:writer.writerow([r[k] for k in ['id','path','source','actual','predicted','correct']])

def pct(v):
    return f'{v*100:.2f}%'.replace('.',',')

rows = [('Số ảnh kiểm thử',str(n['test_images']),str(b['test_images'])),
        ('Dự đoán đúng',str(n['correct']),str(b['correct'])),('Dự đoán sai',str(n['incorrect']),str(b['incorrect'])),
        ('Accuracy',pct(n['accuracy']),pct(b['accuracy'])),
        ('Precision (macro)',pct(n['macro_precision']),pct(b['macro_precision'])),
        ('Recall (macro)',pct(n['macro_recall']),pct(b['macro_recall'])),
        ('F1-score (macro)',pct(n['macro_f1']),pct(b['macro_f1'])),
        ('F1-score (weighted)',pct(n['weighted_f1']),pct(b['weighted_f1']))]
summary = (f'Đánh giá lại trên tập kiểm thử gồm {n["test_images"]} ảnh cho thấy mô hình Keras và ONNX trên Python '
           f'phân loại đúng {n["correct"]} ảnh, đạt Accuracy {pct(n["accuracy"])}. '
           f'Khi chạy đúng đường xử lý ảnh của website trên trình duyệt, hệ thống phân loại đúng {b["correct"]} ảnh, '
           f'đạt Accuracy {pct(b["accuracy"])}. Đây là kết quả đánh giá nội bộ, chưa phải kiểm định thực địa độc lập.')
methods = [
    'Giữ nguyên mô hình, nhãn, ảnh và bộ chia train/validation/test. Kiểm tra SHA-256 của toàn bộ 329 ảnh test, thứ tự 8 lớp và việc các nhóm ảnh gần trùng không xuất hiện ở nhiều tập. Không huấn luyện lại hoặc chọn checkpoint bằng kết quả lần kiểm tra này.',
    'Keras/ONNX Python: ảnh RGB JPEG đã chuẩn bị, đổi kích thước bằng Pillow bilinear về 224 × 224; đầu vào float32 có pixel 0–255. Lớp Rescaling đã nằm trong mô hình.',
    f'Trình duyệt: chạy trực tiếp diagnose() và readImagePixels() của website bằng Chromium {browser["browser_version"]}; đọc ảnh, đổi kích thước bằng Canvas và suy luận bằng ONNX Runtime Web/WASM. Không gửi ảnh lên API. SHA-256 mô hình trùng manifest trên website đang triển khai.',
    f'Keras và ONNX Python khớp top-1 ở cả 329 ảnh; sai số xác suất tối đa {native["parity"]["max_absolute_probability_error"]:.2e}. Hai đường tiền xử lý đổi nhãn ở {transition["top1_disagreements"]} ảnh. Với 10 ảnh khác nhãn được kiểm tra thêm, CPU và WASM khớp khi nhận cùng điểm ảnh Canvas; sai số xác suất tối đa {parity["maximum_probability_error_on_same_pixels"]:.2e}. Kết quả này xác nhận ảnh hưởng của tiền xử lý ở 10 mẫu được đối chiếu.'
]
limits = [
    'Tập test đã được sử dụng trong quá trình phát triển trước đây; chạy lại không tạo thành một kiểm định độc lập mới. Ảnh cùng nguồn và thiếu định danh đầy đủ theo cây/phiên chụp vẫn giới hạn khả năng khái quát.',
    'Nhãn được kế thừa bộ dữ liệu nguồn, chưa được chuyên gia giám định lại toàn bộ tác nhân bệnh. Điểm dự đoán của một ảnh không phải Accuracy của mô hình.',
    'Mức 94,5% trên 400 ảnh trong file Word chưa được tái lập với checkpoint và bộ dữ liệu hiện tại. Nếu dùng mô hình hiện tại, phải ghi rõ 329 ảnh và phân biệt kết quả Python 69,30% với kết quả đường xử lý trên trình duyệt 66,87%.',
    'Các số 96,4%/94,2% và bảng so sánh ResNet50/VGG16 trong file Word không được xác nhận bởi lần đánh giá này; cần đối chiếu nhật ký hoặc cung cấp mô hình, dữ liệu và mã đánh giá tương ứng.'
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'figure.facecolor':'white'})
labels = ['Đốm tảo','Thán thư','Đốm mắt chim','Đốm nâu','Đốm xám','Lá khỏe','Đốm đỏ','Đốm trắng']
maximum = max(np.max(n['confusion_matrix']),np.max(b['confusion_matrix']))
for key, data, title in [('native',n,'Keras / ONNX Python'),('browser',b,'Đường xử lý trên trình duyệt')]:
    cm=np.array(data['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(8.5,7.0));fig.subplots_adjust(left=.20,right=.96,bottom=.25,top=.88)
    ax.imshow(cm,cmap='Greens',vmin=0,vmax=maximum)
    ax.set_xticks(range(8),labels,rotation=40,ha='right',fontsize=11)
    ax.set_yticks(range(8),labels,fontsize=11)
    ax.set_xlabel('Nhãn dự đoán',fontsize=13,labelpad=10)
    ax.set_ylabel('Nhãn thực tế',fontsize=13,labelpad=8)
    ax.set_title(f'{title}\n{data["correct"]}/{len(truth)} ảnh đúng · Accuracy {pct(data["accuracy"])}',fontsize=14,pad=15)
    for y in range(8):
        for x in range(8):
            ax.text(x,y,str(cm[y,x]),ha='center',va='center',fontsize=13,color='white' if cm[y,x]>maximum*.5 else '#183c32')
    fig.savefig(OUT/f'confusion-matrix-{key}.png',dpi=300)
    plt.close(fig)

md=['# Đánh giá lại mô hình TeaLeafAI — 18/11/2026','',summary,'',
    '| Chỉ số | Keras / ONNX Python | Trình duyệt |','| --- | ---: | ---: |']
md.extend('| '+' | '.join(row)+' |' for row in rows)
md.extend(['','## Quy trình và đối chiếu',''])
md.extend(f'- {text}' for text in methods)
for title,data,key in [('Keras / ONNX Python',n,'native'),('Trình duyệt',b,'browser')]:
    md.extend(['',f'## Chi tiết từng lớp — {title}','','| Lớp | Precision | Recall | F1 | Số ảnh |','| --- | ---: | ---: | ---: | ---: |'])
    for c in classes:
        r=data['classification_report'][c]
        md.append(f'| {c} | {pct(r["precision"])} | {pct(r["recall"])} | {pct(r["f1-score"])} | {int(r["support"])} |')
    md.extend(['',f'![Ma trận nhầm lẫn {title}](confusion-matrix-{key}.png)'])
md.extend(['','## Phạm vi sử dụng và số liệu trong báo cáo',''])
md.extend(f'- {text}' for text in limits)
md.extend(['','Macro là trung bình đều giữa 8 lớp; weighted có trọng số theo số ảnh mỗi lớp. Healthy có 101/329 ảnh test nên cần xem cả macro F1.','',
           '## Tái tạo','',
           'Dùng môi trường TensorFlow/ONNX và bộ dữ liệu đã chia của dự án. Không chạy prepare_dataset.py trước lần đánh giá này.','',
           '```bash','python training/reevaluate_expanded.py',
           '# Mở một terminal khác: cd frontend && npm run dev -- --host 127.0.0.1 --port 5179 --strictPort',
           '# Với Playwright và Chromium đã cài:','node frontend/tests/reevaluate-browser.mjs','```','',
           f'Model version: `{native["model_version"]}`.',
           f'ONNX SHA-256: `{native["onnx_sha256"]}`.',
           f'Dataset manifest SHA-256: `{native["dataset_manifest_sha256"]}`.','',
           'Dữ liệu máy đọc được: native.json, browser-metrics.json, browser-predictions.json, pixel-parity.json và các CSV.'])
(OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin,sec.bottom_margin=Cm(2),Cm(2)
sec.left_margin,sec.right_margin=Cm(2.5),Cm(2.5)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
normal.paragraph_format.space_after=Pt(7)
title=doc.add_paragraph('ĐÁNH GIÁ LẠI MÔ HÌNH TEALEAFAI — 18/11/2026')
title.alignment=WD_ALIGN_PARAGRAPH.CENTER;title.runs[0].bold=True
doc.add_paragraph(summary)
table=doc.add_table(rows=1, cols=3);table.style='Table Grid'
for c,text in zip(table.rows[0].cells,['Chỉ số','Keras / ONNX Python','Trình duyệt']):c.text=text
for row in rows:
    for cell,text in zip(table.add_row().cells,row):cell.text=text
doc.add_paragraph('Ghi chú: Precision, Recall và F1 macro là trung bình đều trên 8 lớp. Đây là tỷ lệ trên tập test nội bộ, không phải điểm dự đoán của một ảnh.')
doc.add_page_break()
doc.add_paragraph('QUY TRÌNH VÀ PHẠM VI ĐÁNH GIÁ').runs[0].bold=True
for text in methods+limits:doc.add_paragraph(text)
doc.add_page_break()
for title,data in [('Keras / ONNX Python',n),('Đường xử lý trên trình duyệt',b)]:
    doc.add_paragraph('KẾT QUẢ THEO LỚP — '+title).runs[0].bold=True
    table=doc.add_table(rows=1,cols=5);table.style='Table Grid'
    for c,text in zip(table.rows[0].cells,['Lớp','Precision','Recall','F1','Số ảnh']):c.text=text
    for label,c in zip(labels,classes):
        r=data['classification_report'][c]
        for cell,text in zip(table.add_row().cells,[label,pct(r['precision']),pct(r['recall']),pct(r['f1-score']),str(int(r['support']))]):cell.text=text
    doc.add_paragraph('')
for key,title in [('native','Keras / ONNX Python'),('browser','Đường xử lý trên trình duyệt')]:
    doc.add_page_break()
    doc.add_paragraph('MA TRẬN NHẦM LẪN — '+title).runs[0].bold=True
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(OUT/f'confusion-matrix-{key}.png'),width=Cm(16))
    doc.add_paragraph('Hàng là nhãn thực tế; cột là nhãn dự đoán. Các ô trên đường chéo biểu thị số ảnh được phân loại đúng.')
word=ROOT/'downloads/TeaLeafAI-danh-gia-lai.docx';doc.save(word)
with zipfile.ZipFile(ROOT/'downloads/TeaLeafAI-danh-gia-lai.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(OUT.iterdir()):
        if p.is_file():archive.write(p,'TeaLeafAI-danh-gia-lai/'+p.name)
    archive.write(word,'TeaLeafAI-danh-gia-lai/'+word.name)
print(summary)
print('Created Word, figures, CSV and ZIP.')
