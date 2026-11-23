"""Build editable Vietnamese sections 3.2/3.3 and evidence-based figures from frozen evaluations."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'training/.cache/matplotlib'))
import argparse,collections,csv,hashlib,json,shutil,zipfile
import numpy as np
from PIL import Image,ImageOps
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--output',type=Path,default=ROOT/'training/reports/evaluation-20261122')
args=p.parse_args();OUT=args.output;FIG=OUT/'figures';DATA=OUT/'data'
FIG.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
for source in (args.source_root/'native').iterdir():
    if source.is_file():
        destination=DATA/source.name
        shutil.copy2(source,destination)
        if source.suffix=='.csv':destination.write_bytes(destination.read_bytes().replace(b'\r\n',b'\n'))
for name in ['browser-predictions.json','browser-summary.json','web-result-example.json','web-result-example-panel.json']:
    shutil.copy2(args.source_root/'browser'/name,DATA/name)
shutil.copy2(args.source_root/'browser/web-result-example-panel.png',FIG/'3-2-3-web-example.png')
shutil.copy2(args.source_root/'browser/web-result-example.png',FIG/'web-result-full-page.png')
n=json.loads((DATA/'native-results.json').read_text())
br=json.loads((DATA/'browser-predictions.json').read_text());bp=br['predictions']
classes=n['classes'];N=len(bp)
manifest=json.loads((ROOT/'training/reports/expanded-dataset-manifest.json').read_text())
lookup={r['id']:r for r in manifest}
test=[r for r in manifest if r['split']=='test']
assert [r['id'] for r in bp]==[r['id'] for r in test]
truth=np.array([classes.index(r['actual']) for r in bp]);pred=np.array([classes.index(r['predicted']) for r in bp])
cr=classification_report(truth,pred,labels=list(range(8)),target_names=classes,output_dict=True,zero_division=0)
cm=confusion_matrix(truth,pred,labels=list(range(8)))
b={'accuracy':float((truth==pred).mean()),'correct':int((truth==pred).sum()),'incorrect':int((truth!=pred).sum()),
   'macro_precision':cr['macro avg']['precision'],'macro_recall':cr['macro avg']['recall'],
   'macro_f1':cr['macro avg']['f1-score'],'weighted_f1':cr['weighted avg']['f1-score'],
   'classification_report':cr,'confusion_matrix':cm.tolist()}
(DATA/'browser-metrics.json').write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf-8')
summary=json.loads((ROOT/'training/reports/expanded-dataset-summary.json').read_text())
history=json.loads((ROOT/'training/evaluation.json').read_text())['training']
head_selected=int(np.argmin(history['history']['val_loss']))+1
fine=history['fine_tuning'];fine_selected=int(np.argmin(fine['history']['val_loss']))+1
native=n['splits']['test']['keras'];warm=br['warm_timing']['diagnose'];labels=['Đốm tảo','Thán thư','Đốm mắt chim','Đốm nâu','Đốm xám','Lá khỏe','Đốm đỏ','Đốm trắng']
label=dict(zip(classes,labels));GREEN='#216b50';BLUE='#537f9c';GRAY='#56665e'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'figure.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})

def pct(value):return f'{value*100:.2f}%'.replace('.',',')
def num(value,digits=2):return f'{value:.{digits}f}'.replace('.',',')
def save(fig,stem):
    for suffix in ['png','svg','pdf']:
        dest=FIG/f'{stem}.{suffix}';fig.savefig(dest,dpi=300)
        if suffix=='svg':dest.write_text('\n'.join(line.rstrip() for line in dest.read_text().splitlines())+'\n')
    plt.close(fig)

# Fresh non-augmented scores on the same checkpoint; All explicitly includes train.
fig,ax=plt.subplots(figsize=(9,5.5));fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.83)
split_keys=['train','val','test','all'];values=[n['splits'][s]['keras']['accuracy']*100 for s in split_keys]
bars=ax.bar(['Train','Validation','Test','All*'],values,color=[GREEN,BLUE,'#b8873f','#8d9d92'],width=.6)
for bar,s in zip(bars,split_keys):
    d=n['splits'][s]['keras'];ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{pct(d["accuracy"])}\n{d["correct"]}/{d["test_images"]}',ha='center',fontsize=12,weight='bold')
ax.set_ylim(0,110);ax.set_yticks(range(0,101,20));ax.set_ylabel('Accuracy (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.text(.5,.94,'ĐÁNH GIÁ CHECKPOINT TRÊN ẢNH KHÔNG TĂNG CƯỜNG',ha='center',fontsize=14,weight='bold')
fig.text(.5,.08,'* All = train + validation + test; không dùng để đại diện độ chính xác trên ảnh mới.',ha='center',fontsize=10.5,color=GRAY)
save(fig,'3-2-2-splits')

# Select examples deterministically from the frozen manifest, not by confidence.
chosen=[]
for key in ['healthy','anthracnose','brown_blight','gray_light']:
    chosen.append(next(r for r in bp if r['actual']==key and r['correct']))
correct=[r for r in bp if r['correct']][:2];wrong=[r for r in bp if not r['correct']][:2]
case_metadata={}
def case_figure(entries,stem,title):
    fig,axes=plt.subplots(2,2,figsize=(9,9));fig.subplots_adjust(left=.035,right=.965,top=.91,bottom=.17,hspace=.54,wspace=.18)
    for index,(ax,r) in enumerate(zip(axes.flat,entries),1):
        record=lookup[r['id']];path=ROOT/'training/dataset'/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        with Image.open(path) as im:
            photo=ImageOps.contain(im.convert('RGB'),(512,512))
            frame=Image.new('RGB',(512,512),'#f1f3ef');frame.paste(photo,((512-photo.width)//2,(512-photo.height)//2))
        ax.imshow(frame);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_color(GREEN if r['correct'] else '#b4684d')
        score=max(r['probabilities'].values())
        ax.set_title(f'{index}. '+('Phân loại đúng' if r['correct'] else 'Phân loại sai'),fontsize=12.5,weight='bold',pad=9)
        ax.text(.5,-.065,f'Nhãn dữ liệu: {label[r["actual"]]}\nDự đoán: {label[r["predicted"]]} · Điểm {pct(score)}',transform=ax.transAxes,ha='center',va='top',fontsize=11,linespacing=1.5)
    fig.text(.5,.965,title,ha='center',fontsize=14,weight='bold')
    fig.text(.5,.07,'Ảnh thật trong test; điểm đầu ra không phải Accuracy. Nhãn kế thừa bộ dữ liệu nguồn.',ha='center',fontsize=10,color=GRAY)
    fig.text(.5,.03,'Nguồn ảnh và ID từng mẫu được ghi trong data/illustration-sources.json.',ha='center',fontsize=9.8,color=GRAY)
    save(fig,stem);case_metadata[stem]=[{'record':lookup[r['id']],'prediction':r} for r in entries]
case_figure(chosen,'3-2-4-prediction-examples','VÍ DỤ DỰ ĐOÁN THỰC TẾ TRÊN TẬP TEST')
case_figure(correct+wrong,'3-2-5-correct-and-wrong','CÁC TRƯỜNG HỢP PHÂN LOẠI ĐÚNG VÀ SAI')
(DATA/'illustration-sources.json').write_text(json.dumps(case_metadata,ensure_ascii=False,indent=2),encoding='utf-8')

fig,ax=plt.subplots(figsize=(9,4.8));fig.subplots_adjust(left=.22,right=.96,bottom=.20,top=.79)
for y,(name,d) in enumerate([('Keras / ONNX Python',native),('Trình duyệt Chromium',b)]):
    good=d['accuracy']*100;ax.barh(y,good,color=GREEN,height=.48);ax.barh(y,100-good,left=good,color='#d4c0b5',height=.48)
    ax.text(good/2,y,f'Đúng {d["correct"]}/{N}\n{pct(d["accuracy"])}',ha='center',va='center',color='white',fontsize=13,weight='bold')
    ax.text(good+(100-good)/2,y,f'Sai {N-d["correct"]}',ha='center',va='center',fontsize=12)
ax.set_yticks([0,1],['Keras / ONNX Python','Trình duyệt Chromium'],fontsize=12);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Tỷ lệ ảnh (%)');ax.spines[['top','right']].set_visible(False)
fig.text(.5,.92,'ACCURACY TRÊN CÙNG 329 ẢNH TEST',ha='center',fontsize=15,weight='bold')
fig.text(.5,.065,'Cùng MobileNetV2; khác đường giải mã và đổi kích thước ảnh.',ha='center',fontsize=11,color=GRAY)
save(fig,'3-3-1-accuracy')

fig,ax=plt.subplots(figsize=(9,7));fig.subplots_adjust(left=.33,right=.95,bottom=.16,top=.86)
mat=np.array([[cr[c][k]*100 for k in ['precision','recall','f1-score']] for c in classes])
ax.imshow(mat,cmap='Greens',vmin=0,vmax=100,aspect='auto')
ax.set_xticks([0,1,2],['Precision','Recall','F1-score'],fontsize=13)
ax.set_yticks(range(8),[f'{label[c]} (n={int(cr[c]["support"])})' for c in classes],fontsize=12)
for y in range(8):
    for x in range(3):ax.text(x,y,num(mat[y,x])+'%',ha='center',va='center',fontsize=12,color='white' if mat[y,x]>65 else '#183c32',weight='bold')
fig.text(.5,.94,'PRECISION, RECALL VÀ F1 THEO LỚP — TRÌNH DUYỆT',ha='center',fontsize=13.8,weight='bold')
fig.text(.5,.065,f'Macro: P {pct(b["macro_precision"])} · R {pct(b["macro_recall"])} · F1 {pct(b["macro_f1"])}',ha='center',fontsize=11.5,color=GRAY)
save(fig,'3-3-2-precision-recall-f1')

fig,ax=plt.subplots(figsize=(9,8));fig.subplots_adjust(left=.20,right=.97,bottom=.24,top=.87)
percent=cm/cm.sum(1,keepdims=True)*100;ax.imshow(percent,cmap='Greens',vmin=0,vmax=100)
ax.set_xticks(range(8),labels,rotation=40,ha='right',fontsize=11);ax.set_yticks(range(8),labels,fontsize=11)
ax.set_xlabel('Nhãn dự đoán',fontsize=13,labelpad=12);ax.set_ylabel('Nhãn dữ liệu',fontsize=13)
for y in range(8):
    for x in range(8):
        text=f'{cm[y,x]}\n{percent[y,x]:.1f}%'.replace('.',',') if cm[y,x] else '0'
        ax.text(x,y,text,ha='center',va='center',fontsize=10.5,color='white' if percent[y,x]>60 else '#183c32')
fig.text(.5,.95,'MA TRẬN NHẦM LẪN — ĐƯỜNG XỬ LÝ TRÌNH DUYỆT',ha='center',fontsize=13.5,weight='bold')
fig.text(.5,.06,'Mỗi ô: số ảnh và tỷ lệ theo hàng nhãn dữ liệu. Tổng 329 ảnh.',ha='center',fontsize=11,color=GRAY)
save(fig,'3-3-3-confusion-matrix-browser')

conditions=[{'label':'Ảnh chuẩn','metrics':native}]+n['robustness']['conditions']
fig,ax=plt.subplots(figsize=(9,6.5));fig.subplots_adjust(left=.39,right=.95,bottom=.18,top=.84)
vals=[r['metrics']['accuracy']*100 for r in conditions]
bars=ax.barh(range(len(vals)),vals,color=[GREEN]+[BLUE]*(len(vals)-1),height=.57)
for bar,r in zip(bars,conditions):ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{pct(r["metrics"]["accuracy"])} ({r["metrics"]["correct"]}/{N})',ha='left',va='center',fontsize=11.5)
ax.set_yticks(range(len(vals)),[r['label'] for r in conditions],fontsize=11.5);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Accuracy (%)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.text(.5,.94,'ĐỘ NHẠY VỚI BIẾN ĐỔI ẢNH NHÂN TẠO',ha='center',fontsize=15,weight='bold')
fig.text(.5,.065,'Mỗi điều kiện dùng cùng 329 ảnh và ONNX Python; chỉ đổi một yếu tố mỗi lần.',ha='center',fontsize=10.5,color=GRAY)
save(fig,'3-3-5-sensitivity')

# Evidence tables and prose preserve the user's section numbering.
sections=[]
def section(key,title,paragraphs,tables,figs):sections.append({'key':key,'title':title,'paragraphs':paragraphs,'tables':tables,'figures':figs})
def table(headers,rows):return {'headers':headers,'rows':[[str(v) for v in row] for row in rows]}
def figure(stem,caption,width=16):return {'stem':stem,'caption':caption,'width_cm':width}
section('3.2.1','Kết quả quá trình huấn luyện mô hình CNN',[
    'Mô hình MobileNetV2 được khởi tạo từ trọng số ImageNet và huấn luyện qua hai giai đoạn. Giai đoạn đầu đóng băng backbone, huấn luyện bộ phân loại trên đặc trưng đã lưu; giai đoạn sau tinh chỉnh 30 lớp cuối, giữ cố định BatchNormalization. Nhãn ảnh được mã hóa bằng số nguyên nên dùng Sparse Categorical Crossentropy và bộ tối ưu Adam.',
    f'Giai đoạn đầu chạy {history["epochs_completed"]} epoch trong giới hạn {history["maximum_epochs"]}, chọn checkpoint tại epoch {head_selected}; giai đoạn tinh chỉnh chạy {fine["epochs_completed"]} epoch, chọn checkpoint tại epoch {fine_selected}. Việc chọn checkpoint dựa trên validation loss, không lấy mặc định trọng số ở epoch cuối. Trọng số lớp được dùng để giảm ảnh hưởng của mất cân bằng dữ liệu.',
    'Đường cong hai giai đoạn được trình bày riêng vì cách nạp ảnh và tăng cường khác nhau. Accuracy trong nhật ký train đo trên các biến thể tăng cường; các điểm đánh giá lại checkpoint trên ảnh không tăng cường ở mục tiếp theo là phép đo khác. Validation của giai đoạn đầu dùng Pillow, giai đoạn tinh chỉnh dùng TensorFlow resize; không ghép hai đường loss thành một chuỗi đồng nhất.'
], [table(['Thông số','Bộ phân loại','Tinh chỉnh'],[
    ['Learning rate ban đầu','0,001','0,00001'],['Batch train','32','16'],['Epoch tối đa / thực chạy','20 / 14','6 / 6'],['Epoch checkpoint','9','3'],
    ['Train Accuracy tại checkpoint',pct(history['history']['accuracy'][head_selected-1]),pct(fine['history']['accuracy'][fine_selected-1])],
    ['Validation Accuracy tại checkpoint',pct(history['history']['val_accuracy'][head_selected-1]),pct(fine['history']['val_accuracy'][fine_selected-1])],
    ['Validation loss tại checkpoint',num(history['history']['val_loss'][head_selected-1],4),num(fine['history']['val_loss'][fine_selected-1],4)]])],
    [figure('3-2-1-training-history','Đường cong Accuracy và loss của hai giai đoạn huấn luyện.')])

split_rows=[]
for s,name in [('train','Train'),('val','Validation'),('test','Test'),('all','All (gồm cả train)')]:
    d=n['splits'][s]['keras'];split_rows.append([name,d['test_images'],d['correct'],pct(d['accuracy']),pct(d['macro_f1'])])
section('3.2.2','Kết quả phân loại trên các tập dữ liệu',[
    'Bộ dữ liệu sau lọc gồm 2.197 ảnh, chia thành 1.541 ảnh train, 327 ảnh validation và 329 ảnh test, xấp xỉ 70/15/15. Các nhóm ảnh gần trùng được giữ trong một tập. Đánh giá lại kiểm tra checksum của toàn bộ ảnh và giữ nguyên checkpoint, nhãn và bộ chia.',
    f'Trên ảnh RGB không tăng cường, tiền xử lý Pillow bilinear và pixel 0–255, checkpoint đạt {pct(n["splits"]["train"]["keras"]["accuracy"])} trên train, {pct(n["splits"]["val"]["keras"]["accuracy"])} trên validation và {pct(native["accuracy"])} trên test. Điểm All là {pct(n["splits"]["all"]["keras"]["accuracy"])} nhưng bao gồm ảnh train, chỉ dùng làm thống kê mô tả; không thay thế kết quả test.',
    f'Khi chạy đúng đường xử lý của website trên Chromium, {b["correct"]}/{N} ảnh test được phân loại đúng, đạt {pct(b["accuracy"])}. Tập test đã được dùng trong quá trình phát triển trước đây nên đây là đánh giá nội bộ, chưa phải kiểm định thực địa độc lập.'
], [table(['Tập','Số ảnh','Đúng','Accuracy','Macro F1'],split_rows)],
    [figure('3-2-2-splits','Accuracy trên các tập với cùng checkpoint và tiền xử lý Python.')])

section('3.2.3','Minh họa dự đoán trên website và phạm vi thử nghiệm thực địa',[
    'Website tiếp nhận ảnh tải lên hoặc chụp bằng camera, xử lý bằng React và ONNX Runtime Web ngay trong trình duyệt. Kết quả gồm nhãn dự đoán, điểm của tám nhóm và phần tra cứu hướng xử lý tương ứng; lịch sử được lưu trong trình duyệt.',
    'Trong lần kiểm tra này, hình giao diện dùng một ảnh healthy thuộc tập test cố định; đây là minh họa thao tác website, không phải một cuộc thử nghiệm thực địa độc lập. Chưa có tập ảnh mới ngoài vườn với nhãn chuyên gia để định lượng riêng hiệu quả ngoài thực địa.',
    f'Trên máy đánh giá cục bộ, thời gian diagnose sau khởi tạo, tính trên 328 lần gọi, có trung bình {num(warm["mean_ms"])} ms, trung vị {num(warm["median_ms"])} ms và p95 {num(warm["p95_ms"])} ms. Phép đo gồm đọc/đổi kích thước ảnh trong trình duyệt, chạy mô hình và xử lý kết quả; không gồm đọc file ngoài trình duyệt hay truyền dữ liệu của công cụ tự động. Đây không phải cam kết tốc độ trên điện thoại hoặc Internet.'
], [],[figure('3-2-3-web-example','Giao diện kết quả thật khi phân tích một ảnh thuộc tập test.',width=12)])

case_rows=[[label[r['actual']],label[r['predicted']],pct(max(r['probabilities'].values()))] for r in chosen]
section('3.2.4','Minh họa kết quả phân loại các trường hợp cụ thể',[
    'Bốn ví dụ dưới đây được chọn theo thứ tự manifest: lấy ảnh được phân loại đúng đầu tiên của các lớp healthy, anthracnose, brown_blight và gray_light. Nhãn nguồn, dự đoán và điểm đầu ra được lấy trực tiếp từ kết quả chạy trình duyệt.',
    'Các ví dụ minh họa giao diện và đầu ra của mô hình, không thay thế thống kê trên toàn bộ 329 ảnh. Điểm softmax cao trên một ảnh không có nghĩa hệ thống đạt cùng mức Accuracy hoặc tác nhân bệnh đã được xác minh.'
], [table(['Nhãn dữ liệu','Dự đoán','Điểm đầu ra'],case_rows)],
    [figure('3-2-4-prediction-examples','Ví dụ ảnh test và kết quả dự đoán đo được trên trình duyệt.')])

high_wrong=sum((not r['correct']) and max(r['probabilities'].values())>=.9 for r in bp)
section('3.2.5','Các trường hợp phân loại đúng và phân loại sai',[
    f'Trong {N} ảnh test, website phân loại đúng {b["correct"]} ảnh và sai {b["incorrect"]} ảnh. Hình minh họa lấy hai ảnh đúng đầu tiên và hai ảnh sai đầu tiên theo thứ tự manifest; không lựa chọn dựa trên điểm dự đoán để tạo ấn tượng về hiệu quả.',
    f'Có {high_wrong} ảnh sai vẫn đạt điểm đầu ra từ 90% trở lên. Điều này cho thấy cần phân biệt điểm đầu ra với độ chính xác đo trên tập dữ liệu. Mô hình phân loại đơn nhãn trả về một nhóm có điểm cao nhất; chưa đánh giá khả năng nhận diện đồng thời nhiều bệnh trên cùng lá.',
    'Các nguyên nhân như ảnh mờ, ánh sáng, nền phức tạp hoặc triệu chứng gần giống nhau cần được kiểm tra bằng ảnh cụ thể và thí nghiệm phù hợp. Không thể kết luận nguyên nhân sinh học chỉ từ nhãn dự đoán sai.'
], [],[figure('3-2-5-correct-and-wrong','Hai trường hợp đúng và hai trường hợp sai từ kết quả trình duyệt.')])

metrics_rows=[['Accuracy',pct(native['accuracy']),pct(b['accuracy'])],['Precision macro',pct(native['macro_precision']),pct(b['macro_precision'])],
              ['Recall macro',pct(native['macro_recall']),pct(b['macro_recall'])],['F1 macro',pct(native['macro_f1']),pct(b['macro_f1'])],['F1 weighted',pct(native['weighted_f1']),pct(b['weighted_f1'])]]
section('3.3.1','Đánh giá độ chính xác (Accuracy)',[
    'Accuracy là tỷ lệ số ảnh được phân loại đúng trên tổng số ảnh đánh giá: Accuracy = số dự đoán đúng / tổng số ảnh. Với hệ thống website, cần đánh giá cả tiền xử lý trong trình duyệt và suy luận thay vì chỉ dùng kết quả Python.',
    f'Trên {N} ảnh test nội bộ, đường xử lý trình duyệt đạt {b["correct"]}/{N} ảnh đúng, tương đương {pct(b["accuracy"])}; có {b["incorrect"]} ảnh sai. Cùng checkpoint trên Python đạt {native["correct"]}/{N}, tương đương {pct(native["accuracy"])}. Chênh lệch là {num((native["accuracy"]-b["accuracy"])*100)} điểm phần trăm.',
    'Kết quả cho thấy hệ thống có khả năng phân loại trong phạm vi bộ dữ liệu, nhưng còn nhiều trường hợp nhầm lẫn. Cần tiếp tục cải thiện mô hình và kiểm định trên ảnh thực địa mới trước khi kết luận về khả năng sử dụng rộng rãi.'
], [table(['Chỉ số','Keras/ONNX Python','Trình duyệt'],[metrics_rows[0],['Số đúng / tổng',f'{native["correct"]}/{N}',f'{b["correct"]}/{N}']])],
    [figure('3-3-1-accuracy','So sánh tỷ lệ đúng và sai trên cùng tập test.')])

class_rows=[[c,pct(cr[c]['precision']),pct(cr[c]['recall']),pct(cr[c]['f1-score']),int(cr[c]['support'])] for c in classes]
section('3.3.2','Đánh giá Precision, Recall và F1-score',[
    'Precision của mỗi lớp được tính bằng TP/(TP+FP), Recall bằng TP/(TP+FN), và F1 = 2 × Precision × Recall/(Precision+Recall). Chỉ số macro lấy trung bình đều của tám lớp sau khi tính từng lớp. F1 macro không được thay bằng phép tính trực tiếp từ Precision macro và Recall macro.',
    f'Đường xử lý trình duyệt đạt Precision macro {pct(b["macro_precision"])}, Recall macro {pct(b["macro_recall"])} và F1 macro {pct(b["macro_f1"])}. F1 weighted đạt {pct(b["weighted_f1"])} do có trọng số theo số ảnh từng lớp. Healthy có F1 {pct(cr["healthy"]["f1-score"])}, trong khi gray_light chỉ đạt {pct(cr["gray_light"]["f1-score"])}.',
    'Các chỉ số cho thấy chất lượng giữa các lớp chưa đồng đều. Lớp healthy có 101 ảnh test còn bird_eye_spot chỉ có 14 ảnh, vì vậy cần đọc kết quả cùng số mẫu hỗ trợ và chỉ số macro; không suy diễn độ tin cậy của toàn bộ mô hình từ riêng nhóm lá khỏe.'
], [table(['Chỉ số','Keras/ONNX Python','Trình duyệt'],metrics_rows[1:]),table(['Nhãn nguồn','Precision','Recall','F1','Số ảnh'],class_rows)],
    [figure('3-3-2-precision-recall-f1','Precision, Recall và F1 của từng lớp trên trình duyệt.')])

off=[]
for y in range(8):
    for x in range(8):
        if x!=y and cm[y,x]:off.append((int(cm[y,x]),classes[y],classes[x]))
off.sort(reverse=True)
error_text='; '.join(f'{count} ảnh {actual} → {predicted}' for count,actual,predicted in off[:3])
section('3.3.3','Ma trận nhầm lẫn (Confusion Matrix)',[
    'Ma trận nhầm lẫn có tám hàng và tám cột. Hàng là nhãn dữ liệu, cột là nhãn dự đoán; các ô trên đường chéo biểu thị ảnh được phân loại đúng. Hình trình bày cả số ảnh và tỷ lệ theo từng hàng để hạn chế ảnh hưởng của số mẫu không đều.',
    f'Trên trình duyệt, healthy đúng {cm[5,5]}/{int(cr["healthy"]["support"])}, gray_light đúng {cm[4,4]}/{int(cr["gray_light"]["support"])} và algal_leaf đúng {cm[0,0]}/{int(cr["algal_leaf"]["support"])}. Những nhầm lẫn lớn gồm {error_text}.',
    'Tỷ lệ đúng trên đường chéo khác biệt rõ giữa các lớp. Ma trận xác định các nhóm cần cải thiện, nhưng chưa đủ để giải thích nguyên nhân sinh học hay quy lỗi cho một đặc điểm ảnh riêng lẻ.'
], [],[figure('3-3-3-confusion-matrix-browser','Ma trận nhầm lẫn trình duyệt, chuẩn hóa theo nhãn dữ liệu.')])

comparison_rows=[['MobileNetV2','Có',pct(b['accuracy'])+' (trình duyệt)'],['ResNet50','Chưa thực nghiệm','Chưa có số liệu'],['VGG16','Chưa thực nghiệm','Chưa có số liệu']]
section('3.3.4','So sánh hiệu quả giữa các mô hình phân loại',[
    'Trong phạm vi mã nguồn và kết quả hiện có, MobileNetV2 là kiến trúc có checkpoint, chương trình huấn luyện và dữ liệu đánh giá có thể tái lập. Chưa có checkpoint hoặc kết quả huấn luyện ResNet50/VGG16 trên cùng bộ chia, nên chưa đủ cơ sở so sánh định lượng ba kiến trúc hoặc kết luận MobileNetV2 vượt trội tuyệt đối.',
    f'Keras và ONNX Python là hai cách thực thi cùng MobileNetV2, không phải hai kiến trúc. Hai bản khớp nhãn top-1 trên cả {N} ảnh test. File Keras có {num(n["model_bytes"]["keras"]/1e6)} MB và ONNX có {num(n["model_bytes"]["onnx"]/1e6)} MB (MB = 10⁶ byte); file Keras còn chứa trạng thái phục vụ huấn luyện, nên chênh lệch này không phải so sánh số tham số giữa các kiến trúc.',
    f'Đường cơ sở luôn dự đoán lớp xuất hiện nhiều nhất trong train, healthy, đạt {pct(n["majority_baseline"]["test_metrics"]["accuracy"])} trên test. Đây là đối chứng đơn giản đã tính trên cùng dữ liệu, không thay thế thử nghiệm với các mạng khác. Muốn so sánh kiến trúc, cần huấn luyện và đánh giá các mô hình trên cùng bộ chia và cùng điều kiện đo.'
], [table(['Kiến trúc','Bằng chứng thực nghiệm','Accuracy trên test'],comparison_rows),table(['Cách thực thi MobileNetV2','Dung lượng file','Accuracy test'],[
    ['Keras / Python',num(n['model_bytes']['keras']/1e6)+' MB',pct(native['accuracy'])],['ONNX / Python',num(n['model_bytes']['onnx']/1e6)+' MB',pct(n['splits']['test']['onnx']['accuracy'])],['ONNX Runtime Web',num(n['model_bytes']['onnx']/1e6)+' MB',pct(b['accuracy'])]])],
    [figure('3-3-4-comparison-evidence','Phạm vi bằng chứng thực nghiệm của các kiến trúc được đề cập.')])

robust_rows=[['Ảnh chuẩn',native['correct'],pct(native['accuracy']),'0,00']]
for r in n['robustness']['conditions']:robust_rows.append([r['label'],r['metrics']['correct'],pct(r['metrics']['accuracy']),num(r['delta_accuracy_percentage_points'])])
section('3.3.5','Phân tích các yếu tố ảnh hưởng đến kết quả',[
    'Chất lượng ảnh được kiểm tra bằng sáu điều kiện nhân tạo được xác định trước: giảm/tăng sáng, làm mờ Gaussian, giảm độ phân giải, nén JPEG và xoay ảnh. Mỗi điều kiện được áp dụng riêng cho cùng 329 ảnh test, giữ nguyên checkpoint và đường suy luận ONNX Python; báo cáo đầy đủ các mức đã thử, không chọn mức tốt nhất.',
    'Kết quả cho thấy ảnh giảm độ phân giải và ảnh bị nén/mờ có thể làm giảm Accuracy rõ rệt trong thí nghiệm này. Đây là phép thử độ nhạy với biến đổi nhân tạo, không mô phỏng đầy đủ môi trường thực địa và không chứng minh nguyên nhân của từng ảnh sai ban đầu. Điều kiện ảnh chuẩn trong bảng là ảnh sau tiền xử lý, chưa áp dụng các biến đổi thử nghiệm.',
    'Phân bố dữ liệu cũng là yếu tố cần xem xét: healthy có 673 ảnh, trong khi anthracnose và bird_eye_spot mỗi lớp chỉ có 99 ảnh. Dù đã dùng trọng số lớp, kết quả theo lớp vẫn khác nhau. Ngoài ra, thiếu thông tin đầy đủ theo cây/phiên chụp và nhãn kế thừa nguồn công khai giới hạn độ tin cậy khi khái quát ngoài bộ dữ liệu.',
    'Nền ảnh phức tạp, điều kiện ánh sáng và góc chụp cần được kiểm chứng thêm bằng tập ảnh thực địa có nhãn. Hướng cải thiện gồm kiểm tra nhãn các lớp dễ nhầm, bổ sung ảnh đa dạng theo cây/phiên chụp, thống nhất tiền xử lý giữa Python và trình duyệt, và đánh giá trên tập ngoài nguồn trước khi triển khai rộng.'
], [table(['Điều kiện (ONNX Python)','Đúng /329','Accuracy','Chênh lệch (điểm %)'],robust_rows)],
    [figure('3-3-5-sensitivity','Độ nhạy Accuracy với các biến đổi ảnh nhân tạo.'),figure('3-3-5-class-distribution','Phân bố số ảnh train, validation và test của tám lớp.')])

content={'prepared_date':'23/11/2026','measurements_native_at':n['evaluated_at'],'measurements_browser_at':br['evaluated_at'],
         'model_version':n['model_version'],'model_hashes':n['model_hashes'],
         'sections':sections,'source_commit_before_evaluation':'b36a311a707f4666dd15b3eaa632ba201830f12f',
         'evidence':{'native':'data/native-results.json','browser':'data/browser-predictions.json',
                     'training_history':'training/evaluation.json','dataset':'training/reports/expanded-dataset-summary.json'},
         'notes':['Phép đo diễn ra ngày 22/11/2026; biên soạn tài liệu ngày 23/11/2026.',
                  'Test nội bộ đã sử dụng trong phát triển; không phải kiểm định thực địa độc lập.',
                  'Không huấn luyện lại hay thay đổi bộ chia trong lần đánh giá này.',
                  'Các kiểm tra phần mềm đã đạt 8/8 Node tests, 13/13 Python tests và build frontend; đây là kiểm tra khác Accuracy.']}
(OUT/'report-content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8')

def as_text(items):
    lines=[]
    for s in items:
        lines.extend([s['key']+'. '+s['title'],''])
        for text in s['paragraphs']:lines.extend([text,''])
        for t in s['tables']:
            lines.append(' | '.join(t['headers']))
            lines.extend(' | '.join(row) for row in t['rows']);lines.append('')
        for f in s['figures']:lines.extend(['[Chèn ảnh: '+f['stem']+'.png]',f['caption'],''])
    return '\n'.join(lines)
(OUT/'noi-dung-3-2-va-3-3.txt').write_text(as_text(sections),encoding='utf-8')
(OUT/'noi-dung-3-3.txt').write_text(as_text([s for s in sections if s['key'].startswith('3.3.')]),encoding='utf-8')

def make_word(items,path):
    doc=Document();sec=doc.sections[0];sec.page_width,sec.page_height=Cm(21),Cm(29.7)
    sec.top_margin,sec.bottom_margin=Cm(2),Cm(2);sec.left_margin,sec.right_margin=Cm(2.5),Cm(2.5)
    normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
    normal.paragraph_format.line_spacing=1.1;normal.paragraph_format.space_after=Pt(6)
    for i,s in enumerate(items):
        if i==0 or s['key']=='3.3.1':
            chapter='3.2. KẾT QUẢ PHÂN LOẠI BỆNH LÁ CHÈ' if s['key'].startswith('3.2.') else '3.3. ĐÁNH GIÁ HIỆU QUẢ MÔ HÌNH'
            p=doc.add_paragraph(chapter);p.runs[0].bold=True
            p.paragraph_format.page_break_before=bool(i)
            p.paragraph_format.keep_with_next=True
        p=doc.add_paragraph(s['key']+'. '+s['title']);p.runs[0].bold=True;p.paragraph_format.keep_with_next=True
        if i and s['key']!='3.3.1':p.paragraph_format.page_break_before=True
        for text in s['paragraphs']:doc.add_paragraph(text)
        for t in s['tables']:
            tab=doc.add_table(rows=1,cols=len(t['headers']));tab.style='Table Grid'
            for c,text in zip(tab.rows[0].cells,t['headers']):c.text=text
            for row in t['rows']:
                for c,text in zip(tab.add_row().cells,row):c.text=text
            for row in tab.rows:
                pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
                for c in row.cells:
                    for p in c.paragraphs:
                        for run in p.runs:run.font.size=Pt(10.5)
            doc.add_paragraph('')
        for f in s['figures']:
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
            p.add_run().add_picture(str(FIG/(f['stem']+'.png')),width=Cm(f['width_cm']))
            p=doc.add_paragraph(f['caption']);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.runs[0].italic=True;p.runs[0].font.size=Pt(11)
    p=doc.add_paragraph('NGUỒN SỐ LIỆU VÀ PHẠM VI ĐÁNH GIÁ');p.runs[0].bold=True
    p.paragraph_format.page_break_before=True;p.paragraph_format.keep_with_next=True
    for text in content['notes']:doc.add_paragraph(text)
    doc.add_paragraph('Mã nguồn: https://github.com/khanh1023/TeaLeafAI. Số liệu, checksum, dự đoán từng ảnh và định nghĩa phép đo thời gian được lưu cùng bộ tài liệu. Hình 3.2.3 là ảnh chụp giao diện thật bằng một ảnh test; không phải ảnh thử nghiệm độc lập ngoài vườn.')
    doc.add_paragraph('Nguồn ảnh minh họa: ảnh gốc của TeaLeafAI; Billah và cộng sự (2025), DOI 10.17632/rrff4vb35m.2; Ahmad và cộng sự (2024), DOI 10.17632/tt2smzrzrs.4, bản sao Project-AgML. Nguồn bổ sung có giấy phép CC BY 4.0; dữ liệu gốc giữ điều kiện nguồn. ID và đường dẫn từng ảnh nằm trong data/illustration-sources.json.')
    doc.save(path)

full=ROOT/'downloads/TeaLeafAI-chuong-3-ket-qua-danh-gia.docx'
only=ROOT/'downloads/TeaLeafAI-muc-3-3-danh-gia.docx'
make_word(sections,full);make_word([s for s in sections if s['key'].startswith('3.3.')],only)
with zipfile.ZipFile(ROOT/'downloads/TeaLeafAI-chuong-3-anh-va-so-lieu.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():archive.write(path,'TeaLeafAI-chuong-3/'+path.relative_to(OUT).as_posix())
    for path in [full,only]:archive.write(path,'TeaLeafAI-chuong-3/'+path.name)
print('Built',len(sections),'sections;',len(list(FIG.glob('3-*.png'))),'figures; Word full/3.3, text and ZIP.')
