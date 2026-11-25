"""Publish small, auditable training evidence; ONNX weights live in frontend/public/model/comparison."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);args=p.parse_args()
run=args.source/'run';out=ROOT/'training/reports/model-comparison-20261125';out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'training/reports/expanded-dataset-manifest.json').read_text())
ids=[r['id'] for r in manifest if r['split']=='test']
truth=np.array([['algal_leaf','anthracnose','bird_eye_spot','brown_blight','gray_light','healthy','red_leaf_spot','white_spot'].index(r['label']) for r in manifest if r['split']=='test'])
results=[]
for name in ['MobileNetV2','ResNet50','VGG16']:
    report=json.loads((run/name/'evaluation.json').read_text())
    data=np.load(run/name/'predictions.npz',allow_pickle=False)
    assert data['ids'].tolist()==ids and np.array_equal(data['truth'],truth)
    pred=data['probabilities'].argmax(1);cm=np.bincount(truth*8+pred,minlength=64).reshape(8,8)
    assert cm.tolist()==report['confusion_matrix'] and int(np.trace(cm))==report['correct']
    assert abs(float((truth==pred).mean())-report['accuracy'])<1e-12
    precision=np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(8),where=cm.sum(0)>0)
    recall=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(8),where=cm.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(8),where=precision+recall>0)
    assert abs(float(f1.mean())-report['macro_f1'])<1e-12
    dest=out/name;dest.mkdir(exist_ok=True)
    for file in ['evaluation.json','predictions.csv','predictions.npz']:shutil.copy2(run/name/file,dest/file)
    results.append(report)
for source in run.iterdir():
    if source.is_file():shutil.copy2(source,out/source.name)
shutil.copy2(args.source/'runtime-environment.json',out/'runtime-environment.json')
for source in (args.source/'export').iterdir():
    if source.is_file():shutil.copy2(source,out/source.name)
qa=out/'browser-qa';qa.mkdir(exist_ok=True)
for file in ['browser-qa.json','three-models-desktop.png','three-models-mobile.png']:
    shutil.copy2(args.source/'browser-qa'/file,qa/file)
for source in out.rglob('*.svg'):
    source.write_text('\n'.join(line.rstrip() for line in source.read_text().splitlines())+'\n')
for source in out.rglob('*.csv'):
    source.write_bytes(source.read_bytes().replace(b'\r\n',b'\n'))
lines=['# Thực nghiệm ba mô hình và chức năng thử trong trình duyệt — 25/11/2026','',
    '[Dùng trên website](https://tea-leaf-ai.vercel.app/so-sanh) · [Hướng dẫn](../../../docs/TRY_THREE_MODELS.md)','',
    'Thực nghiệm mới: ImageNet backbone đóng băng, cùng bộ phân loại, dữ liệu, tăng cường chỉ train và quy tắc chọn checkpoint theo validation loss. Seed42, tối đa20 epoch. Test329 ảnh, không dùng test để chọn epoch. Đây là đánh giá nội bộ đã dùng trong phát triển, chưa phải kiểm định thực địa độc lập.','',
    '| Mô hình | Đúng/329 | Accuracy | Macro F1 | Keras không optimizer (MB) | TF CPU warm median (ms) |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
for r in results:lines.append(f'| {r["name"]} | {r["correct"]}/329 | {r["accuracy"]*100:.2f}% | {r["macro_f1"]*100:.2f}% | {r["model_bytes"]/1e6:.2f} | {r["latency"]["median_ms"]:.2f} |')
lines.extend(['','Các số trên đo bằng Python/TensorFlow, không phải kết quả Canvas/WASM của trình duyệt. MobileNetV2 của thí nghiệm này khác checkpoint đã fine-tune ở trang Phân tích. Không ghép hai lần đánh giá để xếp hạng. Thời gian CPU là60 lời gọi warm batch1, bỏ10 warm-up, cùng30 ảnh×2 lượt; không gồm đọc/resize/tải và dùng p95 NumPy method=higher. Môi trường CPU2 vCPU, TF intra-op2/inter-op1 được lưu riêng.','',
    'Trang Thử3 mô hình chạy các bản ONNX đã chuyển từ chính ba checkpoint này, tải khi bấm chạy và giữ một session so sánh trong RAM. Nhãn do người dùng nhập được dùng để tính Accuracy trên đúng giao của các ảnh đã chạy đủ cả ba mạng; không lấy điểm softmax làm Accuracy.','',
    '- [Bảng, hình và Word từ thí nghiệm Python](So-sanh-mo-hinh-muc-3-3-4.docx) · [Biểu đồ](comparison.png) · [CSV](comparison.csv).',
    '- [Cấu hình](protocol.json), [môi trường CPU](runtime-environment.json), [kết quả tổng hợp](comparison.json).',
    '- [Catalog ONNX và kết quả kiểm tra chuyển đổi](browser-model-catalog.json); mỗi mạng có JSON parity riêng.',
    '- Thư mục từng mạng lưu ma trận, chỉ số, lịch sử và329 dự đoán CSV/NPZ. Trọng số Keras được giữ trong workspace; ONNX được cung cấp ở frontend/public/model/comparison.',
    '- browser-qa ghi kiểm tra chức năng trên **một ảnh test healthy**, không phải đánh giá Accuracy trên ảnh thực địa. Ảnh chụp giao diện là ảnh thật, không được dùng làm bằng chứng100% Accuracy của mô hình.',
    '- Nguồn ảnh và giấy phép: [TRAINING_DATA.md](../../../docs/TRAINING_DATA.md).',''])
text='\n'.join(lines)
for a,b in [('Seed42','Seed 42'),('đa20','đa 20'),('Test329','Test 329'),('là60','là 60'),('batch1','batch 1'),('bỏ10','bỏ 10'),('cùng30','cùng 30'),('CPU2','CPU 2'),('intra-op2/inter-op1','intra-op 2/inter-op 1'),('Thử3','Thử 3'),('và329','và 329'),('bằng chứng100','bằng chứng 100')]:text=text.replace(a,b)
(out/'README.md').write_text(text)
word=ROOT/'downloads/TeaLeafAI-thuc-nghiem-3-mo-hinh.docx';shutil.copy2(out/'So-sanh-mo-hinh-muc-3-3-4.docx',word)
with zipfile.ZipFile(ROOT/'downloads/TeaLeafAI-thuc-nghiem-3-mo-hinh.zip','w',zipfile.ZIP_DEFLATED) as z:
    for source in sorted(out.rglob('*')):
        if source.is_file():z.write(source,'TeaLeafAI-thuc-nghiem-3-mo-hinh/'+source.relative_to(out).as_posix())
print('Recomputed all three confusion matrices/accuracy/macroF1 and packaged verified results without duplicating large model files.')
