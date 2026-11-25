import { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, HeadingLevel } from 'docx';
import { comparisonSummary } from './comparison-history.js';
import { MODEL_IDS } from './comparison-api.js';
import { LABELS } from './classes.js';

const order = ['resnet50', 'vgg16', 'mobilenetv2'];
const names = { mobilenetv2: 'MobileNetV2', resnet50: 'ResNet50', vgg16: 'VGG16' };
const number = value => value.toLocaleString('vi-VN', { maximumFractionDigits: 2 });
const paragraph = text => new Paragraph({ children: [new TextRun({ text, font: 'Times New Roman', size: 24 })], spacing: { after: 140 } });
const table = rows => new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: rows.map((row, index) => new TableRow({
  tableHeader: index === 0,
  children: row.map(text => new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: index === 0, font: 'Times New Roman', size: 22 })] })] }))
})) });

export async function exportComparisonWord(trials, catalog) {
  const summary = comparisonSummary(trials);
  const lookup = id => summary.find(row => row.id === id);
  const rows = [['Tiêu chí', ...order.map(id => names[id])],
    ['Ảnh chung có nhãn người dùng', ...order.map(id => lookup(id).labelled)],
    ['Số ảnh đúng', ...order.map(id => lookup(id).correct)],
    ['Accuracy trên ảnh chung có nhãn', ...order.map(id => lookup(id).accuracy === null ? 'Chưa có dữ liệu' : `${number(lookup(id).accuracy * 100)}%`)],
    ['Suy luận trung bình trên ảnh chung (ms)', ...order.map(id => lookup(id).mean_inference_ms === null ? 'Chưa có dữ liệu' : number(lookup(id).mean_inference_ms))],
    ['Dung lượng ONNX (MB)', ...order.map(id => number(catalog.models.find(model => model.id === id).size_bytes / 1e6))],
    ['Vai trò', 'Mô hình đối chiếu', 'Mô hình đối chiếu', 'Lựa chọn của đề tài']];
  const details = [['Ảnh / nhãn người dùng', ...order.map(id => names[id])], ...trials.map(trial => [
    `${trial.filename}\n${trial.actual_label ? LABELS[trial.actual_label] : 'Chưa xác định nhãn'}`,
    ...order.map(id => { const result = trial.results[id]; return result && !result.error
      ? `${result.predicted_label || LABELS[result.predicted_class]}\nĐiểm ${number(result.confidence * 100)}%\nSuy luận ${number(result.inference_ms)} ms`
      : 'Chưa có kết quả'; })])];
  const children = [new Paragraph({ text: '3.3.4. So sánh hiệu quả giữa các mô hình phân loại', heading: HeadingLevel.HEADING_1 }),
    paragraph('Báo cáo được xuất từ chức năng Thử ba mô hình của TeaLeafAI. Cùng một ảnh được phân tích bằng MobileNetV2, ResNet50 và VGG16 trên trình duyệt. Mô hình được tải khi chạy; ảnh không được gửi lên máy chủ.'),
    paragraph(`Đã lưu ${trials.length} ảnh; ${summary[0].shared_runs} ảnh có đủ kết quả của ba mô hình và ${summary[0].labelled} ảnh chung có nhãn do người dùng cung cấp. Cùng một ảnh và phiên bản thí nghiệm chỉ được tính một lần.`),
    paragraph('Bảng 3.9. Đối chiếu các mô hình trên ảnh đã thử trong trình duyệt.'), table(rows),
    paragraph('Accuracy chỉ được tính khi ảnh có nhãn người dùng và đủ kết quả của cả ba mô hình. Đây là kết quả trên các ảnh đã thử, chưa phải độ chính xác chung hay kiểm định độc lập. Nhãn do người dùng nhập chưa mặc nhiên là nhãn được chuyên gia xác nhận.'),
    paragraph('Thời gian suy luận được đo quanh lời gọi mô hình trong trình duyệt, không gồm đọc ảnh, tải và khởi tạo mô hình. Số trung bình có thể gồm lần chạy đầu sau tải; tải và khởi tạo được lưu riêng trong CSV. Thời gian phụ thuộc thiết bị và trạng thái trình duyệt. MB = 1.000.000 byte.'),
    paragraph('Bảng 3.10. Kết quả trên từng ảnh đã thử.'), table(details),
    paragraph(`Xuất lúc: ${new Date().toLocaleString('vi-VN')}. Trình duyệt: ${navigator.userAgent}. WASM dùng một luồng. Mã thí nghiệm: ${catalog.protocol_id}.`),
    ...MODEL_IDS.map(id => { const model = catalog.models.find(item => item.id === id); return paragraph(`${model.name}: ${model.model_version}; SHA-256: ${model.sha256}.`); })];
  const doc = new Document({ sections: [{ properties: {}, children }] });
  const blob = await Packer.toBlob(doc);
  const url = URL.createObjectURL(blob); const link = document.createElement('a');
  link.href = url; link.download = 'TeaLeafAI-ket-qua-thu-ba-mo-hinh.docx'; document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
