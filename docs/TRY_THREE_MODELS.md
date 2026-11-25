# Thử MobileNetV2, ResNet50 và VGG16 trên website

Mở **https://tea-leaf-ai.vercel.app/so-sanh** hoặc chọn **Thử 3 mô hình** trong menu.

1. Chọn ảnh JPG/PNG/WEBP tối đa 8 MB, hoặc chụp bằng camera.
2. Nếu biết nhãn thực tế, chọn nhãn tương ứng. Nếu chưa biết, giữ **Chưa xác định nhãn**.
3. Bấm **Chạy MobileNetV2**, **Chạy ResNet50**, **Chạy VGG16**, hoặc **Chạy cả ba mô hình**.
4. Xem nhãn dự đoán, điểm của tám nhóm và thời gian suy luận. Thời gian tải/khởi tạo được tách riêng.
5. Thử thêm ảnh rồi bấm **Tải Word** hoặc **Tải CSV**. Word có bảng chỉnh sửa được và kết quả từng ảnh.

MobileNetV2 được đánh dấu **Lựa chọn của đề tài**. Dấu này không phải tuyên bố mạng luôn có Accuracy cao nhất. Ba mô hình đều được huấn luyện và suy luận thật; không điều chỉnh số liệu để thay đổi thứ hạng.

## Cách tính dữ liệu xuất báo cáo

- Lịch sử giữ tối đa 500 tệp trên thiết bị, không lưu ảnh. SHA-256 tệp và mã phiên bản thí nghiệm giúp cập nhật lần chạy lại thay vì tăng số mẫu.
- Chỉ tính Accuracy trên **cùng những tệp có nhãn và đủ kết quả cả ba mạng**. Chưa có nhãn thì không suy ra đúng/sai. Có thể sửa nhãn sau khi chạy; bảng và dữ liệu lưu sẽ cập nhật.
- Nhãn do người dùng nhập chưa mặc nhiên là nhãn chuyên gia. Thử ít ảnh không xác nhận độ chính xác chung của mô hình.
- Điểm softmax của một ảnh không phải Accuracy, và điểm cao hơn giữa hai mạng không chứng minh mạng đó tốt hơn.
- Thời gian suy luận là thời gian gọi ONNX Runtime Web với đầu vào 224×224, không gồm đọc ảnh, tải/khởi tạo. Lần chạy đầu có thể chậm; CSV lưu cờ và thời gian tải riêng.
- Các tệp được phân tích tại thiết bị, không gửi ảnh lên máy chủ. Kết quả Word/CSV tải về chứa tên tệp, nhãn, dự đoán và số đo, không nhúng ảnh.

## Mô hình và tải lần đầu

Ba mạng trong trang này được huấn luyện cùng quy trình ImageNet backbone đóng băng và đầu phân loại tám lớp. Đây là bộ thí nghiệm mới; MobileNetV2 có thể cho kết quả khác trang Phân tích đang dùng checkpoint đã fine-tune.

File ONNX khoảng 9,57 MB (MobileNetV2), 95,02 MB (ResNet50), 59,14 MB (VGG16), tải khi bấm chạy. App chỉ giữ một session so sánh trong RAM và có thể dùng lại file từ HTTP cache. ResNet50/VGG16 cần thiết bị có đủ bộ nhớ; nên thử trên máy tính khi điện thoại báo lỗi.

Ứng dụng dùng file tĩnh và ONNX Runtime Web, không cần API trả phí. Việc lưu trữ/băng thông vẫn phụ thuộc hạn mức gói hosting. Không tự thay đổi mô hình ở trang Phân tích khi chọn mạng đối chiếu.

[Kết quả kiểm thử Python trên 329 ảnh và toàn bộ bằng chứng](../training/reports/model-comparison-20261125/README.md) · [Quy trình tái lập](MODEL_COMPARISON.md).

## Chạy mã nguồn hiện tại trên máy

```bash
git clone https://github.com/khanh1023/TeaLeafAI.git
cd TeaLeafAI/frontend
npm install
npm run dev
```

Mở URL được in và vào `/so-sanh`. Không cần Python để dùng ba ONNX đã đóng gói. Bản ZIP chạy sẵn cũ chỉ có MobileNetV2; dùng website hoặc mã nguồn mới để thử cả ba.
