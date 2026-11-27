# So sánh ba backbone đóng băng trên dữ liệu v2

Cùng 6.840 ảnh, bộ chia 4.788/1.026/1.026, seed42, tối đa30epoch, ImageNet backbone đóng băng, cùng head và quy tắc chọn theo validation.

| Kiến trúc | Accuracy Python test | Macro-F1 |
|---|---:|---:|
| MobileNetV2 |82,16%|0,7173|
| ResNet50 |85,58%|0,7958|
| VGG16 |83,14%|0,7737|

Cả ba được đo trên cùng1.026ảnh. MobileNetV2 trong bảng này là checkpoint đóng băng, khác mô hình fine-tune mặc định của trang Phân tích. Các số đo không phải điểm của từng ảnh người dùng tải lên.

[Báo cáo tổng hợp và Word](../retraining-v2-20261127/README.md). Các thư mục theo tên mạng có dự đoán CSV/NPZ, chỉ số và checksum checkpoint. Thư mục onnx có kiểm tra chuyển đổi trên đầy đủ test.
