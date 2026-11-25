# Thực nghiệm ba mô hình và chức năng thử trong trình duyệt — 25/11/2026

[Dùng trên website](https://tea-leaf-ai.vercel.app/so-sanh) · [Hướng dẫn](../../../docs/TRY_THREE_MODELS.md)

Thực nghiệm mới: ImageNet backbone đóng băng, cùng bộ phân loại, dữ liệu, tăng cường chỉ train và quy tắc chọn checkpoint theo validation loss. Seed 42, tối đa 20 epoch. Test 329 ảnh, không dùng test để chọn epoch. Đây là đánh giá nội bộ đã dùng trong phát triển, chưa phải kiểm định thực địa độc lập.

| Mô hình | Đúng/329 | Accuracy | Macro F1 | Keras không optimizer (MB) | TF CPU warm median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2 | 222/329 | 67.48% | 62.90% | 10.30 | 152.49 |
| ResNet50 | 247/329 | 75.08% | 72.28% | 96.08 | 315.82 |
| VGG16 | 229/329 | 69.60% | 66.07% | 59.23 | 282.13 |

Các số trên đo bằng Python/TensorFlow, không phải kết quả Canvas/WASM của trình duyệt. MobileNetV2 của thí nghiệm này khác checkpoint đã fine-tune ở trang Phân tích. Không ghép hai lần đánh giá để xếp hạng. Thời gian CPU là 60 lời gọi warm batch 1, bỏ 10 warm-up, cùng 30 ảnh×2 lượt; không gồm đọc/resize/tải và dùng p95 NumPy method=higher. Môi trường CPU 2 vCPU, TF intra-op 2/inter-op 1 được lưu riêng.

Trang Thử 3 mô hình chạy các bản ONNX đã chuyển từ chính ba checkpoint này, tải khi bấm chạy và giữ một session so sánh trong RAM. Nhãn do người dùng nhập được dùng để tính Accuracy trên đúng giao của các ảnh đã chạy đủ cả ba mạng; không lấy điểm softmax làm Accuracy.

- [Bảng, hình và Word từ thí nghiệm Python](So-sanh-mo-hinh-muc-3-3-4.docx) · [Biểu đồ](comparison.png) · [CSV](comparison.csv).
- [Cấu hình](protocol.json), [môi trường CPU](runtime-environment.json), [kết quả tổng hợp](comparison.json).
- [Catalog ONNX và kết quả kiểm tra chuyển đổi](browser-model-catalog.json); mỗi mạng có JSON parity riêng.
- Thư mục từng mạng lưu ma trận, chỉ số, lịch sử và 329 dự đoán CSV/NPZ. Trọng số Keras được giữ trong workspace; ONNX được cung cấp ở frontend/public/model/comparison.
- browser-qa ghi kiểm tra chức năng trên **một ảnh test healthy**, không phải đánh giá Accuracy trên ảnh thực địa. Ảnh chụp giao diện là ảnh thật, không được dùng làm bằng chứng 100% Accuracy của mô hình.
- Nguồn ảnh và giấy phép: [TRAINING_DATA.md](../../../docs/TRAINING_DATA.md).
