# Huấn luyện và đánh giá TeaLeafAI — dữ liệu v2

[Tải Word có bảng chỉnh sửa được](TeaLeafAI-ket-qua-huan-luyen-v2.docx) · [Số liệu JSON](summary.json) · [Bảng CSV](comparison.csv)

Bộ v2 có 6.840 ảnh: 4.788 train, 1.026 validation và 1.026 test, đúng tỷ lệ tổng 70/15/15. Ảnh tăng cường chỉ áp dụng cho train; bốn ảnh dự phòng không tham gia đánh giá.

MobileNetV2 dùng checkpoint sau fine-tuning các lớp cuối, được chọn theo validation loss trước khi đánh giá test.

Với Python/Pillow và Keras, checkpoint mặc định phân loại đúng 868/1.026 ảnh test, Accuracy 84,60%, Precision macro 72,83%, Recall macro 80,81% và F1 macro 75,64%.

Quy tắc luôn đoán lớp nhiều nhất trong train, healthy, đúng 467/1.026 ảnh (45,52%). Đây là đường cơ sở trên cùng v2, không phải một mạng CNN được huấn luyện.

Ba mô hình so sánh bên dưới đều dùng backbone ImageNet đóng băng và cùng quy trình head/augmentation/budget. MobileNetV2 trong bảng là checkpoint thí nghiệm so sánh, được phân biệt với checkpoint mặc định. Kết quả của hai phiên bản không được ghép để xếp hạng.

Thời gian đối chiếu là lời gọi toàn mô hình TensorFlow eager sau warm-up, batch 1, không gồm đọc ảnh, resize, tải mô hình hoặc mạng. Không dùng thời gian này làm tốc độ trình duyệt.

Kết quả thuộc bộ test nội bộ theo nhóm ảnh, chưa phải kiểm định thực địa độc lập. Thông tin cây/phiên chụp nguồn còn thiếu. Không so sánh trực tiếp số Accuracy v1 và v2 để kết luận mức cải thiện, vì ảnh và bộ chia đã thay đổi.

ONNX/Python của checkpoint mặc định được kiểm tra trên đủ 1.026 tensor test; đúng 868/1.026, Accuracy 84,60%. Đối chiếu Keras–ONNX là kiểm tra chuyển đổi cùng mô hình, không phải hai kiến trúc.

Đường xử lý Canvas và ONNX Runtime Web trong Chromium đạt 863/1.026, Accuracy 84,11%, F1 macro 75,86%. Đây là phép đo riêng trên đủ test; khác biệt với Python có thể đến từ đọc/đổi kích thước ảnh.

| Mô hình | Đúng/test | Accuracy | Precision macro | Recall macro | F1 macro | Keras (MB) | TF warm median (ms) | ONNX (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2 | 843/1026 | 82,16% | 69,25% | 75,63% | 71,73% | 10.30 | 151.92 | 9.57 |
| ResNet50 | 878/1026 | 85,58% | 77,30% | 82,98% | 79,58% | 96.08 | 305.77 | 95.02 |
| VGG16 | 853/1026 | 83,14% | 74,75% | 81,13% | 77,37% | 59.23 | 184.24 | 59.14 |

## Hình và bằng chứng

- [Ma trận mặc định](01-default-confusion.png) · [Lịch sử huấn luyện](02-training-history.png) · [So sánh có kiểm soát](03-controlled-comparison.png).
- [Báo cáo mặc định gốc](evidence/default/evaluation.json) và [kết quả so sánh gốc](evidence/comparison/comparison.json).
- Thư mục evidence giữ lịch sử, dự đoán và thông tin xuất mô hình thật; không nhân đôi checkpoint lớn.
- Không có kết luận cải thiện v1–v2 trên hai bộ test khác nhau; không khẳng định MobileNetV2 luôn vượt trội.
- Báo cáo không tự xác nhận trạng thái triển khai. Cần đối chiếu manifest trên tên miền thật với bản xuất đã kiểm tra.

Dataset manifest SHA-256: `47a24012c4f67957c9ec98b997e37404575738d068ec004c415ab7bd03ecf001`.
Checkpoint mặc định SHA-256: `223a253d6619d19117c0c44e411f86916149a5d7f29851e72d284cff83febb03`.
