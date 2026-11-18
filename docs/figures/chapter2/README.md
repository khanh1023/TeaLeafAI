# Hình minh họa chương 2 — TeaLeafAI

PNG: 300 dpi, phù hợp chèn Word. SVG/PDF: bản vector cho sơ đồ và biểu đồ; phần ảnh mẫu vẫn là ảnh raster.
File Word có sáu hình và chú thích dạng văn bản có thể sửa. Không có ảnh AI tạo ra thay cho mẫu bệnh.

## Hình 2.1. Quy trình phân tích ảnh lá chè của hệ thống TeaLeafAI

- Vị trí chèn: **2.1.3. Đầu vào, đầu ra và yêu cầu hệ thống**.
- PNG: [hinh-2-1-quy-trinh-he-thong.png](hinh-2-1-quy-trinh-he-thong.png).
- Nguồn: Tác giả xây dựng từ mã nguồn TeaLeafAI.
- Ghi chú: Đặt sau đoạn giới thiệu quy trình xử lý. Lịch sử chỉ lưu thông tin kết quả, không lưu ảnh.

## Hình 2.2. Ảnh minh họa tám nhóm phân loại trong bộ dữ liệu

- Vị trí chèn: **2.2.3. Phân loại và gán nhãn dữ liệu**.
- PNG: [hinh-2-2-tam-nhom-la-che.png](hinh-2-2-tam-nhom-la-che.png).
- Nguồn: Bộ dữ liệu gốc TeaLeafAI và Billah cùng cộng sự (2025), Mendeley Data V2, DOI: 10.17632/rrff4vb35m.2, CC BY 4.0.
- Ghi chú: Đây là minh họa nhãn dữ liệu. Không sử dụng hình này như tài liệu xác minh tác nhân bệnh.

## Hình 2.3. Các bước tiền xử lý ảnh đầu vào

- Vị trí chèn: **2.2.4. Tiền xử lý ảnh và chuẩn hóa kích thước**.
- PNG: [hinh-2-3-tien-xu-ly-anh.png](hinh-2-3-tien-xu-ly-anh.png).
- Nguồn: Tác giả xây dựng từ ảnh UNADJUSTEDNONRAW_thumb_20d.jpg trong bộ dữ liệu gốc TeaLeafAI.
- Ghi chú: Ảnh gốc có kích thước 768 × 1024 pixel. Việc thay đổi kích thước không phải phép cắt ảnh; không chuẩn hóa đầu vào hai lần.

## Hình 2.4. Minh họa tăng cường dữ liệu trên ảnh lá chè

- Vị trí chèn: **2.2.5. Tăng cường dữ liệu (Data Augmentation)**.
- PNG: [hinh-2-4-tang-cuong-du-lieu.png](hinh-2-4-tang-cuong-du-lieu.png).
- Nguồn: Tác giả tạo bằng đúng phép biến đổi trong training/train_expanded.py từ ảnh healthy của bộ dữ liệu gốc.
- Ghi chú: Hai biến thể minh họa giai đoạn huấn luyện bộ phân loại trên backbone đóng băng. Giai đoạn tinh chỉnh có cấu hình tăng cường riêng trong finetune_expanded.py.

## Hình 2.5. Phân chia bộ dữ liệu huấn luyện, kiểm định và kiểm thử

- Vị trí chèn: **2.2.6. Chia tập dữ liệu huấn luyện, kiểm định và kiểm thử**.
- PNG: [hinh-2-5-chia-tap-du-lieu.png](hinh-2-5-chia-tap-du-lieu.png).
- Nguồn: Tác giả xây dựng từ training/reports/expanded-dataset-summary.json của TeaLeafAI.
- Ghi chú: Tỷ lệ thực tế là 70,14% / 14,88% / 14,97%; sai lệch nhỏ do số ảnh nguyên và phải giữ nhóm ảnh gần trùng cùng một tập.

## Hình 2.6. Kiến trúc mô hình MobileNetV2 được sử dụng trong đề tài

- Vị trí chèn: **2.3.3. Lựa chọn mô hình MobileNetV2 cho bài toán phân loại bệnh lá chè**.
- PNG: [hinh-2-6-kien-truc-mobilenetv2.png](hinh-2-6-kien-truc-mobilenetv2.png).
- Nguồn: Tác giả xây dựng từ training/train_expanded.py và training/finetune_expanded.py của TeaLeafAI.
- Ghi chú: Đọc sơ đồ từ trái sang phải ở hàng trên, xuống hàng dưới rồi từ phải sang trái. GAP là phép lấy trung bình toàn cục trên hai chiều không gian.

## Sử dụng trong Word

Chọn Insert → Pictures → This Device và chọn PNG. Giữ đúng tỷ lệ, đặt chiều rộng khoảng 15–16 cm.
Chú thích hình đặt ngoài ảnh bằng References → Insert Caption để Word đánh số tự động.
Các hình dùng nhãn kế thừa bộ dữ liệu, không khẳng định mọi ảnh đã được giám định tác nhân bệnh.
Ảnh gốc thuộc bộ dữ liệu đã có của dự án; ảnh bổ sung Mendeley được ghi nguồn theo CC BY 4.0.
Xem thêm [nguồn dữ liệu](../../TRAINING_DATA.md).

## Tạo lại hình

Cần bộ dữ liệu đã chia tại training/dataset và ảnh gốc trong DuLieuGoc.
Cài matplotlib, pillow, numpy, python-docx rồi chạy `python scripts/generate_chapter2_figures.py` từ thư mục gốc.
