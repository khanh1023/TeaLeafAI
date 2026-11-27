# Dữ liệu v3 — bổ sung ảnh, giữ nguyên tập cũ

Phiên bản `dataset-v3-20261128` có **8.280 ảnh dùng trong thí nghiệm**,
tăng **1.440 ảnh** so với v2. Tổng chia chính xác **5.796 / 1.242 / 1.242**
(train 70%, validation 15%, test 15%). Có 10 ảnh khỏe dự phòng không dùng
huấn luyện hay đánh giá, để tỷ lệ phần trăm khớp số nguyên.

| Nhãn | Bổ sung | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|---:|
| algal_leaf | 297 | 722 | 155 | 155 | 1.032 |
| anthracnose | 0 | 69 | 15 | 15 | 99 |
| bird_eye_spot | 0 | 69 | 15 | 15 | 99 |
| brown_blight | 127 | 687 | 147 | 148 | 982 |
| gray_light | 311 | 1.159 | 248 | 248 | 1.655 |
| healthy | 323 | 2.405 | 515 | 515 | 3.435 |
| red_leaf_spot | 382 | 586 | 125 | 125 | 836 |
| white_spot | 0 | 99 | 22 | 21 | 142 |
| **Tổng** | **1.440** | **5.796** | **1.242** | **1.242** | **8.280** |

Toàn bộ 6.840 ảnh v2 giữ nguyên nhãn, nội dung và tập đã chia. Không chuyển ảnh
test/validation cũ vào train. Ảnh mới được kiểm tra SHA-256, pixel và pHash so với
toàn bộ dữ liệu; ảnh cùng nhóm thị giác hoặc cùng phút chụp Datta ở cùng lớp
được giữ trong một tập. Không dùng kết quả dự đoán để chọn ảnh hay chia dữ liệu.

## Nguồn bổ sung và quyền sử dụng

1. [Datta, Tea leaf disease](https://www.kaggle.com/datasets/saikatdatta1994/tea-leaf-disease),
   qua [Project-AgML mirror](https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_classification)
   tại revision `d44a9aff2f0f71aae523b22b5f0f821c68bf18ea`.
   Mirror công bố **CC BY-NC 4.0**: chỉ sử dụng học thuật/phi thương mại và ghi nguồn.
   Chỉ lấy các tệp có mẫu tên ngày giờ chụp; bỏ mọi tệp `image_*` tăng cường ngoại tuyến
   và lớp helopeltis. Tên tệp là dấu vết nguồn, chưa chứng minh đây là ảnh camera chưa chỉnh sửa.
   Các ảnh đã được nguồn thu về khoảng 256 × 256; không được gọi là ảnh gốc độ phân giải cao.
2. [Hu, Wu, Zhang và Wan (2019)](https://data.mendeley.com/datasets/dbjyfkn6jr/1),
   DOI `10.17632/dbjyfkn6jr.1`, **CC BY 4.0**. Chỉ nhận thư mục `Tea red leaf spot`.
   Thư mục train/test do nhà cung cấp đặt tên được gộp để lọc trùng rồi chia theo
   quy trình của dự án; không coi hai bản ảnh trùng là hai mẫu độc lập.

Các nguồn v2 tiếp tục giữ điều kiện ban đầu. Không áp dụng lại một giấy phép
chung cho toàn bộ dữ liệu. Việc mở rộng này **không bổ sung mẫu độc lập** cho
than thư, đốm mắt chim và đốm trắng vì chưa tìm được nguồn nhãn tương ứng đáng tin cậy.

21 ảnh ứng viên bị loại do trùng. 3.436 bản tăng cường ngoại tuyến và 1.000 ảnh
helopeltis từ nguồn Datta không được nhận làm ứng viên. Ảnh được EXIF-correct,
RGB, giới hạn cạnh dài 512 và mã hóa JPEG; nguồn băm trước/sau xử lý được lưu
trong manifest. Tăng cường khi huấn luyện chỉ áp dụng sau chia tập cho train.

## Khôi phục bộ dữ liệu

Từ thư mục gốc mã nguồn, chạy từng dòng:

```powershell
python training/download_dataset_v2.py
python training/download_dataset_v3.py
```

Nếu v2 đã có thì chỉ chạy dòng thứ hai. Mặc định tạo `training/dataset-v3`;
không ghi đè thư mục đã có. Gói `training/data/dataset-v3-additions.zip` chỉ chứa
ảnh bổ sung và metadata v3, phụ thuộc bộ v2 đã công bố. Các ảnh cũ được xác minh
rồi hardlink (hoặc copy nếu hệ điều hành không hỗ trợ). SHA-256 của ảnh,
manifest và gói đều được kiểm tra.

```powershell
python training/audit_dataset_v3.py --data training/dataset-v3 --previous training/dataset-v2 --output training/reports/dataset-v3-20261128/audit-local.json
```

## Huấn luyện và cách đọc kết quả

`train_dataset_v3.py` chỉ cho phép tiếp tục từ mô hình v2 khi chứng minh được
mọi ảnh cũ giữ nguyên tập. Chọn checkpoint bằng **validation loss**, giữ mô hình
ban đầu nếu validation không cải thiện. Giảm mức trọng số lớp từ nghịch đảo đầy
đủ sang căn bậc hai, thích nghi phần phân loại rồi tinh chỉnh cuối MobileNetV2,
đóng băng BatchNormalization. Test chỉ được tính sau khi chọn checkpoint xong.

Mục tiêu >90% phải được xác nhận bằng dự đoán thật; thêm ảnh không bảo đảm đạt
mốc này. Báo cáo cả Accuracy, balanced accuracy, macro F1 và kết quả từng lớp.
Validation/test v2 đã từng được xem trong quá trình phát triển, nên số liệu
mở rộng vẫn là đánh giá nội bộ, chưa phải kiểm định thực địa độc lập tại Việt Nam.
