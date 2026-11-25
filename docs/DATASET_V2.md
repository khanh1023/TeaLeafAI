# Bộ dữ liệu lá chè mở rộng v2 — 25/11/2026

Đây là phiên bản dữ liệu mới, dùng để huấn luyện lại và đánh giá lại mô hình. Thư mục `training/dataset` và các kết quả cũ được giữ nguyên; dữ liệu mới nằm ở `training/dataset-v2`. Việc thêm dữ liệu không tự thay trọng số trên website và không chứng minh độ chính xác tăng lên.

## Tải bộ đã chia sẵn

Từ thư mục TeaLeafAI sau khi cập nhật mã nguồn:

```powershell
python training/download_dataset_v2.py
```

Lệnh này chỉ cần Python, tự tải ba phần dữ liệu từ GitHub, đối chiếu SHA-256 từng phần, ghép thành ZIP và giải nén vào `training/dataset-v2`. [Trang tải dataset-v2-20261125](https://github.com/khanh1023/TeaLeafAI/releases/tag/dataset-v2-20261125) có hướng dẫn tương ứng. Các phần được lưu riêng ở tag dữ liệu để không đưa ảnh vào lịch sử mã nguồn chính. Không ghi đè thư mục đã có. Nếu đã tải ZIP hoàn chỉnh từ cuộc trò chuyện về máy, dùng:

```powershell
python training/download_dataset_v2.py --archive "C:\duong-dan\TeaLeafAI-dataset-v2-20261125.zip"
```

Cũng có thể giải nén ZIP hoàn chỉnh bằng Windows: bên trong có thư mục `dataset-v2`; đặt thư mục này trong `TeaLeafAI/training`. Các tệp `.part01` đến `.part03` trên GitHub là từng đoạn của một ZIP, cần tải đủ và ghép theo đúng thứ tự; lệnh Python ở trên tự làm việc này. Không trộn ảnh của phiên bản này với các tập train/val/test cũ.

## Số lượng và bộ chia

Tăng từ **2.197 lên 6.840 ảnh**, thêm **4.643 ảnh sau lọc**, khoảng **3,11 lần** bộ cũ. Lớp lá khỏe có **3.112 ảnh**, lớn hơn từng lớp bệnh. Giữ riêng **4 ảnh lá khỏe dự phòng**; chúng không được tính vào 6.840 ảnh của bộ chia.

| Lớp | Bộ cũ | Train | Validation | Test | Tổng v2 |
|---|---:|---:|---:|---:|---:|
| algal_leaf | 265 | 515 | 110 | 110 | 735 |
| anthracnose | 99 | 69 | 15 | 15 | 99 |
| bird_eye_spot | 99 | 69 | 15 | 15 | 99 |
| brown_blight | 305 | 599 | 128 | 128 | 855 |
| gray_light | 281 | 941 | 201 | 202 | 1.344 |
| healthy | 673 | 2.178 | 467 | 467 | 3.112 |
| red_leaf_spot | 333 | 318 | 68 | 68 | 454 |
| white_spot | 142 | 99 | 22 | 21 | 142 |
| **Tổng** | **2.197** | **4.788 (70%)** | **1.026 (15%)** | **1.026 (15%)** | **6.840** |

Sau khi loại bản sao, bộ chia có **6.470 nhóm pHash**, nhóm lớn nhất 14 ảnh. Tổng nhật ký loại trừ có 1.907 mục, gồm 1.608 bản sao byte, 297 bản sao nén lại/xoay/lật và 2 ảnh có nhãn mâu thuẫn. Không dùng số tệp thô mà tác giả công bố làm số ảnh độc lập: riêng nguồn Bangladesh có 561 tệp đốm đỏ nhưng chỉ 82 giá trị hash byte khác nhau trước khi đối chiếu các nguồn khác.

Số ảnh tổng là số tệp sau lọc, không đồng nghĩa với số cây hoặc phiên chụp độc lập. Các nhóm ảnh gần trùng được ghi riêng trong manifest. Mỗi lớp được phân chia trong phạm vi làm tròn một ảnh so với 70/15/15; tổng toàn bộ tập chia đúng 70/15/15. Phần `reserve` chứa tối đa 19 ảnh lá khỏe mới để tổng dùng cho huấn luyện/đánh giá chia hết cho 20. Ảnh dự phòng được giữ lại nhưng không tham gia huấn luyện, chọn mô hình hoặc tính chỉ số test.

Các báo cáo đầy đủ nằm trong [training/reports/dataset-v2-20261125](https://github.com/khanh1023/TeaLeafAI/tree/main/training/reports/dataset-v2-20261125): `class-counts.csv`, `split_summary.json`, `manifest.json`, `exclusions.json`, `reserve.json`, `audit.json`, `package.json` và `SHA256SUMS.txt`. CSV dùng UTF-8 có BOM để mở bằng Excel. Trong ZIP cũng có các manifest, bảng số lượng và báo cáo kiểm tra tương ứng.

## Nguồn ảnh và cách ánh xạ nhãn

Giữ nguyên 8 nhãn và thứ tự đầu ra: `algal_leaf`, `anthracnose`, `bird_eye_spot`, `brown_blight`, `gray_light`, `healthy`, `red_leaf_spot`, `white_spot`. Tên `gray_light` là mã lớp cũ của dự án, ánh xạ từ các nhãn Grey/Gray Blight của nguồn.

1. Bộ v1 đã lọc, bao gồm ảnh người dùng cung cấp và hai nguồn bên dưới. Ảnh người dùng giữ điều kiện nguồn gốc; không tự cấp lại giấy phép. Manifest v1 được giữ nguyên để tái lập các kết quả đã công bố.
2. [Billah, Sagor, Sultana & Uddin (2025), Mendeley V2](https://data.mendeley.com/datasets/rrff4vb35m/2), DOI 10.17632/rrff4vb35m.2. CC BY 4.0. Các lớp Tea Algal Spot, Tea Brown Blight, Tea Grey Blight, Tea Healthy và Tea Red Spot đã được dùng trong v1; không tính lại như nguồn ảnh hoàn toàn mới.
3. [Ahmad (2024), Mendeley V4](https://data.mendeley.com/datasets/tt2smzrzrs/4), DOI 10.17632/tt2smzrzrs.4, CC BY 4.0; bản sao [Project-AgML](https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_classification_bangladesh), revision `eb53d418d10a4bdf8cbab95928aad905aa62a368`. Duyệt toàn bộ 2.270 ảnh Healthy và 561 ảnh Tea Red Leaf Spot, thay cho giới hạn 400 ảnh Healthy của v1. Loại nhãn Tea leaf blight gộp và Tea red scab vì không ánh xạ chắc chắn vào 8 lớp.
4. [Ahammed và cộng sự (2025), teaLeafBD V4](https://data.mendeley.com/datasets/744vznw5k2/4), DOI 10.17632/744vznw5k2.4, CC BY 4.0; bản sao [Project-AgML](https://huggingface.co/datasets/Project-AgML/teaLeafBD_disease_classification), revision `068914a550e65791dcb488e4ef9a828d710bff3c`. Duyệt 508 Brown Blight, 1.013 Gray Blight, 935 Healthy leaf và 418 Tea algal leaf spot. Loại Green mirid bug, Helopeltis và Red spider.
5. [Chowdhury & Arman (2025), Mendeley V1](https://data.mendeley.com/datasets/3x42rbj8yv/1), DOI 10.17632/3x42rbj8yv.1, CC BY 4.0; bản sao [Project-AgML](https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_classification_bangladesh_2), revision `cb677887ee7b52ee0b084e7a2e3dc8cd3ec989fd`. Duyệt toàn bộ 650 Healthy_Leaf. Không tự đổi nhãn Blight/Red_Rust thành một loại bệnh cụ thể của dự án.
6. [Billah, Sagor, Uddin & Hossain (2025), Hibiscus and Tea V4](https://data.mendeley.com/datasets/5bzy89brkv/4), DOI 10.17632/5bzy89brkv.4, CC BY 4.0; bản sao [Project-AgML](https://huggingface.co/datasets/Project-AgML/Hibiscus_Tea_disease_classification), revision `995934b58fe3530bfdf8e9ec87689b12d6426165`. Chỉ duyệt cấu hình `raw`: 54 Tea Algal Spot, 48 Tea Brown Blight, 53 Tea Grey Blight và 44 Tea Red Spot. Loại toàn bộ lá Hibiscus, bao gồm lớp Healthy của Hibiscus, và toàn bộ cấu hình augmented. Ảnh trùng với nguồn khác tiếp tục bị loại.

Các con số ở danh sách nguồn là số ứng viên trước lọc; số ảnh thực dùng được tính từ các tệp tải về. Không tìm thấy nguồn bổ sung đáng tin cậy cho ba lớp `anthracnose`, `bird_eye_spot`, `white_spot` trong các nguồn đã kiểm tra. Không gán bệnh khác vào các lớp này để làm tăng số lượng. Bộ 5.867 ảnh sáu lớp trên Kaggle/AgML được bỏ qua vì có ảnh tăng cường sẵn và thiếu định danh ảnh cha đủ rõ để bảo đảm bộ chia mới.

Giấy phép ảnh bổ sung: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Khi phân phối lại, giữ nguồn, tác giả, DOI, giấy phép và thông báo thay đổi. Các tác giả nguồn không bảo trợ ứng dụng này. Giấy phép mã nguồn không tự áp dụng cho dữ liệu ảnh.

## Tiền xử lý, lọc trùng và kiểm tra

- Xoay đúng EXIF, chuyển RGB, thu cạnh dài về tối đa 512 pixel và lưu JPEG quality 92; ảnh gốc của người dùng không bị sửa. Khi huấn luyện mới đổi kích thước về 224 × 224.
- Loại trùng byte, pixel đã chuẩn hóa và tệp JPEG. Giữ lại hash byte gốc, hash pixel, hash ảnh lưu, nguồn, nhãn và kích thước gốc trong manifest.
- Loại thêm bản sao nén lại/xoay/lật khi tỷ lệ khung hình khớp trong 1%, pHash gần nhau và sai khác pixel trung bình không quá 1,5/255, phân vị 95 không quá 6/255 trên ảnh 128 × 128. Nhãn mâu thuẫn của cùng ảnh được cách ly, không tự sửa theo dự đoán.
- Các ảnh gần trùng còn lại được gom bằng hai pHash 64 bit trên ảnh đầy đủ và vùng xanh của lá, xét xoay/lật: một khoảng cách Hamming ≤4 và cả hai khoảng cách ≤10. Điều kiện kết hợp giúp hạn chế ghép nhầm các lá khác nhau chỉ vì có chung nền trắng. Một nhóm nằm trọn trong một tập; nhóm được nối bắc cầu.
- Kiểm tra toàn bộ ảnh đọc được, đúng SHA-256, đủ 8 lớp, đúng tổng 70/15/15, lá khỏe nhiều hơn từng bệnh và không có nhóm hoặc cặp pHash gần trùng vượt ranh giới train/val/test.

Thông tin định danh cây/phiên chụp của các nguồn còn thiếu. Vì vậy kiểm tra ảnh gần trùng giúp giảm rò rỉ dữ liệu nhưng chưa chứng minh tập test độc lập hoàn toàn theo cây hoặc địa điểm. Nhãn kế thừa từ tác giả dữ liệu; không coi tất cả ảnh đã được dự án xét nghiệm tác nhân bệnh.

## Huấn luyện trên phiên bản mới

Cần môi trường TensorFlow cùng các thư viện trong `training/expanded-requirements.txt` và thư viện đánh giá của dự án. Dùng checkpoint mới, khởi tạo backbone ImageNet; không tiếp tục checkpoint v1 rồi tính test trên bộ chia v2 vì có thể gặp lại ảnh cũ ở tập test mới.

```powershell
python training/train_expanded.py --data training/dataset-v2 --output training/runs/v2-head --epochs 30 --augmented-views 2
python training/finetune_expanded.py --data training/dataset-v2 --initial training/runs/v2-head --output training/runs/v2-final --epochs 10
```

Hai biến thể tăng cường mỗi ảnh chỉ được tạo trong tập train, không được tính như ảnh gốc bổ sung vào bộ chia 70/15/15. Validation/test không tăng cường. Chương trình dùng trọng số lớp vì healthy chủ ý nhiều hơn các bệnh. Khi báo cáo, cần xem thêm balanced accuracy, macro-F1, precision/recall từng lớp và ma trận nhầm lẫn.

Chọn mô hình theo validation; chỉ sử dụng test cho lần đánh giá cuối của phiên bản thực nghiệm. Các chỉ số v1 không chuyển sang v2. Hướng dẫn trên chuẩn bị cho lần huấn luyện mới; lần mở rộng dữ liệu này chưa tự huấn luyện hoặc thay mô hình trên web.

## Tái tạo và kiểm tra bằng mã nguồn

```powershell
python training/download_expansion_sources.py --cache-dir training/source-cache
python training/download_dataset_v2_sources.py --cache-dir training/source-cache/v2
python training/build_dataset_v2.py --old-cache training/source-cache --new-cache training/source-cache/v2
python training/audit_dataset_v2.py --data training/dataset-v2 --output training/reports/dataset-v2-20261125/audit.json
```

Bước tái tạo cần bộ v1 tại `training/dataset`. Nếu đã tải gói v2 thì không cần tái tạo. Các trình tạo dữ liệu từ chối ghi đè thư mục đã tồn tại; dùng đường dẫn mới khi muốn chạy lại. Bản sao Hugging Face được ghim theo revision và nguồn tải được kiểm tra trước khi nhận ảnh.
