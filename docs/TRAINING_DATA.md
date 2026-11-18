# Bộ dữ liệu bổ sung và quy trình huấn luyện

Bản huấn luyện này có 2.197 ảnh sau lọc, gồm 673 ảnh lá khỏe, nhiều hơn từng lớp bệnh. Các ảnh gốc trong DuLieuGoc không bị sửa, xóa hoặc tự đổi nhãn. Ảnh bị trùng nội dung nhưng có nhãn mâu thuẫn được loại khỏi lần huấn luyện; nhật ký nằm trong exclusions.json.

## Nguồn ảnh và giấy phép

1. Ảnh gốc do người dùng cung cấp trong repo khanh1023/TeaLeafAI. Giữ nguyên nguồn và các điều kiện sử dụng của dữ liệu gốc; không tự gán lại giấy phép cho các ảnh này.
2. Billah, Md Masum; Sagor, Saifuddin; Sultana, Azmery; Uddin, Mohammad Shorif (2025), Real-Life Dataset of Tea Plant Leaf Disease Classification, Mendeley Data V2, DOI https://doi.org/10.17632/rrff4vb35m.2. Dùng gói Tea Compressed Dataset.zip, SHA-256 90e694b5eb3d0ad22ea018d578f39775a94ea9ab919172e8ea627bc12d49215b. Giấy phép CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. Kế thừa 5 nhãn tương ứng: algal, brown, grey, healthy, red. Kho lưu có 965 file ảnh trước lọc; số thực tế được ghi theo file tải về, không dùng số mô tả làm số đếm.
3. Ahmad, MD Hasan và cộng sự (2024), Advanced Tea Crop Disease Study: High-Resolution Dataset for Precision Agriculture and Pathological Insight, Mendeley Data V4, DOI https://doi.org/10.17632/tt2smzrzrs.4; bản sao Project-AgML tại https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_classification_bangladesh, revision eb53d418d10a4bdf8cbab95928aad905aa62a368. Giấy phép CC BY 4.0. Chỉ lấy 400 ảnh có nhãn Healthy sau kiểm tra gần trùng; không tự ánh xạ lớp Tea leaf blight gộp thành đốm nâu hoặc đốm xám.

Nguồn công khai mới được chuyển thành RGB JPEG tối đa 512 px, xoay đúng theo EXIF và tổ chức theo nhãn; đây là các thay đổi so với dữ liệu của tác giả. Việc sử dụng không hàm ý tác giả nguồn xác nhận hoặc bảo trợ TeaLeafAI. Nhãn được kế thừa từ nguồn, không phải toàn bộ ảnh đã được nhóm dự án giám định lại tác nhân bệnh.

## Chia dữ liệu

Mục tiêu train/validation/test là 70/15/15, seed 42. Ảnh gần trùng được gom nhóm bằng perceptual hash trên vùng lá, có xét xoay/lật; một nhóm chỉ xuất hiện trong một tập. Chia 20 fold phân tầng theo lớp và nguồn, lấy 14 fold train, 3 validation, 3 test. Số thực tế: train 1.541 (70,14%), validation 327 (14,88%), test 329 (14,97%). Sai lệch nhỏ do số ảnh nguyên và phải giữ nhóm gần trùng cùng tập.

Giữ riêng validation/test và không tăng cường hai tập này. Tăng cường chỉ ở train. Trọng số lớp được dùng để xử lý mất cân bằng khi số ảnh healthy lớn hơn các lớp khác. Kiểm tra ảnh trùng giúp giảm rò rỉ, nhưng không thay thế việc chia theo cây/phiên chụp nếu sau này có đầy đủ thông tin nguồn.

## Dùng bản dữ liệu đã chia

Tải TeaLeafAI-training-data.zip trong thư mục downloads của repo. Giải nén rồi chép thư mục dataset vào TeaLeafAI/training/dataset. Manifest chứa nhãn, nguồn, hash, nhóm và đường dẫn từng file. File TRAINING_SHA256SUMS.txt dùng kiểm tra toàn vẹn bản tải.

Bản ZIP website chạy sẵn không chứa toàn bộ dữ liệu này để giữ dung lượng gọn.

## Tái tạo từ nguồn và huấn luyện

Từ thư mục gốc, dùng Python trong môi trường riêng đã cài training/browser-export-requirements.txt và training/expanded-requirements.txt. Các lệnh dưới đây viết theo Python của môi trường đã cài:

```bash
python training/download_expansion_sources.py --cache-dir training/source-cache
python training/expand_dataset.py --cache-dir training/source-cache --healthy-extra 400
python training/train_expanded.py --output training/runs/head --epochs 20 --augmented-views 2
python training/finetune_expanded.py --initial training/runs/head --output training/runs/final --epochs 6
python training/export_browser_model.py --model training/runs/final/tea_leaf_mobilenetv2.keras --classes training/runs/final/classes.json --output frontend/public/model
```

Nếu cần tạo lại bộ chia đã sinh, truyền --replace cho expand_dataset.py. Cache và candidate model không đưa lên Git. Bản đầu huấn luyện head mới trên backbone ImageNet đóng băng; bản sau tinh chỉnh một số lớp cuối, chọn theo validation loss và báo cáo kết quả test. Có đánh giá sơ bộ head trên cùng phân vùng test trước giai đoạn tinh chỉnh; kết quả này là đánh giá phát triển nội bộ, không phải kiểm định thực địa độc lập. Báo cáo kết quả cuối ở training/REPORT.md.
