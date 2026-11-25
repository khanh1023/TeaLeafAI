# So sánh MobileNetV2, ResNet50 và VGG16 — mục 3.3.4

**Cập nhật 25/11/2026:** đã huấn luyện và kiểm thử thật cả ba mạng. Có thể [chạy từng mạng hoặc cả ba ngay trên website](https://tea-leaf-ai.vercel.app/so-sanh), gán nhãn ảnh đã thử và xuất Word/CSV. Xem [hướng dẫn sử dụng](TRY_THREE_MODELS.md) và [bằng chứng thực nghiệm Python](../training/reports/model-comparison-20261125/README.md). Các bản Word ngày 23/11 bên dưới là hồ sơ trước khi có thực nghiệm này.

Đã có [hình và văn bản ngắn mục 3.3.4](../training/reports/model-comparison-20261123/README.md) để dán vào báo cáo: phạm vi kiểm chứng hiện có và dung lượng ONNX khoảng 9,58 MB. [Tải Word](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-muc-3-3-4-so-sanh-mo-hinh.docx). Phần này chưa xếp hạng Accuracy hoặc tốc độ của ba mạng.

[Mở trên Google Colab](https://colab.research.google.com/github/khanh1023/TeaLeafAI/blob/main/training/Compare_Models_Colab.ipynb) · [Tải notebook](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-so-sanh-mo-hinh.ipynb)

Notebook chuẩn bị một **thí nghiệm mới**, chưa chứa kết quả huấn luyện. Chọn **Runtime → Change runtime type → T4 GPU**, rồi **Runtime → Run all**. Dùng gói Colab miễn phí khi Google cấp GPU; hạn mức và thời lượng phiên do Google quyết định. Không cần Colab Pro, API key hoặc kết nối Drive. Máy Cloud dùng chuẩn bị notebook chỉ có CPU; chưa thực hiện huấn luyện ba mạng trên GPU.

## Quy trình chung

| Thành phần | Quy định |
| --- | --- |
| Dữ liệu | Cùng 2.197 ảnh; checksum và nhóm gần trùng được kiểm tra |
| Bộ chia | 1.541 train, 327 validation, 329 test, xấp xỉ 70/15/15 |
| Đầu vào | RGB 224×224, Pillow bilinear, float32 |
| Trọng số | ImageNet chính thức, bỏ bộ phân loại gốc, đóng băng toàn bộ backbone |
| Bộ phân loại | GAP → Dropout 0,3 → Dense128 ReLU → Dropout 0,2 → Dense8 Softmax |
| Tăng cường | Bản gốc + hai biến thể xác định trước; lật ngang, xoay ±15°, sáng/tương phản 0,85–1,15; chỉ train |
| Huấn luyện | Seed 42, batch 32, Adam LR 0,001, class weights, tối đa 20 epoch |
| Chọn checkpoint | Validation loss nhỏ nhất; EarlyStopping patience 5, giảm LR khi không cải thiện |
| Test | Chạy sau khi chọn checkpoint; không dùng test chọn epoch hoặc điều chỉnh cấu hình |

Số chiều đặc trưng khác nhau làm số tham số đầu Dense khác nhau; cấu trúc lớp và quy tắc huấn luyện là chung. Lưu đặc trưng giúp giảm chi phí khi backbone đóng băng và gọi ở chế độ inference. Không tăng cường validation/test.

**Chuẩn hóa theo kiến trúc:** MobileNetV2 đưa pixel về [−1,1]; ResNet50/VGG16 chuyển RGB sang BGR và trừ trung bình ImageNet theo quy ước Caffe. Các bước nằm trong mô hình đầy đủ, không chuẩn hóa thêm bên ngoài.

Đây là so sánh dưới quy trình transfer learning cơ sở, chưa tìm cấu hình tối ưu riêng cho từng mạng hay fine-tune toàn bộ mạng. Một seed chưa phản ánh biến động giữa các lần huấn luyện. Test là bộ nội bộ đã dùng trong phát triển, chưa phải kiểm định thực địa độc lập; nguồn nhãn và danh tính cây/phiên chụp còn giới hạn.

## Các phép đo

- Accuracy, Precision macro, Recall macro, F1 macro và weighted F1 trên cùng 329 ảnh; ma trận nhầm lẫn và chỉ số từng lớp.
- Dung lượng file **Keras cùng định dạng, không gồm optimizer**, tổng tham số và số tham số được huấn luyện. Không dùng dung lượng phiên bản nguyên bản trên Internet thay cho phép đo thực tế.
- Latency warm batch 1: TensorFlow eager float32, đồng bộ bằng `.numpy()`, cùng máy. Làm nóng 10 lần, đo 30 ảnh trải đều theo manifest, hai lượt; báo mean/median/p95. Không gồm đọc/resize ảnh, khởi tạo, mạng hoặc xử lý trình duyệt.
- Thời gian trích xuất train/validation và huấn luyện đầu phân loại, tách khỏi tải trọng số và đánh giá test.
- Seed, checksum code/dữ liệu/mô hình, phiên bản thư viện/GPU, lịch sử và dự đoán từng ảnh.

Notebook tải code/dữ liệu **ghim theo commit** và xác minh checksum. Không thay checkpoint của website hoặc tự đẩy trọng số mới lên GitHub/Vercel. MobileNetV2 của thí nghiệm mới có thể khác 69,30% Python hoặc 66,87% trình duyệt đã báo cáo; không ghép kết quả cũ với hai mạng mới để xếp hạng.

## Kết quả để dán vào Word

Sau khi cả ba mạng chạy xong, ô cuối tải ZIP gồm:

- `So-sanh-mo-hinh-muc-3-3-4.docx`: văn bản, bảng chỉnh sửa được và hình so sánh.
- `comparison.png` 300 dpi, SVG/PDF và `comparison.csv`/JSON.
- `protocol.json` và các thư mục mô hình: trọng số, nhật ký, chỉ số, ma trận nhầm lẫn và dự đoán CSV/NPZ.

Mặc định ZIP giữ cả trọng số thật, có thể lớn khoảng vài trăm MB. Có thể đặt `INCLUDE_TRAINED_MODELS=False` ở ô cuối để chỉ lấy tài liệu/số liệu; cần tự lưu mô hình nếu muốn dùng lại trước khi Colab kết thúc. Notebook không bảo đảm khôi phục dữ liệu sau khi Colab ngắt phiên.

Nếu chưa chạy xong, chưa điền Accuracy/F1 hoặc kết luận mô hình nào vượt trội. Các mốc 94,5%, 95,1%, 94,8% trong bản nháp chưa được kiểm chứng bằng thí nghiệm này.

## Chạy trên máy riêng

Cài TensorFlow phù hợp GPU, Pillow, NumPy, scikit-learn, Matplotlib và python-docx; giải nén bộ dữ liệu vào `training/dataset`. Từ gốc repo:

```bash
python training/compare_models.py --output training/comparison-results
```

`--resume` chỉ dùng lại kết quả có cùng code, cấu hình, bộ chia, môi trường và file mô hình. Đổi môi trường/quy trình phải chọn thư mục mới. `--allow-cpu` cho phép chạy CPU nhưng có thể lâu. File kết quả mới không tự đưa lên GitHub.

Kiểm tra phần mềm đã chạy với trọng số ngẫu nhiên trên dữ liệu nhỏ để kiểm tra chuẩn hóa, bộ phân loại và lưu/nạp lại cả ba kiến trúc. Đây không phải kết quả Accuracy và không được dùng làm số liệu báo cáo. Nguồn ảnh/giấy phép: [TRAINING_DATA.md](TRAINING_DATA.md).
