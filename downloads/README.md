# Tải TeaLeafAI chạy sẵn

## So sánh ba mô hình — Google Colab

[Mở notebook MobileNetV2/ResNet50/VGG16](https://colab.research.google.com/github/khanh1023/TeaLeafAI/blob/main/training/Compare_Models_Colab.ipynb) · [Tải notebook](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-so-sanh-mo-hinh.ipynb) · [Quy trình và cách đọc kết quả](../docs/MODEL_COMPARISON.md).

Chọn GPU rồi Run all để chạy thí nghiệm mới trên cùng bộ chia 70/15/15; sau khi hoàn thành sẽ xuất Word, bảng, biểu đồ và số liệu thật. Notebook hiện chưa có kết quả huấn luyện. GPU miễn phí phụ thuộc hạn mức Colab.

[Tải TeaLeafAI-ready.zip](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-ready.zip) — khoảng 43 MiB, gồm mã nguồn, mô hình và website đã build.

Windows: cài Node.js LTS 22+ → giải nén → nhấp đúp `CHAY-WEBSITE.cmd`. Không cần Python hoặc npm install cho bản đã build. Đọc README trong ZIP.

File SHA256SUMS.txt chứa mã kiểm tra tính toàn vẹn. Không có dữ liệu gốc, tài khoản, token, môi trường ảo hoặc node_modules trong ZIP. Không cần file mô hình riêng khác để sử dụng.

[Tải TeaLeafAI-training-data.zip](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-training-data.zip) — khoảng 75 MiB, gồm 2.197 ảnh đã chia train/validation/test và thông tin nguồn. Giải nén, chép thư mục `dataset` vào `TeaLeafAI/training/dataset` nếu muốn huấn luyện lại. File `TRAINING_SHA256SUMS.txt` chứa mã kiểm tra riêng. Xem [nguồn ảnh và hướng dẫn huấn luyện](../docs/TRAINING_DATA.md) và [kết quả](../training/REPORT.md).

## Hình minh họa chương 2

[Sơ đồ đầu vào và đầu ra của hệ thống](../docs/figures/chapter2/dau-vao-dau-ra-he-thong.png) · [File Word kèm hình và chú thích](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-dau-vao-dau-ra.docx). Dùng cho mục 2.1.3; tách rõ đầu ra dự đoán một ảnh và đầu ra đánh giá trên tập kiểm thử.

- [File Word gồm 11 hình, vị trí chèn, chú thích và nguồn ảnh](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-hinh-chuong-2.docx).
- [Trọn bộ hình PNG 300 dpi, SVG, PDF và Word](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-hinh-chuong-2.zip).
- [Xem từng hình và hướng dẫn chèn vào báo cáo](../docs/figures/chapter2/README.md).

Bản cập nhật bổ sung sơ đồ tổng quan CNN, ví dụ tích chập–ReLU–Max pooling, khối Inverted Residual của MobileNetV2, tham số và quy trình huấn luyện/theo dõi mô hình. Chú thích trong Word được đánh số theo thứ tự mục; các đường dẫn ảnh cũ vẫn dùng được.

[Hình minh họa sáu ảnh về các phép tăng cường dữ liệu phổ biến](../docs/figures/chapter2/minh-hoa-cac-phep-tang-cuong-du-lieu.png) · [File Word kèm hình và chú thích](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-tang-cuong-du-lieu.docx). Dùng cho mục 2.2.5; đây là hình minh họa phương pháp, có ghi rõ khác biệt với cấu hình thực nghiệm.

[Hai hình giải thích các lớp và chu trình huấn luyện CNN](../docs/figures/chapter2/cnn-cau-truc-va-huan-luyen.md) · [File Word kèm hai hình và công thức](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-cau-truc-va-huan-luyen-CNN.docx). Dùng cho mục 2.3.2; có phân biệt sơ đồ CNN tổng quát với MobileNetV2 thực tế.

## Đặc điểm hình ảnh của lá chè bị bệnh — mục 1.3.3

- [Tải hình PNG 300 dpi](../docs/figures/chapter1/dac-diem-hinh-anh-la-che-bi-benh.png).
- [Tải file Word kèm hình và nguồn ảnh](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-dac-diem-la-che-bi-benh.docx).

## Phân loại ảnh bằng phương pháp truyền thống — mục 1.4.1

- [Tải hình PNG 300 dpi](../docs/figures/chapter1/phan-loai-anh-phuong-phap-truyen-thong.png).
- [Tải file Word kèm hình và chú thích](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-phan-loai-anh-truyen-thong.docx).

## Phân loại ảnh sử dụng học sâu — mục 1.4.3

- [Tải hình PNG 300 dpi](../docs/figures/chapter1/phan-loai-anh-su-dung-hoc-sau.png).
- [Tải file Word kèm hình và chú thích](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-phan-loai-anh-hoc-sau.docx).
- [Xem nguồn và bản SVG/PDF](../docs/figures/chapter1/README.md).

## Đánh giá lại độ chính xác

- [Tải bảng đánh giá và ma trận nhầm lẫn trong Word](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-danh-gia-lai.docx).
- [Tải trọn bộ kết quả JSON, CSV, ảnh và Word](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-danh-gia-lai.zip).
- [Xem báo cáo đối chiếu Keras, ONNX Python và đường xử lý trên trình duyệt](../training/reports/reevaluation/REPORT.md).

## Nội dung và hình chương 3 — kết quả đo ngày 22/11/2026

- [Word riêng mục 3.3: Đánh giá hiệu quả mô hình](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-muc-3-3-danh-gia.docx).
- [Word đầy đủ mục 3.2 và 3.3: 10 tiểu mục, 11 hình, bảng chỉnh sửa được](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-chuong-3-ket-qua-danh-gia.docx).
- [ZIP gồm Word, từng hình PNG/SVG/PDF, văn bản và số liệu JSON/CSV](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-chuong-3-anh-va-so-lieu.zip).
- [Xem kết quả và nguồn bằng chứng](../training/reports/evaluation-20261122/README.md).

Accuracy trên 329 ảnh test là 69,30% với Keras/ONNX Python và 66,87% với đường xử lý thật của trình duyệt. Điểm All bao gồm cả train; test là bộ nội bộ đã dùng trong phát triển. ResNet50/VGG16 được ghi rõ chưa thực nghiệm. Biểu đồ có bản PNG 300 dpi và vector; ảnh giao diện giữ độ phân giải chụp thật.
