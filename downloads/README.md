# Tải TeaLeafAI chạy sẵn

[Tải TeaLeafAI-ready.zip](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-ready.zip) — khoảng 43 MiB, gồm mã nguồn, mô hình và website đã build.

Windows: cài Node.js LTS 22+ → giải nén → nhấp đúp `CHAY-WEBSITE.cmd`. Không cần Python hoặc npm install cho bản đã build. Đọc README trong ZIP.

File SHA256SUMS.txt chứa mã kiểm tra tính toàn vẹn. Không có dữ liệu gốc, tài khoản, token, môi trường ảo hoặc node_modules trong ZIP. Không cần file mô hình riêng khác để sử dụng.

[Tải TeaLeafAI-training-data.zip](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-training-data.zip) — khoảng 75 MiB, gồm 2.197 ảnh đã chia train/validation/test và thông tin nguồn. Giải nén, chép thư mục `dataset` vào `TeaLeafAI/training/dataset` nếu muốn huấn luyện lại. File `TRAINING_SHA256SUMS.txt` chứa mã kiểm tra riêng. Xem [nguồn ảnh và hướng dẫn huấn luyện](../docs/TRAINING_DATA.md) và [kết quả](../training/REPORT.md).

## Hình minh họa chương 2

- [File Word gồm 11 hình, vị trí chèn, chú thích và nguồn ảnh](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-hinh-chuong-2.docx).
- [Trọn bộ hình PNG 300 dpi, SVG, PDF và Word](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-hinh-chuong-2.zip).
- [Xem từng hình và hướng dẫn chèn vào báo cáo](../docs/figures/chapter2/README.md).

Bản cập nhật bổ sung sơ đồ tổng quan CNN, ví dụ tích chập–ReLU–Max pooling, khối Inverted Residual của MobileNetV2, tham số và quy trình huấn luyện/theo dõi mô hình. Chú thích trong Word được đánh số theo thứ tự mục; các đường dẫn ảnh cũ vẫn dùng được.

## Đánh giá lại độ chính xác

- [Tải bảng đánh giá và ma trận nhầm lẫn trong Word](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-danh-gia-lai.docx).
- [Tải trọn bộ kết quả JSON, CSV, ảnh và Word](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-danh-gia-lai.zip).
- [Xem báo cáo đối chiếu Keras, ONNX Python và đường xử lý trên trình duyệt](../training/reports/reevaluation/REPORT.md).
