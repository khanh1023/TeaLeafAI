# Tải TeaLeafAI chạy sẵn

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
