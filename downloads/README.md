# Tải TeaLeafAI chạy sẵn

[Tải TeaLeafAI-ready.zip](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-ready.zip) — khoảng 43 MiB, gồm mã nguồn, mô hình và website đã build.

Windows: cài Node.js LTS 22+ → giải nén → nhấp đúp `CHAY-WEBSITE.cmd`. Không cần Python hoặc npm install cho bản đã build. Đọc README trong ZIP.

File SHA256SUMS.txt chứa mã kiểm tra tính toàn vẹn. Không có dữ liệu gốc, tài khoản, token, môi trường ảo hoặc node_modules trong ZIP. Không cần file mô hình riêng khác để sử dụng.

[Tải TeaLeafAI-training-data.zip](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-training-data.zip) — khoảng 75 MiB, gồm 2.197 ảnh đã chia train/validation/test và thông tin nguồn. Giải nén, chép thư mục `dataset` vào `TeaLeafAI/training/dataset` nếu muốn huấn luyện lại. File `TRAINING_SHA256SUMS.txt` chứa mã kiểm tra riêng. Xem [nguồn ảnh và hướng dẫn huấn luyện](../docs/TRAINING_DATA.md) và [kết quả](../training/REPORT.md).
