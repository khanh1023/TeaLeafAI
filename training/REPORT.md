# Huấn luyện hiện tại — dữ liệu v2

Mô hình chính MobileNetV2 được khởi tạo từ ImageNet, huấn luyện head và fine-tune, chọn checkpoint theo validation loss trên bộ chia cố định 4.788 train / 1.026 validation / 1.026 test.

- Keras/ONNX Python: 868/1.026 ảnh đúng, Accuracy 84,60%, macro-F1 0,7564.
- Chromium qua đúng luồng website: 863/1.026 ảnh đúng, Accuracy 84,11%.
- Cả bốn ONNX khớp top-1 với Keras trên đủ 1.026 ảnh; không dùng số liệu v1 để thay thế kết quả v2.

[Báo cáo đầy đủ, Word, ma trận nhầm lẫn và bằng chứng](reports/retraining-v2-20261127/README.md). [V1 lịch sử](reports/retraining-v1-20261117/REPORT.md).
