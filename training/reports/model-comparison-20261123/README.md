# Mục 3.3.4 — bản ngắn, ảnh và nội dung

- [Tải Word](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-muc-3-3-4-so-sanh-mo-hinh.docx).
- [PNG 300 dpi](3-3-4-pham-vi-kiem-chung.png), [SVG](3-3-4-pham-vi-kiem-chung.svg), [PDF](3-3-4-pham-vi-kiem-chung.pdf).
- [Văn bản](noi-dung-muc-3-3-4.txt), [nguồn và số liệu](report-content.json).

MobileNetV2 có trọng số và kết quả kiểm thử được lưu trong hồ sơ. ResNet50/VGG16 chưa có thực nghiệm so sánh trên cùng bộ chia. Keras và ONNX là hai định dạng của cùng mô hình. File ONNX có 9.580.347 byte, tương đương khoảng 9,58 MB thập phân.

Sinh lại bằng `python scripts/generate_model_comparison_report.py`. Không huấn luyện mới hoặc dựng số liệu Accuracy cho các mạng chưa thực nghiệm. [Quy trình thực nghiệm bổ sung trên Colab](../../../docs/MODEL_COMPARISON.md).
