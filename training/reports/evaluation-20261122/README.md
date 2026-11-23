# Nội dung báo cáo 3.2–3.3 và đánh giá train/validation/test/All

Phép đo ngày **22/11/2026**, biên soạn ngày **23/11/2026**. Giữ nguyên checkpoint `mobilenetv2-expanded-20261117`, tám nhãn và manifest. Không huấn luyện lại, đổi bộ chia hoặc lựa chọn checkpoint theo kết quả lần này. Mã nguồn trước đánh giá: `b36a311a707f4666dd15b3eaa632ba201830f12f`.

## Tải vào Word

- [Word riêng mục 3.3](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-muc-3-3-danh-gia.docx).
- [Word đầy đủ 3.2 và 3.3, 10 tiểu mục và 11 hình](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-chuong-3-ket-qua-danh-gia.docx).
- [ZIP gồm Word, hình, số liệu và văn bản](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-chuong-3-anh-va-so-lieu.zip).
- [Văn bản 3.2–3.3](noi-dung-3-2-va-3-3.txt) · [Văn bản riêng 3.3](noi-dung-3-3.txt).

Các bảng trong Word có thể chỉnh sửa. Mỗi tiểu mục có hình và chú thích. Biểu đồ có PNG 300 dpi, SVG và PDF; ảnh chụp giao diện là ảnh thật, giữ độ phân giải gốc. Hai tấm ví dụ dự đoán dùng ảnh thật trong tập test; nhãn kế thừa nguồn dữ liệu, chưa phải xác nhận tác nhân bệnh trên từng mẫu.

## Kết quả đo

| Tập, đường xử lý | Số ảnh | Đúng | Accuracy |
| --- | ---: | ---: | ---: |
| Train, Keras/ONNX Python | 1541 | 1439 | 93,38% |
| Validation, Keras/ONNX Python | 327 | 209 | 63,91% |
| Test, Keras/ONNX Python | 329 | 228 | 69,30% |
| All, Keras/ONNX Python | 2197 | 1876 | 85,39% |
| Test, trình duyệt Chromium | 329 | 220 | 66,87% |

**All gồm cả train, chỉ là thống kê gộp; không dùng để biểu thị khả năng phân loại ảnh mới.** Điểm train trong bảng đo lại checkpoint trên ảnh không tăng cường, khác phép đo train trong nhật ký huấn luyện. Tập test đã được dùng trong quá trình phát triển; chưa có kiểm định thực địa độc lập. Kiểm tra phát hiện nhóm gần trùng và các hash trùng không giao nhau giữa ba tập, nhưng chưa có đầy đủ danh tính cây/phiên chụp.

| Chỉ số test | Python | Trình duyệt |
| --- | ---: | ---: |
| Precision macro | 64,24% | 61,44% |
| Recall macro | 66,05% | 63,97% |
| F1 macro | 64,53% | 62,23% |
| F1 weighted | 69,33% | 67,45% |

Python dùng Pillow bilinear RGB 224×224, trình duyệt dùng `createImageBitmap` và Canvas; đều đưa pixel float32 0–255 vào mô hình. Keras và ONNX Python khớp top-1 trên toàn bộ 2.197 ảnh. Số đo trình duyệt đánh giá đường xử lý thật của website, không đồng nhất với tiền xử lý Python.

Chưa có thực nghiệm ResNet50 hoặc VGG16 cùng bộ chia. Tài liệu không sử dụng các mức 94,5%, 95,1%, 94,8% hoặc thời gian suy luận của những mô hình này trong bản nháp làm kết quả thực nghiệm.

Sáu phép thử độ nhạy nhân tạo áp dụng riêng cho cùng 329 ảnh, giữ nguyên mô hình ONNX Python: giảm sáng ×0,65 đạt 65,96%; tăng sáng ×1,35 đạt 63,22%; mờ Gaussian bán kính 2 đạt 57,45%; giảm về 32×32 rồi phóng lại đạt 37,39%; JPEG quality 35 đạt 54,41%; xoay +15° đạt 67,48%. Đây là phép thử chẩn đoán độ nhạy, không mô phỏng đầy đủ thực địa hay chứng minh nguyên nhân của từng lỗi gốc.

Thời gian `diagnose` sau khởi tạo: 328 lần gọi, trung bình 41,86 ms, trung vị 40,30 ms, p95 51,40 ms trên máy Cloud Linux và Chromium cục bộ. Bao gồm đọc ảnh/Canvas trong trình duyệt, suy luận và xử lý đầu ra; không gồm disk I/O, base64 hoặc Playwright transport ngoài trình duyệt. Không dùng số đo này để cam kết tốc độ trên điện thoại hoặc mạng Internet. Định nghĩa và thông tin môi trường nằm trong `browser-summary.json`.

## Số liệu và hình tương ứng

| Mục | Hình | Bằng chứng |
| --- | --- | --- |
| 3.2.1 | [Đường cong huấn luyện](figures/3-2-1-training-history.png) | [Nhật ký huấn luyện](../../evaluation.json) |
| 3.2.2 | [Train/validation/test/All](figures/3-2-2-splits.png) | [Kết quả native](data/native-results.json) |
| 3.2.3 | [Kết quả trên giao diện](figures/3-2-3-web-example.png) | [Mẫu và thông tin chụp](data/web-result-example.json) |
| 3.2.4 | [Ví dụ phân loại](figures/3-2-4-prediction-examples.png) | [ID, nguồn ảnh và dự đoán](data/illustration-sources.json) |
| 3.2.5 | [Phân loại đúng và sai](figures/3-2-5-correct-and-wrong.png) | [Dự đoán trình duyệt](data/browser-predictions.json) |
| 3.3.1 | [Accuracy](figures/3-3-1-accuracy.png) | [Native](data/native-results.json), [browser](data/browser-metrics.json) |
| 3.3.2 | [Precision/Recall/F1](figures/3-3-2-precision-recall-f1.png) | [Chỉ số và ma trận](data/browser-metrics.json) |
| 3.3.3 | [Ma trận nhầm lẫn](figures/3-3-3-confusion-matrix-browser.png) | [329 dự đoán riêng lẻ](data/browser-predictions.json) |
| 3.3.4 | [Phạm vi so sánh](figures/3-3-4-comparison-evidence.png) | Checkpoint và kết quả hiện có; hai kiến trúc còn lại chưa thực nghiệm |
| 3.3.5 | [Độ nhạy](figures/3-3-5-sensitivity.png), [phân bố lớp](figures/3-3-5-class-distribution.png) | [Kết quả từng biến đổi](data/robustness-predictions.csv), [phân bố gốc](../expanded-dataset-summary.json) |

Toàn bộ dự đoán native nằm trong [CSV](data/native-predictions.csv) và [NPZ](data/native-predictions.npz). [Nội dung máy đọc được](report-content.json) lưu văn bản, bảng, tên hình và checksum. [Thông tin sinh các hình bối cảnh](data/context-figure-sources.json) ghi nguồn nhật ký và phân bố dữ liệu. Nguồn và giấy phép dữ liệu: [TRAINING_DATA.md](../../../docs/TRAINING_DATA.md).

## Đánh giá lại

Tải và giải nén [bộ dữ liệu đã chia](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-training-data.zip) vào `training/dataset`, chuẩn bị Python với TensorFlow, ONNX Runtime, NumPy, Pillow, scikit-learn và Matplotlib. Chạy từ gốc repo:

```bash
python training/evaluate_all_splits.py --output /tmp/tealeaf-evaluation/native --robustness
```

Đánh giá trình duyệt cần Node.js, Playwright và Chromium. Khởi động Vite từ `frontend` bằng `npm run dev -- --host 127.0.0.1 --port 5179`, sau đó từ gốc repo:

```bash
TEALEAF_BASE_URL=http://127.0.0.1:5179 TEALEAF_EVAL_OUTPUT=/tmp/tealeaf-evaluation/browser node frontend/tests/reevaluate-browser.mjs
```

Chọn thư mục đầu ra mới để không ghi đè kết quả. Tốc độ có thể khác do phần cứng và tải máy. Các phép đo giữ nguyên trọng số, nhãn và ảnh gốc; so sánh theo checksum được lưu trong JSON. Sinh lại ba hình bối cảnh bằng `python scripts/generate_evaluation_context_figures.py`. Script `training/build_evaluation_chapter.py --source-root <thư-mục-kết-quả>` đóng gói báo cáo từ hai thư mục `native`, `browser`; ngoài JSON, thư mục `browser` cần hai ảnh chụp giao diện và JSON thông tin chụp như bộ bằng chứng đã công bố.
