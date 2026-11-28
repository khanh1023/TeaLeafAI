# TeaLeafAI · Nhận diện bệnh lá chè

## Chạy thử ba mô hình trên website

**[Mở trang Thử 3 mô hình](https://tea-leaf-ai.vercel.app/so-sanh)**: chọn ảnh/camera, chạy MobileNetV2, ResNet50 hoặc VGG16, đối chiếu cùng ảnh và tải Word/CSV từ kết quả đã thử. Có thể gán nhãn thực tế để tính Accuracy trên các ảnh chung; không lấy điểm softmax làm Accuracy. [Hướng dẫn](docs/TRY_THREE_MODELS.md) · [Huấn luyện lại và bằng chứng trên dữ liệu v2](docs/RETRAINING_V2.md).

Ba mạng là checkpoint thật được huấn luyện mới trên bộ **6.840 ảnh v2**, cùng quy trình chuyển giao backbone đóng băng. Trang Phân tích dùng MobileNetV2 v3 đã được fine-tune và chọn theo validation loss; kết quả của mô hình chính được phân biệt với bảng so sánh ba backbone đóng băng.


Website tiếng Việt phân loại ảnh lá chè bằng MobileNetV2 thành **8 lớp: 7 bệnh và lá khỏe mạnh**. Bản mặc định chạy mô hình **ngay trong trình duyệt** bằng ONNX Runtime Web: không cần tài khoản, API key, dịch vụ AI trả phí hay máy chủ Python.

- Website: https://tea-leaf-ai.vercel.app
- Mã nguồn phiên bản hiện tại và hướng dẫn chạy: https://github.com/khanh1023/TeaLeafAI/releases/latest
- Mã nguồn: https://github.com/khanh1023/TeaLeafAI

## 1. Cách chạy dễ nhất trên Windows

1. Cài **Node.js LTS 22 trở lên** từ https://nodejs.org nếu máy chưa có.
2. Tải **Source code (zip)** của bản phát hành hiện tại trong mục **Releases**, giải nén toàn bộ.
3. Nhấp đúp **`CHAY-WEBSITE.cmd`**. Trình duyệt mở tại http://127.0.0.1:5173.
4. Vào **Phân tích** → chọn hoặc kéo thả ảnh → **Phân tích ảnh**.

Giữ cửa sổ chạy mở trong lúc sử dụng; nhấn Ctrl+C để dừng. Bản mã nguồn có các mô hình ONNX hiện tại. Lần chạy đầu, file `.cmd` cài thư viện và build giao diện, cần Internet; không cần Python để dùng website. Không mở trực tiếp `index.html` bằng `file://` vì trình duyệt cần HTTP để tải mô hình.

Các file `TeaLeafAI-ready.zip` và release `v2.0.0` trước đây là bản lưu v1, không chứa mô hình mặc định hiện tại. Nếu báo cổng 5173 đã dùng, đóng cửa sổ TeaLeafAI cũ rồi chạy lại.

## 2. Chạy từ mã nguồn để phát triển

Tại thư mục dự án, Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

Lệnh chỉ áp dụng ExecutionPolicy cho tiến trình chạy file này, không đổi thiết lập hệ thống. Linux/macOS:

```bash
bash scripts/start.sh
```

Hoặc chạy từng lệnh riêng:

```bash
cd frontend
npm ci
npm run dev
```

Sau đó mở địa chỉ Vite hiển thị, thường là http://localhost:5173. Xem hướng dẫn chi tiết tại [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md).

## 3. Chức năng có sẵn

- Chọn hoặc kéo thả ảnh JPG, PNG, WEBP, tối đa 8 MB.
- Chụp lá chè bằng camera ngay trên website: ưu tiên camera sau, đổi camera, xem lại ảnh rồi phân tích; có tùy chọn dùng camera gốc của điện thoại. Camera trực tiếp cần HTTPS (Vercel đã có) hoặc localhost, và quyền camera của bạn. Không thu âm.
- AI chạy trên thiết bị, hiển thị tiến trình khởi tạo và thông báo lỗi rõ ràng.
- Kết quả gồm nhãn dự đoán, điểm mô hình và xác suất của đủ 8 lớp.
- Thư viện 8 thẻ tóm tắt; bấm từng thẻ để xem dấu hiệu, xử lý ban đầu, phòng ngừa và tài liệu tham khảo.
- Gợi ý xử lý/chăm sóc tương ứng hiển thị ngay dưới kết quả phân tích, kể cả nhóm lá khỏe; kết quả điểm thấp và nhãn cần xác minh có lưu ý riêng.
- Hướng dẫn chụp ảnh và trang kiểm tra hệ thống.
- Lịch sử lưu trong trình duyệt (tối đa 100 kết quả); xóa từng mục hoặc toàn bộ, xuất CSV; xuất kết quả riêng ra JSON.
- Giao diện thích ứng điện thoại và máy tính, có hỗ trợ bàn phím.
- Tài khoản/đồng bộ không nằm trong bản này; không có trang quản trị huấn luyện công khai. Huấn luyện chỉ chạy bằng Python trên máy riêng.

**Riêng tư:** ở chế độ mặc định `browser`, ảnh không được gửi lên API hoặc lưu lên máy chủ. Trình duyệt tải mô hình ONNX khoảng 9 MB và runtime WASM khoảng 11 MB ở lần dùng đầu, rồi xử lý ảnh trong bộ nhớ của thiết bị. Lịch sử chỉ giữ tên ảnh, thời gian và kết quả, không giữ ảnh. Đổi trình duyệt hoặc xóa dữ liệu trình duyệt sẽ mất lịch sử. Dữ liệu này không được đồng bộ.

## 4. Tám nhãn theo đúng thứ tự mô hình

| Nhãn | Tên hiển thị |
| --- | --- |
| `algal_leaf` | Bệnh đốm tảo |
| `anthracnose` | Bệnh thán thư |
| `bird_eye_spot` | Bệnh đốm mắt chim |
| `brown_blight` | Bệnh đốm nâu |
| `gray_light` | Bệnh đốm xám |
| `healthy` | Lá khỏe mạnh |
| `red_leaf_spot` | Bệnh đốm đỏ |
| `white_spot` | Bệnh đốm trắng |

Giữ nguyên `gray_light` theo bộ dữ liệu nguồn; cần xác minh tên bệnh tương ứng. Không đổi thứ tự `classes.json` mà không huấn luyện và xuất lại mô hình.

## 5. Triển khai miễn phí

Website mặc định chỉ gồm file tĩnh, vì vậy không cần Render hay máy chủ TensorFlow. Có thể dùng Vercel Hobby hoặc Cloudflare Pages Free trong giới hạn gói hiện hành. Xem [docs/FREE_HOSTING.md](docs/FREE_HOSTING.md).

**Vercel (repo này đã có website):**

- Import `khanh1023/TeaLeafAI` hoặc dùng project hiện có.
- Root Directory: `frontend`; Framework: `Vite`.
- Build Command: `npm run build`; Output Directory: `dist`.
- Node.js: `22.x` trở lên.
- Không cần biến môi trường. Mặc định là `browser`, kể cả khi còn `VITE_API_URL` cũ. Chỉ đặt `VITE_INFERENCE_MODE=api` khi chủ động sử dụng API riêng.
- Push nhánh `main` để Vercel tự build nếu đã liên kết repo.

## 6. Kiến trúc

```text
frontend/                   React + Vite + onnxruntime-web
  public/model/             ONNX, manifest kiểm tra SHA-256, thứ tự nhãn
  scripts/prepare-runtime   Sao chép runtime từ dependency đã khóa phiên bản
  scripts/serve-dist.mjs    Chạy bản build sẵn, không cần npm install
backend/                    FastAPI tùy chọn + mô hình Keras gốc
training/                   Chuẩn bị dữ liệu, train, đánh giá, xuất ONNX
scripts/                    Chạy phát triển trên Windows/Linux/macOS
tests/                      Kiểm tra backend và chuẩn bị dataset
.github/workflows/          Kiểm tra tự động khi push/PR
```

Ảnh đầu vào: RGB float32, kích thước 224×224, **pixel thô 0–255**. Mỗi mô hình chứa chuẩn hóa phù hợp kiến trúc: MobileNetV2 rescale về [-1,1], ResNet50/VGG16 đổi RGB sang BGR và trừ mean ImageNet. Không chuẩn hóa lần nữa ở ứng dụng. Mô hình chính v3 được đối chiếu Keras/ONNX trên **1.242 ảnh test v3**; ba mô hình so sánh được đối chiếu trên **1.026 ảnh test v2**. Không có nhãn top-1 bất đồng khi chuyển định dạng; sai số xác suất nhỏ hơn `1e-4`. Đây là kiểm tra chuyển định dạng; Accuracy được báo riêng theo mô hình và cách tiền xử lý.

## 7. Backend Python tùy chọn

Chỉ cần khi tích hợp hệ thống khác hoặc chủ động chuyển sang API. Python 3.11/3.12 được khuyến nghị.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 -WithBackend
```

Hoặc Linux/macOS:

```bash
bash scripts/start.sh --with-backend
```

Backend chạy cổng 9000, frontend chạy 5173. Khi dùng API, ảnh được gửi tới URL API cấu hình; giao diện hiển thị chế độ này. Không đưa `.env`, mật khẩu hoặc token lên GitHub.

## 8. Huấn luyện, dữ liệu và giới hạn

**Mô hình mặc định hiện tại dùng dữ liệu v3:** **8.280 ảnh**, bổ sung **1.440 ảnh** sau lọc so với v2;
gồm **3.435 ảnh lá khỏe**, chia chính xác **5.796 train / 1.242 validation / 1.242 test (70/15/15)**.
Mọi ảnh v2 giữ nguyên nhãn, nội dung và tập đã chia. [Nguồn ảnh, giấy phép và bảng 8 lớp](docs/DATASET_V3.md).

Keras/ONNX Python với tiền xử lý Pillow đạt **89.29% (1109/1.242 ảnh đúng)**,
macro-F1 **0.8495**. Luồng thực tế trên Chromium đạt
**89.29% (1109/1.242 ảnh đúng)**, macro-F1 **0.8495**.
**Kết quả trên trình duyệt hiện chưa đạt mục tiêu trên 90%**; Accuracy đo được là **89.29%**.

Checkpoint và phương án suy luận được chọn bằng validation loss. Mô hình dùng **trung bình xác suất từ ảnh gốc và ba hướng lật của cùng ảnh, sau đó hiệu chỉnh theo trọng số lớp với β=0.75 chọn trên validation**;
các hướng lật không được tính thành mẫu dữ liệu độc lập. Website dùng resize bilinear chống răng cưa theo quy tắc Pillow;
JPEG decoder và quản lý màu của trình duyệt vẫn có thể tạo chênh lệch nhỏ.
[Báo cáo, ảnh và bằng chứng dự đoán](training/reports/retraining-v3-20261128/README.md) ·
[Tải báo cáo Word](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-ket-qua-huan-luyen-v3.docx).

Ba mô hình ở trang **Thử ba mô hình** tiếp tục thuộc thí nghiệm **v2 với 6.840 ảnh**, cùng backbone đóng băng:
MobileNetV2 **82,16%**, ResNet50 **85,58%**, VGG16 **83,14%** trên test v2.
Không dùng điểm MobileNetV2 v3 đã tinh chỉnh để thay vào bảng so sánh kiến trúc v2.
[Quy trình v2](docs/RETRAINING_V2.md) · [Lượt v3 đầu được lưu đầy đủ](training/reports/retraining-v3-preliminary-20261128/README.md).

[Tải gói khôi phục đầy đủ dữ liệu v3](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-khoi-phuc-du-lieu-v3.zip),
giải nén rồi chạy `KHOI-PHUC-DU-LIEU.cmd`, hoặc từ mã nguồn chạy `python training/restore_dataset_v3.py`.
Gói nhỏ chứa ảnh bổ sung và tải phần v2 đã công bố qua Internet; cần Python chuẩn, không cần TensorFlow để khôi phục.
Xem [lệnh tái lập huấn luyện và lựa chọn suy luận](docs/DATASET_V3.md).

Đây là đánh giá phát triển nội bộ: test đã được báo cáo ở lượt trước, không phải kiểm thử mù mới hay kiểm định thực địa
độc lập tại Việt Nam. Ba lớp thán thư, đốm mắt chim và đốm trắng chưa có ảnh độc lập mới được bổ sung.

### Các kết quả v1 được giữ để đối chiếu lịch sử

Mô hình v1 trước đây được huấn luyện với **2.197 ảnh**, gồm **673 ảnh healthy**. Bộ chia có 1.541 ảnh train, 327 validation và 329 test. Dữ liệu và kết quả này không phải phiên bản đang triển khai.

- [Tải bộ dữ liệu đã chia](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-training-data.zip), khoảng 75 MiB; tải riêng với bản website.
- [Nguồn ảnh, giấy phép và lệnh tái tạo/huấn luyện](docs/TRAINING_DATA.md).
- [Báo cáo v1 đã lưu](training/reports/retraining-v1-20261117/REPORT.md), [số liệu v1](training/reports/retraining-v1-20261117/evaluation.json) và [môi trường phần mềm](docs/MOI_TRUONG_PHAN_MEM.md).

Accuracy v1 là **69,30% trên 329 ảnh**, macro F1 **0,6453**. Đây là kết quả lịch sử trên bộ v1; báo cáo trước v1 vẫn được giữ tại `training/reports/previous-evaluation.json`.

[Đánh giá lại ngày 18/11/2026](training/reports/reevaluation/REPORT.md) tái lập **228/329 ảnh đúng (69,30%)** với Keras/ONNX Python. Đường xử lý ảnh trên trình duyệt Chromium đạt **220/329 (66,87%)**; cần phân biệt hai cách tiền xử lý khi trích số liệu. Mức 94,5% trên 400 ảnh trong bản Word người dùng cung cấp chưa được tái lập bằng mô hình và bộ test hiện tại.

[Đánh giá train/validation/test/All ngày 22/11/2026](training/reports/evaluation-20261122/README.md) giữ nguyên mô hình và bộ chia, kèm sáu phép thử biến đổi ảnh. [Tải nội dung mục 3.3 trong Word](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-muc-3-3-danh-gia.docx) hoặc [bản đầy đủ 3.2–3.3 gồm 11 hình và bảng chỉnh sửa được](https://github.com/khanh1023/TeaLeafAI/raw/refs/heads/main/downloads/TeaLeafAI-chuong-3-ket-qua-danh-gia.docx). Tài liệu được biên soạn ngày 23/11/2026 từ các kết quả đo, không huấn luyện lại.

[Notebook Colab so sánh MobileNetV2, ResNet50 và VGG16](https://colab.research.google.com/github/khanh1023/TeaLeafAI/blob/main/training/Compare_Models_Colab.ipynb) chuẩn bị thí nghiệm chuyển giao cùng bộ chia, xuất bảng Word và hình sau khi chạy đủ ba mạng. [Xem quy trình](docs/MODEL_COMPARISON.md). Notebook chưa chứa kết quả thực nghiệm mới và không thay mô hình đang triển khai.

[Ảnh và nội dung ngắn mục 3.3.4](training/reports/model-comparison-20261123/README.md) nêu phạm vi kiểm chứng hiện có, file ONNX khoảng 9,58 MB và có [bản Word](https://raw.githubusercontent.com/khanh1023/TeaLeafAI/main/downloads/TeaLeafAI-muc-3-3-4-so-sanh-mo-hinh.docx). Chưa xếp hạng Accuracy hoặc tốc độ của ba kiến trúc.

Quy trình v2 được tái tạo bằng `training/run_v2_pipeline.py --run-dir training/runs/v2` sau khi khôi phục dữ liệu và cài thư viện huấn luyện. Script huấn luyện, đánh giá và xuất vào staging, không tự thay file website hoặc đẩy GitHub. Các script chuẩn bị dữ liệu và báo cáo v1 được giữ để tái lập theo đúng phiên bản cũ.

Sau khi huấn luyện lại, xuất mô hình browser bằng [training/export_browser_model.py](training/export_browser_model.py), theo [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md). Commit ONNX mới và manifest cùng nhau, rồi build lại website.

Mô hình chỉ phân loại trong 8 nhóm đã học; **không có bộ phát hiện “không phải lá chè” đáng tin cậy**. Ảnh vật thể khác vẫn có thể nhận điểm cao. Điểm mô hình không phải xác suất bệnh đã được kiểm định. Không dùng kết quả riêng lẻ để quyết định thuốc hoặc điều trị cây trồng; cần kiểm tra thực địa và chuyên gia.

## 9. Kiểm tra

```bash
cd frontend
npm test
npm run build
```

Python, tại thư mục gốc:

```bash
python -m pip install -r tests/requirements.txt
python -m unittest discover -s tests -v
```

Bộ kiểm tra nhẹ không cần tải TensorFlow. CI chạy các lệnh này khi push hoặc mở PR.
