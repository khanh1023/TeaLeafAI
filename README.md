# TeaLeafAI · Nhận diện bệnh lá chè

Website tiếng Việt phân loại ảnh lá chè bằng MobileNetV2 thành **8 lớp: 7 bệnh và lá khỏe mạnh**. Bản mặc định chạy mô hình **ngay trong trình duyệt** bằng ONNX Runtime Web: không cần tài khoản, API key, dịch vụ AI trả phí hay máy chủ Python.

- Website: https://tea-leaf-ai.vercel.app
- Tải bản chạy sẵn: https://github.com/khanh1023/TeaLeafAI/releases/latest
- Mã nguồn: https://github.com/khanh1023/TeaLeafAI

## 1. Cách chạy dễ nhất trên Windows

1. Cài **Node.js LTS 22 trở lên** từ https://nodejs.org nếu máy chưa có.
2. Tải `TeaLeafAI-ready.zip` trong mục **Releases**, giải nén toàn bộ.
3. Nhấp đúp **`CHAY-WEBSITE.cmd`**. Trình duyệt mở tại http://127.0.0.1:5173.
4. Vào **Phân tích** → chọn hoặc kéo thả ảnh → **Phân tích ảnh**.

Giữ cửa sổ chạy mở trong lúc sử dụng; nhấn Ctrl+C để dừng. Bản ZIP đã có giao diện được build, mô hình và WebAssembly nên không cần cài Python hoặc tải thư viện npm để dùng bản chạy sẵn. Node chỉ dùng để phục vụ file website trên máy. Không mở trực tiếp `index.html` bằng `file://` vì trình duyệt cần HTTP để tải mô hình.

Nếu tải mã nguồn thay vì bản Release, file `.cmd` sẽ cài thư viện và build ở lần chạy đầu; cần Internet cho bước này. Nếu báo cổng 5173 đã dùng, đóng cửa sổ TeaLeafAI cũ rồi chạy lại.

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

Ảnh đầu vào: RGB float32, kích thước 224×224, **pixel thô 0–255**. Mô hình đã chứa lớp `Rescaling(1/127.5, offset=-1)`, vì vậy không chuẩn hóa ảnh thêm lần nữa. Đã sửa lỗi backend cũ chuẩn hóa hai lần. Bản ONNX được so với Keras trên 9 ảnh; nhãn top-1 khớp toàn bộ, sai số xác suất lớn nhất khoảng `2.64e-6`. Kiểm tra này chứng minh việc chuyển định dạng, **không phải đánh giá độ chính xác trên dữ liệu mới**.

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

Đặt ảnh gốc trong `DuLieuGoc`, theo tên lớp tương ứng (dấu cách hoặc gạch dưới được chấp nhận). Chỉ chạy khi cần huấn luyện lại:

```powershell
cd training
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_dataset.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
```

Linux/macOS: thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`.

`prepare_dataset.py` yêu cầu gõ YES trước khi tạo lại dataset chia 70/15/15. Script loại ảnh trùng nội dung trong cùng lớp và dừng nếu cùng một ảnh có nhãn mâu thuẫn; không sửa hay xóa dữ liệu gốc. Nếu các ảnh cùng cây/lá/phiên chụp, phải chia theo nhóm nguồn để tránh rò rỉ dữ liệu.

**Vấn đề thực tế đã tìm thấy:** `Anthracnose/IMG_20220503_150405.jpg` và `bird eye spot/IMG_20220503_150405.jpg` có cùng nội dung nhưng khác nhãn. Cần người có chuyên môn xác minh trước khi huấn luyện lại. Dữ liệu gốc được giữ nguyên. File `training/evaluation.json` là báo cáo cũ của dự án, chưa được coi là kiểm định độc lập hoặc cam kết độ chính xác ngoài thực tế.

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
