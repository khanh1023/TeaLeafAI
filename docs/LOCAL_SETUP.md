# Chạy TeaLeafAI trên máy của bạn

## Cách đơn giản: AI chạy trong trình duyệt

Bản website mặc định chỉ cần **Node.js 20.19+ hoặc 22+**, không cần tài khoản trả phí, Python hay máy chủ AI. Tải và giải nén ZIP, cài Node.js từ <https://nodejs.org>, mở thư mục vừa giải nén trong VS Code rồi chọn **Terminal → New Terminal**.

Windows PowerShell, chạy từng dòng riêng và nhấn Enter:

```powershell
npm ci --prefix frontend
npm run dev --prefix frontend -- --open
```

Hoặc chạy script hỗ trợ từ thư mục gốc:

```powershell
.\scripts\start.ps1
```

Nếu máy có chính sách chặn script PowerShell, dùng hai lệnh `npm` ở trên; không cần đổi chính sách bảo mật của máy.

macOS/Linux:

```bash
bash scripts/start.sh
```

Trang sẽ mở tại <http://localhost:5173>. Lần đầu dùng cần tải mô hình; chờ quá trình tải hoàn tất rồi phân tích ảnh. Giữ Terminal mở trong khi sử dụng. Đừng mở `frontend/index.html` bằng cách nhấp đúp vì mô hình cần máy chủ HTTP.

## Backend Python (tùy chọn)

Chỉ dùng khi muốn chạy dịch vụ API. Cài Python **3.11 hoặc 3.12** rồi chạy:

Windows PowerShell:

```powershell
.\scripts\start.ps1 -WithBackend
```

macOS/Linux:

```bash
bash scripts/start.sh --with-backend
```

Script tạo môi trường ảo trong `backend/.venv`, cài thư viện, khởi động API ở cổng 9000 và giao diện ở cổng 5173. Không cần kích hoạt môi trường ảo thủ công. Cài TensorFlow lần đầu có thể mất vài phút và tải nhiều dữ liệu.

Hoặc chạy API thủ công trong Terminal riêng:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 9000
```

API docs: <http://127.0.0.1:9000/docs>. Đặt đồng thời `VITE_INFERENCE_MODE=api` và `VITE_API_URL=http://127.0.0.1:9000` khi khởi động giao diện để chọn dịch vụ API. Mặc định AI chạy tại trình duyệt; `VITE_API_URL` riêng lẻ không bật chế độ API.

## Một số lỗi thường gặp

- **`cd TeaLeafAIcd backend`**: hai lệnh bị dán dính. Chạy mỗi lệnh trên một dòng hoặc mở Terminal đúng thư mục gốc rồi dùng script.
- **`npm` không nhận diện**: cài Node.js rồi đóng, mở lại Terminal.
- **Cổng 5173 đang được dùng**: dừng cửa sổ chạy trước bằng Ctrl+C hoặc đổi cổng bằng `--port 5174`.
- **Không có mô hình**: dùng ZIP đầy đủ hoặc tải toàn bộ mã nguồn. Các file trong `frontend/public/model` và `backend/models` phải đi cùng bản mã tương ứng; không chỉ đổi `classes.json` của mô hình khác.
- **Trình duyệt quá cũ**: dùng Chrome, Edge hoặc Firefox bản hiện tại. Nếu máy ít RAM, đóng bớt thẻ trình duyệt rồi thử lại.

## Huấn luyện và đánh giá

Dữ liệu gốc không nằm trong bản tải gọn để website chạy. Để huấn luyện lại, bổ sung ảnh đủ 8 lớp vào `DuLieuGoc`, cài `training/requirements.txt`, chạy `prepare_dataset.py`, `train.py`, rồi `evaluate.py`. Script chia dữ liệu loại bản sao giống hệt nhau và từ chối ảnh trùng nhưng mang nhãn khác nhau. Với nhiều ảnh cùng lá/cây/phiên chụp, vẫn phải tách theo nhóm nguồn; kiểm tra trùng file không bảo đảm tránh mọi rò rỉ dữ liệu.

Mô hình `.keras` hiện có lớp chuẩn hóa `Rescaling` bên trong. Đầu vào đúng là pixel RGB float32 trong khoảng **0–255**, kích thước **224×224**. Không chia 127.5 và trừ 1 lần nữa ở API hoặc trình duyệt.

Các chỉ số trong `training/evaluation.json` là báo cáo đã có trong repo, chưa được tái xác minh bằng một tập ảnh độc lập mới. Dự đoán chỉ hỗ trợ tham khảo, không thay thế chuyên gia cây trồng.

## Xuất mô hình cho trình duyệt sau khi huấn luyện lại

Bản ZIP đã có ONNX nên **không cần chạy bước này để sử dụng website**. Chỉ chạy khi thay mô hình `.keras` và cần đồng bộ website. Dùng Python 3.11 hoặc 3.12, tạo môi trường riêng để tránh thay đổi thư viện huấn luyện:

```powershell
python -m venv training/.export-venv
.\training\.export-venv\Scripts\python.exe -m pip install -r training/browser-export-requirements.txt
.\training\.export-venv\Scripts\python.exe training/export_browser_model.py
```

Trên macOS/Linux, thay đường dẫn Python thành `training/.export-venv/bin/python`. Script tạo `frontend/public/model/tea-leaf.onnx`, `classes.json`, `manifest.json` và so sánh số học giữa ONNX với Keras (sai số tối đa < 0.0001). Đây là kiểm tra chuyển định dạng, không phải đánh giá độ chính xác bệnh học. Sau đó build và triển khai lại giao diện.

## Kiểm tra mã Python

Có thể chạy các kiểm tra hợp đồng API và chia dữ liệu mà không cài TensorFlow:

```powershell
python -m pip install -r tests/requirements.txt
python -m unittest discover -s tests -v
```

### Vấn đề dữ liệu đã phát hiện khi rà soát

Trong bộ dữ liệu gốc hiện tại có ít nhất một ảnh trùng hệt nội dung nhưng nằm ở hai nhãn khác nhau:

- `DuLieuGoc/Anthracnose/IMG_20220503_150405.jpg`
- `DuLieuGoc/bird eye spot/IMG_20220503_150405.jpg`

Không tự đổi nhãn hoặc xóa dữ liệu gốc. Cần người có chuyên môn xác minh nhãn và dọn các bản sao trước khi huấn luyện lại. Script chuẩn bị dữ liệu sẽ dừng và nêu rõ hai đường dẫn khi gặp trường hợp này. Website vẫn sử dụng mô hình đã có, với giới hạn dự đoán tham khảo được hiển thị.

## Chụp ảnh bằng camera

Vào Phân tích → Chụp bằng camera → cho phép camera → giữ lá rõ nét trong khung hình → Chụp ảnh. Xem lại ảnh đầu vào rồi nhấn Phân tích ảnh. Nút Đổi camera chuyển giữa camera trước/sau nếu thiết bị hỗ trợ. Camera tắt khi đóng cửa sổ, chụp xong hoặc chuyển trang ra nền; website không dùng microphone.

Nếu camera trực tiếp bị chặn, chọn Dùng camera của điện thoại hoặc đóng cửa sổ để chọn ảnh có sẵn. Camera trực tiếp cần HTTPS hoặc localhost; mở website bằng HTTP qua địa chỉ IP trong mạng nội bộ có thể không cấp quyền camera. Lần dùng đầu vẫn cần Internet để tải mô hình và runtime; không tuyên bố ứng dụng đã có chế độ ngoại tuyến đầy đủ.
