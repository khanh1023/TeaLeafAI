# Tea Leaf AI

Website hỗ trợ phân loại ảnh lá chè thành 5 lớp bằng MobileNetV2.

## 1. Kiến trúc
- `frontend/`: React + Vite, triển khai giao diện trên Vercel.
- `backend/`: FastAPI, nhận ảnh và chạy dự đoán.
- `training/`: chia dataset, huấn luyện và đánh giá mô hình local.
- `DuLieuGoc/`: bạn tự tạo và đặt dữ liệu gốc ở đây (không đưa lên Git).

## 2. Năm lớp dữ liệu
1. `blister_blight` - Bệnh phồng lá
2. `grey_blight` - Bệnh đốm xám
3. `algal_spot` - Bệnh đốm tảo
4. `brown_blight` - Bệnh đốm nâu
5. `healthy` - Lá khỏe mạnh

Lưu ý: thư mục nguồn `gray light` được ánh xạ sang `grey_blight`. Kiểm tra ảnh trong thư mục có đúng bệnh đốm xám không. Không đổi nhãn Anthracnose, bird eye spot, red leaf spot hoặc white spot thành một trong 5 lớp trên. Cần bổ sung ảnh Blister Blight có nhãn được xác minh.

## 3. Chuẩn bị dữ liệu
Tạo cấu trúc:
```
TeaLeafAI/
  DuLieuGoc/
    algal leaf/
    brown blight/
    gray light/
    healthy/
    blister blight/   # thêm ảnh bệnh phồng lá đã xác minh
  training/
```
Các thư mục nguồn có thể chứa ảnh trực tiếp hoặc thư mục con. Cài Python 3.10 hoặc 3.11, mở terminal tại thư mục `training`:
```
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_dataset.py
```
Chương trình yêu cầu gõ YES để tạo lại `training/dataset`. Tập gốc không bị xóa. Nếu thêm ảnh, hãy thêm vào `DuLieuGoc` rồi chạy lại script. Việc chia ngẫu nhiên từng file phù hợp khi ảnh độc lập; nếu nhiều ảnh cùng một cây/lá/phiên chụp, cần chia theo nhóm nguồn để tránh rò rỉ dữ liệu.

## 4. Huấn luyện local
Từ thư mục `training`:
```
python train.py
```
Mô hình dùng MobileNetV2 pretrained ImageNet, đầu vào 224x224, batch 32, tối đa 20 epoch, đóng băng backbone ở giai đoạn đầu. Kết quả tốt nhất lưu tại `backend/models/tea_leaf_mobilenetv2.keras` và danh sách lớp lưu trong `backend/models/classes.json`.

Không thể đảm bảo độ chính xác trước khi có dữ liệu đủ tốt. Không huấn luyện 5 lớp khi một lớp đang trống. Nên đánh giá bằng test set độc lập:
```
python evaluate.py
```

## 5. Chạy backend
Mở terminal mới tại `backend`:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
API docs: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/api/health

## 6. Chạy frontend
Mở terminal mới tại `frontend`:
```
npm install
copy .env.example .env
npm run dev
```
Mở URL do Vite hiển thị, thường là http://localhost:5173. `.env` cần `VITE_API_URL=http://127.0.0.1:8000`.

## 7. GitHub
Tại thư mục gốc:
```
git init
git add .
git commit -m "Initial Tea Leaf AI project"
git branch -M main
git remote add origin https://github.com/<TEN_TAI_KHOAN>/<TEN_REPO>.git
git push -u origin main
```
Không commit dataset gốc, dataset đã chia, file mô hình, `.venv`, `node_modules` hoặc file `.env`.

## 8. Vercel
1. Push source lên GitHub.
2. Vercel > Add New > Project > Import repository.
3. Root Directory chọn `frontend`.
4. Framework preset chọn Vite.
5. Build command `npm run build`, Output Directory `dist`.
6. Environment Variables thêm `VITE_API_URL` bằng URL backend được triển khai công khai.
7. Deploy.

**Quan trọng:** Vercel chỉ host frontend trong cấu hình này. Backend FastAPI và file mô hình cần chạy ở một máy chủ Python có thể truy cập từ Internet. Nếu `VITE_API_URL` vẫn là `127.0.0.1`, website công khai sẽ không thể gọi backend trên máy tính của bạn. Có thể demo local bằng cách chạy cả frontend và backend trên máy cá nhân; để chẩn đoán công khai, cần triển khai backend riêng và cấu hình CORS cho domain Vercel.

## 9. Giới hạn phiên bản khởi đầu
- Lịch sử hiện lưu trong localStorage của trình duyệt, chưa có tài khoản hoặc đồng bộ.
- Trang Admin là trang hướng dẫn/kiểm tra API; việc huấn luyện thực hiện bằng lệnh Python local.
- Dự đoán chỉ là hỗ trợ tham khảo. Cần kiểm định dữ liệu, mô hình và tư vấn chuyên gia trước khi dùng quyết định xử lý cây trồng.
