# Đưa TeaLeafAI lên website bằng công cụ miễn phí

Bản mặc định đã đóng gói mô hình chạy ngay trong trình duyệt. Không cần mua API, GPU, tên miền riêng hoặc thuê máy chủ Python. Người dùng cần trình duyệt hiện đại, đủ RAM và kết nối để tải website/mô hình. Đây là ứng dụng hỗ trợ tham khảo, không phải hệ thống chẩn đoán đã kiểm định.

## Vercel Hobby — dùng repo GitHub hiện tại

1. Tạo tài khoản Vercel hoặc dùng tài khoản đã có và kết nối GitHub.
2. Add New → Project → chọn repo `khanh1023/TeaLeafAI`.
3. Root Directory `frontend`, framework `Vite`, build `npm run build`, output `dist`, Node 22.x trở lên.
4. Không cần Environment Variables. Nếu project cũ có `VITE_INFERENCE_MODE=api`, xóa biến đó hoặc đổi thành `browser` rồi Redeploy. `VITE_API_URL` đơn lẻ không bật chế độ API.
5. Deploy. URL miễn phí có dạng do Vercel cấp. Repo này hiện dùng https://tea-leaf-ai.vercel.app.
6. Mỗi lần push nhánh `main`, integration Vercel sẽ tự build nếu kết nối đang bật. Xem Deployments để kiểm tra.

`frontend/vercel.json` cho phép mở trực tiếp `/chan-doan`, `/thu-vien`, `/lich-su` và các đường dẫn React. Runtime WASM được sao chép lúc build; mô hình ONNX nằm trong `public/model`, vì vậy không phụ thuộc CDN AI ngoài.

## Cloudflare Pages Free — triển khai cùng bản build

1. Tại thư mục `frontend`, chạy `npm ci` và `npm run build`.
2. Trong Cloudflare Dashboard, Workers & Pages → Pages → tạo project bằng tải tài nguyên tĩnh hoặc kết nối GitHub.
3. Nếu kết nối GitHub: Root `frontend`, build `npm run build`, output `dist`, Node 22.
4. Nếu tải trực tiếp: tải toàn bộ **nội dung** `frontend/dist` (hoặc ZIP nội dung này). `index.html` phải ở gốc bản tải.
5. Mở URL `.pages.dev` được cấp và thử Chẩn đoán. File `_redirects` có sẵn giúp các đường dẫn React hoạt động.

## Chi phí và giới hạn

Các thư viện React, Vite, ONNX Runtime là mã nguồn mở; website không gọi API AI trả phí. Hosting miễn phí có giới hạn băng thông, build, điều khoản và phạm vi sử dụng; kiểm tra gói hiện hành của nhà cung cấp. Website tải khoảng 20 MB mô hình/runtime ở lần dùng đầu; lượng truy cập lớn có thể chạm giới hạn miễn phí. Không cần thêm thẻ thanh toán để code này hoạt động tại máy cá nhân. Việc đăng ký tài khoản hosting, nếu chưa có, do bạn thực hiện.

## Kiểm tra sau khi deploy

- Trang Hệ thống báo đang ở chế độ trình duyệt và đọc được manifest.
- Mở Chẩn đoán; tải ảnh JPG, PNG hoặc WEBP nhỏ hơn 8 MB.
- Lần đầu chờ khởi tạo WebAssembly; kết quả hiển thị đủ 8 lớp.
- Xem tab Network của DevTools: chế độ browser tải model/runtime, không POST ảnh tới Render hay API ngoài.
- Mở trực tiếp `/chan-doan` và tải lại trang để kiểm tra SPA fallback.
