# Nội dung nhận biết và hướng xử lý

Dữ liệu giao diện nằm trong `frontend/src/diseases.js`, dùng đúng 8 nhãn và thứ tự trong `classes.js`. Mỗi mục có bản tóm tắt, dấu hiệu quan sát, điều kiện cần kiểm tra, việc nên làm, phòng ngừa, lúc cần chuyên gia và đường dẫn nguồn.

## Luồng sử dụng

- Thư viện và thẻ trang chủ dẫn đến `/thu-vien/<nhãn>`; thẻ chỉ hiện thông tin ngắn.
- Trang chi tiết đọc cùng dữ liệu với `DiseaseAdvice` bên dưới kết quả phân tích.
- Kết quả lá khỏe hiển thị chăm sóc/theo dõi; không được biến thành một chẩn đoán bệnh.
- Điểm mô hình thấp hơn 0.7 có nhắc đối chiếu và chụp lại. Mốc này chỉ là nhắc UX, không phải ngưỡng được kiểm định bệnh học.
- gray_light, red_leaf_spot và white_spot có ghi chú về việc xác minh nhãn; không khẳng định đã giám định tác nhân từ ảnh.
- Không đưa liều thuốc hoặc lịch phun tự động theo nhãn AI. Hướng dẫn ban đầu ưu tiên quan sát, chăm sóc và nhờ cán bộ kỹ thuật khi cần.

## Nguồn và phạm vi

- TNAU Agritech, Algal leaf spot: dấu hiệu cơ bản và vai trò của thoát nước, tán thông thoáng, dinh dưỡng. https://agritech.tnau.ac.in/crop_protection/tea_diseases_1.html
- TNAU Agritech, Brown blight/grey blight: mô tả đốm nâu/xám và giảm stress, thời gian lá ướt. https://agritech.tnau.ac.in/crop_protection/tea_diseases_2.html
- KKHSOU Tea World, Leaf Diseases: mô tả đốm nâu, xám và thán thư. https://www.kkhsou.ac.in/web/teaworld/page-details.php?name=Leaf-Diseases&page=42433deb7ea680602fcc3ded2
- Sinniah & Mahadevan (2024), Disease Diagnosis in Tea, DOI 10.5772/intechopen.1004903: nhận dạng đốm mắt chim, sự phức tạp của giám định bệnh chè. https://www.intechopen.com/chapters/1160040
- Chen, Liu & Gao (2019), Symmetry 11(3),343, DOI 10.3390/sym11030343: đối chiếu tên 7 nhóm bệnh, sự tương tự giữa một số biểu hiện. Bài báo CC BY 4.0; website chỉ tóm tắt, không dùng hình ảnh hoặc số liệu độ chính xác của nghiên cứu làm chỉ số của TeaLeafAI. https://www.mdpi.com/2073-8994/11/3/343
- UPASI Tea Research Foundation: vết thương trong thu hái, sâu/nhện và các biểu hiện dễ nhầm khi quan sát lá. https://www.upasitearesearch.org/pests-diseases/

Các thao tác ghi ảnh, đối chiếu nhiều lá, theo dõi thay đổi và chuẩn bị thông tin cho cán bộ kỹ thuật là hướng dẫn hỗ trợ sử dụng tại vườn. Nguồn tham khảo không chứng minh bộ dữ liệu hoặc mô hình hiện tại đã được kiểm định, cũng không xác nhận mọi ảnh mang một nhãn đều đúng bệnh. Không sao chép các lịch phun/thuốc ở nước khác thành hướng dẫn tự áp dụng tại Việt Nam.
