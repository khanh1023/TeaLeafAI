import { CLASSES } from './classes.js';

export const SOURCES = {
  algal: {title:'TNAU Agritech — Algal leaf spot',url:'https://agritech.tnau.ac.in/crop_protection/tea_diseases_1.html'},
  blight: {title:'TNAU Agritech — Brown blight, grey blight',url:'https://agritech.tnau.ac.in/crop_protection/tea_diseases_2.html'},
  teaWorld: {title:'KKHSOU Tea World — Leaf Diseases',url:'https://www.kkhsou.ac.in/web/teaworld/page-details.php?name=Leaf-Diseases&page=42433deb7ea680602fcc3ded2'},
  diagnosis: {title:'Sinniah & Mahadevan (2024) — Disease Diagnosis in Tea',url:'https://www.intechopen.com/chapters/1160040'},
  labels: {title:'Chen, Liu & Gao (2019) — Các nhóm bệnh trong nghiên cứu nhận diện lá chè',url:'https://www.mdpi.com/2073-8994/11/3/343'},
  upasi: {title:'UPASI Tea Research Foundation — Pests & Diseases',url:'https://www.upasitearesearch.org/pests-diseases/'}
};

const content = {
  algal_leaf: {
    summary:'Đốm hơi nổi trên mặt lá, thường có sắc nâu đỏ hoặc màu gỉ. Ưu tiên kiểm tra độ ẩm và thoát nước.',
    signs:['Đốm gần tròn, hơi gồ trên bề mặt lá, có thể mang màu nâu đỏ.', 'Vết cũ có thể chuyển xanh xám; cần nhìn rõ cả hai mặt lá.'],
    conditions:['Tán ẩm, bí và đất thoát nước kém là những yếu tố cần kiểm tra.'],
    actions:['Khơi thông chỗ đọng nước quanh bụi, tránh để đất úng kéo dài.', 'Điều chỉnh tán ở mức vừa phải để lá thông thoáng; tránh tỉa mạnh làm cây suy.', 'Kiểm tra nước tưới và dinh dưỡng, chăm sóc cân đối theo tình trạng đất và cây.'],
    prevention:['Duy trì thoát nước tốt và độ thông thoáng của tán.', 'Theo dõi các lá mới để biết đốm có tiếp tục xuất hiện hay không.'],
    expert:'Nhờ cán bộ kỹ thuật kiểm tra nếu nhiều lá mới bị ảnh hưởng, lá rụng hoặc cành có biểu hiện suy yếu.',
    sources:['algal']
  },
  anthracnose: {
    summary:'Tổn thương nâu đỏ có thể lan rộng và làm rụng lá. Dễ nhầm với các nhóm cháy, đốm lá khác.',
    signs:['Mảng tổn thương nâu đỏ, có thể mở rộng trên phiến lá.', 'Lá bị nặng có thể khô và rụng; chỉ nhìn màu đốm chưa đủ để xác định bệnh.'],
    conditions:['Cần lưu ý khi thời tiết mưa ẩm và khi cây đang suy yếu.'],
    actions:['Chụp lại ranh giới giữa mô xanh và mô bị tổn thương, kiểm tra thêm các lá quanh đó.', 'Thu gom lá bệnh rụng; vệ sinh dụng cụ trước khi chuyển sang bụi khác.', 'Giảm thời gian lá bị ướt kéo dài và kiểm tra tình trạng nước, dinh dưỡng của bụi.'],
    prevention:['Theo dõi lá và chồi mới sau đợt mưa, chăm sóc để cây không bị suy.', 'Hạn chế làm xây xát lá, cành khi chăm sóc.'],
    expert:'Cần kiểm tra thực địa nếu vết lan nhanh, rụng nhiều lá hoặc chồi có dấu hiệu khô chết. Thán thư và đốm nâu có thể được gọi tên khác nhau trong tài liệu.',
    sources:['teaWorld','diagnosis']
  },
  bird_eye_spot: {
    summary:'Đốm nhỏ gần tròn, tâm nhạt và viền sẫm, có dạng giống mắt chim. Cần quan sát kỹ hình dạng từng đốm.',
    signs:['Đốm gần tròn, tâm xám nhạt hoặc trắng xám, viền nâu đỏ sẫm hơn.', 'Khi phát triển, hình dạng đốm có thể gợi hình mắt chim; không phải mọi đốm tròn đều thuộc nhóm này.'],
    conditions:['Kiểm tra độ ẩm trong tán và xem đốm mới có xuất hiện trên nhiều lá hay không.'],
    actions:['Chụp cận cảnh đốm có tâm và viền rõ, thêm một ảnh toàn lá để đối chiếu.', 'Ghi lại vài lá đang có đốm và theo dõi sự thay đổi thay vì chỉ dựa vào một ảnh.', 'Giữ tán thông thoáng, thu gom lá bệnh rụng và tránh tưới làm lá ướt lâu.'],
    prevention:['Kiểm tra mặt dưới lá và các bụi lân cận trong các lần thăm vườn.', 'Vệ sinh dụng cụ và hạn chế di chuyển lá có tổn thương sang khu vực khác.'],
    expert:'Nhờ chuyên gia xác minh nếu đốm tăng nhanh hoặc giống nhiều nhóm bệnh khác; có thể cần kiểm tra mẫu để xác định tác nhân.',
    sources:['diagnosis','labels']
  },
  brown_blight: {
    summary:'Mảng nâu trên lá, đôi khi có vòng đồng tâm hoặc chấm đen nhỏ. Cần kiểm tra cả tình trạng stress và vết thương.',
    signs:['Vết đổi màu mở rộng thành mảng nâu, có thể có viền rõ.', 'Có thể thấy vòng đồng tâm hoặc chấm đen nhỏ ở mô bị tổn thương.'],
    conditions:['Lá ướt lâu, cây suy và tổn thương khi thu hái là các yếu tố cần lưu ý.'],
    actions:['Kiểm tra bụi có thiếu nước, úng hoặc suy sau thu hái hay không.', 'Giảm làm xây xát lá, cành; vệ sinh dụng cụ giữa các khu vực.', 'Giữ tán thông thoáng và thu gom lá rụng có tổn thương.'],
    prevention:['Tránh để cây chịu stress nước hoặc dinh dưỡng kéo dài.', 'Theo dõi tình trạng lá sau thu hái và sau đợt mưa.'],
    expert:'Cần hỗ trợ kỹ thuật nếu diện tích lá khô tăng, rụng nhiều hoặc khó phân biệt với đốm xám và thán thư.',
    sources:['blight','upasi']
  },
  gray_light: {
    summary:'Vết có tâm xám và phần rìa sẫm có thể gợi nhóm đốm xám. Nhãn dữ liệu này cần được đối chiếu thêm.',
    signs:['Mảng tròn hoặc bầu dục có tâm xám, phần rìa nâu.', 'Có thể xuất hiện vòng hoặc chấm đen nhỏ; biểu hiện dễ chồng lấn với đốm nâu.'],
    conditions:['Kiểm tra thời gian lá bị ướt, độ bí của tán và các vết thương trên lá.'],
    actions:['Chụp cả tâm, viền đốm và toàn lá trước khi quyết định cách xử lý.', 'Giảm lá ướt lâu, giữ tán thông thoáng và chăm sóc cây cân đối.', 'Đối chiếu mẫu với cán bộ kỹ thuật nếu tổn thương tiếp tục lan.'],
    prevention:['Hạn chế gây thương tích khi thu hái và vệ sinh dụng cụ.', 'Theo dõi lá mới thay vì chỉ xem các vết đã khô.'],
    expert:'Cần xác minh nếu lá rụng nhiều, chồi suy hoặc hình ảnh không khớp mô tả.',
    identityNote:'Hãy đối chiếu kết quả với dấu hiệu thực tế trên cây trước khi chọn cách xử lý.',
    sources:['blight','teaWorld','labels']
  },
  healthy: {
    summary:'Ảnh được xếp vào nhóm lá khỏe. Tiếp tục theo dõi cây vì một ảnh không loại trừ bệnh tiềm ẩn hoặc sâu hại.',
    signs:['Lá không có vùng hoại tử, đốm bất thường hay biến dạng rõ trong ảnh quan sát.', 'Màu và hình dạng cần được so với lá cùng tuổi, cùng giống trên bụi.'],
    conditions:['Kiểm tra cả chồi, mặt dưới lá, cành và vùng rễ khi thăm vườn.'],
    actions:['Tiếp tục chăm sóc nước, thoát nước và dinh dưỡng phù hợp với vườn.', 'Quan sát thêm các lá quanh vị trí chụp và kiểm tra dấu hiệu sâu, nhện ở mặt dưới.', 'Nếu thực tế cây vẫn bất thường, chụp lại phần tổn thương rõ; không bỏ qua dấu hiệu chỉ vì AI trả về lá khỏe.'],
    prevention:['Theo dõi định kỳ và lưu ảnh để so sánh thay đổi của cây.', 'Không cần dùng thuốc chỉ dựa trên kết quả “lá khỏe mạnh” của mô hình.'],
    expert:'Nhờ cán bộ kỹ thuật nếu cây héo, rụng lá, chậm phát triển hoặc có dấu hiệu ở cành, rễ dù ảnh lá được xếp nhóm khỏe.',
    sources:['upasi','diagnosis']
  },
  red_leaf_spot: {
    summary:'Các vùng đỏ hoặc nâu đỏ cần được đối chiếu với sâu hại và tác động môi trường, không chỉ với bệnh nấm.',
    signs:['Trên lá có vùng hoặc đốm đỏ, nâu đỏ khác với mô xung quanh.', 'Cần kiểm tra cả mặt dưới lá, vết chích hút và sự thay đổi màu trên toàn phiến.'],
    conditions:['Ghi nhận ánh sáng, độ ẩm và tình trạng sâu, nhện quanh lá; riêng màu đỏ không xác định được nguyên nhân.'],
    actions:['Chụp lại dưới ánh sáng tự nhiên, không dùng bộ lọc màu hay đèn flash mạnh.', 'Kiểm tra mặt dưới lá và các lá bên cạnh để tìm dấu hiệu sâu, nhện hoặc tổn thương khác.', 'Theo dõi đốm mới và hỏi cán bộ kỹ thuật trước khi chọn biện pháp đặc trị.'],
    prevention:['Duy trì chăm sóc cân đối, kiểm tra vườn định kỳ.', 'Ghi lại diễn biến tổn thương để phân biệt bệnh với stress hoặc sâu hại.'],
    expert:'Cần xác minh nếu sắc đỏ lan rộng, lá rụng hoặc nhiều bụi cùng bị ảnh hưởng.',
    identityNote:'Hãy đối chiếu kết quả với dấu hiệu thực tế trên cây trước khi chọn cách xử lý.',
    sources:['labels','upasi']
  },
  white_spot: {
    summary:'Đốm hoặc mảng sáng trên lá cần được phân biệt với phản sáng và cặn bám trước khi đánh giá tổn thương.',
    signs:['Có vùng trắng hoặc xám nhạt khác với phần lá xung quanh.', 'Quan sát lại khi đổi góc chụp để phân biệt tổn thương thật với phản quang, bụi hoặc cặn bám.'],
    conditions:['Ghi nhận đốm có tiếp tục xuất hiện trên lá mới và có kèm khô, rụng lá hay không.'],
    actions:['Chụp lại không dùng flash, lấy nét ở ranh giới vùng sáng và mô xanh.', 'Kiểm tra cả hai mặt lá; không tự coi mọi mảng trắng là một bệnh nấm.', 'Nếu là tổn thương thật, thu gom lá bệnh đã rụng, giữ tán thoáng và mang ảnh hoặc mẫu tới cán bộ kỹ thuật khi lan rộng.'],
    prevention:['Giảm lá ướt kéo dài và giữ vệ sinh khu vực quanh bụi.', 'Lưu ảnh trong các lần quan sát để phân biệt vết cũ với đốm mới.'],
    expert:'Cần hỗ trợ nếu nhiều lá mới có đốm sáng, lá biến dạng hoặc rụng, hoặc không phân biệt được tổn thương với cặn bám.',
    identityNote:'Hãy đối chiếu kết quả với dấu hiệu thực tế trên cây trước khi chọn cách xử lý.',
    sources:['labels','teaWorld']
  }
};

export const DISEASES = CLASSES.map(item => ({...item,...content[item.key]}));
export const DISEASE_BY_KEY = Object.freeze(Object.assign(Object.create(null), Object.fromEntries(DISEASES.map(item => [item.key,item]))));
export const CARE_NOTE = 'Hướng dẫn hỗ trợ quan sát và chăm sóc ban đầu, không phải đơn điều trị. Chỉ cân nhắc thuốc bảo vệ thực vật sau khi xác định nguyên nhân, theo hướng dẫn kỹ thuật và nhãn được phép dùng cho chè, đúng thời gian cách ly trước thu hái.';
