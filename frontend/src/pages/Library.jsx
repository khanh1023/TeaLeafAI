import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Leaf, Search, ArrowUpRight, Info } from 'lucide-react';
import { CLASSES } from '../classes';

const notes = {
  algal_leaf: 'Quan sát các vùng đốm trên bề mặt lá và chụp ảnh rõ cả vùng lá xung quanh.',
  anthracnose: 'Các biểu hiện tổn thương có thể giống nhóm bệnh khác. Cần đối chiếu mẫu và chuyên gia.',
  bird_eye_spot: 'Ưu tiên ảnh rõ chi tiết đốm, tránh ánh sáng phản chiếu làm mất màu tổn thương.',
  brown_blight: 'Chụp cả mặt lá và vùng chuyển tiếp giữa phần lá bình thường với vùng tổn thương.',
  gray_light: 'Tên kỹ thuật được giữ nguyên từ bộ dữ liệu. Cần xác minh nhãn gốc trước khi diễn giải chuyên môn.',
  healthy: 'Nhãn khỏe mạnh chỉ phản ánh những đặc điểm mà mô hình đã học; không loại trừ bệnh chưa biểu hiện.',
  red_leaf_spot: 'Chụp dưới ánh sáng tự nhiên để hạn chế sai lệch sắc đỏ và màu của lá.',
  white_spot: 'Tránh đèn flash vì vùng phản sáng có thể bị nhầm với đốm sáng trên bề mặt lá.'
};

export default function Library() {
  const [query, setQuery] = useState('');
  const filtered = CLASSES.filter((item) => `${item.label} ${item.name} ${item.key}`.toLocaleLowerCase('vi').includes(query.toLocaleLowerCase('vi')));
  return <section className="page"><div className="page-title"><span className="eyebrow">HIỂU CÁC NHÓM NHẬN DIỆN</span><h1>Thư viện lá chè</h1><p>7 nhóm bệnh và 1 nhóm lá khỏe mạnh trong dữ liệu huấn luyện. Đây là thông tin về nhãn phân loại, không phải cẩm nang điều trị.</p></div><label className="search-field library-search"><Search size={19}/><input aria-label="Tìm nhóm phân loại" placeholder="Tìm tên bệnh hoặc nhãn dữ liệu…" value={query} onChange={(event) => setQuery(event.target.value)}/></label><div className="library-grid">{filtered.map((item) => <article className={`library-card ${item.key === 'healthy' ? 'healthy-card' : ''}`} key={item.key}><div className="class-card-top"><span className="leaf-mini"><Leaf size={25}/></span><span className="number">{String(CLASSES.findIndex((x) => x.key === item.key) + 1).padStart(2, '0')}</span></div><span className="english-name">{item.name}</span><h2>{item.label}</h2><code className="label-code">{item.key}</code><p>{item.description}</p><div className="library-tip"><Info size={16}/><p>{notes[item.key]}</p></div></article>)}</div>{!filtered.length && <div className="empty compact"><Search size={30}/><h3>Chưa tìm thấy nhóm này</h3><p>Thử tìm “đốm”, “healthy” hoặc tên nhãn tiếng Anh.</p></div>}<div className="notice"><Info size={18}/><span>Nhiều bệnh có biểu hiện giống nhau. Mô hình luôn chọn trong 8 nhãn, kể cả khi ảnh không thuộc những nhóm này. Hãy xem kết quả cùng tình trạng cây và tư vấn chuyên gia.</span></div><div className="center-action"><Link className="btn primary" to="/chan-doan">Bạn có ảnh lá chè cần xem? <ArrowUpRight size={18}/></Link></div></section>;
}
