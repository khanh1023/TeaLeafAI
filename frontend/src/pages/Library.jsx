import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Leaf, Search, ArrowUpRight, Info, ArrowRight } from 'lucide-react';
import { DISEASES } from '../diseases';

export default function Library() {
  const [query, setQuery] = useState('');
  const filtered = DISEASES.filter(item => `${item.label} ${item.name} ${item.key} ${item.summary}`.toLocaleLowerCase('vi').includes(query.toLocaleLowerCase('vi')));
  return <section className="page"><div className="page-title"><span className="eyebrow">NHẬN BIẾT VÀ CHĂM SÓC</span><h1>Thư viện lá chè</h1><p>Xem nhanh 7 nhóm bệnh và 1 nhóm lá khỏe. Chọn một thẻ để đọc dấu hiệu, hướng xử lý và cách phòng ngừa.</p></div>
    <label className="search-field library-search"><Search size={19}/><input aria-label="Tìm nhóm phân loại" placeholder="Tìm tên bệnh, dấu hiệu hoặc nhãn dữ liệu…" value={query} onChange={event => setQuery(event.target.value)}/></label>
    <div className="library-grid">{filtered.map(item => <Link to={`/thu-vien/${item.key}`} className={`library-card library-card-link ${item.key === 'healthy' ? 'healthy-card' : ''}`} key={item.key} aria-label={`Xem chi tiết ${item.label}`}>
      <div className="class-card-top"><span className="leaf-mini"><Leaf size={25}/></span><span className="number">{String(DISEASES.findIndex(x => x.key === item.key)+1).padStart(2,'0')}</span></div>
      <span className="english-name">{item.name}</span><h2>{item.label}</h2><p className="library-summary">{item.summary}</p>
      <span className="library-card-cta">{item.key === 'healthy' ? 'Xem cách chăm sóc' : 'Dấu hiệu & hướng xử lý'} <ArrowRight size={18}/></span>
    </Link>)}</div>
    {!filtered.length && <div className="empty compact"><Search size={30}/><h3>Chưa tìm thấy nhóm này</h3><p>Thử tên bệnh hoặc một dấu hiệu như “đốm”, “nâu”, “lá khỏe”.</p></div>}
    <div className="notice"><Info size={18}/><span>Dấu hiệu giữa các bệnh có thể giống nhau. Thông tin giúp bạn đối chiếu tại vườn; kết quả AI vẫn cần được kiểm tra trước khi quyết định xử lý.</span></div>
    <div className="center-action"><Link className="btn primary" to="/chan-doan">Chụp hoặc chọn ảnh để phân tích <ArrowUpRight size={18}/></Link></div>
  </section>;
}
