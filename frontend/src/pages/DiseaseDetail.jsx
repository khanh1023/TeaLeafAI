import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Leaf, Eye, Sprout, ClipboardCheck, ShieldCheck, AlertCircle, ExternalLink } from 'lucide-react';
import { DISEASE_BY_KEY, DISEASES, SOURCES, CARE_NOTE } from '../diseases';

export default function DiseaseDetail() {
  const { diseaseKey } = useParams();
  const disease = DISEASE_BY_KEY[diseaseKey];
  if (!disease) return <section className="page"><div className="page-title"><h1>Chưa có thông tin cho nhóm này</h1><p>Vui lòng chọn một trong 8 nhóm của thư viện.</p></div><Link className="btn outline" to="/thu-vien"><ArrowLeft size={18}/> Về thư viện</Link></section>;
  const healthy = disease.key === 'healthy';
  return <section className="page disease-page" data-disease-key={disease.key}>
    <Link className="text-link disease-back" to="/thu-vien"><ArrowLeft size={18}/> Về thư viện 8 nhóm</Link>
    <header className="disease-intro"><div className="disease-intro-icon"><Leaf size={38}/></div><div><span className="eyebrow">{healthy ? 'CHĂM SÓC VÀ THEO DÕI' : 'NHẬN BIẾT VÀ HƯỚNG XỬ LÝ'}</span><h1>{disease.label}</h1><span className="english-name">{disease.name}</span><p>{disease.summary}</p></div></header>
    {disease.identityNote && <p className="warning-note"><AlertCircle size={20}/><span>{disease.identityNote}</span></p>}
    <div className="disease-detail-grid">
      <article className="panel"><h2><Eye size={22}/> {healthy ? 'Dấu hiệu để đối chiếu' : 'Dấu hiệu nhận biết cơ bản'}</h2><ul className="disease-list">{disease.signs.map(text => <li key={text}>{text}</li>)}</ul><h3>Quan sát thêm tại vườn</h3><ul className="disease-list">{disease.conditions.map(text => <li key={text}>{text}</li>)}</ul></article>
      <article className="panel care-steps-panel"><h2><ClipboardCheck size={22}/> {healthy ? 'Chăm sóc tiếp theo' : 'Hướng xử lý ban đầu'}</h2><ol className="care-steps">{disease.actions.map(text => <li key={text}>{text}</li>)}</ol></article>
      <article className="panel"><h2><Sprout size={22}/> {healthy ? 'Duy trì cây khỏe' : 'Phòng ngừa và theo dõi'}</h2><ul className="disease-list">{disease.prevention.map(text => <li key={text}>{text}</li>)}</ul></article>
      <article className="panel"><h2><ShieldCheck size={22}/> Khi nào cần hỗ trợ kỹ thuật?</h2><p className="disease-body">{disease.expert}</p><p className="disease-body">Khi cần kiểm tra, chuẩn bị ảnh toàn bụi, ảnh hai mặt lá và thông tin về diễn biến, mưa tưới, chăm sóc gần đây.</p></article>
    </div>
    <p className="notice"><AlertCircle size={20}/><span>{CARE_NOTE}</span></p>
    <section className="disease-sources" aria-labelledby="disease-sources-title"><h2 id="disease-sources-title">Tài liệu tham khảo</h2><p>Nội dung được tóm tắt để hỗ trợ quan sát. Các nguồn này không xác nhận riêng nhãn ảnh hoặc độ chính xác của mô hình TeaLeafAI.</p><ul>{disease.sources.map(key => <li key={key}><a href={SOURCES[key].url} target="_blank" rel="noreferrer">{SOURCES[key].title} <ExternalLink size={15}/></a></li>)}</ul></section>
    <div className="disease-bottom-actions"><Link className="btn primary" to="/chan-doan">Chụp ảnh để đối chiếu <ArrowRight size={18}/></Link><Link className="btn outline" to="/thu-vien">Xem các nhóm khác</Link></div>
    <nav className="related-diseases" aria-label="Các nhóm phân loại khác">{DISEASES.filter(item => item.key !== disease.key).map(item => <Link key={item.key} to={`/thu-vien/${item.key}`}>{item.label}</Link>)}</nav>
  </section>;
}
