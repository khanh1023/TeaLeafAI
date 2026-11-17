import { Link } from 'react-router-dom';
import { ClipboardCheck, Eye, ArrowRight, AlertCircle } from 'lucide-react';
import { DISEASE_BY_KEY } from '../diseases';

export default function DiseaseAdvice({ diseaseKey, confidence }) {
  const disease = DISEASE_BY_KEY[diseaseKey];
  if (!disease) return <section className="panel result-advice"><h2>Đối chiếu thêm trước khi xử lý</h2><p>Chưa có hướng dẫn cho nhãn này. Hãy xem các nhóm trong thư viện hoặc nhờ cán bộ kỹ thuật kiểm tra.</p><Link className="text-link" to="/thu-vien">Mở thư viện <ArrowRight size={18}/></Link></section>;
  const healthy = disease.key === 'healthy';
  return <section className="panel result-advice" data-testid="result-advice" data-disease-key={disease.key} aria-labelledby="result-advice-title">
    <div className="advice-heading"><span className="step-badge">03</span><div><span className="eyebrow">GỢI Ý THEO KẾT QUẢ PHÂN TÍCH</span><h2 id="result-advice-title">{healthy ? 'Chăm sóc và theo dõi lá khỏe' : `Hướng xử lý tham khảo: ${disease.label}`}</h2></div></div>
    <p className="advice-intro">{healthy ? 'Ảnh được xếp vào nhóm lá khỏe; vẫn cần quan sát toàn cây và các lá xung quanh.' : 'Trước khi xử lý, hãy đối chiếu các dấu hiệu dưới đây với lá và tình trạng thực tế của bụi chè.'}</p>
    {confidence < 0.7 && <p className="warning-note"><AlertCircle size={18}/><span>Điểm mô hình còn thấp. Ưu tiên chụp lại và đối chiếu dấu hiệu; chưa chọn biện pháp đặc trị chỉ dựa trên kết quả này.</span></p>}
    {disease.identityNote && <p className="advice-identity-note">{disease.identityNote}</p>}
    <div className="advice-grid"><div><h3><ClipboardCheck size={20}/> Việc nên làm tiếp theo</h3><ol className="care-steps">{disease.actions.map(text => <li key={text}>{text}</li>)}</ol></div><div><h3><Eye size={20}/> Dấu hiệu cần đối chiếu</h3><ul className="disease-list">{disease.signs.map(text => <li key={text}>{text}</li>)}</ul><p className="advice-expert">{disease.expert}</p></div></div>
    <Link className="btn outline" to={`/thu-vien/${disease.key}`}>Xem chi tiết, phòng ngừa và tài liệu <ArrowRight size={18}/></Link>
  </section>;
}
