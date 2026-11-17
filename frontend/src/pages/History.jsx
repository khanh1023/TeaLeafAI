import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Trash2, History, Download, Search, ArrowRight, Clock3, AlertCircle } from 'lucide-react';
import { CLASSES, LABELS } from '../classes';
import { readHistory, saveHistory, clearHistory, exportHistory } from '../history';

function displayDate(date) {
  const parsed = new Date(date);
  return Number.isNaN(parsed.getTime()) ? 'Không có thời gian' : parsed.toLocaleString('vi-VN');
}

export default function HistoryPage() {
  const [items, setItems] = useState(readHistory);
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('');
  const [error, setError] = useState('');
  const filtered = useMemo(() => items.filter((item) => (!category || item.predicted_class === category) && item.filename.toLocaleLowerCase('vi').includes(query.toLocaleLowerCase('vi'))), [items, category, query]);

  function remove(id) {
    const next = items.filter((item) => item.id !== id);
    if (saveHistory(next)) { setItems(next); setError(''); }
    else setError('Trình duyệt không cho thay đổi lịch sử. Hãy kiểm tra quyền lưu dữ liệu của trang.');
  }
  function clear() {
    if (!window.confirm('Xóa toàn bộ lịch sử phân tích trên trình duyệt này?')) return;
    if (clearHistory()) { setItems([]); setError(''); }
    else setError('Chưa xóa được lịch sử. Hãy kiểm tra quyền lưu dữ liệu của trình duyệt.');
  }

  return <section className="page"><div className="page-title"><span className="eyebrow">NHÌN LẠI NHỮNG LẦN QUAN SÁT</span><h1>Lịch sử phân tích</h1><p>Lưu tối đa 100 lần phân tích gần nhất trên trình duyệt này. Ảnh gốc không được lưu trong lịch sử.</p></div>
    <div className="history-stats"><div><Clock3 size={22}/><p><b>{items.length}</b><span>Lần phân tích đã lưu</span></p></div><div><History size={22}/><p><b>Chỉ trên thiết bị này</b><span>Không đồng bộ giữa các trình duyệt</span></p></div></div>
    <div className="panel"><div className="history-head"><h2>Kết quả đã lưu <span className="count-tag">{items.length}</span></h2><div className="button-group"><button className="btn pale" disabled={!filtered.length} onClick={() => exportHistory(filtered)}><Download size={16}/> Xuất CSV</button><button className="btn danger-outline" disabled={!items.length} onClick={clear}><Trash2 size={16}/> Xóa tất cả</button></div></div>
      {error && <p className="error" role="alert"><AlertCircle size={17}/>{error}</p>}
      {items.length > 0 && <div className="history-filters"><label className="search-field"><Search size={17}/><input aria-label="Tìm theo tên ảnh" placeholder="Tìm theo tên ảnh…" value={query} onChange={(event) => setQuery(event.target.value)}/></label><select aria-label="Lọc theo nhóm phân loại" value={category} onChange={(event) => setCategory(event.target.value)}><option value="">Tất cả nhóm phân loại</option>{CLASSES.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></div>}
      {!items.length ? <div className="empty"><div className="empty-icon"><History size={32}/></div><h3>Chưa có lần phân tích nào</h3><p>Kết quả sẽ tự động được lưu sau khi phân tích ảnh thành công.</p><Link className="btn primary" to="/chan-doan">Phân tích ảnh đầu tiên <ArrowRight size={16}/></Link></div> : !filtered.length ? <div className="empty compact"><Search size={30}/><h3>Không có kết quả phù hợp</h3><p>Thử tên ảnh khác hoặc chọn tất cả nhóm phân loại.</p></div> : <div className="table-wrap"><table><thead><tr><th>Thời gian</th><th>Tên ảnh</th><th>Kết quả</th><th>Điểm mô hình</th><th><span className="sr-only">Thao tác</span></th></tr></thead><tbody>{filtered.map((item) => <tr key={item.id}><td className="date-cell">{displayDate(item.date)}</td><td className="filename-cell">{item.filename}</td><td><span className={`result-pill ${item.predicted_class === 'healthy' ? 'is-healthy' : ''}`}>{item.predicted_label || LABELS[item.predicted_class] || item.predicted_class}</span></td><td><b>{(item.confidence * 100).toFixed(1)}%</b></td><td><button className="icon-button" aria-label={`Xóa kết quả ${item.filename}`} title="Xóa kết quả" onClick={() => remove(item.id)}><Trash2 size={17}/></button></td></tr>)}</tbody></table></div>}
    </div><p className="notice"><AlertCircle size={18}/><span>Xóa dữ liệu trình duyệt sẽ xóa lịch sử tại đây. Bạn có thể xuất CSV để giữ một bản sao. Các kết quả là gợi ý từ mô hình, chưa được xác minh bởi chuyên gia.</span></p>
  </section>;
}
