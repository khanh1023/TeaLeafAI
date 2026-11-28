import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Laptop, RefreshCw, CheckCircle2, CircleAlert, FileJson, Github, ShieldCheck, ArrowUpRight } from 'lucide-react';
import { health, ENGINE, BASE } from '../api';
import { CLASSES } from '../classes';

export default function Admin() {
  const [info, setInfo] = useState(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState('');
  async function check() {
    setBusy(true); setError('');
    try { setInfo(await health()); }
    catch (exception) { setInfo(null); setError(exception.message || 'Chưa kiểm tra được hệ thống.'); }
    finally { setBusy(false); }
  }
  useEffect(() => { check(); }, []);
  return <section className="page"><div className="page-title"><span className="eyebrow">THÔNG TIN CÔNG KHAI VỀ ỨNG DỤNG</span><h1>Hệ thống Tea Leaf AI</h1><p>Kiểm tra chế độ chạy, mô hình và các nhãn phân loại. Trang này cung cấp thông tin hệ thống, không có quyền quản trị dữ liệu người khác.</p></div>
    <div className={`system-status ${error ? 'has-error' : ''}`}><div className="system-status-icon">{busy ? <RefreshCw className="spin"/> : error ? <CircleAlert/> : <CheckCircle2/>}</div><div><span className="eyebrow">TRẠNG THÁI</span><h2>{busy ? 'Đang kiểm tra hệ thống…' : error ? 'Chưa sẵn sàng' : 'Sẵn sàng để phân tích'}</h2><p>{error || (ENGINE === 'api' ? 'Ứng dụng đang kết nối với máy chủ API.' : 'AI chạy trực tiếp trong trình duyệt, không cần thuê máy chủ.')}</p></div><button className="btn outline" disabled={busy} onClick={check}><RefreshCw size={17} className={busy ? 'spin' : ''}/> Kiểm tra lại</button></div>
    <div className="admin-grid"><article className="panel"><Laptop size={27}/><h2>Chế độ {ENGINE === 'api' ? 'API' : 'trình duyệt'}</h2><p>{ENGINE === 'api' ? 'Ảnh được gửi tới API đã cấu hình để mô hình phân tích.' : 'Bản mặc định chạy mô hình trên thiết bị của bạn. Không cần tài khoản, API trả phí hay môi trường Python.'}</p><dl className="system-details"><div><dt>Bộ xử lý</dt><dd>{ENGINE === 'api' ? 'FastAPI + TensorFlow' : 'ONNX Runtime Web'}</dd></div><div><dt>Mô hình</dt><dd>MobileNetV2 · 8 lớp</dd></div><div><dt>Đầu vào</dt><dd>Ảnh RGB · 224 × 224</dd></div><div><dt>Đã tải vào bộ nhớ</dt><dd>{info?.model_loaded ? 'Đã tải' : 'Tải khi phân tích ảnh'}</dd></div>{ENGINE === 'api' && <div><dt>API</dt><dd className="break-word">{BASE}</dd></div>}</dl></article>
      {info?.dataset_version && <article className="panel"><FileJson size={27}/><h2>Dữ liệu của mô hình hiện tại</h2>
        <dl className="system-details"><div><dt>Phiên bản dữ liệu</dt><dd className="break-word">{info.dataset_version}</dd></div>
          <div><dt>Tổng số ảnh</dt><dd>{info.dataset_images?.toLocaleString('vi-VN')}</dd></div>
          {info.split_totals && <div><dt>Train / Validation / Test</dt><dd>{['train', 'val', 'test'].map(key => info.split_totals[key]?.toLocaleString('vi-VN')).join(' / ')}</dd></div>}
          {Number.isFinite(info.browser_test_accuracy) && <div><dt>Accuracy test trên trình duyệt</dt><dd>{(info.browser_test_accuracy * 100).toLocaleString('vi-VN', {maximumFractionDigits: 2})}% · {info.browser_test_images?.toLocaleString('vi-VN')} ảnh</dd></div>}
        </dl><p>Đánh giá trên tập test nội bộ. Kết quả có thể thay đổi với ảnh thực địa mới; điểm dự đoán của một ảnh không phải Accuracy của mô hình.</p>
      </article>}
      <article className="panel"><FileJson size={27}/><h2>8 nhãn được sử dụng</h2><p>Tên và thứ tự nhãn thống nhất giữa mô hình, mã huấn luyện và giao diện.</p><ol className="class-list">{CLASSES.map((item) => <li key={item.key}><span>{item.label}</span><code>{item.key}</code></li>)}</ol></article>
      <article className="panel"><ShieldCheck size={27}/><h2>Giới hạn cần biết</h2><p>Mô hình chưa có lớp loại trừ ảnh không phải lá chè. Điểm dự đoán chưa được hiệu chuẩn thành xác suất bệnh thực tế.</p><p>Không tuyên bố độ chính xác cho ảnh vườn chè mới. Cần kiểm định bằng dữ liệu độc lập và chuyên gia nông nghiệp trước khi ứng dụng vào xử lý cây.</p><Link className="text-link" to="/huong-dan">Hướng dẫn đọc kết quả <ArrowUpRight size={16}/></Link></article>
      <article className="panel"><Github size={27}/><h2>Mã nguồn và tự triển khai</h2><p>Mã nguồn có quy trình chạy local, huấn luyện Python và bản web tĩnh. Có thể triển khai bản web bằng dịch vụ lưu trữ tĩnh miễn phí theo hướng dẫn trong README.</p><a className="btn pale" href="https://github.com/khanh1023/TeaLeafAI" target="_blank" rel="noreferrer">Xem mã nguồn GitHub <ArrowUpRight size={16}/></a><p><a className="text-link" href="https://github.com/khanh1023/TeaLeafAI/releases/latest" target="_blank" rel="noreferrer">Tải bản chạy sẵn cho máy tính <ArrowUpRight size={16}/></a></p><p className="small-text">Không cần huấn luyện lại để chạy thử mô hình được đóng gói sẵn.</p></article>
    </div>
  </section>;
}
