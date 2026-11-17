import { useCallback, useEffect, useRef, useState } from 'react';
import { Upload, Image as ImageIcon, LoaderCircle, AlertCircle, ScanSearch, X, ShieldCheck, Download, ArrowRight, CheckCircle2, Camera } from 'lucide-react';
import { Link } from 'react-router-dom';
import { diagnose, ENGINE } from '../api';
import { CLASSES, LABELS } from '../classes';
import { addHistory, downloadFile } from '../history';
import CameraCapture from '../components/CameraCapture';

const remoteApi = ENGINE === 'api';

export default function Diagnosis() {
  const input = useRef(null);
  const nativeCamera = useRef(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const closeCamera = useCallback(() => setCameraOpen(false), []);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');
  const [storageWarning, setStorageWarning] = useState('');
  const [progress, setProgress] = useState('');
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function select(next) {
    if (!next || busy) return;
    setError('');
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type)) {
      setError('Vui lòng chọn ảnh JPG, PNG hoặc WEBP.');
      if (input.current) input.current.value = '';
      return;
    }
    if (next.size === 0 || next.size > 8 * 1024 * 1024) {
      setError(next.size === 0 ? 'Tệp ảnh đang trống. Hãy chọn một ảnh khác.' : 'Ảnh quá lớn. Vui lòng chọn ảnh dưới 8 MB.');
      if (input.current) input.current.value = '';
      return;
    }
    setFile(next);
    setResult(null);
    setStorageWarning('');
  }

  function clear() {
    if (busy) return;
    setFile(null); setResult(null); setError(''); setStorageWarning('');
    if (input.current) input.current.value = '';
  }

  function onProgress(value) {
    if (typeof value === 'string') setProgress(value);
    else if (typeof value === 'number') setProgress(`Đang tải mô hình: ${Math.round(value <= 1 ? value * 100 : value)}%`);
    else if (value?.message) setProgress(value.message);
  }

  async function run() {
    if (!file || busy) return;
    setBusy(true); setError(''); setResult(null); setStorageWarning('');
    setProgress(remoteApi ? 'Đang gửi ảnh đến máy chủ AI…' : 'Đang chuẩn bị mô hình AI trên thiết bị…');
    try {
      const data = await diagnose(file, { onProgress });
      setResult(data);
      if (!addHistory(data, file.name)) setStorageWarning('Đã có kết quả nhưng trình duyệt không cho lưu lịch sử. Bạn vẫn có thể tải kết quả về.');
    } catch (exception) {
      setError(exception.message || 'Chưa thể phân tích ảnh. Vui lòng thử lại.');
    } finally {
      setBusy(false); setProgress('');
    }
  }

  const probabilities = result ? [...CLASSES].map((item) => ({ ...item, value: result.probabilities?.[item.key] || 0 })).sort((a, b) => b.value - a.value) : [];
  return <section className="page">
    <div className="page-title"><span className="eyebrow">TỪ MỘT CHIẾC LÁ, HIỂU THÊM VỀ VƯỜN CHÈ</span><h1>Phân tích ảnh lá chè</h1><p>Chọn ảnh có sẵn hoặc chụp bằng camera. AI sẽ đối chiếu với 8 nhóm trong dữ liệu huấn luyện.</p></div>
    <div className="diagnosis-layout">
      <div className="panel upload-panel">
        <div className="panel-heading"><span className="step-badge">01</span><h2>Ảnh lá chè của bạn</h2></div>
        <div className={`upload-area ${dragging ? 'is-dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); select(event.dataTransfer.files?.[0]); }}>
          {!preview ? <button className="dropzone" disabled={busy} onClick={() => input.current?.click()}><span className="upload-symbol"><Upload size={30}/></span><b>Kéo thả ảnh vào đây</b><span>hoặc <u>chọn ảnh từ thiết bị</u></span><small>JPG, PNG, WEBP · tối đa 8 MB</small></button> : <div className="preview"><img src={preview} alt={`Ảnh lá chè đã chọn: ${file?.name}`}/><button className="remove" type="button" aria-label="Bỏ ảnh đã chọn" disabled={busy} onClick={clear}><X size={18}/></button></div>}
        </div>
        <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" aria-label="Chọn ảnh lá chè" hidden onChange={(event) => select(event.target.files?.[0])}/>
        <input ref={nativeCamera} type="file" accept="image/jpeg" capture="environment" aria-label="Chụp ảnh bằng camera điện thoại" hidden onChange={event => { select(event.target.files?.[0]); event.target.value = ''; }}/>
        <div className="photo-source-actions"><button type="button" className="btn outline" disabled={busy} onClick={() => setCameraOpen(true)}><Camera size={20}/> Chụp bằng camera</button>{file && <button type="button" className="btn pale" disabled={busy} onClick={() => input.current?.click()}><Upload size={18}/> Chọn ảnh khác</button>}</div>
        <p className="camera-source-hint">Đi thực địa? Chạm “Chụp bằng camera”, cho phép camera rồi chụp lá rõ nét. Bạn sẽ được xem lại ảnh trước khi phân tích.</p>
        {file && <div className="file-caption"><ImageIcon size={16}/><span>{file.name}</span><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></div>}
        <button className="btn primary full" disabled={!file || busy} onClick={run}>{busy ? <><LoaderCircle className="spin" size={19}/> Đang phân tích…</> : <><ScanSearch size={19}/> Phân tích ảnh <ArrowRight size={17}/></>}</button>
        {busy && <p className="progress-message" role="status">{progress}<small>Lần đầu cần tải mô hình. Vui lòng giữ trang mở.</small></p>}
        {error && <p className="error" role="alert"><AlertCircle size={18}/><span>{error}</span></p>}
        <div className="privacy-note"><ShieldCheck size={19}/><p>{remoteApi ? 'Ảnh được gửi đến API đã cấu hình để phân tích.' : 'Ảnh được phân tích ngay trên thiết bị, không tải lên máy chủ.'}<span>Lịch sử chỉ lưu tên ảnh và kết quả trong trình duyệt này.</span></p></div>
        <Link className="text-link" to="/huong-dan">Cách chụp ảnh để phân tích tốt hơn <ArrowRight size={14}/></Link>
      </div>
      <div className="panel result-panel" aria-live="polite" aria-busy={busy}>
        <div className="panel-heading"><span className="step-badge">02</span><h2>Kết quả phân tích</h2>{result && <CheckCircle2 className="result-check" size={20}/>}</div>
        {!result ? <div className="empty result-empty"><div className="empty-icon">{busy ? <LoaderCircle className="spin" size={34}/> : <ScanSearch size={34}/>}</div><h3>{busy ? 'AI đang xem ảnh của bạn' : 'Kết quả sẽ xuất hiện ở đây'}</h3><p>{busy ? 'Mô hình đang tìm các đặc điểm hình ảnh của lá chè.' : 'Tải ảnh lên và nhấn “Phân tích ảnh” để bắt đầu.'}</p><div className="empty-chips"><span>7 nhóm bệnh</span><span>1 nhóm lá khỏe</span></div></div> : <>
          <div className={`result-main ${result.confidence < 0.7 ? 'uncertain' : ''}`}><span>Nhóm có điểm cao nhất</span><h3>{result.predicted_label || LABELS[result.predicted_class] || result.predicted_class}</h3><strong>{(result.confidence * 100).toFixed(1)}<small>%</small></strong><span>Điểm đầu ra mô hình</span></div>
          {result.confidence < 0.7 && <div className="warning-note"><AlertCircle size={17}/><span>Kết quả còn chưa rõ. Hãy chụp lại ở ánh sáng tốt và tham khảo chuyên gia.</span></div>}
          <h3 className="prob-title">So sánh 8 nhóm phân loại</h3>
          {probabilities.map(({ key, label, value }) => <div className={`prob ${key === result.predicted_class ? 'top-prob' : ''}`} key={key}><div><span>{label}</span><b>{(value * 100).toFixed(1)}%</b></div><div className="bar"><i style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}/></div></div>)}
          <button className="btn pale full" onClick={() => downloadFile('tea-leaf-ai-ket-qua.json', JSON.stringify({ filename: file.name, date: new Date().toISOString(), ...result }, null, 2))}><Download size={17}/> Tải kết quả JSON</button>
          {storageWarning && <p className="warning-note" role="status">{storageWarning}</p>}
        </>}
      </div>
    </div>
    {cameraOpen && <CameraCapture onCapture={select} onClose={closeCamera} onNativeCamera={() => { closeCamera(); nativeCamera.current?.click(); }}/>}
    <p className="notice"><AlertCircle size={18}/><span>Đây là công cụ hỗ trợ tham khảo, chưa phải kết luận bệnh. Điểm mô hình không phải xác suất bệnh đã được kiểm định. Ảnh không phải lá chè vẫn có thể bị gán vào một trong 8 nhóm; không dùng kết quả này để tự quyết định phun thuốc.</span></p>
  </section>;
}
