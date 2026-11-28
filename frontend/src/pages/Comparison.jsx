import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, Upload, X, LoaderCircle, Play, Download, AlertCircle, CheckCircle2, ShieldCheck, Trash2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { comparisonCatalog, compareImage, MODEL_IDS } from '../comparison-api.js';
import { comparisonSummary, comparisonCSV, readComparisonTrials, saveComparisonTrial, clearComparisonTrials } from '../comparison-history.js';
import { CLASSES, LABELS } from '../classes.js';
import { downloadFile } from '../history.js';
import CameraCapture from '../components/CameraCapture.jsx';

const names = { mobilenetv2: 'MobileNetV2', resnet50: 'ResNet50', vgg16: 'VGG16' };
const number = value => value.toLocaleString('vi-VN', { maximumFractionDigits: 2 });

export default function Comparison() {
  const picker = useRef(null);
  const nativeCamera = useRef(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const closeCamera = useCallback(() => setCameraOpen(false), []);
  const [catalog, setCatalog] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [actual, setActual] = useState('');
  const [results, setResults] = useState({});
  const [lastRun, setLastRun] = useState(null);
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [progress, setProgress] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [trials, setTrials] = useState([]);
  const [confirmClear, setConfirmClear] = useState(false);

  useEffect(() => { let live = true;
    comparisonCatalog().then(data => { if (live) { setCatalog(data); setTrials(readComparisonTrials(data.protocol_id)); } })
      .catch(exception => { if (live) setError(exception.message); });
    return () => { live = false; };
  }, []);
  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function select(next) {
    if (!next || busy) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type) || !next.size || next.size > 8 * 1024 * 1024) {
      setError('Chọn ảnh JPG, PNG hoặc WEBP hợp lệ, dung lượng tối đa 8 MB.'); return;
    }
    setFile(next); setActual(''); setResults({}); setLastRun(null); setError(''); setNotice('');
  }
  async function run(ids) {
    if (!file || busy || !catalog) return;
    setBusy(true); setError(''); setNotice('');
    setResults(previous => Object.fromEntries(Object.entries(previous).filter(([id]) => !ids.includes(id))));
    try {
      const data = await compareImage(file, ids, { onProgress: setProgress,
        onResult: (id, result) => setResults(previous => ({ ...previous, [id]: result })) });
      const saved = saveComparisonTrial(data, file.name, actual);
      setLastRun(data);
      setTrials(readComparisonTrials(catalog.protocol_id));
      setNotice(saved ? 'Đã lưu kết quả trên thiết bị. Chạy lại cùng ảnh sẽ cập nhật, không đếm trùng ảnh.' : 'Chưa lưu được kết quả. Hãy kiểm tra quyền lưu trữ của trình duyệt.');
    } catch (exception) { setError(exception.message || 'Chưa phân tích được ảnh.'); }
    finally { setBusy(false); setProgress(''); }
  }
  async function exportWord() {
    setExporting(true); setError('');
    try {
      const { exportComparisonWord } = await import('../comparison-export.js');
      await exportComparisonWord(trials, catalog);
    } catch { setError('Chưa xuất được Word. Bạn có thể tải CSV và thử lại.'); }
    finally { setExporting(false); }
  }
  const summary = comparisonSummary(trials);
  const totalBytes = catalog?.models.reduce((sum, model) => sum + model.size_bytes, 0);
  return <section className="page comparison-page">
    <div className="page-title"><span className="eyebrow">CÙNG MỘT ẢNH · BA GÓC NHÌN</span><h1>Thử ba mô hình phân loại</h1>
      <p>Chọn MobileNetV2, ResNet50 hoặc VGG16 để phân tích cùng một ảnh lá chè, đối chiếu kết quả rồi tải dữ liệu vào báo cáo.</p>
    </div>
    <div className="comparison-intro"><ShieldCheck size={22}/><p>Ảnh được xử lý trên thiết bị. Mỗi mô hình chỉ tải khi bạn bấm chạy; cả ba khoảng {totalBytes ? `${number(totalBytes / 1e6)} MB` : '… MB'}.
      <span>Các mô hình ở đây dùng chung quy trình huấn luyện cho thí nghiệm so sánh; kết quả có thể khác <Link to="/chan-doan">trang Phân tích</Link>.</span>
      {catalog?.dataset_version && <span>Bộ dữ liệu của thí nghiệm này: {catalog.dataset_version}{catalog.dataset_images ? ` · ${number(catalog.dataset_images)} ảnh` : ''}. Trang Phân tích có thể dùng phiên bản được cập nhật riêng.</span>}</p></div>
    <div className="comparison-upload panel">
      <div className="comparison-picture">{preview ? <><img src={preview} alt={`Ảnh đối chiếu: ${file.name}`}/><button className="remove" aria-label="Bỏ ảnh đối chiếu" disabled={busy} onClick={() => { setFile(null); setResults({}); setLastRun(null); setActual(''); }}><X size={18}/></button></> :
        <button className="dropzone" disabled={busy} onClick={() => picker.current?.click()} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); select(event.dataTransfer.files?.[0]); }}><Upload size={28}/><b>Chọn hoặc kéo thả ảnh lá chè</b><span>JPG, PNG, WEBP · tối đa 8 MB</span></button>}</div>
      <div className="comparison-controls"><h2>Ảnh dùng chung cho cả ba mô hình</h2><p>Chọn ảnh rõ nét. Bạn có thể chạy từng mô hình trước, hoặc đối chiếu cả ba trong một lượt.</p>
        {file && <p className="comparison-filename">{file.name} · {number(file.size / 1e6)} MB</p>}
        <div className="button-group"><button className="btn outline" disabled={busy} onClick={() => picker.current?.click()}><Upload size={18}/>{file ? 'Đổi ảnh' : 'Chọn ảnh'}</button><button className="btn outline" disabled={busy} onClick={() => setCameraOpen(true)}><Camera size={18}/>Chụp bằng camera</button></div>
        <label className="comparison-label" htmlFor="actual-label">Nhãn thực tế, nếu bạn đã biết <span>(không bắt buộc)</span></label>
        <select id="actual-label" value={actual} disabled={busy} onChange={event => {
          const label = event.target.value; setActual(label);
          if (lastRun && file && catalog) {
            if (saveComparisonTrial(lastRun, file.name, label)) setTrials(readComparisonTrials(catalog.protocol_id));
            else setError('Chưa lưu được nhãn. Bạn có thể chạy lại và thử lưu kết quả.');
          }
        }}><option value="">Chưa xác định nhãn</option>{CLASSES.map(item => <option key={item.key} value={item.key}>{item.label}</option>)}</select>
        <p className="comparison-helper">Nhãn do bạn cung cấp được dùng để tính tỷ lệ đúng. Nếu chưa biết, hãy để “Chưa xác định nhãn”.</p>
        <button className="btn primary full" disabled={!file || !catalog || busy} onClick={() => run(MODEL_IDS)}>{busy ? <LoaderCircle size={19} className="spin"/> : <Play size={18}/>}Chạy cả ba mô hình</button>
      </div>
    </div>
    <input type="file" hidden ref={picker} accept="image/jpeg,image/png,image/webp" aria-label="Chọn ảnh đối chiếu ba mô hình" onChange={event => { select(event.target.files?.[0]); event.target.value = ''; }}/>
    <input type="file" hidden ref={nativeCamera} accept="image/jpeg" capture="environment" onChange={event => { select(event.target.files?.[0]); event.target.value = ''; }}/>
    {busy && <p role="status" className="comparison-progress"><LoaderCircle className="spin" size={19}/>{progress}<span>Vui lòng giữ trang mở. ResNet50 và VGG16 cần nhiều bộ nhớ hơn.</span></p>}
    {error && <p role="alert" className="error"><AlertCircle size={19}/>{error}</p>}
    {!catalog && <button className="btn outline" disabled={busy} onClick={async () => {
      setError('');
      try { const data = await comparisonCatalog(); setCatalog(data); setTrials(readComparisonTrials(data.protocol_id)); }
      catch (exception) { setError(exception.message); }
    }}>Tải lại danh sách mô hình</button>}
    <div className="comparison-models">{MODEL_IDS.map(id => {
      const model = catalog?.models.find(item => item.id === id); const result = results[id];
      return <article className={`panel comparison-model ${id === 'mobilenetv2' ? 'comparison-choice' : ''}`} key={id}>
        <div className="comparison-model-heading"><h2>{names[id]}</h2>{id === 'mobilenetv2' && <span>Lựa chọn của đề tài</span>}</div>
        <p className="comparison-model-size">{model ? `${number(model.size_bytes / 1e6)} MB · ONNX` : 'Đang tải thông tin…'}</p>
        {result?.error ? <p className="error" role="alert">{result.error}</p> : result ? <>
          <div className="comparison-prediction"><span>Nhóm có điểm cao nhất</span><h3>{result.predicted_label}</h3><strong>{number(result.confidence * 100)}%</strong><small>Điểm đầu ra mô hình</small></div>
          <dl className="comparison-times"><div><dt>Suy luận</dt><dd>{number(result.inference_ms)} ms</dd></div><div><dt>Tải và khởi tạo</dt><dd>{number(result.load_ms)} ms</dd></div></dl>
          {actual && <p className={`comparison-correct ${actual === result.predicted_class ? 'is-correct' : ''}`}>{actual === result.predicted_class ? <CheckCircle2 size={17}/> : <AlertCircle size={17}/>} {actual === result.predicted_class ? 'Khớp nhãn bạn cung cấp' : `Khác nhãn bạn cung cấp: ${LABELS[actual]}`}</p>}
          <details className="comparison-scores"><summary>Xem điểm của 8 nhóm</summary>{CLASSES.map(item => <div className="prob" key={item.key}><div><span>{item.label}</span><b>{number(result.probabilities[item.key] * 100)}%</b></div><div className="bar"><i style={{ width: `${result.probabilities[item.key] * 100}%` }}/></div></div>)}</details>
        </> : <p className="comparison-placeholder">Chọn ảnh và bấm chạy để xem dự đoán, điểm từng nhóm và thời gian suy luận.</p>}
        <button className={`btn ${id === 'mobilenetv2' ? 'primary' : 'outline'} full`} disabled={!file || !model || busy} onClick={() => run([id])}><Play size={17}/>Chạy {names[id]}</button>
      </article>;
    })}</div>
    {notice && <p className="comparison-storage-note" role="status">{notice}</p>}
    <section className="panel comparison-history"><div className="comparison-history-head"><div><h2>Kết quả thử trên thiết bị của bạn</h2><p>{trials.length} ảnh đã lưu · {summary[0].shared_runs} ảnh có đủ kết quả của ba mô hình</p></div><div className="button-group">
      <button className="btn pale" disabled={!trials.length || busy || exporting} onClick={exportWord}><Download size={18}/>{exporting ? 'Đang xuất…' : 'Tải Word'}</button>
      <button className="btn outline" disabled={!trials.length || busy} onClick={() => downloadFile('tea-leaf-ai-so-sanh.csv', comparisonCSV(trials), 'text/csv;charset=utf-8')}><Download size={18}/>Tải CSV</button>
    </div></div>
      <p className="comparison-helper">Accuracy chỉ tính trên cùng những ảnh có nhãn do bạn cung cấp và đủ kết quả của cả ba mô hình. Điểm đầu ra của một ảnh không phải Accuracy.</p>
      <p className="comparison-table-hint">Vuốt ngang bảng để xem đủ ba mô hình.</p>
      <div className="comparison-table-scroll" tabIndex="0" aria-label="Bảng kết quả thử ba mô hình"><table className="comparison-table"><thead><tr><th scope="col">Tiêu chí</th>{['resnet50', 'vgg16', 'mobilenetv2'].map(id => <th key={id} scope="col" className={id === 'mobilenetv2' ? 'chosen-column' : ''}>{names[id]}{id === 'mobilenetv2' && <small>Lựa chọn của đề tài</small>}</th>)}</tr></thead><tbody>
        {[['Ảnh chung đã gán nhãn', row => row.labelled], ['Accuracy trên các ảnh này', row => row.accuracy === null ? 'Chưa có dữ liệu' : `${number(row.accuracy * 100)}% (${row.correct}/${row.labelled})`],
          ['Suy luận trung bình trên ảnh chung', row => row.mean_inference_ms === null ? 'Chưa có dữ liệu' : `${number(row.mean_inference_ms)} ms`], ['Kết quả riêng đã lưu', row => row.runs]].map(([label, value]) => <tr key={label}><th scope="row">{label}</th>{['resnet50', 'vgg16', 'mobilenetv2'].map(id => <td key={id} className={id === 'mobilenetv2' ? 'chosen-column' : ''}>{value(summary.find(row => row.id === id))}</td>)}</tr>)}
      </tbody></table></div>
      <p className="comparison-helper">Thời gian suy luận không gồm đọc ảnh, tải và khởi tạo mô hình; lần chạy đầu có thể chậm hơn. Các kết quả này phụ thuộc thiết bị và ảnh bạn đã thử, chưa phải kiểm định ngoài thực địa.</p>
      {trials.length > 0 && <div className="comparison-clear">{confirmClear ? <><span>Xóa toàn bộ lịch sử đối chiếu trên thiết bị này?</span><button className="btn danger-outline" disabled={busy} onClick={() => { if (clearComparisonTrials()) { setTrials([]); setConfirmClear(false); } else setError('Chưa xóa được lịch sử.'); }}>Xóa lịch sử đối chiếu</button><button className="btn outline" onClick={() => setConfirmClear(false)}>Giữ lại</button></> : <button className="text-link" disabled={busy} onClick={() => setConfirmClear(true)}><Trash2 size={16}/>Xóa lịch sử đối chiếu</button>}</div>}
    </section>
    <p className="notice"><AlertCircle size={18}/><span>Cả ba mô hình đều phân loại vào 8 nhóm đã học, kể cả khi ảnh không phải lá chè. Kết quả dùng để tham khảo và đối chiếu, không thay thế chẩn đoán chuyên môn.</span></p>
    {cameraOpen && <CameraCapture onCapture={select} onClose={closeCamera} onNativeCamera={() => { closeCamera(); nativeCamera.current?.click(); }}/>}
  </section>;
}
