import { useEffect, useRef, useState } from 'react';
import { Camera, SwitchCamera, X, LoaderCircle, AlertCircle } from 'lucide-react';

export default function CameraCapture({ onCapture, onClose, onNativeCamera }) {
  const dialog = useRef(null);
  const video = useRef(null);
  const activeStream = useRef(null);
  const [facing, setFacing] = useState('environment');
  const [attempt, setAttempt] = useState(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const [capturing, setCapturing] = useState(false);

  useEffect(() => {
    dialog.current.showModal();
    return () => dialog.current?.close();
  }, []);

  useEffect(() => {
    let disposed = false;
    let stream;
    setReady(false);
    setError('');
    async function start() {
      try {
        if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
          throw new Error('Trình duyệt chưa hỗ trợ camera trực tiếp. Hãy mở website bằng HTTPS hoặc dùng camera của điện thoại bên dưới.');
        }
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } }
        });
        if (disposed) { stream.getTracks().forEach(track => track.stop()); return; }
        activeStream.current = stream;
        video.current.srcObject = stream;
        await video.current.play();
        if (!disposed) setReady(true);
      } catch (exception) {
        stream?.getTracks().forEach(track => track.stop());
        if (disposed) return;
        const messages = {
          NotAllowedError: 'Bạn chưa cho phép camera. Hãy cho phép trong cài đặt trang rồi thử lại, hoặc dùng camera của điện thoại bên dưới.',
          NotFoundError: 'Không tìm thấy camera trên thiết bị này. Bạn có thể chọn ảnh có sẵn.',
          NotReadableError: 'Camera đang bận hoặc chưa mở được. Hãy đóng ứng dụng đang dùng camera và thử lại.',
          OverconstrainedError: 'Camera này chưa đáp ứng được yêu cầu. Hãy dùng camera của điện thoại hoặc chọn ảnh có sẵn.'
        };
        setError(messages[exception.name] || exception.message || 'Không mở được camera. Hãy thử lại hoặc chọn ảnh có sẵn.');
      }
    }
    start();
    return () => {
      disposed = true;
      stream?.getTracks().forEach(track => track.stop());
      if (activeStream.current === stream) activeStream.current = null;
      if (video.current) video.current.srcObject = null;
    };
  }, [facing, attempt]);

  useEffect(() => {
    const hide = () => { if (document.hidden) onClose(); };
    document.addEventListener('visibilitychange', hide);
    return () => document.removeEventListener('visibilitychange', hide);
  }, [onClose]);

  async function capture() {
    if (!ready || capturing) return;
    setCapturing(true);
    try {
      const source = video.current;
      if (!source.videoWidth || !source.videoHeight) throw new Error('Camera chưa có hình. Vui lòng chờ một chút rồi chụp lại.');
      const scale = Math.min(1, 1600 / Math.max(source.videoWidth, source.videoHeight));
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(source.videoWidth * scale);
      canvas.height = Math.round(source.videoHeight * scale);
      canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
      if (!blob) throw new Error('Chưa tạo được ảnh. Hãy chụp lại.');
      if (!video.current) return;
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      onCapture(new File([blob], `la-che-${stamp}.jpg`, { type: 'image/jpeg' }));
      onClose();
    } catch (exception) {
      setError(exception.message || 'Không chụp được ảnh. Vui lòng thử lại.');
    } finally { setCapturing(false); }
  }

  return <dialog ref={dialog} className="camera-dialog" aria-labelledby="camera-title" onCancel={event => { event.preventDefault(); onClose(); }}>
    <div className="camera-header"><h2 id="camera-title"><Camera size={22}/> Chụp lá chè tại vườn</h2><button type="button" className="icon-button" aria-label="Đóng camera" onClick={onClose}><X/></button></div>
    <p className="camera-tip">Đưa lá vào khung hình, giữ máy ổn định và chụp dưới ánh sáng tự nhiên.</p>
    <div className="camera-view"><video ref={video} autoPlay playsInline muted aria-label="Hình ảnh trực tiếp từ camera"/>{!ready && !error && <div className="camera-loading" role="status"><LoaderCircle className="spin"/> Đang mở camera…</div>}</div>
    {error && <p className="error" role="alert"><AlertCircle size={18}/><span>{error}</span></p>}
    <div className="camera-actions"><button type="button" className="btn outline" disabled={!ready || capturing} onClick={() => setFacing(value => value === 'environment' ? 'user' : 'environment')}><SwitchCamera size={18}/> Đổi camera</button><button type="button" className="btn primary" disabled={!ready || capturing} onClick={capture}><Camera size={19}/>{capturing ? 'Đang lấy ảnh…' : 'Chụp ảnh'}</button>{error && <button type="button" className="btn outline" onClick={() => setAttempt(value => value + 1)}>Thử lại camera</button>}</div>
    <button type="button" className="btn pale full" disabled={capturing} onClick={() => { activeStream.current?.getTracks().forEach(track => track.stop()); dialog.current.close(); onNativeCamera(); }}>Dùng camera của điện thoại</button>
    <p className="camera-footnote">Camera chỉ bật khi bạn mở cửa sổ này và sẽ tắt khi đóng hoặc chụp xong. Ảnh chụp sẽ hiện ở phần ảnh đầu vào để bạn xem lại trước khi phân tích. Nếu thiết bị không hỗ trợ camera, hãy đóng cửa sổ và chọn ảnh có sẵn.</p>
  </dialog>;
}
