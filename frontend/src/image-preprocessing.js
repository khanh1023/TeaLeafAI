import { rgbaToRGB } from './inference-utils.js';

async function loadHtmlImage(file) {
  const url = URL.createObjectURL(file);
  const image = document.createElement('img');
  try {
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { image.onload = image.onerror = null; reject(new Error('Đọc ảnh quá lâu. Hãy tải lại trang và chọn lại ảnh.')); }, 20000);
      image.onload = () => { clearTimeout(timer); resolve(); };
      image.onerror = () => { clearTimeout(timer); reject(new Error('Trình duyệt chưa đọc được ảnh này. Hãy thử lưu lại thành JPG hoặc PNG rồi chọn lại.')); };
      image.src = url;
    });
    if (!image.naturalWidth || !image.naturalHeight) throw new Error('Ảnh không có kích thước hợp lệ. Hãy chọn ảnh khác.');
    return { image, release: () => { image.removeAttribute('src'); URL.revokeObjectURL(url); } };
  } catch (error) {
    image.removeAttribute('src');
    URL.revokeObjectURL(url);
    throw error;
  } finally {
    image.onload = image.onerror = null;
  }
}

async function loadImage(file) {
  if (typeof createImageBitmap === 'function') {
    try {
      const image = await createImageBitmap(file, {imageOrientation:'from-image'});
      return { image, release: () => image.close() };
    } catch {
      // Some browsers cannot use the bitmap decoder for otherwise valid JPEGs.
      // The HTML image decoder is the same path used by the visible preview.
    }
  }
  return loadHtmlImage(file);
}

export async function readImagePixels(file) {
  const decoded = await loadImage(file);
  try {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 224;
    let context;
    try { context = canvas.getContext('2d', {willReadFrequently:true}); }
    catch { context = canvas.getContext('2d'); }
    if (!context) throw new Error('Trình duyệt không tạo được vùng xử lý ảnh. Hãy thử Chrome, Edge hoặc Firefox mới nhất.');
    context.fillStyle = '#fff';
    context.fillRect(0,0,224,224);
    context.drawImage(decoded.image,0,0,224,224);
    return rgbaToRGB(context.getImageData(0,0,224,224).data);
  } catch (error) {
    if (error.name === 'SecurityError') {
      throw new Error('Trình duyệt đang chặn đọc điểm ảnh để phân tích. Hãy thử mở website bằng trình duyệt khác.', {cause:error});
    }
    if (/vùng xử lý ảnh/.test(error.message)) throw error;
    throw new Error('Ảnh đã mở được nhưng chưa xử lý được trên trình duyệt này. Hãy tải lại trang hoặc thử trình duyệt khác.', {cause:error});
  } finally { decoded.release(); }
}
