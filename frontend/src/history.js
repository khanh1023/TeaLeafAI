const KEY = 'teaLeafHistory';
const isRecord = (x) => x && typeof x === 'object' && typeof x.predicted_class === 'string' && Number.isFinite(x.confidence) && typeof x.filename === 'string';

export function readHistory() {
  try {
    const data = JSON.parse(localStorage.getItem(KEY) || '[]');
    return Array.isArray(data) ? data.filter(isRecord).slice(0, 100) : [];
  } catch {
    return [];
  }
}

export function saveHistory(items) {
  try {
    localStorage.setItem(KEY, JSON.stringify(items.filter(isRecord).slice(0, 100)));
    return true;
  } catch {
    return false;
  }
}

export function addHistory(result, filename) {
  return saveHistory([{
    ...result,
    id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    date: new Date().toISOString(),
    filename
  }, ...readHistory()]);
}

export function clearHistory() {
  try {
    localStorage.removeItem(KEY);
    return true;
  } catch {
    return false;
  }
}

export function downloadFile(filename, text, type = 'application/json;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function exportHistory(items) {
  const cell = (value) => {
    let text = String(value ?? '');
    // Spreadsheet formulas must remain plain text, including user-supplied filenames.
    if (/^[=+@\-\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  };
  const rows = [['Thời gian', 'Tên ảnh', 'Nhãn', 'Kết quả', 'Điểm mô hình (%)'], ...items.map((x) => [
    x.date, x.filename, x.predicted_class, x.predicted_label || x.predicted_class, (x.confidence * 100).toFixed(2)
  ])];
  downloadFile('tea-leaf-ai-lich-su.csv', '\ufeff' + rows.map((row) => row.map(cell).join(',')).join('\r\n'), 'text/csv;charset=utf-8');
}
