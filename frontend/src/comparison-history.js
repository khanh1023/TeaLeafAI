import { CLASSES } from './classes.js';
import { MODEL_IDS } from './comparison-api.js';

const KEY = 'teaLeafComparisonTrialsV1';
const labels = new Set(CLASSES.map(item => item.key));
const validResult = result => result && MODEL_IDS.includes(result.model_id) && labels.has(result.predicted_class) &&
  Number.isFinite(result.confidence) && result.confidence >= 0 && result.confidence <= 1 && Number.isFinite(result.inference_ms) && result.inference_ms >= 0;
const valid = trial => trial && typeof trial.image_sha256 === 'string' && typeof trial.protocol_id === 'string' &&
  typeof trial.filename === 'string' && (!trial.actual_label || labels.has(trial.actual_label)) && trial.results && typeof trial.results === 'object';

export function readComparisonTrials(protocolId) {
  try {
    const data = JSON.parse(localStorage.getItem(KEY) || '[]');
    return Array.isArray(data) ? data.filter(valid).filter(trial => !protocolId || trial.protocol_id === protocolId).slice(0, 500) : [];
  } catch { return []; }
}

export function saveComparisonTrial(run, filename, actualLabel) {
  const trials = readComparisonTrials();
  const previous = trials.find(trial => trial.image_sha256 === run.image_sha256 && trial.protocol_id === run.protocol_id);
  const results = { ...previous?.results };
  for (const id of MODEL_IDS) if (validResult(run.results[id])) results[id] = run.results[id];
  if (!Object.keys(results).length) return false;
  const trial = { image_sha256: run.image_sha256, protocol_id: run.protocol_id, filename,
    actual_label: labels.has(actualLabel) ? actualLabel : '', results, date: new Date().toISOString() };
  try {
    localStorage.setItem(KEY, JSON.stringify([trial, ...trials.filter(item => item !== previous)].slice(0, 500)));
    return true;
  } catch { return false; }
}

export function clearComparisonTrials() {
  try { localStorage.removeItem(KEY); return true; } catch { return false; }
}

export function comparisonSummary(trials) {
  const shared = trials.filter(trial => MODEL_IDS.every(id => validResult(trial.results?.[id])));
  const labelled = shared.filter(trial => labels.has(trial.actual_label));
  return MODEL_IDS.map(id => {
    const measured = trials.filter(trial => validResult(trial.results?.[id]));
    const correct = labelled.filter(trial => trial.results[id].predicted_class === trial.actual_label).length;
    const warm = shared.filter(trial => !trial.results[id].first_run_after_load);
    return { id, runs: measured.length, shared_runs: shared.length, labelled: labelled.length, correct,
      accuracy: labelled.length ? correct / labelled.length : null,
      mean_inference_ms: shared.length ? shared.reduce((sum, trial) => sum + trial.results[id].inference_ms, 0) / shared.length : null,
      warm_runs: warm.length };
  });
}

export function comparisonCSV(trials) {
  const cell = value => {
    let text = String(value ?? '');
    if (/^[=+@\-\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  };
  const rows = [['Thời gian', 'Ảnh', 'SHA256 ảnh', 'Nhãn người dùng', 'Mô hình', 'Phiên bản', 'SHA256 mô hình',
    'Dự đoán', 'Điểm đầu ra', 'Đúng (chỉ khi có nhãn)', 'Suy luận (ms)', 'Tải/khởi tạo (ms)', 'Lần đầu sau tải']];
  for (const trial of trials) for (const id of MODEL_IDS) {
    const result = trial.results[id];
    if (!validResult(result)) continue;
    rows.push([trial.date, trial.filename, trial.image_sha256, trial.actual_label, result.model_name, result.model_version,
      result.model_sha256, result.predicted_class, result.confidence, trial.actual_label ? result.predicted_class === trial.actual_label : '',
      result.inference_ms, result.load_ms, result.first_run_after_load]);
  }
  return '\ufeff' + rows.map(row => row.map(cell).join(',')).join('\r\n');
}
