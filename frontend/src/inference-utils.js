import { CLASSES, LABELS } from './classes.js';

export function rgbaToRGB(data) {
  const rgb = new Float32Array(data.length / 4 * 3);
  for (let source = 0, target = 0; source < data.length; source += 4) {
    rgb[target++] = data[source];
    rgb[target++] = data[source + 1];
    rgb[target++] = data[source + 2];
  }
  return rgb; // Raw 0–255. Rescaling is already embedded in the model.
}

export function predictionResult(values) {
  const scores = Array.from(values);
  if (scores.length !== CLASSES.length || scores.some(x => !Number.isFinite(x) || x < 0 || x > 1) || Math.abs(scores.reduce((a,b) => a+b,0) - 1) > 0.01) {
    throw new Error('Đầu ra mô hình không hợp lệ. Vui lòng tải lại trang và thử lại.');
  }
  const index = scores.indexOf(Math.max(...scores));
  return {
    predicted_class: CLASSES[index].key,
    predicted_label: LABELS[CLASSES[index].key],
    confidence: scores[index],
    probabilities: Object.fromEntries(CLASSES.map((item,i) => [item.key,scores[i]])),
    engine: 'browser',
    model_version: 'mobilenetv2-8-onnx-v1'
  };
}
