"""Shared evidence validation and final-test parity for versioned ONNX exports."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind_checkpoint_evaluation(report, checkpoint_sha):
    """A requested evaluation must explicitly identify the checkpoint it measures."""
    if not isinstance(report, dict):
        raise ValueError('Evaluation must be a JSON object bound to a checkpoint.')
    if report.get('model_sha256') != checkpoint_sha:
        raise ValueError('Checkpoint checksum differs from its evaluation.')
    return report


def validate_evaluation(report, classes, test_count=None):
    """Reject stale counts, class ordering, or metrics inconsistent with the CM."""
    if 'classes' in report and report['classes'] != classes:
        raise ValueError('Evaluation class order differs from the exported class order.')
    matrix = report.get('confusion_matrix')
    if not isinstance(matrix, list) or len(matrix) != len(classes):
        raise ValueError('Evaluation requires a square confusion matrix.')
    if any(not isinstance(row, list) or len(row) != len(classes) for row in matrix):
        raise ValueError('Evaluation requires a square confusion matrix.')
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for row in matrix for n in row):
        raise ValueError('Confusion-matrix counts must be nonnegative integers.')
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(classes)))
    if not total or report.get('test_images') != total or (test_count is not None and total != test_count):
        raise ValueError('Evaluation test count differs from the manifest or confusion matrix.')
    if not math.isclose(report['accuracy'], correct / total, abs_tol=1e-12):
        raise ValueError('Evaluation Accuracy differs from the confusion matrix.')
    if 'correct' in report and report['correct'] != correct:
        raise ValueError('Evaluation correct count differs from the confusion matrix.')
    f1 = []
    for i in range(len(classes)):
        predicted = sum(row[i] for row in matrix)
        support = sum(matrix[i])
        denominator = predicted + support
        f1.append(2 * matrix[i][i] / denominator if denominator else 0.)
    if 'macro_f1' in report and not math.isclose(report['macro_f1'], sum(f1) / len(f1), abs_tol=1e-10):
        raise ValueError('Evaluation macro-F1 differs from the confusion matrix.')
    return dict(test_images=total, correct=correct, accuracy=correct / total,
                macro_f1=sum(f1) / len(f1))


def load_dataset(data, classes, manifest_path=None, expected_sha=None):
    data = Path(data).resolve()
    manifest_path = Path(manifest_path) if manifest_path else data / 'manifest.json'
    digest = sha_file(manifest_path)
    if expected_sha and digest != expected_sha:
        raise ValueError('Dataset manifest differs from the checkpoint training protocol.')
    if (data / 'classes.json').exists() and read_json(data / 'classes.json') != classes:
        raise ValueError('Dataset class order differs from the exported class order.')
    records = read_json(manifest_path)
    seen_paths = set()
    identities = {key: defaultdict(set) for key in ['group', 'sha256', 'raw_sha256', 'pixel_sha256']}
    for record in records:
        if record['label'] not in classes or record['split'] not in ['train', 'val', 'test']:
            raise ValueError('Unknown dataset class or split.')
        path = (data / record['path']).resolve()
        if not path.is_relative_to(data) or record['path'] in seen_paths:
            raise ValueError('Unsafe or repeated dataset path.')
        seen_paths.add(record['path'])
        for field, lookup in identities.items():
            lookup[record[field]].add(record['split'])
    if any(len(splits) != 1 for lookup in identities.values() for splits in lookup.values()):
        raise ValueError('Cross-split duplication in export dataset.')
    totals = Counter(record['split'] for record in records)
    if set(totals) != {'train', 'val', 'test'}:
        raise ValueError('Export dataset must contain train, validation and test.')
    if any(not any(r['label'] == label and r['split'] == 'test' for r in records) for label in classes):
        raise ValueError('Every exported class needs held-out test images.')
    summary = read_json(data / 'split_summary.json') if (data / 'split_summary.json').exists() else {}
    if summary.get('total_images', len(records)) != len(records):
        raise ValueError('Dataset summary total differs from the manifest.')
    if summary.get('manifest_sha256', digest) != digest:
        raise ValueError('Dataset summary checksum differs from the manifest.')
    for split, counts in summary.get('split_counts', {}).items():
        if dict(Counter(r['label'] for r in records if r['split'] == split)) != counts:
            raise ValueError('Dataset summary class counts differ from the manifest.')
    return dict(data=data, manifest_path=manifest_path, records=records, summary=summary,
                dataset_manifest_sha256=digest, dataset_images=len(records), split_totals=dict(totals))


def evidence_metadata(evidence, report=None):
    result = dict(dataset_manifest_sha256=evidence['dataset_manifest_sha256'],
                  dataset_version=evidence['summary'].get('version', 'dataset-' + evidence['dataset_manifest_sha256'][:12]),
                  dataset_images=evidence['dataset_images'],
                  # Historic manifest meaning is total source split images, not only train.
                  training_images=evidence['dataset_images'],
                  train_images=evidence['split_totals']['train'],
                  validation_images=evidence['split_totals']['val'],
                  test_images=evidence['split_totals']['test'],
                  split_totals=evidence['split_totals'],
                  evaluation_scope='Internal held-out group split; not independently field-validated.')
    if report is not None:
        metrics = validate_evaluation(report, read_json(evidence['data'] / 'classes.json'), result['test_images'])
        if report.get('dataset_manifest_sha256', result['dataset_manifest_sha256']) != result['dataset_manifest_sha256']:
            raise ValueError('Evaluation was measured on a different dataset version.')
        result.update(test_accuracy=metrics['accuracy'], test_macro_f1=metrics['macro_f1'],
                      trained_at=report.get('trained_at', report.get('completed_at')))
    return result


def select_test_records(evidence, classes, count=16):
    test = [r for r in evidence['records'] if r['split'] == 'test']
    size = min(count, len(test))
    selected = {round(i * (len(test) - 1) / max(1, size - 1)) for i in range(size)}
    selected.update(next(i for i, record in enumerate(test) if record['label'] == label) for label in classes)
    return [test[i] for i in sorted(selected)]


def image_tensor(evidence, record):
    import numpy as np
    from PIL import Image
    path = evidence['data'] / record['path']
    if sha_file(path) != record['sha256']:
        raise ValueError(f'Parity image checksum mismatch: {record["path"]}')
    with Image.open(path) as image:
        image = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)[None]


def parity_check(model, session, evidence, records, classes, threshold=1e-4):
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    samples = []; truth = []; keras_probs = []; onnx_probs = []
    maximum = 0.; disagreements = 0
    for index, record in enumerate(records):
        array = image_tensor(evidence, record)
        expected = model(array, training=False).numpy()
        actual = session.run([output_name], {input_name: array})[0]
        if expected.shape != (1, len(classes)) or actual.shape != expected.shape:
            raise ValueError('Exported output shape differs from the class count.')
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('Nonfinite model probability.')
        for probabilities in (expected, actual):
            if (np.any(probabilities < -1e-6) or np.any(probabilities > 1 + 1e-6)
                    or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5, rtol=0.)):
                raise ValueError('Model output must contain normalized class probabilities.')
        delta = float(np.max(np.abs(expected - actual)))
        maximum = max(maximum, delta)
        disagreements += int(expected.argmax() != actual.argmax())
        samples.append(dict(id=record['id'], actual_class=record['label'], max_absolute_error=delta,
                            keras_class=classes[int(expected.argmax())], onnx_class=classes[int(actual.argmax())]))
        truth.append(classes.index(record['label']))
        keras_probs.append(expected[0]); onnx_probs.append(actual[0])
        if (index + 1) % 100 == 0:
            print(f'Final export parity: {index + 1}/{len(records)}', flush=True)
    if maximum >= threshold or disagreements:
        raise ValueError(f'ONNX parity failed: max_error={maximum}, top1_disagreements={disagreements}.')
    truth = np.asarray(truth, dtype=np.int32)
    keras_probs = np.asarray(keras_probs); onnx_probs = np.asarray(onnx_probs)

    def metrics(probabilities):
        pred = probabilities.argmax(1)
        report = classification_report(truth, pred, labels=list(range(len(classes))), target_names=classes,
                                       output_dict=True, zero_division=0)
        return dict(accuracy=float((pred == truth).mean()), correct=int((pred == truth).sum()),
                    test_images=len(truth), macro_f1=report['macro avg']['f1-score'],
                    macro_precision=report['macro avg']['precision'], macro_recall=report['macro avg']['recall'],
                    weighted_f1=report['weighted avg']['f1-score'], classification_report=report,
                    confusion_matrix=confusion_matrix(truth, pred, labels=list(range(len(classes)))).tolist())
    return dict(parity_samples=len(records), parity_max_absolute_error=maximum,
                parity_top1_disagreements=disagreements, samples=samples,
                keras_test=metrics(keras_probs), onnx_test=metrics(onnx_probs)), dict(
                    truth=truth, keras_probabilities=keras_probs, onnx_probabilities=onnx_probs,
                    ids=np.asarray([r['id'] for r in records]))
