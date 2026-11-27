"""Publication evidence must not accidentally mix dataset/model versions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from training.export_metadata import (evidence_metadata, load_dataset, read_json,
                                      bind_checkpoint_evaluation, parity_check, select_test_records, sha_file, validate_evaluation)

ROOT = Path(__file__).resolve().parents[1]


class ExportMetadataTests(unittest.TestCase):
    def test_requested_evaluation_must_identify_the_checkpoint_even_when_json_is_empty(self):
        digest = 'a' * 64
        for report in [None, [], {}, {'accuracy': 1.}, {'model_sha256': 'b' * 64}]:
            with self.subTest(report=report), self.assertRaises(ValueError):
                bind_checkpoint_evaluation(report, digest)
        report = {'model_sha256': digest}
        self.assertIs(bind_checkpoint_evaluation(report, digest), report)

    def fixture(self, root):
        classes = ['a', 'b']
        records = []
        for split in ['train', 'val', 'test']:
            for label in classes:
                identity = split + label
                records.append(dict(id=identity, path=f'{split}/{label}.jpg', label=label,
                                    split=split, group=identity, sha256=identity,
                                    raw_sha256=identity, pixel_sha256=identity))
        (root / 'classes.json').write_text(json.dumps(classes))
        (root / 'manifest.json').write_text(json.dumps(records))
        (root / 'split_summary.json').write_text(json.dumps(dict(version='fixture-v2', total_images=6)))
        return classes, records

    def test_frozen_v1_and_v2_metadata_derive_counts_from_the_actual_manifests(self):
        classes = read_json(ROOT / 'training/classes.json')
        for version, folder, manifest, counts in [
                ('v1', 'training/reports', 'expanded-dataset-manifest.json', (2197, 1541, 327, 329)),
                ('v2', 'training/reports/dataset-v2-20261125', 'manifest.json', (6840, 4788, 1026, 1026))]:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                data = Path(temp)
                (data / 'classes.json').write_text(json.dumps(classes))
                source = ROOT / folder / manifest
                evidence = load_dataset(data, classes, source, sha_file(source))
                metadata = evidence_metadata(evidence)
                self.assertEqual((metadata['dataset_images'], metadata['train_images'],
                                  metadata['validation_images'], metadata['test_images']), counts)
                self.assertEqual(metadata['training_images'], metadata['dataset_images'])
                selection = select_test_records(evidence, classes)
                self.assertEqual({r['label'] for r in selection}, set(classes))
                self.assertTrue(all(r['split'] == 'test' for r in selection))

    def test_export_rejects_changed_manifest_wrong_class_order_and_cross_split_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); classes, records = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'training protocol'):
                load_dataset(root, classes, expected_sha='0' * 64)
            with self.assertRaisesRegex(ValueError, 'class order'):
                load_dataset(root, list(reversed(classes)))
            records[-1]['group'] = records[0]['group']
            (root / 'manifest.json').write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'Cross-split'):
                load_dataset(root, classes)

    def test_metrics_must_be_supported_by_the_confusion_matrix_and_test_count(self):
        report = dict(classes=['a', 'b'], test_images=2, correct=1,
                      accuracy=.5, macro_f1=1 / 3, confusion_matrix=[[1, 0], [1, 0]])
        metrics = validate_evaluation(report, ['a', 'b'], test_count=2)
        self.assertEqual(metrics['correct'], 1)
        for field, value, pattern in [('test_images', 329, 'test count'),
                                      ('accuracy', .945, 'Accuracy'),
                                      ('macro_f1', .935, 'macro-F1'),
                                      ('correct', 2, 'correct count')]:
            with self.subTest(field=field):
                changed = copy.deepcopy(report); changed[field] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_evaluation(changed, ['a', 'b'], test_count=2)

    def test_evaluation_cannot_be_attached_to_another_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); classes, _ = self.fixture(root)
            evidence = load_dataset(root, classes)
            report = dict(classes=classes, test_images=2, correct=2, accuracy=1., macro_f1=1.,
                          confusion_matrix=[[1, 0], [0, 1]], dataset_manifest_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'different dataset'):
                evidence_metadata(evidence, report)

    def test_parity_verifies_actual_images_probabilities_and_held_out_metrics(self):
        import numpy as np
        from PIL import Image

        class Model:
            def __call__(self, images, training=False):
                probability = [[.9, .1]] if images[0, 0, 0, 0] == 0 else [[.2, .8]]
                return SimpleNamespace(numpy=lambda: np.asarray(probability, dtype=np.float32))

        class Session:
            def __init__(self, transform=lambda values: values):
                self.transform = transform

            def get_inputs(self):
                return [SimpleNamespace(name='images')]

            def get_outputs(self):
                return [SimpleNamespace(name='probabilities')]

            def run(self, names, inputs):
                return [self.transform(Model()(inputs['images']).numpy())]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); records = []
            for label, brightness in [('a', 0), ('b', 255)]:
                path = root / (label + '.png')
                Image.new('RGB', (8, 8), (brightness,) * 3).save(path)
                records.append(dict(id=label, label=label, split='test', path=path.name, sha256=sha_file(path)))
            evidence = {'data': root}
            result, predictions = parity_check(Model(), Session(), evidence, records, ['a', 'b'])
            self.assertEqual(result['parity_samples'], 2)
            self.assertEqual(result['parity_top1_disagreements'], 0)
            self.assertEqual(result['keras_test']['confusion_matrix'], [[1, 0], [0, 1]])
            self.assertEqual(result['onnx_test']['accuracy'], 1.)
            self.assertEqual(predictions['ids'].tolist(), ['a', 'b'])
            with self.assertRaisesRegex(ValueError, 'normalized class probabilities'):
                parity_check(Model(), Session(lambda values: values * 2), evidence, records, ['a', 'b'])
            records[0]['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'image checksum'):
                parity_check(Model(), Session(), evidence, records, ['a', 'b'])

    def test_probability_tolerance_does_not_allow_different_top_one_labels(self):
        import numpy as np
        from PIL import Image

        class Model:
            def __call__(self, images, training=False):
                return SimpleNamespace(numpy=lambda: np.array([[.50001, .49999]], dtype=np.float32))

        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name='images')]

            def get_outputs(self):
                return [SimpleNamespace(name='probabilities')]

            def run(self, names, inputs):
                return [np.array([[.49999, .50001]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'leaf.png'
            Image.new('RGB', (8, 8), 'green').save(path)
            record = dict(id='leaf', label='a', split='test', path=path.name, sha256=sha_file(path))
            with self.assertRaisesRegex(ValueError, 'top1_disagreements=1'):
                parity_check(Model(), Session(), {'data': root}, [record], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
