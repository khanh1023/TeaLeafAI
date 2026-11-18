"""Validate the published dataset split and the evaluation tied to the browser model."""
import collections
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TrainingArtifactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_bytes = (ROOT / 'training/reports/expanded-dataset-manifest.json').read_bytes()
        cls.records = json.loads(cls.manifest_bytes)
        cls.summary = json.loads((ROOT / 'training/reports/expanded-dataset-summary.json').read_text())
        cls.evaluation = json.loads((ROOT / 'training/evaluation.json').read_text())
        cls.model = json.loads((ROOT / 'frontend/public/model/manifest.json').read_text())

    def test_groups_and_duplicate_content_do_not_cross_splits(self):
        for field in ['group', 'raw_sha256', 'pixel_sha256', 'sha256']:
            splits = collections.defaultdict(set)
            for record in self.records:
                splits[record[field]].add(record['split'])
            self.assertTrue(all(len(values) == 1 for values in splits.values()), field)

    def test_counts_and_healthy_majority_match_the_report(self):
        counts = collections.Counter(record['label'] for record in self.records)
        self.assertEqual(counts, self.summary['counts'])
        self.assertEqual(sum(counts.values()), self.summary['total_images'])
        self.assertGreater(counts['healthy'], max(v for k, v in counts.items() if k != 'healthy'))
        self.assertEqual(set(counts), set(self.model['classes']))
        for split, target in [('train', .70), ('val', .15), ('test', .15)]:
            subset = [r for r in self.records if r['split'] == split]
            self.assertEqual(collections.Counter(r['label'] for r in subset), self.summary['split_counts'][split])
            self.assertLess(abs(len(subset) / len(self.records) - target), .01)

    def test_evaluation_matches_dataset_and_served_model(self):
        digest = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.assertEqual(digest, self.evaluation['dataset_manifest_sha256'])
        self.assertEqual(digest, self.model['dataset_manifest_sha256'])
        matrix = self.evaluation['confusion_matrix']
        n = sum(sum(row) for row in matrix)
        self.assertEqual(n, sum(r['split'] == 'test' for r in self.records))
        accuracy = sum(matrix[i][i] for i in range(len(matrix))) / n
        self.assertAlmostEqual(accuracy, self.evaluation['accuracy'])
        self.assertAlmostEqual(accuracy, self.model['test_accuracy'])


if __name__ == '__main__':
    unittest.main()
