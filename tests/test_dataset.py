import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prepare_dataset", Path(__file__).resolve().parents[1] / "training" / "prepare_dataset.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class DatasetTests(unittest.TestCase):
    def test_identical_files_deduplicate_and_conflicting_labels_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "algal_leaf").mkdir()
            (root / "healthy").mkdir()
            (root / "algal_leaf" / "a.jpg").write_bytes(b"same-image")
            (root / "algal_leaf" / "b.jpg").write_bytes(b"same-image")
            with patch.object(prepare, "SOURCE", root):
                collected = prepare.collect_images()
                self.assertEqual(len(collected["algal_leaf"]), 1)
                (root / "healthy" / "c.jpg").write_bytes(b"same-image")
                with self.assertRaisesRegex(ValueError, "khác nhãn"):
                    prepare.collect_images()

    def test_three_images_produce_nonempty_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            images = []
            for index in range(3):
                path = source / f"{index}.jpg"
                path.write_bytes(str(index).encode())
                images.append(path)
            dest = Path(temp) / "dataset"
            with patch.object(prepare, "DEST", dest), patch("builtins.input", return_value="YES"):
                prepare.split_dataset({"healthy": images})
            for split in ["train", "val", "test"]:
                self.assertEqual(len(list((dest / split / "healthy").iterdir())), 1)
            self.assertTrue(all(path.exists() for path in images))


if __name__ == "__main__":
    unittest.main()
