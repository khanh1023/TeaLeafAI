"""Checks for the comparison workflow, never scientific accuracy measurements."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    from training import compare_models as comparison
except ImportError:
    tf = None


@unittest.skipIf(tf is None, 'TensorFlow comparison dependencies are optional')
class ComparisonTests(unittest.TestCase):
    def test_caffe_preprocessing_matches_official_models(self):
        pixels = np.array([[[[255., 128., 0.], [1., 2., 3.]]]], dtype=np.float32)
        actual = comparison.CaffePreprocessing()(pixels).numpy()
        for official in [tf.keras.applications.resnet50.preprocess_input,
                         tf.keras.applications.vgg16.preprocess_input]:
            np.testing.assert_allclose(actual, official(pixels.copy()), atol=1e-5)

    def test_pinned_manifest_images_and_group_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixels = b'fixture bytes; integrity test only'
            (root/'image.bin').write_bytes(pixels)
            record = {'path':'image.bin', 'label':'healthy', 'split':'train', 'group':'g',
                      'sha256':hashlib.sha256(pixels).hexdigest(), 'raw_sha256':'r', 'pixel_sha256':'p'}
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps([record]))
            comparison.validate_dataset(root, manifest, comparison.sha(manifest))
            (root/'image.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                comparison.validate_dataset(root, manifest, comparison.sha(manifest))
            manifest.write_text(json.dumps([record, {**record, 'split':'test'}]))
            with self.assertRaisesRegex(ValueError, 'Cross-split'):
                comparison.validate_dataset(root, manifest, comparison.sha(manifest))

    def test_augmentation_is_repeatable_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp); record = {'path':'leaf.png'}
            Image.fromarray(np.random.default_rng(42).integers(0,256,(80,120,3),dtype=np.uint8)).save(data/'leaf.png')
            original_hash = comparison.sha(data/'leaf.png')
            a = comparison.image_array(data, record, 1, 1, 42)
            b = comparison.image_array(data, record, 1, 1, 42)
            original = comparison.image_array(data, record, 1, 0, 42)
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.array_equal(a, original))
            self.assertEqual(original_hash, comparison.sha(data/'leaf.png'))

    def test_all_backbones_head_training_and_full_model_roundtrip(self):
        # Random weights validate plumbing/serialization only; no reported model metrics.
        with tempfile.TemporaryDirectory() as tmp:
            for name in comparison.MODELS:
                with self.subTest(name=name):
                    tf.keras.backend.clear_session()
                    tf.keras.utils.set_random_seed(42)
                    extractor = comparison.build_extractor(name, weights=None)
                    raw = np.random.default_rng(42).uniform(0,255,(2,224,224,3)).astype('float32')
                    features = extractor(raw, training=False).numpy()
                    head = comparison.build_head(features.shape[1])
                    head.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
                    loss = head.train_on_batch(features, np.array([0,5]))
                    self.assertTrue(np.isfinite(loss))
                    model = tf.keras.Model(extractor.input, head(extractor.output))
                    expected = model(raw, training=False).numpy()
                    self.assertEqual(expected.shape, (2,8))
                    np.testing.assert_allclose(expected.sum(1), 1., atol=1e-5)
                    path = Path(tmp)/(name+'.keras'); model.save(path)
                    loaded = tf.keras.models.load_model(path, compile=False)
                    np.testing.assert_allclose(loaded(raw, training=False).numpy(), expected, atol=1e-6)
                    self.assertFalse(extractor.trainable_weights)


if __name__ == '__main__':
    unittest.main()
