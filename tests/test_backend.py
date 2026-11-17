"""Contract tests run without installing TensorFlow or loading model weights."""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import main


def make_image(color=(255, 128, 0), fmt="PNG"):
    data = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(data, format=fmt)
    return data.getvalue()


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_preprocessing_keeps_raw_pixel_range(self):
        batch = main.preprocess_image(make_image())
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_array_equal(batch[0, 0, 0], [255, 128, 0])

    def test_reordered_class_list_is_rejected(self):
        with self.assertRaises(ValueError):
            main.validate_classes(main.FALLBACK_CLASSES[::-1])

    def test_bad_image_is_rejected_before_model_load(self):
        with patch.object(main, "get_model") as model:
            response = self.client.post("/api/diagnosis", files={"file": ("x.jpg", b"fake", "image/jpeg")})
            self.assertEqual(response.status_code, 400)
            model.assert_not_called()

    def test_oversized_upload_is_rejected(self):
        with patch.object(main, "MAX_BYTES", 10), patch.object(main, "get_model") as model:
            response = self.client.post("/api/diagnosis", files={"file": ("x.png", make_image(), "image/png")})
            self.assertEqual(response.status_code, 413)
            model.assert_not_called()

    def test_actual_image_format_is_checked(self):
        with self.assertRaises(HTTPException) as caught:
            main.preprocess_image(make_image(fmt="BMP"))
        self.assertEqual(caught.exception.status_code, 415)

    def test_nonfinite_probabilities_are_rejected(self):
        with self.assertRaises(HTTPException):
            main.format_prediction([float("nan")] * 8, main.FALLBACK_CLASSES)

    def test_eight_class_output_labels_and_order(self):
        values = [0.01, 0.01, 0.01, 0.01, 0.01, 0.93, 0.01, 0.01]
        result = main.format_prediction(values, main.FALLBACK_CLASSES)
        self.assertEqual(result["predicted_class"], "healthy")
        self.assertEqual(result["predicted_label"], "Lá khỏe mạnh")
        self.assertEqual(list(result["probabilities"]), main.FALLBACK_CLASSES)

    def test_empty_and_unsupported_uploads(self):
        self.assertEqual(self.client.post("/api/diagnosis", files={"file": ("x.png", b"", "image/png")}).status_code, 400)
        self.assertEqual(self.client.post("/api/diagnosis", files={"file": ("x.txt", b"text", "text/plain")}).status_code, 415)


if __name__ == "__main__":
    unittest.main()
