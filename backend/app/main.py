"""Optional FastAPI inference service; the public website can also run AI locally."""
from pathlib import Path
import io
import json
import logging
import os
from threading import Lock

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tea_leaf_mobilenetv2.keras"
CLASSES_PATH = ROOT / "models" / "classes.json"
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 25_000_000
IMG_SIZE = (224, 224)
FALLBACK_CLASSES = [
    "algal_leaf", "anthracnose", "bird_eye_spot", "brown_blight",
    "gray_light", "healthy", "red_leaf_spot", "white_spot",
]
LABELS = {
    "algal_leaf": "Bệnh đốm tảo", "anthracnose": "Bệnh thán thư",
    "bird_eye_spot": "Bệnh đốm mắt chim", "brown_blight": "Bệnh đốm nâu",
    "gray_light": "Bệnh đốm xám", "healthy": "Lá khỏe mạnh",
    "red_leaf_spot": "Bệnh đốm đỏ", "white_spot": "Bệnh đốm trắng",
}
app = FastAPI(title="Tea Leaf AI API", version="2.0.0")
origins = [x.strip() for x in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)
_model = None
_classes = FALLBACK_CLASSES.copy()
_model_lock = Lock()
logger = logging.getLogger(__name__)


def validate_classes(classes):
    """The class order is part of the trained model's output contract."""
    if classes != FALLBACK_CLASSES:
        raise ValueError("classes.json phải đúng tên và thứ tự 8 lớp của mô hình.")
    return classes


def get_model():
    global _model, _classes
    with _model_lock:
        if _model is not None:
            return _model
        if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
            raise HTTPException(503, "Thiếu mô hình hoặc classes.json trong backend/models.")
        try:
            classes = validate_classes(json.loads(CLASSES_PATH.read_text(encoding="utf-8")))
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            if tuple(model.input_shape[1:]) != (*IMG_SIZE, 3):
                raise ValueError("Mô hình cần ảnh RGB 224x224.")
            if model.output_shape[-1] != len(classes):
                raise ValueError("Số đầu ra mô hình không khớp 8 lớp.")
            _classes, _model = classes, model
        except HTTPException:
            raise
        except Exception:
            logger.exception("Cannot load tea leaf model")
            raise HTTPException(503, "Không tải được mô hình AI. Kiểm tra mô hình và thư viện TensorFlow.")
    return _model


def preprocess_image(raw):
    """Return raw RGB float pixels; Rescaling is already INSIDE the saved model."""
    try:
        with Image.open(io.BytesIO(raw)) as original:
            if original.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Chỉ chấp nhận ảnh JPG, PNG hoặc WEBP.")
            if original.width * original.height > MAX_PIXELS:
                raise HTTPException(413, "Ảnh có quá nhiều điểm ảnh (tối đa 25 megapixel).")
            image = ImageOps.exif_transpose(original).convert("RGB")
            image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
            return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Tệp tải lên không phải ảnh hợp lệ.")


def format_prediction(predictions, classes):
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(classes),) or not np.all(np.isfinite(predictions)):
        raise HTTPException(500, "Đầu ra mô hình không hợp lệ.")
    if np.any(predictions < 0) or np.any(predictions > 1) or not np.isclose(predictions.sum(), 1, atol=1e-3):
        raise HTTPException(500, "Xác suất đầu ra mô hình không hợp lệ.")
    key = classes[int(np.argmax(predictions))]
    return {
        "predicted_class": key, "predicted_label": LABELS[key],
        "confidence": float(np.max(predictions)),
        "probabilities": {c: float(p) for c, p in zip(classes, predictions)},
    }


def predict(raw):
    arr = preprocess_image(raw)
    model = get_model()
    try:
        predictions = model(arr, training=False).numpy()[0]
    except Exception:
        logger.exception("Tea leaf inference failed")
        raise HTTPException(500, "Không thể phân tích ảnh. Vui lòng thử lại.")
    return format_prediction(predictions, _classes)


@app.get("/")
def root():
    return {"name": "Tea Leaf AI API", "status": "running", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {
        "status": "ok", "model_loaded": _model is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "classes_file_exists": CLASSES_PATH.exists(), "classes": _classes,
        "input_range": "0..255 (normalization is embedded in the model)",
    }


@app.post("/api/diagnosis")
async def diagnosis(file: UploadFile = File(...)):
    try:
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(415, "Chỉ chấp nhận ảnh JPG, PNG hoặc WEBP.")
        # Bound the memory used even when Content-Length is missing or forged.
        raw = await file.read(MAX_BYTES + 1)
        if not raw:
            raise HTTPException(400, "Bạn chưa tải ảnh lên.")
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "Ảnh vượt quá giới hạn 8 MB.")
        return await run_in_threadpool(predict, raw)
    finally:
        await file.close()
