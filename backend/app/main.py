
from pathlib import Path
import json
import io
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "tea_leaf_mobilenetv2.keras"
CLASSES_PATH = ROOT / "models" / "classes.json"

MAX_BYTES = 8 * 1024 * 1024
IMG_SIZE = (224, 224)

FALLBACK_CLASSES = [
    "algal_leaf",
    "anthracnose",
    "bird_eye_spot",
    "brown_blight",
    "gray_light",
    "healthy",
    "red_leaf_spot",
    "white_spot"
]

LABELS = {
    "algal_leaf": "Bệnh đốm tảo",
    "anthracnose": "Bệnh thán thư",
    "bird_eye_spot": "Bệnh đốm mắt chim",
    "brown_blight": "Bệnh đốm nâu",
    "gray_light": "Bệnh đốm xám",
    "healthy": "Lá khỏe mạnh",
    "red_leaf_spot": "Bệnh đốm đỏ",
    "white_spot": "Bệnh đốm trắng"
}


# ==========================================
# 2. KHỞI TẠO FASTAPI
# ==========================================

app = FastAPI(
    title="Tea Leaf AI API",
    version="1.0.0"
)


# ==========================================
# 3. CẤU HÌNH CORS
# ==========================================

origins = [
    x.strip()
    for x in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# 4. BIẾN MÔ HÌNH
# ==========================================

_model = None
_classes = FALLBACK_CLASSES


# ==========================================
# 5. TẢI MÔ HÌNH AI
# ==========================================

def get_model():

    global _model, _classes

    if _model is None:

        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Không tìm thấy mô hình AI. "
                       "Vui lòng kiểm tra file "
                       "tea_leaf_mobilenetv2.keras."
            )

        if not CLASSES_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Không tìm thấy file classes.json."
            )

        try:
            import tensorflow as tf

            loaded_classes = json.loads(
                CLASSES_PATH.read_text(encoding="utf-8")
            )

            if not isinstance(loaded_classes, list) or not loaded_classes:
                raise ValueError("classes.json không hợp lệ.")

            loaded_model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )

            output_shape = loaded_model.output_shape

            if output_shape[-1] != len(loaded_classes):
                raise ValueError(
                    "Số đầu ra mô hình không khớp số lớp trong classes.json."
                )

            _classes = loaded_classes
            _model = loaded_model

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Không tải được mô hình AI: {str(exc)}"
            )

    return _model


# ==========================================
# 6. API TRANG CHỦ
# ==========================================

@app.get("/")
def root():

    return {
        "name": "Tea Leaf AI API",
        "status": "running",
        "docs": "/docs"
    }


# ==========================================
# 7. API KIỂM TRA KẾT NỐI
# ==========================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "classes_file_exists": CLASSES_PATH.exists(),
        "classes": _classes
    }


# ==========================================
# 8. API PHÂN TÍCH ẢNH LÁ CHÈ
# ==========================================

@app.post("/api/diagnosis")
async def diagnosis(file: UploadFile = File(...)):

    # Kiểm tra định dạng ảnh
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp"
    }:

        raise HTTPException(
            status_code=415,
            detail="Chỉ chấp nhận ảnh JPG, PNG hoặc WEBP."
        )

    # Đọc ảnh
    raw = await file.read()

    if not raw:

        raise HTTPException(
            status_code=400,
            detail="Bạn chưa tải ảnh lên."
        )

    # Kiểm tra dung lượng
    if len(raw) > MAX_BYTES:

        raise HTTPException(
            status_code=413,
            detail="Ảnh vượt quá giới hạn 8 MB."
        )

    # Đọc và xử lý ảnh
    try:

        image = (
            Image.open(io.BytesIO(raw))
            .convert("RGB")
            .resize(IMG_SIZE)
        )

    except (UnidentifiedImageError, OSError):

        raise HTTPException(
            status_code=400,
            detail="Tệp tải lên không phải ảnh hợp lệ."
        )

    # Tải mô hình
    model = get_model()

    # ======================================
    # TIỀN XỬ LÝ ẢNH
    # Đồng bộ với train.py
    # ======================================

    arr = np.asarray(
        image,
        dtype=np.float32
    )

    arr = arr / 127.5 - 1.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    # ======================================
    # DỰ ĐOÁN
    # ======================================

    try:

        predictions = model.predict(
            arr,
            verbose=0
        )[0]

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích ảnh: {str(exc)}"
        )

    if len(predictions) != len(_classes):

        raise HTTPException(
            status_code=500,
            detail="Số đầu ra mô hình không khớp số lớp."
        )

    # Xác định lớp có xác suất cao nhất
    idx = int(np.argmax(predictions))

    key = _classes[idx]

    confidence = float(predictions[idx])

    # ======================================
    # TRẢ KẾT QUẢ
    # ======================================

    return {
        "predicted_class": key,

        "predicted_label": LABELS.get(
            key,
            key
        ),

        "confidence": confidence,

        "probabilities": {
            c: float(p)
            for c, p in zip(
                _classes,
                predictions
            )
        }
    }
