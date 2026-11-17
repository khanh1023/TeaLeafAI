from pathlib import Path
import json, io, os
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tea_leaf_mobilenetv2.keras"
CLASSES_PATH = ROOT / "models" / "classes.json"
FALLBACK_CLASSES = ["blister_blight", "grey_blight", "algal_spot", "brown_blight", "healthy"]
LABELS = {"blister_blight":"Bệnh phồng lá","grey_blight":"Bệnh đốm xám",
          "algal_spot":"Bệnh đốm tảo","brown_blight":"Bệnh đốm nâu","healthy":"Lá khỏe mạnh"}
MAX_BYTES = 8 * 1024 * 1024

app = FastAPI(title="Tea Leaf AI API", version="1.0.0")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

_model = None
_classes = FALLBACK_CLASSES

def get_model():
    global _model, _classes
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Máy chủ chưa có mô hình. Hãy huấn luyện local và đặt file .keras vào backend/models.")
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(MODEL_PATH)
            if CLASSES_PATH.exists():
                _classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Không tải được mô hình: {exc}")
    return _model

@app.get("/")
def root():
    return {"name":"Tea Leaf AI API","docs":"/docs"}

@app.get("/api/health")
def health():
    return {"status":"ok","model_loaded":_model is not None,"model_file_exists":MODEL_PATH.exists(),"classes":_classes}

@app.post("/api/diagnosis")
async def diagnosis(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg","image/png","image/webp"}:
        raise HTTPException(status_code=415, detail="Chỉ chấp nhận ảnh JPG, PNG hoặc WEBP.")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn 8 MB.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize((224,224))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải ảnh hợp lệ.")
    model = get_model()
    arr = np.asarray(image, dtype=np.float32)[None, ...]
    predictions = model.predict(arr, verbose=0)[0]
    if len(predictions) != len(_classes):
        raise HTTPException(status_code=500, detail="Số đầu ra mô hình không khớp số lớp trong classes.json.")
    idx = int(np.argmax(predictions))
    key = _classes[idx]
    return {
        "predicted_class": key,
        "predicted_label": LABELS.get(key, key),
        "confidence": float(predictions[idx]),
        "probabilities": {c: float(p) for c,p in zip(_classes, predictions)}
    }
