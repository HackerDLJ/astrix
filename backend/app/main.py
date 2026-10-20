import io
import os
import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

APP_NAME = "ASTRIX API"
MODEL_PATH = os.getenv("ASTRIX_MODEL_PATH", "yolo11n.pt")
CONF = float(os.getenv("ASTRIX_CONFIDENCE", "0.25"))
IOU = float(os.getenv("ASTRIX_IOU", "0.45"))
MAX_DETECTIONS = int(os.getenv("ASTRIX_MAX_DETECTIONS", "40"))
ENABLE_DEPTH = os.getenv("ASTRIX_ENABLE_DEPTH", "false").lower() == "true"
METERS_PER_DEPTH_UNIT = float(os.getenv("ASTRIX_METERS_PER_DEPTH_UNIT", "1.0"))
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

origins = os.getenv(
    "ASTRIX_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app = FastAPI(title=APP_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in origins if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
model_error = None
depth_pipeline = None
depth_error = None


def get_model():
    global model, model_error
    if model is None and model_error is None:
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
    return model


def get_depth_pipeline():
    global depth_pipeline, depth_error
    if not ENABLE_DEPTH:
        return None
    if depth_pipeline is None and depth_error is None:
        try:
            from transformers import pipeline
            depth_pipeline = pipeline(
                task="depth-estimation",
                model=os.getenv(
                    "ASTRIX_DEPTH_MODEL",
                    "depth-anything/Depth-Anything-V2-Small-hf"
                )
            )
        except Exception as exc:
            depth_error = f"{type(exc).__name__}: {exc}"
    return depth_pipeline


def estimate_pseudo_distance(depth_map: np.ndarray, xyxy: list[float]) -> dict[str, Any]:
    """
    Relative monocular depth is not metric distance. This returns a clearly marked
    pseudo-distance only; the scale must be calibrated for a specific camera/model.
    """
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = xyxy
    xa = max(0, min(w - 1, int(x1 / max(1, w) * w)))
    ya = max(0, min(h - 1, int(y1 / max(1, h) * h)))
    xb = max(xa + 1, min(w, int(x2 / max(1, w) * w)))
    yb = max(ya + 1, min(h, int(y2 / max(1, h) * h)))
    crop = depth_map[ya:yb, xa:xb]
    if crop.size == 0:
        return {"distance_m": None, "distance_quality": "unavailable"}
    # Depth Anything output is commonly inverse/relative depth. Do not claim meters.
    rel = float(np.nanmedian(crop))
    if not np.isfinite(rel):
        return {"distance_m": None, "distance_quality": "unavailable"}
    return {
        "distance_m": None,
        "relative_depth": round(rel, 4),
        "distance_quality": "relative_only",
        "note": "Metric distance requires a calibrated metric-depth model."
    }


@app.get("/health")
def health():
    loaded = model is not None
    return {
        "status": "ok",
        "app": APP_NAME,
        "detector_loaded": loaded,
        "detector_error": model_error,
        "model_path": MODEL_PATH,
        "depth_enabled": ENABLE_DEPTH,
        "depth_loaded": depth_pipeline is not None,
        "depth_error": depth_error,
    }


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(default=CONF, ge=0.0, le=1.0),
):
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 8 MB limit.")

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        frame = np.asarray(pil)
    except Exception:
        raise HTTPException(status_code=400, detail="Image could not be decoded.")

    detector = get_model()
    if detector is None:
        raise HTTPException(status_code=503, detail=f"Detector unavailable: {model_error}")

    started = time.perf_counter()
    try:
        result = detector.predict(
            source=frame,
            conf=confidence,
            iou=IOU,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}")

    names = result.names
    detections = []
    boxes = result.boxes
    if boxes is not None:
        for idx in range(min(len(boxes), MAX_DETECTIONS)):
            try:
                xyxy = boxes.xyxy[idx].detach().cpu().numpy().astype(float).tolist()
                cls_id = int(boxes.cls[idx].item())
                score = float(boxes.conf[idx].item())
                x1, y1, x2, y2 = xyxy
                label = str(names.get(cls_id, f"class_{cls_id}"))
                detections.append({
                    "id": f"frame-{idx}",
                    "class_id": cls_id,
                    "label": label,
                    "confidence": round(score, 4),
                    "bbox": [
                        max(0, min(1, x1 / frame.shape[1])),
                        max(0, min(1, y1 / frame.shape[0])),
                        max(0, min(1, x2 / frame.shape[1])),
                        max(0, min(1, y2 / frame.shape[0])),
                    ],
                    "distance_m": None,
                    "distance_quality": "unavailable",
                    "risk": "unknown",
                })
            except Exception:
                # One malformed result must not crash the entire frame.
                continue

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "detections": detections,
        "image": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "inference_ms": round(elapsed_ms, 2),
        "model": MODEL_PATH,
        "distance_note": "Metric distance unavailable unless a calibrated metric-depth system is configured.",
    }


@app.post("/api/depth")
async def depth(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    raw = await image.read()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Invalid image or image exceeds 8 MB.")
    pipeline = get_depth_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=depth_error or "Depth estimation is disabled. Set ASTRIX_ENABLE_DEPTH=true."
        )
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        result = pipeline(pil)
        depth_image = result["depth"]
        arr = np.asarray(depth_image, dtype=np.float32)
        return {
            "width": int(arr.shape[1]),
            "height": int(arr.shape[0]),
            "relative_depth": arr.tolist(),
            "distance_quality": "relative_only",
            "note": "This model output is not metric distance. Do not present it as meters."
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Depth inference failed: {type(exc).__name__}")
