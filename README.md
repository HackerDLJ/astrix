# ASTRIX — Real-time obstacle detection

ASTRIX is a web application starter with a premium dark UI, browser camera capture, local YOLO inference in the browser, optional server-side YOLO inference, monocular depth estimation, multi-object tracking, and voice alerts.

## Important accuracy/safety notes

- This is an executable **starter implementation**, not a validated safety device.
- A monocular depth model often predicts relative depth, not reliable metric distance. This implementation uses a calibration scale (`ASTRIX_METERS_PER_DEPTH_UNIT`) and marks distance as estimated. Calibrate against measured distances in the target camera setup.
- Standard COCO YOLO models do not identify safety pins or Apple Watches as dedicated classes. Fine-grained classes require a model trained on labeled examples for those classes.
- The Hugging Face dataset is an input to the training workflow, not proof that its labels include watches or safety pins.
- Browser support for WebGPU varies. ONNX Runtime Web uses WebGPU where available and can fall back to WASM, but a suitable ONNX model and compatible output decoder are required.

## Stack

- Frontend: React + Vite + TypeScript + Tailwind-style CSS
- Local detection: ONNX Runtime Web (optional local model)
- Backend: FastAPI + Ultralytics YOLO + optional depth model
- Voice: Web Speech API
- Tracking: IoU-based lightweight tracker
- Camera: `getUserMedia`

## Run locally

Requirements: Node.js 20+, Python 3.11+.

### 1. Backend

```bash
cd backend
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first Ultralytics run may download a model. Set `ASTRIX_MODEL_PATH` to a custom trained weights file. For CPU use, `yolo11n.pt` is a reasonable prototype starting point.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL. Camera access requires `localhost` or HTTPS and user permission. Set `VITE_API_URL=http://localhost:8000` in `frontend/.env` if your backend is hosted elsewhere.

## Browser-local model (optional)

Place an exported ONNX detector at `frontend/public/models/obstacles.onnx`, then create `frontend/public/models/classes.json` as a JSON array of class names in the same order as the model. Set `VITE_LOCAL_MODEL_URL=/models/obstacles.onnx`.

**Important:** the provided browser decoder expects YOLO-style output shaped like `[1, 4 + classes, candidates]` or `[1, candidates, 4 + classes]`, with xywh box coordinates in the input image coordinate system (typically 640x640) and class scores following the first four box values. Some exported YOLO models include objectness or NMS outputs; adapt `decodeYoloOutput()` to your exact export. Test the decoder before relying on it.

If a local model is unavailable or fails to load, ASTRIX can use the backend endpoint.

## Dataset and fine-tuning

Use the requested dataset:
https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO

The `backend/scripts/train.py` script expects a YOLO-format dataset folder containing `data.yaml` and image/label directories. Download the dataset from Hugging Face and inspect its README, license, class names and splits before training. Do not assume the dataset contains safety-pin or Apple Watch classes.

```bash
pip install -r requirements.txt
python scripts/train.py --data /path/to/dataset/data.yaml --model yolo11n.pt --epochs 100 --imgsz 960
```

For small objects such as safety pins:
- gather high-resolution close-up images under varied backgrounds and lighting
- label safety pins precisely; include hard negatives (clips, needles, wires, jewelry)
- train a dedicated small-object model or add the class to the detector
- evaluate precision/recall on a held-out test set, including tiny-object recall
- consider tiled/high-resolution inference, with a latency trade-off

## Backend API

- `GET /health`
- `POST /api/detect` multipart form field `image` (JPEG/PNG) -> detections, optional estimated depth values, model status
- `POST /api/depth` optional depth map endpoint (if configured in code)

## Environment variables

- `ASTRIX_MODEL_PATH`: YOLO weights path (default `yolo11n.pt`)
- `ASTRIX_CONFIDENCE`: fallback detector confidence threshold (default `0.25`; the live UI sensitivity is sent per frame)
- `ASTRIX_IOU`: NMS IoU threshold (default `0.45`)
- `ASTRIX_MAX_DETECTIONS`: max results per frame (default `40`)
- `ASTRIX_METERS_PER_DEPTH_UNIT`: calibration factor for monocular depth estimates (default `1.0`, **must be calibrated**)
- `ASTRIX_CORS_ORIGINS`: comma-separated allowed origins (default localhost Vite)
- `VITE_API_URL`: backend URL
- `VITE_LOCAL_MODEL_URL`: optional browser ONNX model URL

## Distance estimation

The backend optionally loads Transformers `depth-anything/Depth-Anything-V2-Small-hf` when `ASTRIX_ENABLE_DEPTH=true`. Its output is relative depth; this starter converts it into a rough pseudo-distance using a configurable scale. This is **not metric depth** and is not a substitute for calibration or a depth sensor. To obtain metric estimates, use a metric-depth model trained for the camera domain, known camera intrinsics, and real measured calibration targets. Displaying “Unknown” is safer than fabricating a number.

## Deployment

Deploy frontend and backend separately. The browser camera requires HTTPS in production. The backend should be deployed behind HTTPS with rate limits, upload-size limits, authentication if private, and a GPU if low-latency server inference is needed. Do not expose unrestricted model inference to the public internet.

## References to evaluate during development

- https://github.com/ultralytics/yolo-ios-app
- https://github.com/lrettenberger/obstacle-detection-on-sidewalks-with-a-single-camera
- https://modelnova.ai/blueprints/details/obstacle-detection-system
- https://repository.gatech.edu/server/api/core/bitstreams/a24e6c75-d28e-4094-bf93-9d372ab7818f/content
- https://github.com/AryanTN05/obstacle-detection-and-classification
- https://github.com/tcotte/Computer_vision_UC3M
- https://github.com/yahyatawil/WODA_Paper
- https://universe.roboflow.com/visually-impaired-obstacle-detection-uxdze/obstacle-detection-yeuzf
- https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO

Review each project's license and dataset terms before copying code, weights, or annotations.
