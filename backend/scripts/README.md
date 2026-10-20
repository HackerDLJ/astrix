# Dataset preparation

1. Download the dataset from https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO.
2. Inspect dataset card/license and class map.
3. Ensure a YOLO `data.yaml` points to train/val/test image directories.
4. Run training from `backend/`:
   `python scripts/train.py --data /absolute/path/data.yaml --model yolo11n.pt --epochs 100 --imgsz 960`
5. Copy the best checkpoint from `runs/astrix/obstacle-detector/weights/best.pt` and set `ASTRIX_MODEL_PATH` to it.
6. Validate precision, recall, mAP50-95, class-specific recall, small-object recall and latency on the target hardware.
