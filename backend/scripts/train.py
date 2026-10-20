import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ASTRIX YOLO obstacle detector")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    parser.add_argument("--project", default="runs/astrix")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="obstacle-detector",
        patience=20,
        cache=False,
        plots=True,
    )
    model.val(data=args.data, imgsz=args.imgsz)
    model.export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True)

if __name__ == "__main__":
    main()
