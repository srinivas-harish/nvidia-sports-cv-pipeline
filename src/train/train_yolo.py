from ultralytics import YOLO
import argparse
import os

def train_yolo(
    model_size="n",
    data_yaml="data/football-players-v14/data.yaml",
    img_size=640,
    epochs=50,
    batch_size=16,
    name="football_yolov8_640"
):
    assert os.path.exists(data_yaml), f"Dataset YAML not found: {data_yaml}"
    model = YOLO(f"yolov8{model_size}.pt")  # Load base YOLOv8 model (e.g. yolov8n.pt)

    model.train(
        data=data_yaml,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        name=name,
        project="runs/train",
        workers=4,
        verbose=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on custom dataset")
    parser.add_argument("--model_size", default="n", help="YOLOv8 model size: n/s/m/l/x")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=640, help="Image size (square)")
    parser.add_argument("--data_yaml", type=str, default="data/football-players-v14/data.yaml", help="Path to data.yaml")
    parser.add_argument("--name", type=str, default="football_yolov8_640", help="Training run name")

    args = parser.parse_args()
    train_yolo(
        model_size=args.model_size,
        data_yaml=args.data_yaml,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        name=args.name
    )
