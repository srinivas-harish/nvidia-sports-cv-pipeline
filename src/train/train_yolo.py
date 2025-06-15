from ultralytics import YOLO
import argparse
import os
import time


def train_yolo(
    data_yaml="data/football-players-v14/data.yaml",
    img_size=640,
    epochs=50,
    batch_size=16,
    name="football_yolov8l_640"
):
    #   absolute path to data.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))  # → ~/sport-cv-pipeline/src/train
    root_dir = os.path.dirname(os.path.dirname(script_dir))  # → ~/sport-cv-pipeline
    abs_data_yaml = os.path.join(root_dir, data_yaml)

    assert os.path.exists(abs_data_yaml), f"Dataset YAML not found: {abs_data_yaml}"

    #  YOLOv8l
    model = YOLO("yolov8l.pt")

    model.train(
        data=abs_data_yaml,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        project="runs/train",
        name=name,
        workers=4,
        verbose=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8l on custom dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=640, help="Image size (square)")
    parser.add_argument("--data_yaml", type=str, default="data/football-players-v14/data.yaml", help="Path to data.yaml")
    parser.add_argument("--name", type=str, default="football_yolov8l_640", help="Training run name")

    args = parser.parse_args()
    train_yolo(
        data_yaml=args.data_yaml,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        name=args.name
    )
