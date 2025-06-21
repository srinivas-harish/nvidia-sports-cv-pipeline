# train_yolo.py

import os
from ultralytics import YOLO


def train_yolo_model():
    # Define paths and training config
    data_yaml_path = "/home/srini/sport-cv-pipeline/data/football-players-v13/data.yaml"
    model_arch = "yolov9c.pt"  # Using YOLOv9-C
    project_dir = "/home/srini/sport-cv-pipeline/runs"
    run_name = "yolov9c_football_v13"

    # Create the model
    print(f"Loading model: {model_arch}")
    model = YOLO(model_arch)

    # Kick off training
    print(f"Starting training on dataset: {data_yaml_path}")
    model.train(
        data=data_yaml_path,
        epochs=60,
        imgsz=1280,
        batch=12,
        name=run_name,
        project=project_dir,
        workers=4,
        device=0  # Use CUDA device 0
    )

    print(f"Training complete. Best weights saved to: {project_dir}/{run_name}/weights/best.pt")


if __name__ == "__main__":
    train_yolo_model()
