import cv2
from ultralytics import YOLO
import os

# Absolute paths for WSL  
VIDEO_PATH = "/home/srini/sport-cv-pipeline/data/test_clip.mp4"
MODEL_PATH = "/home/srini/sport-cv-pipeline/runs/train/football_yolov8l_640/weights/best.pt"
OUTPUT_PATH = "/home/srini/sport-cv-pipeline/data/test_clip_output.mp4"
 
def main():
    assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
 
    model = YOLO(MODEL_PATH)
 
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Failed to open video: {VIDEO_PATH}"

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("Running inference...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        results = model.predict(source=frame, save=False, conf=0.4, verbose=False)
 
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Inference complete. Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
