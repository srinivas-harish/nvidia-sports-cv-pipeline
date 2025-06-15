from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path="runs/train/train/weights/best.pt"):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model.predict(source=frame, save=False, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        return boxes, classes, confidences
