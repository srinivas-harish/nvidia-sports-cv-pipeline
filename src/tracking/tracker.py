'''
Uses YOLOv8 model to run inference frame by frame 
Feeds results to ByteTrack for multi-object tracking
Draws bounding boxes with consistent track IDs
Saves output video
'''


'''
Uses YOLOv8 model to run inference frame by frame 
Feeds results to ByteTrack for multi-object tracking
Draws bounding boxes with consistent track IDs
Saves output video
'''


import cv2
import os
import json
import csv
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker 
from easydict import EasyDict

# Paths
VIDEO_PATH = "/home/srini/sport-cv-pipeline/data/test_clip.mp4"
MODEL_PATH = "/home/srini/sport-cv-pipeline/runs/train/football_yolov8_6402/weights/best.pt"
OUTPUT_VIDEO = "/home/srini/sport-cv-pipeline/data/test_clip_tracked.mp4"
OUTPUT_JSON = "/home/srini/sport-cv-pipeline/data/tracking_output.json"
OUTPUT_CSV = "/home/srini/sport-cv-pipeline/data/tracking_output.csv"
CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]


 

def main():
    model = YOLO(MODEL_PATH)
    tracker_args = EasyDict(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        min_box_area=10,
        frame_rate=30
    )
    tracker = BYTETracker(tracker_args)


    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Failed to open video: {VIDEO_PATH}"

    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_id = 0
    json_out = []
    csv_out = [["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2"]]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.3, save=False, verbose=False)[0]
        detections, classes = [], []

        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(float, box.tolist())
            conf = float(conf)
            cls = int(cls)
            detections.append([x1, y1, x2, y2, conf])
            classes.append(cls)

        dets = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)
        tracks = tracker.update(dets, classes, [h, w], (h, w))

        for t in tracks:
            x1, y1, x2, y2 = map(int, t.tlbr)
            cls_id = t.cls
            track_id = t.track_id
            conf = t.score
            label = f"{CLASS_NAMES[cls_id]} ID:{track_id}"

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

            # Logs
            json_out.append({
                "frame": frame_id,
                "track_id": track_id,
                "class": CLASS_NAMES[cls_id],
                "conf": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })
            csv_out.append([frame_id, track_id, CLASS_NAMES[cls_id], round(conf, 3), x1, y1, x2, y2])

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    with open(OUTPUT_JSON, "w") as jf:
        json.dump(json_out, jf, indent=2)

    with open(OUTPUT_CSV, "w", newline="") as cf:
        csv.writer(cf).writerows(csv_out)

    print(f" Saved: {OUTPUT_VIDEO}")
    print(f" JSON: {OUTPUT_JSON}")
    print(f" CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
