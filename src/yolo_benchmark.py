import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import read_video    
from trackers import Tracker    

import csv
import os 

VIDEO_PATH = 'data/test_clip.mp4'

#  base assets  
print('Loading frames …')
frames = read_video(VIDEO_PATH)
#frames = frames[10:750]  #   frames 10 to 749
tracker = Tracker('models/best.engine')    

#  viewer state  
N_FR = len(frames)
idx = 0
play = False
inference_times = []

cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Demo', 1920, 1080)
print('\nKeys: ←/→ frame | SPACE play/pause | Q/ESC quit\n')

target_frame_time = 1 / 30  # 30 FPS
BATCH = 4  # Tune batch size for your GPU

# processing logic  
def process_batch(batch_frames):
    start = time.time()
    results = tracker.model.predict(batch_frames, device=0, half=True, imgsz=1280, conf=0.3, verbose=False)
    end = time.time()

    elapsed_per_frame = ((end - start) * 1000) / len(batch_frames)
    inference_times.extend([elapsed_per_frame] * len(batch_frames))

    processed = []
    for frame, result in zip(batch_frames, results):
        for box, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_name = result.names[int(cls_id)]
            label = f'{class_name} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
        processed.append(frame)
    return processed

#  main OpenCV loop  
buffer = []
running = True

while running and idx < N_FR:
    start_time = time.time()

    if play and len(buffer) < BATCH:
        for i in range(BATCH):
            next_idx = idx + i
            if next_idx < N_FR:
                buffer.append(frames[next_idx])
        processed = process_batch(buffer)
        buffer.clear()

        for frame in processed:
            idx += 1
            if idx >= N_FR:
                running = False
                break
            cv2.imshow('Demo', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                running = False
                break
    else:
        frame = frames[idx]
        processed = process_batch([frame])
        cv2.imshow('Demo', processed[0])

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            play = not play
        elif key in (81, ord('a')):
            idx = max(0, idx - 1)
        elif key in (83, ord('d')):
            idx = min(N_FR - 1, idx + 1)

    elapsed = time.time() - start_time
    delay = max(0, target_frame_time - elapsed)

cv2.destroyAllWindows()
 

#  report inference performance  
if inference_times:
    avg_time = np.mean(inference_times[100:700]) # Frame 100 to 700 to ignore initial spike
    print(f"\nAverage YOLOv9c inference time: {avg_time:.2f} ms per frame (batch size {BATCH})")

    # Save per-frame latencies to CSV
    csv_path = os.path.join(os.getcwd(), "inference_latency.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame Index", "Latency (ms)"])
        for i, latency in enumerate(inference_times, start=10):
            writer.writerow([i, latency])

    print(f"Saved per-frame latencies to '{csv_path}'")
else:
    print(" No inference timings recorded.")
