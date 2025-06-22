import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import read_video    
from trackers.tracker import Tracker
import csv, os, pickle

VIDEO_PATH = 'data/test_clip.mp4'
print('Loading frames …')
frames = read_video(VIDEO_PATH)

tracker = Tracker('models/best.engine')  # TensorRT engine

# Viewer state
N_FR = len(frames)
idx = 0
play = False
inference_times = []
frame_tracks = []

BATCH = 4

cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Demo', 1920, 1080)
print('\nKeys: ←/→ frame | SPACE play/pause | Q/ESC quit\n')

target_frame_time = 1 / 30  # 30 FPS

def process_frame(frame):
    start = time.time()
    track_result = tracker.track_frame(frame)
    end = time.time()

    inference_times.append((end - start) * 1000)  # ms
    frame_tracks.append(track_result)

    sliced_tracks = {k: [v] for k, v in track_result.items()}
    control_dummy = np.zeros(len(frame_tracks), dtype=int)
    annotated = tracker.draw_annotations([frame.copy()], sliced_tracks, control_dummy)[0]
    return annotated

def process_batch(frames_slice):
    start = time.time()
    tracked = tracker.track_batch(frames_slice)
    end = time.time()

    latency = ((end - start) * 1000) / len(frames_slice)
    inference_times.extend([latency] * len(frames_slice))

    control_dummy = np.zeros(len(frame_tracks) + len(frames_slice), dtype=int)
    processed = []

    for i, (f, t) in enumerate(zip(frames_slice, tracked)):
        frame_tracks.append(t)
        sliced = {k: [v] for k, v in t.items()}
        img = tracker.draw_annotations([f.copy()], sliced, control_dummy[:len(frame_tracks)])[0]
        processed.append(img)

    return processed

running = True
while running and idx < N_FR:
    start_time = time.time()

    if play:
        # Grab a batch of up to BATCH frames
        frames_batch = frames[idx:idx+BATCH]
        if not frames_batch:
            break

        processed_batch = process_batch(frames_batch)
        for processed in processed_batch:
            cv2.imshow('Demo', processed)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                running = False
                break
        idx += len(frames_batch)

    else:
        frame = frames[idx]
        processed = process_frame(frame)
        cv2.imshow('Demo', processed)

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
    time.sleep(delay)

cv2.destroyAllWindows()

# Save latency report
if inference_times:
    avg_time = np.mean(inference_times[100:700]) if len(inference_times) > 700 else np.mean(inference_times)
    print(f"\n***Average YOLOv9c+Tracking time: {avg_time:.2f} ms per frame")

    csv_path = os.path.join(os.getcwd(), "inference_latency.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame Index", "Latency (ms)"])
        for i, latency in enumerate(inference_times, start=0):
            writer.writerow([i, latency])

    print(f"Saved per-frame latencies to '{csv_path}'")