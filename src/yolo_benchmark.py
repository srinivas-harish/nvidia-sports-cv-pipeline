import cv2
import numpy as np
import time
from utils import read_video    
from trackers.tracker import Tracker
import csv, os

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

# Configurable batch size
BATCH_SIZE = 4  # Use 4 for playback, 1 for manual navigation

cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Demo', 1920, 1080)
print('\nKeys: ←/→ frame | SPACE play/pause | Q/ESC quit\n')

target_frame_time = 1 / 30  # 30 FPS

def process_batch(frames_slice):
    """Process a batch of frames using track_batch."""
    if not frames_slice:
        return []
    
    start = time.time()
    tracked = tracker.track_batch(frames_slice)  # Use track_batch for all cases
    end = time.time()

    # Record per-frame latency
    latency = ((end - start) * 1000) / len(frames_slice)  # ms per frame
    inference_times.extend([latency] * len(frames_slice))

    # Prepare annotations
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

    # Determine batch size based on mode
    batch_size = BATCH_SIZE if play else 1
    frames_batch = frames[idx:idx + batch_size]
    if not frames_batch:
        break

    # Process the batch (1 frame for manual navigation, BATCH_SIZE for playback)
    processed_batch = process_batch(frames_batch)
    for processed in processed_batch:
        cv2.imshow('Demo', processed)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # Quit
            running = False
            break
        elif not play:  # Handle key inputs only when paused
            if key == ord(' '):  # Toggle play/pause
                play = not play
            elif key in (81, ord('a')):  # Previous frame
                idx = max(0, idx - 1)
                break  # Exit batch to process single frame
            elif key in (83, ord('d')):  # Next frame
                idx = min(N_FR - 1, idx + 1)
                break  # Exit batch to process single frame
    
    idx += len(frames_batch)

    # Enforce 30 FPS
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