import cv2
import numpy as np
import time
from utils import read_video    
from trackers.tracker import Tracker
import csv, os

VIDEO_PATH = 'data/test_clip.mp4'
print('Loading frames …')
frames = read_video(VIDEO_PATH)

tracker = Tracker('models/best_backup_epoch32.pt')  # TensorRT engine to be rendered

# Viewer state
N_FR = len(frames)
idx = 0
play = False
inference_times = []
frame_tracks = []

BATCH_SIZE = 4 # based on TensorRT export size

cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Demo', 1920, 1080)
print('\nKeys: ←/→ frame | SPACE play/pause | Q/ESC quit\n')

target_frame_time = 1 / 30  # not yet used

def process_single_frame(frame_idx):
    """Process a single frame to maintain tracking continuity."""
    if frame_idx >= len(frames):
        return None
    
    start = time.time() 

    tracked = tracker.track_batch([frames[frame_idx]], start_frame_idx=frame_idx)
    end = time.time()
    
    # Record latency
    latency = (end - start) * 1000  # ms
    inference_times.append(latency)
    
    # Store tracks
    frame_tracks.append(tracked[0])
    
    #  annotation
    control_dummy = np.zeros(len(frame_tracks), dtype=int)
    sliced = {k: [v] for k, v in tracked[0].items()}
    img = tracker.draw_annotations([frames[frame_idx].copy()], sliced, control_dummy)[0]
    
    return img

def process_continuous_batch(start_idx, batch_size):
    """Process consecutive frames for smooth playback."""
    end_idx = min(start_idx + batch_size, len(frames))
    frames_slice = frames[start_idx:end_idx]
    
    if not frames_slice:
        return []
    
    start = time.time()
     
    tracked = tracker.track_batch(frames_slice, start_frame_idx=start_idx)
    end = time.time()
    
    # per-frame latency
    latency = ((end - start) * 1000) / len(frames_slice)  # ms per frame
    inference_times.extend([latency] * len(frames_slice))
    
    #  annotations
    processed = []
    for i, (f, t) in enumerate(zip(frames_slice, tracked)):
        frame_tracks.append(t)
        control_dummy = np.zeros(len(frame_tracks), dtype=int)
        sliced = {k: [v] for k, v in t.items()}
        img = tracker.draw_annotations([f.copy()], sliced, control_dummy)[0]
        processed.append(img)
    
    return processed

running = True
while running and idx < N_FR:
    start_time = time.time()

    if play:
        # Continuous playback mode - process batch
        batch_size = min(BATCH_SIZE, N_FR - idx)
        processed_batch = process_continuous_batch(idx, batch_size)
        
        for i, processed in enumerate(processed_batch):
            cv2.imshow('Demo', processed)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # Quit
                running = False
                break
            elif key == ord(' '):  # Toggle play/pause
                play = False
                idx += i + 1  # Update to current position
                break
        else:
            idx += len(processed_batch)
    else:
        # Manual navigation mode - process single frame
        processed = process_single_frame(idx)
        if processed is None:
            break
            
        cv2.imshow('Demo', processed)
        key = cv2.waitKey(0) & 0xFF  # Wait for key press
        
        if key in (27, ord('q')):  # Quit
            running = False
        elif key == ord(' '):  # Toggle play/pause
            play = True
        elif key in (81, ord('a')):  # Previous frame
            idx = max(0, idx - 1)
        elif key in (83, ord('d')):  # Next frame
            idx = min(N_FR - 1, idx + 1)

    # Enforce 30 FPS only during playback, not yet implemented for testing*
    if play:
        elapsed = time.time() - start_time
        delay = max(0, target_frame_time - elapsed)
        time.sleep(delay)

cv2.destroyAllWindows()

# latency report
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