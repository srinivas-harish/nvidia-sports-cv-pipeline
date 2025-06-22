import cv2
import numpy as np
import time
from utils import read_video    
from trackers.tracker import Tracker
import csv, os

VIDEO_PATH = 'data/test_clip.mp4'
print('Loading frames …')
frames = read_video(VIDEO_PATH)

tracker = Tracker('models/128060ep.pt')  # TensorRT engine to be rendered

N_FR = len(frames)
idx = 0
play = False
inference_times = []
frame_tracks = []

BATCH_SIZE = 4
target_frame_time = 1 / 30

cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Demo', 1920, 1080)
print('\nKeys: ←/→ frame | SPACE play/pause | Q/ESC quit | R reset | B show ball stats\n')

def process_single_frame(frame_idx):
    if frame_idx >= len(frames): return None
    start = time.time()
    tracked = tracker.track_batch([frames[frame_idx]], start_frame_idx=frame_idx)
    end = time.time()
    latency = (end - start) * 1000
    inference_times.append(latency)

    while len(frame_tracks) <= frame_idx:
        frame_tracks.append({"players": {}, "referees": {}, "ball": {}})
    frame_tracks[frame_idx] = tracked[0]

    control_dummy = np.zeros(frame_idx + 1, dtype=int)
    sliced = {k: [v] for k, v in tracked[0].items()}
    img = tracker.draw_annotations([frames[frame_idx].copy()], sliced, control_dummy)[0]
    img = add_frame_info_overlay(img, frame_idx, tracked[0])
    return img

def process_continuous_batch(start_idx, batch_size):
    end_idx = min(start_idx + batch_size, len(frames))
    frames_slice = frames[start_idx:end_idx]
    if not frames_slice: return []

    start = time.time()
    tracked = tracker.track_batch(frames_slice, start_frame_idx=start_idx)
    end = time.time()
    latency = ((end - start) * 1000) / len(frames_slice)
    inference_times.extend([latency] * len(frames_slice))

    processed = []
    for i, (f, t) in enumerate(zip(frames_slice, tracked)):
        frame_idx = start_idx + i
        while len(frame_tracks) <= frame_idx:
            frame_tracks.append({"players": {}, "referees": {}, "ball": {}})
        frame_tracks[frame_idx] = t

        control_dummy = np.zeros(frame_idx + 1, dtype=int)
        sliced = {k: [v] for k, v in t.items()}
        img = tracker.draw_annotations([f.copy()], sliced, control_dummy)[0]
        img = add_frame_info_overlay(img, frame_idx, t)
        processed.append(img)
    return processed

def add_frame_info_overlay(img, frame_idx, tracks):
    h, w = img.shape[:2]
    info_text = f"Frame: {frame_idx:04d}/{N_FR-1:04d}"
    (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (15, 5), (25 + text_w, 35 + text_h), (0, 0, 0), -1)
    cv2.putText(img, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if tracks["ball"]:
        ball_info = list(tracks["ball"].values())[0]
        is_interpolated = ball_info.get("interpolated", False)
        bbox = ball_info["bbox"]
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        status_text = "Ball: INTERPOLATED" if is_interpolated else "Ball: DETECTED"
        color = (0, 165, 255) if is_interpolated else (0, 255, 0)
        bg_color = (0, 40, 80) if is_interpolated else (0, 40, 0)
        coord_text = f"({cx}, {cy})"

        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (15, 40), (25 + tw, 65 + th), bg_color, -1)
        cv2.putText(img, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_interpolated:
            cv2.rectangle(img, (10, 35), (30 + tw, 70 + th), color, 2)

        (cw, ch), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (15, 70), (25 + cw, 95 + ch), (0, 0, 0), -1)
        cv2.putText(img, coord_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        status_text = "Ball: NOT FOUND"
        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (15, 40), (25 + tw, 65 + th), (0, 0, 40), -1)
        cv2.putText(img, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img

def show_ball_statistics():
    if not frame_tracks:
        print("\nNo tracking data available yet")
        return
    total_frames = len([ft for ft in frame_tracks if ft])
    ball_detected = ball_interpolated = ball_missing = 0
    for ft in frame_tracks:
        if not ft: continue
        if ft["ball"]:
            ball_info = list(ft["ball"].values())[0]
            if ball_info.get("interpolated", False):
                ball_interpolated += 1
            else:
                ball_detected += 1
        else:
            ball_missing += 1
    print(f"\n=== Ball Tracking Statistics ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Ball detected: {ball_detected} ({ball_detected/total_frames*100:.1f}%)")
    print(f"Ball interpolated: {ball_interpolated} ({ball_interpolated/total_frames*100:.1f}%)")
    print(f"Ball missing: {ball_missing} ({ball_missing/total_frames*100:.1f}%)")
    print(f"Ball coverage: {(ball_detected + ball_interpolated)/total_frames*100:.1f}%")

    gaps, current_gap = [], 0
    for ft in frame_tracks:
        if not ft or not ft["ball"]:
            current_gap += 1
        else:
            if current_gap > 0:
                gaps.append(current_gap)
                current_gap = 0
    if gaps:
        print(f"Detection gaps: {len(gaps)} gaps, avg length: {np.mean(gaps):.1f}, max: {max(gaps)}")
    else:
        print("No detection gaps found")
    print("=" * 33)

def reset_tracking():
    global frame_tracks, inference_times, idx
    frame_tracks.clear()
    inference_times.clear()
    tracker.ball_tracker = tracker.__class__.BallTracker(max_lost_frames=8, ball_radius=16)
    tracker.active_tracks.clear()
    tracker.stable_id_history.clear()
    tracker.next_stable_id = 1
    idx = 0
    print("Tracking state reset")

#  main loop
try:
    running = True
    while running and idx < N_FR:
        start_time = time.time()

        if play:
            batch_size = min(BATCH_SIZE, N_FR - idx)
            processed_batch = process_continuous_batch(idx, batch_size)
            for i, processed in enumerate(processed_batch):
                cv2.imshow('Demo', processed)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    running = False
                    break
                elif key == ord(' '):
                    play = False
                    idx += i + 1
                    break
                elif key == ord('b'):
                    show_ball_statistics()
                elif key == ord('r'):
                    reset_tracking()
                    break
            else:
                idx += len(processed_batch)
        else:
            processed = process_single_frame(idx)
            if processed is None:
                break
            cv2.imshow('Demo', processed)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):
                running = False
            elif key == ord(' '):
                play = True
            elif key in (81, ord('a')):
                idx = max(0, idx - 1)
            elif key in (83, ord('d')):
                idx = min(N_FR - 1, idx + 1)
            elif key == ord('b'):
                show_ball_statistics()
            elif key == ord('r'):
                reset_tracking()

        if play:
            elapsed = time.time() - start_time
            delay = max(0, target_frame_time - elapsed)
            time.sleep(delay)

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C pressed — Exiting cleanly...")

finally:
    cv2.destroyAllWindows()
    show_ball_statistics()
    if inference_times:
        avg_time = np.mean(inference_times[100:700]) if len(inference_times) > 700 else np.mean(inference_times)
        print(f"\n***Average YOLOv9c+Tracking time: {avg_time:.2f} ms per frame")
        csv_path = os.path.join(os.getcwd(), "inference_latency.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame Index", "Latency (ms)", "Ball Status"])
            for i, latency in enumerate(inference_times):
                ball_status = "unknown"
                if i < len(frame_tracks) and frame_tracks[i] and frame_tracks[i]["ball"]:
                    ball_info = list(frame_tracks[i]["ball"].values())[0]
                    ball_status = "interpolated" if ball_info.get("interpolated", False) else "detected"
                elif i < len(frame_tracks) and frame_tracks[i]:
                    ball_status = "missing"
                writer.writerow([i, latency, ball_status])
