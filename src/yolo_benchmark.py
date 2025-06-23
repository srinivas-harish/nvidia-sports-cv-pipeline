
import cv2
import numpy as np
import time
from utils import read_video    
from trackers.tracker import Tracker
from assign_team.assign_team import TeamAssigner
import csv, os
from assign_acquisition.assign_acquisition import BallAcquisition
from camera_motion.camera_motion import CameraMotionEstimator

player_assigner = BallAcquisition()
control_history = []

VIDEO_PATH = 'data/test_clip.mp4'
#VIDEO_PATH = 'data/test_clip_2_cut.mp4'

print('Loading frames …')
frames = read_video(VIDEO_PATH)
cam_est = CameraMotionEstimator(frames[0])


tracker = Tracker('models/128060ep.pt')  # TensorRT engine to be rendered
teamer = TeamAssigner()

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

def _annotate_teams(frame_idx):
    if frame_idx == 0 and frame_tracks[0]["players"]:
        teamer.fit(frames[0], frame_tracks[0]["players"])

    if teamer.kmeans is None:
        return

    for pid, info in frame_tracks[frame_idx]["players"].items():
        # Predict once per ID
        if pid not in teamer.player_team_cache:
            team = teamer.predict(frames[frame_idx], info["bbox"], pid)
            teamer.player_team_cache[pid] = team
        else:
            team = teamer.player_team_cache[pid]

        info["team"] = team
        info["team_color"] = teamer.color_for_team(team)


def process_single_frame(frame_idx):
    result = process_continuous_batch(frame_idx, 1)
    return result[0] if result else None

def process_continuous_batch(start_idx, batch_size):
    end_idx = min(start_idx + batch_size, len(frames))
    frames_slice = frames[start_idx:end_idx]
    if not frames_slice:
        return []

    start = time.time()
    tracked_batch = tracker.track_batch(frames_slice, start_frame_idx=start_idx)
    end = time.time()
    latency = ((end - start) * 1000) / len(frames_slice)
    inference_times.extend([latency] * len(frames_slice))

    processed = []
    team1_pct_list = []
    team2_pct_list = []

    for i, (frame, tracked) in enumerate(zip(frames_slice, tracked_batch)):
        frame_idx = start_idx + i
        while len(frame_tracks) <= frame_idx:
            frame_tracks.append({"players": {}, "referees": {}, "ball": {}})
        frame_tracks[frame_idx] = tracked

        tracker.add_position_to_tracks(frame_tracks)

        # Real-time camera motion estimation and adjustment
        dx, dy = cam_est.estimate(frames[frame_idx])
        cam_est.apply_adjustment(frame_tracks[frame_idx], frame_idx, dx, dy)





        _annotate_teams(frame_idx)

        # Assign ball and update control history
        team = 0
        if tracked["ball"]:
            ball_info = list(tracked["ball"].values())[0]
            assigned_pid = player_assigner.assign_ball_to_player(tracked["players"], ball_info["bbox"])
            if assigned_pid != -1:
                tracked["players"][assigned_pid]["has_ball"] = True
                team = tracked["players"].get(assigned_pid, {}).get("team", 0)
        control_history.append(team)

        # Compute possession percentages
        valid = [t for t in control_history if t in (1, 2)]
        total_valid = len(valid)
        if total_valid > 0:
            team1_pct = (valid.count(1) / total_valid) * 100
            team2_pct = (valid.count(2) / total_valid) * 100
            #print(f"[Frame {frame_idx}] ⚽ Possession → Team 1: {team1_pct:.1f}%, Team 2: {team2_pct:.1f}%")
        else:
            team1_pct = team2_pct = 50.0
            #print(f"[Frame {frame_idx}] ⚽ Possession → No valid team frames yet")

        team1_pct_list.append(team1_pct)
        team2_pct_list.append(team2_pct)

        sliced = {k: [v] for k, v in tracked.items()}
        img = tracker.draw_annotations(
            [frame.copy()],
            sliced,
            np.array(control_history[:frame_idx + 1]),
            team1_pct_list[:i + 1],
            team2_pct_list[:i + 1]
        )[0]


        # DRAW CAMERA MOTION
        img = cam_est.draw_camera_motion(img, dx, dy)



        img = add_frame_info_overlay(img, frame_idx, tracked)
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
    team_ball_control.clear()

    inference_times.clear()
    tracker.ball_tracker = tracker.__class__.BallTracker(max_lost_frames=8, ball_radius=16)
    tracker.active_tracks.clear()
    tracker.stable_id_history.clear()
    tracker.next_stable_id = 1
    teamer.reset()
    idx = 0
    print("Tracking state reset")









####



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
    # Final Ball Possession Assignment
    player_assigner = BallAcquisition()
    team_ball_control = []
    last_valid_team = 0

    for frame_num, frame_data in enumerate(frame_tracks):
        players = frame_data.get("players", {})
        ball_info = frame_data.get("ball", {}).get(1, {})
        ball_bbox = ball_info.get("bbox", None)

        if ball_bbox and isinstance(ball_bbox, (list, tuple)) and len(ball_bbox) == 4:
            assigned_player = player_assigner.assign_ball_to_player(players, ball_bbox)
        else:
            assigned_player = -1
            print(f"[Frame {frame_num}] ⚠️  Invalid/missing ball bbox → Neutral")

        if assigned_player != -1:
            players[assigned_player]["has_ball"] = True
            last_valid_team = players[assigned_player].get("team", 0)
            team_ball_control.append(last_valid_team)
            print(f"[Frame {frame_num}] ✅ Ball → Player {assigned_player}, Team {last_valid_team}")
        else:
            team_ball_control.append(0)
            print(f"[Frame {frame_num}] ⚪ No player assigned → Neutral")

    team_ball_control = np.array(team_ball_control)
 
    control_history[:] = team_ball_control.tolist()

    # Summary
    print("\n=== Team Ball Possession Summary ===")
    print("Unique values:", np.unique(team_ball_control))
    print("First 10:", team_ball_control[:10])
    print(f"Team 1: {np.sum(team_ball_control == 1)} frames")
    print(f"Team 2: {np.sum(team_ball_control == 2)} frames")
    print(f"Neutral: {np.sum(team_ball_control == 0)} frames")

    # Proceed with original clean-up
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
