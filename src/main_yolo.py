# main_yolo.py

# ------- Imports -------
import cv2
import numpy as np
import time
from utils import read_video, reshape_frame_tracks, annotate_teams
from trackers.tracker import Tracker
from assign_team.assign_team import TeamAssigner
import csv, os
from assign_acquisition.assign_acquisition import BallAcquisition
from camera_motion.camera_motion import CameraMotionEstimator
from speed_distance.speed_distance import ViewTransformer, SpeedAndDistanceEstimator
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from analytics import PlayerPossessionTracker
from overlays.overlay_helper import (
    add_toggle_display,
    add_player_stats_overlay,
    add_frame_info_overlay,
    draw_with_toggles
)



# ------- Constants -------
VIDEO_PATH = 'data/test_clip_2_cut.mp4'
MODEL_PATH = 'models/128060ep.pt'
BATCH_SIZE = 1
FONT_CACHE = ImageFont.truetype("DejaVuSans.ttf", size=16)


print('Loading frames …')
frames = read_video(VIDEO_PATH)
cam_est = CameraMotionEstimator(frames[0])
view_transformer = ViewTransformer()
speed_estimator = SpeedAndDistanceEstimator(frame_rate=30, frame_window=5)


teamer = TeamAssigner()
N_FR = len(frames)



# ------- Global State & Player Tracking -------
idx = 0
play = False 
frame_tracks = []
frame_times = []  # Store individual frame processing times
fps_history = []  # Store FPS calculations
control_history = []
player_possession = {}  # {player_id: {'times_held': int, 'total_frames': int}}
selected_player = None  # Store currently selected player for stats display
target_frame_time = 1 / 30

player_assigner = BallAcquisition()
possession_tracker = PlayerPossessionTracker()

tracker = Tracker(
    player_model_path=MODEL_PATH,
    batch_size=BATCH_SIZE,
    player_conf=0.25,
    player_iou=0.45,
    ball_conf=0.3,
    ball_iou=0.5
)
 
TOGGLES = {
    'player_halos': True,
    'speed_distance': True,
    'camera_motion': True,
    'ball_possession': True,
    'frame_info': True,
}

 

# ------- UI Setup (OpenCV window and callback) -------

cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Demo', 1920, 1080)

def print_toggle_status():
    print("\n=== Toggle Status ===")
    print(f"Player Halos (T): {'ON' if TOGGLES['player_halos'] else 'OFF'}")
    print(f"Speed & Distance (S): {'ON' if TOGGLES['speed_distance'] else 'OFF'}")
    print(f"Camera Motion (C): {'ON' if TOGGLES['camera_motion'] else 'OFF'}")
    print(f"Ball Possession (P): {'ON' if TOGGLES['ball_possession'] else 'OFF'}")
    print(f"Frame Info (F): {'ON' if TOGGLES['frame_info'] else 'OFF'}")
    print("=" * 21)

def handle_key_press(key):
    global play, idx, running
    if key in (27, ord('q')):
        return False
    elif key == ord(' '):
        play = not play
        return True
    elif key in (81, ord('a')):
        idx = max(0, idx - 1)
        return True
    elif key in (83, ord('d')):
        idx = min(N_FR - 1, idx + 1)
        return True
    elif key == ord('b'):
        show_ball_statistics()
        return True
    elif key == ord('r'):
        reset_tracking()
        return True
    elif key == ord('t'):
        TOGGLES['player_halos'] = not TOGGLES['player_halos']
        print(f"Player Halos: {'ON' if TOGGLES['player_halos'] else 'OFF'}")
        return True
    elif key == ord('s'):
        TOGGLES['speed_distance'] = not TOGGLES['speed_distance']
        print(f"Speed & Distance: {'ON' if TOGGLES['speed_distance'] else 'OFF'}")
        return True
    elif key == ord('c'):
        TOGGLES['camera_motion'] = not TOGGLES['camera_motion']
        print(f"Camera Motion: {'ON' if TOGGLES['camera_motion'] else 'OFF'}")
        return True
    elif key == ord('p'):
        TOGGLES['ball_possession'] = not TOGGLES['ball_possession']
        print(f"Ball Possession: {'ON' if TOGGLES['ball_possession'] else 'OFF'}")
        return True
    elif key == ord('f'):
        TOGGLES['frame_info'] = not TOGGLES['frame_info']
        print(f"Frame Info: {'ON' if TOGGLES['frame_info'] else 'OFF'}")
        return True
    elif key == ord('h'):
        print('\nKeys: ←/→ frame | SPACE play/pause | Q/ESC quit | R reset | B show ball stats')
        print('Toggles: T=halos | S=speed&distance | C=camera motion | P=ball possession | F=frame info | H=help')
        print('Click a player to view possession stats')
        print_toggle_status()
        return True
    return True

print('\nKeys: ←/→ frame | SPACE play/pause | Q/ESC quit | R reset | B show ball stats')
print('Toggles: T=halos | S=speed&distance | C=camera motion | P=ball possession | F=frame info')
print('Click a player to view possession stats\n')

# ------- Mouse Interaction -------

def mouse_callback(event, x, y, flags, param):
    global selected_player
    if event == cv2.EVENT_LBUTTONDOWN and frame_tracks and idx < len(frame_tracks):
        # Check if click is within any player's bbox
        players = frame_tracks[idx].get("players", {})
        for pid, info in players.items():
            bbox = info.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_player = pid
                    print(f"Selected Player {pid}")
                    return
        selected_player = None  # Clear selection if no player clicked

cv2.setMouseCallback('Demo', mouse_callback)

# ------- Core Frame Processing -------

def process_single_frame(frame_idx): # TO BE REMOVED
    result = process_continuous_batch(frame_idx, 1)
    return result[0] if result else None

def process_continuous_batch(start_idx, batch_size):
    timing_log = {}

     

    global player_possession
    end_idx = min(start_idx + batch_size, len(frames))
    frames_slice = frames[start_idx:end_idx]
    if not frames_slice:
        return []

    # ------- TIMING - Fixed to track per-frame 
    start = time.time()
    tracked_batch = tracker.track_batch(frames_slice, start_frame_idx=start_idx)
    timing_log['inference'] = time.time() - start

 



    for i, tracked in enumerate(tracked_batch):
        frame_idx = start_idx + i
        while len(frame_tracks) <= frame_idx:
            frame_tracks.append({"players": {}, "referees": {}, "ball": {}})
        frame_tracks[frame_idx] = tracked

    # ------- Add position to tracks
    start = time.time()
    tracker.add_position_to_tracks(tracked_batch)
    timing_log['add_position'] = time.time() - start


    # ------- Camera estimation + adjustment
    start = time.time()
    for i in range(len(frames_slice)):
        frame_idx = start_idx + i
        dx, dy = cam_est.estimate(frames[frame_idx])
        cam_est.apply_adjustment(frame_tracks[frame_idx], frame_idx, dx, dy)
    timing_log['camera_motion'] = time.time() - start

    # ------- View transformation + speed/distance
    start = time.time()
    objectwise_tracks = reshape_frame_tracks(tracked_batch)
    view_transformer.add_transformed_position_to_tracks(objectwise_tracks)
    speed_estimator.add_speed_and_distance_to_tracks(objectwise_tracks)
    timing_log['transform+speed'] = time.time() - start

    processed = []
    team1_pct_list = []
    team2_pct_list = []

    # ------- Track previous possession state
    prev_player = None
    current_streak = 0

    # ------- Annotate and draw overlays
    start = time.time()

    for i, (frame, tracked) in enumerate(zip(frames_slice, tracked_batch)):
        frame_idx = start_idx + i
        annotate_teams(frame_idx, frames, frame_tracks, teamer)

        team = 0
        assigned_pid = -1
        if tracked["ball"]:
            ball_info = list(tracked["ball"].values())[0]
            assigned_pid = player_assigner.assign_ball_to_player(tracked["players"], ball_info["bbox"])
            if assigned_pid != -1:
                tracked["players"][assigned_pid]["has_ball"] = True
                team = tracked["players"].get(assigned_pid, {}).get("team", 0)
        
        # Update possession tracking
        possession_tracker.update_possession(assigned_pid, frame_idx)
        
        # Print possession stats every 30 frames (1 second at 30fps) or when possession changes
        if frame_idx % 30 == 0 or assigned_pid != getattr(possession_tracker, '_last_printed_player', None):
            possession_tracker.print_stats(frame_idx)
            possession_tracker._last_printed_player = assigned_pid
            
        control_history.append(team)

        valid = [t for t in control_history if t in (1, 2)]
        total_valid = len(valid)
        team1_pct = (valid.count(1) / total_valid * 100) if total_valid > 0 else 50.0
        team2_pct = (valid.count(2) / total_valid * 100) if total_valid > 0 else 50.0

        team1_pct_list.append(team1_pct)
        team2_pct_list.append(team2_pct)

        sliced = {k: [v] for k, v in tracked.items()}
        img = draw_with_toggles(
            frame, sliced, np.array(control_history[:frame_idx + 1]),
            team1_pct_list[:i + 1], team2_pct_list[:i + 1], TOGGLES, tracker, selected_player, player_possession
        )

        if TOGGLES['speed_distance']:
            objectwise_single_frame = {
                "players": [frame_tracks[frame_idx]["players"]],
                "referees": [frame_tracks[frame_idx]["referees"]],
                "ball": [frame_tracks[frame_idx]["ball"]],
            }
            img = speed_estimator.draw_speed_and_distance([img], objectwise_single_frame)[0]

        if TOGGLES['camera_motion']:
            dx, dy = cam_est.estimate(frames[frame_idx])
            img = cam_est.draw_camera_motion(img, dx, dy)

        if TOGGLES['frame_info']:
            img = add_frame_info_overlay(img, frame_idx, tracked, N_FR, possession_tracker)

        img = add_toggle_display(img, TOGGLES)
        processed.append(img)
    timing_log['overlay+drawing'] = time.time() - start


    print(f"\n⏱ Batch {start_idx}-{end_idx-1} Timing Breakdown:")
    for k, v in timing_log.items():
        print(f"  {k:<18}: {v*1000:.2f} ms total, {v/len(frames_slice)*1000:.2f} ms/frame")


    return processed











# ------- Stats and Utility Functions -------

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
    # Print player possession stats
    #print("\n=== Player Possession Statistics ===")
    for pid, stats in player_possession.items():
        team = frame_tracks[-1]["players"].get(pid, {}).get("team", "Unknown") if frame_tracks else "Unknown"
        print(f"Player {pid} (Team {team}): {stats['times_held']} times, {stats['total_frames']} frames ({stats['total_frames']/30:.2f}s)")
    print("=" * 33)

def reset_tracking():
    global frame_tracks, idx, player_possession, selected_player
    frame_tracks.clear() 
    player_possession.clear()
    selected_player = None
    tracker.ball_tracker = tracker.__class__.BallTracker(max_lost_frames=8, ball_radius=16)
    tracker.active_tracks.clear()
    tracker.stable_id_history.clear()
    tracker.next_stable_id = 1
    teamer.reset()
    idx = 0
    print("Tracking state reset")

# ------- MAIN EXECUTION LOOP -------
global_start_time = time.time()
total_frames_processed = 0


try:
    running = True
    print_toggle_status()
    while running:
        start_time = time.time()
        if play:
            batch_size = min(BATCH_SIZE, N_FR - idx)
            
            # Time the entire batch processing
            batch_start = time.time()
            processed_batch = process_continuous_batch(idx, batch_size)
            batch_end = time.time()
            
            total_batch_time_ms = (batch_end - batch_start) * 1000
            actual_fps = len(processed_batch) / ((batch_end - batch_start)) if (batch_end - batch_start) > 0 else 0
            
            #print(f"Batch {idx}-{idx+len(processed_batch)-1}: {total_batch_time_ms:.2f}ms total, {total_batch_time_ms/len(processed_batch):.2f}ms per frame, ACTUAL FPS: {actual_fps:.1f}")
            
            for i, processed in enumerate(processed_batch):
                cv2.imshow('Demo', processed)
                
                key = cv2.waitKey(1) & 0xFF
                if not handle_key_press(key):
                    running = False
                    break
                elif key == ord(' '):
                    idx += i + 1
                    break
                elif key == ord('r'):
                    break
            else:
                idx += len(processed_batch)
                if idx >= N_FR:
                    print("\n[INFO] End of video reached. Restarting demo...\n")
                    cv2.destroyAllWindows()
                    import sys
                    os.execv(sys.executable, ['python'] + sys.argv)
                total_frames_processed += len(processed_batch)
                
        else:
            # Use batch processing even when paused, but only show current frame
            batch_size = min(BATCH_SIZE, N_FR - idx)
            
            # Time the processing even when paused
            frame_start = time.time()
            processed_batch = process_continuous_batch(idx, batch_size)
            frame_end = time.time()
            
            processing_time_ms = (frame_end - frame_start) * 1000
            actual_fps = len(processed_batch) / (frame_end - frame_start) if (frame_end - frame_start) > 0 else 0
            
            print(f"Paused - Frame {idx}: {processing_time_ms:.2f}ms processing, ACTUAL FPS: {actual_fps:.1f}")
            
            # Show only the first frame (current frame)
            if processed_batch:
                cv2.imshow('Demo', processed_batch[0])
                total_frames_processed += len(processed_batch)

            
            key = cv2.waitKey(0) & 0xFF
            if not handle_key_press(key):
                running = False

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C pressed — Exiting cleanly...")

finally:
    player_assigner = BallAcquisition()
    team_ball_control = []
    last_valid_team = 0

    for frame_num, frame_data in enumerate(frame_tracks):
        players = frame_data.get("players", {})
        ball_info = frame_data.get("ball", {}).get(1, {})
        ball_bbox = ball_info.get("bbox", None)

        if ball_bbox and isinstance(ball_bbox, (list, tuple)) and len(ball_bbox) == 4:
            assigned_pid = player_assigner.assign_ball_to_player(players, ball_bbox)
        else:
            assigned_pid = -1
            #print(f"[Frame {frame_num}] ⚠️ Invalid/missing ball bbox → Neutral")

        if assigned_pid != -1:
            players[assigned_pid]["has_ball"] = True
            last_valid_team = players[assigned_pid].get("team", 0)
            team_ball_control.append(last_valid_team)
            #print(f"[Frame {frame_num}] ✅ Ball → Player {assigned_pid}, Team {last_valid_team}")
        else:
            team_ball_control.append(0)
            #print(f"[Frame {frame_num}] ⚪ No player assigned → Neutral")

    team_ball_control = np.array(team_ball_control)
    control_history[:] = team_ball_control.tolist()

    print("\n=== Team Ball Possession Summary ===")
    print("Unique values:", np.unique(team_ball_control))
    print("First 10:", team_ball_control[:10])
    print(f"Team 1: {np.sum(team_ball_control == 1)} frames")
    print(f"Team 2: {np.sum(team_ball_control == 2)} frames")
    print(f"Neutral: {np.sum(team_ball_control == 0)} frames")

    cv2.destroyAllWindows()
    show_ball_statistics()

    global_end_time = time.time()
    total_time_sec = global_end_time - global_start_time

    if total_frames_processed > 0:
        avg_latency_ms = (total_time_sec / total_frames_processed) * 1000
        avg_fps = total_frames_processed / total_time_sec
        print("\n=== Overall Performance Stats ===")
        print(f"Total frames processed: {total_frames_processed}")
        print(f"Total time taken: {total_time_sec:.2f} sec")
        print(f"Average latency per frame: {avg_latency_ms:.2f} ms")
        print(f"Average FPS: {avg_fps:.2f}")
    else:
        print("No frames were processed.")
