
import cv2
import numpy as np
import time
from utils import read_video    
from trackers.tracker import Tracker
from assign_team.assign_team import TeamAssigner
import csv, os
from assign_acquisition.assign_acquisition import BallAcquisition
from camera_motion.camera_motion import CameraMotionEstimator
from speed_distance.speed_distance import ViewTransformer, SpeedAndDistanceEstimator
from PIL import Image, ImageDraw, ImageFont, ImageFilter





def add_toggle_display(img, TOGGLES):
    h, w = img.shape[:2]
    toggle_lines = [
        "TOGGLES:",
        f"T - Halos: {'ON' if TOGGLES['player_halos'] else 'OFF'}",
        f"S - Speed: {'ON' if TOGGLES['speed_distance'] else 'OFF'}",
        f"C - Camera: {'ON' if TOGGLES['camera_motion'] else 'OFF'}",
        f"P - Possession: {'ON' if TOGGLES['ball_possession'] else 'OFF'}",
        f"F - Frame Info: {'ON' if TOGGLES['frame_info'] else 'OFF'}"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    padding = 5

    max_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in toggle_lines)
    bg_height = len(toggle_lines) * line_height + 2 * padding
    bg_width = max_width + (2 * padding)
    start_y = h - bg_height - 10
    start_x = 10

    overlay = img.copy()
    cv2.rectangle(overlay, (start_x, start_y), (start_x + bg_width, start_y + bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    for i, line in enumerate(toggle_lines):
        y_pos = start_y + padding + (i + 1) * line_height - 5
        color = (255, 255, 255) if i == 0 else (200, 200, 200)
        cv2.putText(img, line, (start_x + padding, y_pos), font, font_scale, color, thickness)

    return img




def add_player_stats_overlay(img, player_id, player_possession):
    """Add possession stats overlay for selected player."""
    if player_id is None or player_id not in player_possession:
        return img

    h, w = img.shape[:2]
    stats = player_possession[player_id]
    team = frame_tracks[idx]["players"].get(player_id, {}).get("team", "Unknown")

    # Define text content
    stats_text = [
        f"Player {player_id} (Team {team})",
        f"Times Held: {stats['times_held']}",
        f"Total Frames: {stats['total_frames']}",
        f"Total Time: {(stats['total_frames']/30):.2f}s"
    ]

    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()

    # Create temporary Pillow image to calculate text sizes
    temp_img = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(temp_img)

    # Calculate text bounding boxes
    text_bboxes = [draw.textbbox((0, 0), line, font=font) for line in stats_text]
    text_widths = [bbox[2] - bbox[0] for bbox in text_bboxes]
    text_heights = [bbox[3] - bbox[1] for bbox in text_bboxes]
    max_width = max(text_widths) + 20
    total_height = sum(text_heights) + 40

    # Position overlay in top-left (below toggle display)
    overlay_x = 10
    overlay_y = 150  # Adjust to avoid overlap with toggle display

    # Convert OpenCV image to Pillow
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Crop region for glass effect
    crop_box = (overlay_x, overlay_y, overlay_x + max_width, overlay_y + total_height)
    crop_box = (
        max(0, crop_box[0]),
        max(0, crop_box[1]),
        min(w, crop_box[2]),
        min(h, crop_box[3])
    )
    overlay_region = pil_img.crop(crop_box)

    # Apply blur for glass effect
    overlay_region = overlay_region.filter(ImageFilter.GaussianBlur(radius=5))

    # Create overlay image
    overlay = Image.new('RGBA', (max_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw semi-transparent background
    draw.rectangle((0, 0, max_width, total_height), fill=(255, 255, 255, 100))

    # Paste blurred region
    overlay.paste(overlay_region, (0, 0))

    # Draw text
    y_offset = 10
    for line in stats_text:
        draw.text((10, y_offset), line, font=font, fill=(255, 255, 255, 255))
        y_offset += text_heights[stats_text.index(line)] + 10

    # Paste overlay onto main image
    pil_img.paste(overlay, (overlay_x, overlay_y), overlay)

    # Convert back to OpenCV
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def draw_with_toggles(frame, tracked_data, control_hist, team1_pct, team2_pct, TOGGLES, tracker, selected_player, player_possession):
    img = frame.copy()
    modified_tracked = {}
    for key, value in tracked_data.items():
        if key == 'players' and isinstance(value, list) and len(value) > 0:
            modified_players = {}
            for pid, pinfo in value[0].items():
                modified_pinfo = pinfo.copy()
                if not TOGGLES['player_halos']:
                    modified_pinfo['team'] = 999
                    modified_pinfo['team_color'] = (128, 128, 128)
                if not TOGGLES['ball_possession']:
                    modified_pinfo.pop('has_ball', None)
                if TOGGLES['ball_possession']:
                    #img = add_possession_overlay(img, possession_tracker, 0)  # frame_idx will be passed properly
                    pass
                modified_players[pid] = modified_pinfo
            modified_tracked[key] = [modified_players]
        else:
            modified_tracked[key] = value

    if TOGGLES['ball_possession'] and len(team1_pct) > 0 and len(team2_pct) > 0:
        safe_control_hist = control_hist if len(control_hist) > 0 else np.array([0])
        img = tracker.draw_annotations(
            [img], modified_tracked, safe_control_hist, team1_pct, team2_pct
        )[0]
    else:
        img = tracker.draw_annotations(
            [img], modified_tracked, np.array([0]), [50.0], [50.0]
        )[0]

    # Add player stats overlay if a player is selected
    img = add_player_stats_overlay(img, selected_player, player_possession)
    return img



def add_frame_info_overlay(img, frame_idx, tracks, N_FR, possession_tracker):
    h, w = img.shape[:2]
    
    # Frame info
    info_text = f"Frame: {frame_idx:04d}/{N_FR-1:04d}"
    
    # Ball info
    if tracks["ball"]:
        ball_info = list(tracks["ball"].values())[0]
        is_interpolated = ball_info.get("interpolated", False)
        bbox = ball_info["bbox"]
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        status_text = "Ball: INTERPOLATED" if is_interpolated else "Ball: DETECTED"
        coord_text = f"({cx}, {cy})"
    else:
        status_text = "Ball: NOT FOUND"
        coord_text = ""

    # Possession stats
    possession_lines = []
    if possession_tracker.player_stats:
        possession_lines.append("POSSESSION STATS:")
        sorted_players = sorted(possession_tracker.player_stats.items(), 
                              key=lambda x: x[1]['total_frames'], 
                              reverse=True)[:3]
        for player_id, stats in sorted_players:
            current_indicator = " [CURR]" if player_id == possession_tracker.current_possessor else ""
            possession_lines.append(f"Player {player_id}{current_indicator}: "
                                  f"{stats['count']} poss, {stats['total_frames']}f")
    
    # Load font (increased size for better readability)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
    except IOError:
        font = ImageFont.load_default(size=20)

    # Calculate text dimensions for vertical sizing only
    temp_img = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(temp_img)
    
    all_text_lines = [info_text, status_text]
    if coord_text:
        all_text_lines.append(coord_text)
    all_text_lines.extend(possession_lines)
    
    text_bboxes = [draw.textbbox((0, 0), line, font=font) for line in all_text_lines]
    text_heights = [bbox[3] - bbox[1] for bbox in text_bboxes]
    total_height = sum(text_heights) + (len(all_text_lines) * 10) + 20  # More spacing

    # Set constant horizontal width
    fixed_width = 330  # Constant 330px width
    overlay_x = w - fixed_width - 15  # Maintain right margin
    overlay_y = 15

    # Create blurred background
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    crop_box = (
        max(0, overlay_x),
        max(0, overlay_y),
        min(w, overlay_x + fixed_width),
        min(h, overlay_y + total_height)
    )
    overlay_region = pil_img.crop(crop_box)
    overlay_region = overlay_region.filter(ImageFilter.GaussianBlur(radius=6))  # Smoother blur

    # Create semi-transparent overlay with subtle border
    overlay = Image.new('RGBA', (fixed_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, fixed_width, total_height), fill=(30, 30, 30, 180))  # Darker background
    draw.rectangle((0, 0, fixed_width, total_height), outline=(200, 200, 200, 200), width=2)  # Subtle border

    overlay.paste(overlay_region, (0, 0), mask=overlay)

    # Draw text with consistent white color (except header)
    y_offset = 15  # Increased top padding
    
    # Frame info
    draw.text((15, y_offset), info_text, font=font, fill=(255, 255, 255, 255))
    y_offset += text_heights[0] + 10
    
    # Ball status
    draw.text((15, y_offset), status_text, font=font, fill=(255, 255, 255, 255))
    y_offset += text_heights[1] + 10
    
    # Ball coordinates
    if coord_text:
        draw.text((15, y_offset), coord_text, font=font, fill=(255, 255, 255, 255))
        y_offset += text_heights[2] + 10
    
    # Possession stats
    for i, line in enumerate(possession_lines):
        line_idx = len([info_text, status_text]) + (1 if coord_text else 0) + i
        if i == 0:  # Header
            draw.text((15, y_offset), line, font=font, fill=(0, 255, 255, 255))  # Cyan header
        else:  # Player stats
            draw.text((15, y_offset), line, font=font, fill=(255, 255, 255, 255))
        y_offset += text_heights[line_idx] + 10

    pil_img.paste(overlay, (overlay_x, overlay_y), overlay)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr