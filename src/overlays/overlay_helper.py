import cv2
import numpy as np
import time
from utils import read_video     
from assign_team.assign_team import TeamAssigner
import csv, os
from assign_acquisition.assign_acquisition import BallAcquisition 
from speed_distance.speed_distance import ViewTransformer, SpeedAndDistanceEstimator
from PIL import Image, ImageDraw, ImageFont, ImageFilter
  


# Global font cache — initialized once
try:
    FONT_CACHE = ImageFont.truetype("DejaVuSans.ttf", size=20)
except IOError:
    FONT_CACHE = ImageFont.load_default()

def build_toggle_overlay_once(TOGGLES, shape):
    h, w = shape[:2]
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

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (start_x, start_y), (start_x + bg_width, start_y + bg_height), (0, 0, 0), -1)

    for i, line in enumerate(toggle_lines):
        y_pos = start_y + padding + (i + 1) * line_height - 5
        color = (255, 255, 255) if i == 0 else (200, 200, 200)
        cv2.putText(overlay, line, (start_x + padding, y_pos), font, font_scale, color, thickness)

    return overlay

def apply_toggle_overlay(img, toggle_overlay):
    if toggle_overlay is None:
        return img
    alpha = 0.6
    return cv2.addWeighted(toggle_overlay, alpha, img, 1 - alpha, 0)

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
    if player_id is None or player_id not in player_possession:
        return img

    global FONT_CACHE
    font = FONT_CACHE

    h, w = img.shape[:2]
    stats = player_possession[player_id]
    team = stats.get("team", "Unknown")

    stats_text = [
        f"Player {player_id} (Team {team})",
        f"Times Held: {stats['times_held']}",
        f"Total Frames: {stats['total_frames']}",
        f"Total Time: {(stats['total_frames']/30):.2f}s"
    ]

    temp_img = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(temp_img)
    text_bboxes = [draw.textbbox((0, 0), line, font=font) for line in stats_text]
    text_widths = [bbox[2] - bbox[0] for bbox in text_bboxes]
    text_heights = [bbox[3] - bbox[1] for bbox in text_bboxes]
    max_width = max(text_widths) + 20
    total_height = sum(text_heights) + 40

    overlay_x = 10
    overlay_y = 150

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    crop_box = (
        max(0, overlay_x),
        max(0, overlay_y),
        min(w, overlay_x + max_width),
        min(h, overlay_y + total_height)
    )
    overlay_region = pil_img.crop(crop_box)
    overlay_region = overlay_region.filter(ImageFilter.GaussianBlur(radius=5))

    overlay = Image.new('RGBA', (max_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, max_width, total_height), fill=(255, 255, 255, 100))
    overlay.paste(overlay_region, (0, 0))

    y_offset = 10
    for i, line in enumerate(stats_text):
        draw.text((10, y_offset), line, font=font, fill=(255, 255, 255, 255))
        y_offset += text_heights[i] + 10

    pil_img.paste(overlay, (overlay_x, overlay_y), overlay)
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

    img = add_player_stats_overlay(img, selected_player, player_possession)
    return img















def add_frame_info_overlay(img, frame_idx, tracks, N_FR, possession_tracker):
    global FONT_CACHE
    font = FONT_CACHE
    h, w = img.shape[:2]

    info_text = f"Frame: {frame_idx:04d}/{N_FR-1:04d}"

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

    possession_lines = []
    if possession_tracker.player_stats:
        possession_lines.append("POSSESSION STATS:")
        sorted_players = sorted(
            possession_tracker.player_stats.items(),
            key=lambda x: x[1]['total_frames'],
            reverse=True
        )[:3]
        for player_id, stats in sorted_players:
            current_indicator = " [CURR]" if player_id == possession_tracker.current_possessor else ""
            possession_lines.append(f"Player {player_id}{current_indicator}: "
                                    f"{stats['count']} poss, {stats['total_frames']}f")

    temp_img = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(temp_img)
    all_text_lines = [info_text, status_text]
    if coord_text:
        all_text_lines.append(coord_text)
    all_text_lines.extend(possession_lines)
    text_bboxes = [draw.textbbox((0, 0), line, font=font) for line in all_text_lines]
    text_heights = [bbox[3] - bbox[1] for bbox in text_bboxes]
    total_height = sum(text_heights) + (len(all_text_lines) * 10) + 20

    fixed_width = 330
    overlay_x = w - fixed_width - 15
    overlay_y = 15

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    crop_box = (
        max(0, overlay_x),
        max(0, overlay_y),
        min(w, overlay_x + fixed_width),
        min(h, overlay_y + total_height)
    )
    overlay_region = pil_img.crop(crop_box)
    overlay_region = overlay_region.filter(ImageFilter.GaussianBlur(radius=6))

    overlay = Image.new('RGBA', (fixed_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, fixed_width, total_height), fill=(30, 30, 30, 180))
    draw.rectangle((0, 0, fixed_width, total_height), outline=(200, 200, 200, 200), width=2)
    overlay.paste(overlay_region, (0, 0), mask=overlay)

    y_offset = 15
    for i, line in enumerate(all_text_lines):
        fill = (0, 255, 255, 255) if line.startswith("POSSESSION STATS") else (255, 255, 255, 255)
        draw.text((15, y_offset), line, font=font, fill=fill)
        y_offset += text_heights[i] + 10

    pil_img.paste(overlay, (overlay_x, overlay_y), overlay)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


    def draw_camera_motion(self, frame, dx, dy):
        """Optimized widget drawing"""
        img = frame.copy()
        
        # Apply smoothing
        self.smoothing_buffer.append((dx, dy))
        if len(self.smoothing_buffer) >= 2:
            dx = float(np.mean([v[0] for v in self.smoothing_buffer]))
            dy = float(np.mean([v[1] for v in self.smoothing_buffer]))

        mag = float(np.hypot(dx, dy))

        # Extract and blur widget area
        panel = img[self.ORIGIN_Y:self.ORIGIN_Y+self.WIDGET_H, 
                   self.ORIGIN_X:self.ORIGIN_X+self.WIDGET_W].copy()
        
        # Faster blur with smaller kernel
        blur = cv2.GaussianBlur(panel, (21, 21), 10)
        dark = cv2.addWeighted(blur, 0.85, np.zeros_like(panel), 0.15, 0)
        
        # Copy to pre-allocated working array
        self.widget_working[:] = dark

        # Draw ring
        cv2.circle(self.widget_working, self.widget_center, self.RING_R, 
                  (60, 60, 60), self.RING_T, cv2.LINE_AA)

        # Color based on magnitude
        if mag < 5:
            col = (50, 220, 50)
        elif mag < 12:
            col = (50, 220, 220)
        else:
            col = (50, 50, 255)

        # Draw sweep
        sweep = np.clip(mag / self.max_display_magnitude * 360, 0, 360)
        if sweep > 5:  #  if significant
            cv2.ellipse(self.widget_working, self.widget_center, 
                       (self.RING_R, self.RING_R), 0, -90, -90 + sweep, 
                       col, self.RING_T, cv2.LINE_AA)

        #   if significant
        if mag > 1.0:
            u = np.array([-dx, -dy]) / (mag + 1e-6)
            tip = (int(self.widget_center[0] + u[0] * (self.RING_R - 15)),
                   int(self.widget_center[1] + u[1] * (self.RING_R - 15)))
            cv2.arrowedLine(self.widget_working, self.widget_center, tip, col, 2,
                           tipLength=0.3, line_type=cv2.LINE_AA)

  
        cv2.circle(self.widget_working, self.widget_center, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Add text every frame  
        pil = Image.fromarray(cv2.cvtColor(self.widget_working, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((8, 6), "Camera Motion", font=self.font_big, fill=(255, 255, 255, 255))

        txt = f"{mag:4.1f} px/f"
        try:
            bbox = draw.textbbox((0, 0), txt, font=self.font_small)
            tw = bbox[2] - bbox[0]
        except:
            tw = draw.textsize(txt, font=self.font_small)[0]

        draw.text(((self.WIDGET_W - tw) // 2, self.WIDGET_H - 25),
                 txt, font=self.font_small, fill=(255, 255, 255, 255))

        self.widget_working[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Copy widget back to frame
        img[self.ORIGIN_Y:self.ORIGIN_Y+self.WIDGET_H, 
            self.ORIGIN_X:self.ORIGIN_X+self.WIDGET_W] = self.widget_working

        return img

    def apply_adjustment(self, tracks: dict, frame_idx: int, dx: float, dy: float):
        """Apply motion adjustment to tracks"""
        for obj, obj_tracks in tracks.items():
            for tid, info in obj_tracks.items():
                x, y = info.get("position", (0, 0))
                info["position_adjusted"] = (x - dx, y - dy)




def draw_camera_motion_widget(
    frame,
    dx,
    dy,
    widget_working,
    widget_center,
    ORIGIN_X,
    ORIGIN_Y,
    WIDGET_W,
    WIDGET_H,
    RING_R,
    RING_T,
    max_display_magnitude,
    font_big,
    font_small,
    smoothing_buffer
):
    """Standalone camera motion HUD drawing function."""
    img = frame.copy()
    
    # Apply smoothing
    smoothing_buffer.append((dx, dy))
    if len(smoothing_buffer) >= 2:
        dx = float(np.mean([v[0] for v in smoothing_buffer]))
        dy = float(np.mean([v[1] for v in smoothing_buffer]))

    mag = float(np.hypot(dx, dy))

    # Extract and blur widget area
    panel = img[ORIGIN_Y:ORIGIN_Y+WIDGET_H, ORIGIN_X:ORIGIN_X+WIDGET_W].copy()
    blur = cv2.GaussianBlur(panel, (21, 21), 10)
    dark = cv2.addWeighted(blur, 0.85, np.zeros_like(panel), 0.15, 0)
    widget_working[:] = dark

    # Draw background ring
    cv2.circle(widget_working, widget_center, RING_R, (60, 60, 60), RING_T, cv2.LINE_AA)

    # Color based on motion
    if mag < 5:
        col = (50, 220, 50)
    elif mag < 12:
        col = (50, 220, 220)
    else:
        col = (50, 50, 255)

    # Draw motion sweep arc
    sweep = np.clip(mag / max_display_magnitude * 360, 0, 360)
    if sweep > 5:
        cv2.ellipse(widget_working, widget_center, (RING_R, RING_R), 0, -90, -90 + sweep, col, RING_T, cv2.LINE_AA)

    # Draw direction arrow
    if mag > 1.0:
        u = np.array([-dx, -dy]) / (mag + 1e-6)
        tip = (
            int(widget_center[0] + u[0] * (RING_R - 15)),
            int(widget_center[1] + u[1] * (RING_R - 15))
        )
        cv2.arrowedLine(widget_working, widget_center, tip, col, 2, tipLength=0.3, line_type=cv2.LINE_AA)

    # Center dot
    cv2.circle(widget_working, widget_center, 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Text overlay
    pil = Image.fromarray(cv2.cvtColor(widget_working, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((8, 6), "Camera Motion", font=font_big, fill=(255, 255, 255, 255))

    txt = f"{mag:4.1f} px/f"
    try:
        bbox = draw.textbbox((0, 0), txt, font=font_small)
        tw = bbox[2] - bbox[0]
    except:
        tw = draw.textsize(txt, font=font_small)[0]

    draw.text(((WIDGET_W - tw) // 2, WIDGET_H - 25), txt, font=font_small, fill=(255, 255, 255, 255))

    widget_working[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img[ORIGIN_Y:ORIGIN_Y+WIDGET_H, ORIGIN_X:ORIGIN_X+WIDGET_W] = widget_working

    return img




# overlays/overlay_helper.py

import cv2
import numpy as np
from PIL import ImageFont
from typing import List, Dict, Tuple

from utils import get_center_of_bbox, get_bbox_width, get_foot_position


def draw_ball_arrow(img: np.ndarray, np_bb: List[float]):
    cx = int((np_bb[0] + np_bb[2]) / 2)
    cy = int((np_bb[1] + np_bb[3]) / 2)
    arrow_tip_y = cy - 25
    tri = np.array(
        [[cx, arrow_tip_y], [cx - 12, arrow_tip_y - 22], [cx + 12, arrow_tip_y - 22]], np.int32
    )
    color, border = (0, 255, 0), (0, 0, 0)
    cv2.fillPoly(img, [tri], color)
    cv2.polylines(img, [tri], True, border, 2)


def draw_team_ball_control(frame, frame_idx, control_hist, team1_pct_list, team2_pct_list, hud_bg_cache):
    h, w = frame.shape[:2]
    PW, PH = 380, 150
    PX, PY = w - PW - 40, h - PH - 40
    if hud_bg_cache["bg"] is None:
        roi = frame[PY:PY + PH, PX:PX + PW]
        hud_bg_cache["bg"] = cv2.addWeighted(cv2.GaussianBlur(roi, (0, 0), 15), 0.85,
                                             np.zeros_like(roi), 0.15, 0)
    frame[PY:PY + PH, PX:PX + PW] = hud_bg_cache["bg"].copy()
    local_idx = len(team1_pct_list) - 1
    p1 = team1_pct_list[local_idx] / 100
    p2 = team2_pct_list[local_idx] / 100
    bx, by, ww, hh = PX + 50, PY + 70, 280, 22
    cv2.rectangle(frame, (bx, by), (bx + int(ww * p1), by + hh), (255, 255, 255), -1)
    cv2.rectangle(frame, (bx + int(ww * p1), by), (bx + ww, by + hh), (150, 150, 150), -1)
    cv2.rectangle(frame, (bx, by), (bx + ww, by + hh), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, "Ball Control", (PX + 30, PY + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Team 1: {p1 * 100:5.1f}%", (PX + 30, PY + 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Team 2: {p2 * 100:5.1f}%", (PX + 200, PY + 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_annotations(frames: List[np.ndarray], tracks: Dict[str, List[Dict[int, Dict]]], 
                     control_hist: np.ndarray, team1_pct_list: List[float], 
                     team2_pct_list: List[float], font_big, font_small, hud_bg_cache) -> List[np.ndarray]:
    out = []
    for f_idx, frame in enumerate(frames):
        img, halo = frame.copy(), np.zeros_like(frame)
        for pid, info in tracks["players"][f_idx].items():
            col = info.get("team_color", (0, 0, 255))
            bb = info["bbox"]
            xc, _ = get_center_of_bbox(bb)
            y2 = int(bb[3])
            w = get_bbox_width(bb)
            w_clamped = max(32, min(w, 40))
            axes = (int(w_clamped * 0.6), int(w_clamped * 0.22))
            if info.get("has_ball", False):
                cv2.ellipse(halo, (xc, y2), (axes[0] + 12, axes[1] + 6), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.ellipse(img, (xc, y2), (axes[0] + 12, axes[1] + 6), 0, -45, 235, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.ellipse(halo, (xc, y2), axes, 0, 0, 360, col, -1, cv2.LINE_AA)
            cv2.ellipse(img, (xc, y2), axes, 0, -45, 235, col, 3, cv2.LINE_AA)
            txt = str(pid)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
            pad, bx, by = 6, xc - (tw + 2 * 6) // 2, y2 + 14
            cv2.rectangle(img, (bx, by), (bx + tw + 2 * pad, by + th + pad), col, -1)
            cv2.putText(img, txt, (bx + pad, by + th), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        for ref in tracks["referees"][f_idx].values():
            bb = ref["bbox"]
            xc, _ = get_center_of_bbox(bb)
            y2 = int(bb[3])
            axes = (22, 8)
            cv2.ellipse(halo, (xc, y2), axes, 0, 0, 360, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.ellipse(img, (xc, y2), axes, 0, -45, 235, (0, 255, 255), 3, cv2.LINE_AA)
        img = cv2.addWeighted(halo, 0.35, img, 0.65, 0)
        for ball_id, ball_info in tracks["ball"][f_idx].items():
            draw_ball_arrow(img, ball_info["bbox"])
        img = draw_team_ball_control(img, f_idx, control_hist, team1_pct_list, team2_pct_list, hud_bg_cache)
        out.append(img)
    return out
