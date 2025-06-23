from __future__ import annotations
import os, sys
from typing import List, Dict, Tuple, Optional
from collections import deque 

import cv2, numpy as np, torch, supervision as sv
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# project-local helpers
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils import get_center_of_bbox, get_bbox_width, get_foot_position   
from assign_acquisition.assign_acquisition import BallAcquisition
torch.backends.cudnn.benchmark = True

class _Kalman2D:
    # Unchanged from your code
    def __init__(self):
        kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        kf.measurementMatrix[:]    = np.eye(2, 4, dtype=np.float32)
        kf.transitionMatrix[:]     = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov[:]      = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov[:]  = np.eye(2, dtype=np.float32) * 5e-1
        self.kf, self.initialised = kf, False
        self.last_prediction = None

    def update(self, x: float, y: float):
        m = np.array([[x], [y]], dtype=np.float32)
        if not self.initialised:
            self.kf.statePre[:]  = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost[:] = self.kf.statePre.copy()
            self.initialised = True
        self.kf.predict()
        self.kf.correct(m)
        self.last_prediction = (x, y)

    def predict(self) -> Tuple[float, float]:
        if not self.initialised:
            return (0.0, 0.0)
        p = self.kf.predict()
        prediction = (float(p[0]), float(p[1]))
        self.last_prediction = prediction
        return prediction

    def get_last_position(self) -> Tuple[float, float]:
        if not self.initialised:
            return (0.0, 0.0)
        return self.last_prediction or (0.0, 0.0)

class BallTracker:
    # Unchanged from your code
    def __init__(self, max_lost_frames: int = 12, ball_radius: int = 16, buffer_size: int = 25):
        self.kalman = _Kalman2D()
        self.max_lost_frames = max_lost_frames
        self.ball_radius = ball_radius
        self.detection_buffer = deque(maxlen=buffer_size)
        self.interpolation_cache: Dict[int, Dict] = {}
        self.last_detection_frame = None
        self.frames_since_detection = 0
        self.last_detection_center = None
        
    def update(self, frame_idx: int, ball_bbox: Optional[List[float]] = None) -> Dict:
        result = {}
        if ball_bbox is not None:
            cx, cy = get_center_of_bbox(ball_bbox)
            self.kalman.update(cx, cy)
            if (self.last_detection_frame is not None and 
                frame_idx - self.last_detection_frame > 1):
                self._fill_detection_gaps(frame_idx, (cx, cy))
            self.detection_buffer.append({
                'frame': frame_idx,
                'bbox': ball_bbox,
                'center': (cx, cy)
            })
            self.last_detection_frame = frame_idx
            self.last_detection_center = (cx, cy)
            self.frames_since_detection = 0
            result[1] = {"bbox": ball_bbox}
            self.interpolation_cache[frame_idx] = result
        else:
            self.frames_since_detection += 1
            if (self.frames_since_detection <= self.max_lost_frames and 
                self.last_detection_frame is not None and
                self.kalman.initialised):
                pred_cx, pred_cy = self.kalman.predict()
                if pred_cx == 0 and pred_cy == 0 and self.last_detection_center:
                    pred_cx, pred_cy = self.last_detection_center
                r = self.ball_radius
                interp_bbox = [pred_cx - r, pred_cy - r, pred_cx + r, pred_cy + r]
                result[1] = {"bbox": interp_bbox, "interpolated": True}
                self.interpolation_cache[frame_idx] = result
            elif frame_idx in self.interpolation_cache:
                result = self.interpolation_cache[frame_idx]
        return result
    
    def _fill_detection_gaps(self, current_frame: int, current_center: Tuple[float, float]):
        if self.last_detection_frame is None or self.last_detection_center is None:
            return
        gap_size = current_frame - self.last_detection_frame - 1
        if gap_size > 0 and gap_size <= self.max_lost_frames:
            last_center = self.last_detection_center
            for i in range(1, gap_size + 1):
                gap_frame = self.last_detection_frame + i
                alpha = i / (gap_size + 1)
                interp_cx = last_center[0] * (1 - alpha) + current_center[0] * alpha
                interp_cy = last_center[1] * (1 - alpha) + current_center[1] * alpha
                r = self.ball_radius
                interp_bbox = [interp_cx - r, interp_cy - r, interp_cx + r, interp_cy + r]
                self.interpolation_cache[gap_frame] = {
                    1: {"bbox": interp_bbox, "interpolated": True}
                }
    
    def get_recent_trajectory(self, num_frames: int = 10) -> List[Tuple[float, float]]:
        trajectory = []
        for det in list(self.detection_buffer)[-num_frames:]:
            trajectory.append(det['center'])
        return trajectory
    
    def clear_old_cache(self, current_frame: int, keep_frames: int = 150):
        frames_to_remove = [f for f in self.interpolation_cache.keys() 
                          if f < current_frame - keep_frames]
        for frame in frames_to_remove:
            del self.interpolation_cache[frame]

class Tracker:
    def __init__(self, player_model_path: str, ball_model_path: Optional[str] = None, batch_size: int = 64, 
                 player_conf: float = 0.25, player_iou: float = 0.45, ball_conf: float = 0.3, ball_iou: float = 0.5):
        # Initialize player model
        self.player_model_path = player_model_path
        self.player_model = YOLO(player_model_path)
        if player_model_path.endswith(".pt"):
            self.player_model.to("cuda", non_blocking=True).half()
        self.player_conf = player_conf
        self.player_iou = player_iou

        # Initialize ball model
        self.ball_model_path = ball_model_path
        self.ball_model = YOLO(ball_model_path) if ball_model_path else None
        if ball_model_path and ball_model_path.endswith(".pt"):
            self.ball_model.to("cuda", non_blocking=True).half()
        self.ball_conf = ball_conf
        self.ball_iou = ball_iou

        self.batch_size = batch_size
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.95,
            frame_rate=30
        )
        self.active_tracks: Dict[int, Dict] = {}
        self.stable_id_history: Dict[int, Tuple[float, float]] = {}
        self.next_stable_id = 1
        self.max_lost_frames = 300
        self.proximity_threshold = 100
        self.ball_tracker = BallTracker(max_lost_frames=12, ball_radius=16, buffer_size=25)
        self.font_big = self._load_font(24)
        self.font_small = self._load_font(22)
        self._hud_bg = None
        self.frame_idx = 0

    def track_batch(self, frames: List[np.ndarray], start_frame_idx: int = 0) -> List[Dict[str, Dict[int, Dict]]]:
        if not frames:
            return []

        # Player and referee detections
        player_results = self.player_model.predict(
            frames,
            batch=len(frames),
            conf=self.player_conf,
            iou=self.player_iou,
            device=0,
            half=self.player_model_path.endswith(".pt"),
            verbose=False,
            tracker=None
        )

        # Ball detections
        ball_results = self.ball_model.predict(
            frames,
            batch=len(frames),
            conf=self.ball_conf,
            iou=self.ball_iou,
            device=0,
            half=self.ball_model_path.endswith(".pt") if self.ball_model_path else False,
            verbose=False,
            tracker=None
        ) if self.ball_model else player_results  # Fallback to player model if no ball model

        batch_tracks = []
        for i, (p_res, b_res) in enumerate(zip(player_results, ball_results)):
            # Process player and referee detections
            dets = sv.Detections.from_ultralytics(p_res)
            names, inv = p_res.names, {v: k for k, v in p_res.names.items()}
            for j, cid in enumerate(dets.class_id):
                if names[cid] == "goalkeeper":
                    dets.class_id[j] = inv["player"]
            bt = self.byte_tracker.update_with_detections(dets)
            frame_tracks = {"players": {}, "referees": {}, "ball": {}}
            current_raw_tids = set()
            frame_idx = start_frame_idx + i

            for d in bt:
                bb, cid, raw_tid = d[0].tolist(), d[3], int(d[4])
                center = get_foot_position(bb)
                if cid == inv["player"]:
                    current_raw_tids.add(raw_tid)
                    sid = self._get_stable_id(raw_tid, center, frame_idx)
                    frame_tracks["players"][sid] = {"bbox": bb}
                elif cid == inv["referee"]:
                    frame_tracks["referees"][raw_tid] = {"bbox": bb}

            self._cleanup_lost_tracks(current_raw_tids, frame_idx)

            # Process ball detections
            ball_dets = sv.Detections.from_ultralytics(b_res)
            ball_bbox = self._extract_ball_bbox(ball_dets, {v: k for k, v in b_res.names.items()})
            ball_tracks = self.ball_tracker.update(frame_idx, ball_bbox)
            frame_tracks["ball"] = ball_tracks

            if frame_idx % 50 == 0:
                self.ball_tracker.clear_old_cache(frame_idx)

            batch_tracks.append(frame_tracks)

        self.frame_idx = start_frame_idx + len(frames)
        return batch_tracks

    def _extract_ball_bbox(self, dets, inv):
        # Handle ball model with only 1 class - assume it's the ball class
        if len(dets.xyxy) == 0:
            return None
        
        # Since there's only one class, take the detection with highest confidence
        if len(dets.xyxy) > 0:
            best_idx = int(np.argmax(dets.confidence))
            return dets.xyxy[best_idx].tolist()
        
        return None

    def _get_stable_id(self, raw_tid: int, center: Tuple[float, float], frame_idx: int) -> int:
        # Unchanged
        if raw_tid in self.active_tracks:
            track_info = self.active_tracks[raw_tid]
            track_info['center'] = center
            track_info['last_seen'] = frame_idx
            stable_id = track_info['stable_id']
            self.stable_id_history[stable_id] = center
            return stable_id
        best_stable_id = None
        best_distance = float('inf')
        for stable_id, last_center in list(self.stable_id_history.items()):
            if any(t['stable_id'] == stable_id for t in self.active_tracks.values()):
                continue
            distance = np.linalg.norm(np.array(center) - np.array(last_center))
            if distance < self.proximity_threshold and distance < best_distance:
                best_distance = distance
                best_stable_id = stable_id
        if best_stable_id is not None:
            stable_id = best_stable_id
        else:
            stable_id = self.next_stable_id
            self.next_stable_id += 1
        self.active_tracks[raw_tid] = {
            'stable_id': stable_id,
            'center': center,
            'last_seen': frame_idx
        }
        self.stable_id_history[stable_id] = center
        return stable_id

    def _cleanup_lost_tracks(self, current_raw_tids: set, frame_idx: int):
        # Unchanged
        lost_raw_tids = []
        for raw_tid, track_info in self.active_tracks.items():
            if raw_tid not in current_raw_tids:
                if frame_idx - track_info['last_seen'] > self.max_lost_frames:
                    lost_raw_tids.append(raw_tid)
        for raw_tid in lost_raw_tids:
            self.active_tracks.pop(raw_tid, None)

    def draw_annotations(self, frames: List[np.ndarray], tracks: Dict[str, List[Dict[int, Dict]]], 
                        control_hist: np.ndarray, team1_pct_list: List[float], team2_pct_list: List[float]) -> List[np.ndarray]:
        # Unchanged
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
                self._draw_ball_arrow(img, ball_info["bbox"])
            img = self._draw_team_ball_control(img, f_idx, control_hist, team1_pct_list, team2_pct_list)
            out.append(img)
        return out

    def add_position_to_tracks(self, frame_tracks: List[Dict[str, Dict[int, Dict]]]):
        # Unchanged
        for obj in ["players", "referees", "ball"]:
            for frame_idx, frame_data in enumerate(frame_tracks):
                for tid, track_info in frame_data[obj].items():
                    bbox = track_info.get("bbox")
                    if not bbox:
                        continue
                    if obj == "ball":
                        center = get_center_of_bbox(bbox)
                    else:
                        center = get_foot_position(bbox)
                    track_info["position"] = center

    def _load_font(self, size: int):
        # Unchanged
        p = "camera_movement_estimator/assets/fonts/Roboto-Bold.ttf"
        if not os.path.isfile(p):
            p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        return ImageFont.truetype(p, size)

    def _draw_ball_arrow(self, img: np.ndarray, np_bb: List[float]):
        # Unchanged
        cx = int((np_bb[0] + np_bb[2]) / 2)
        cy = int((np_bb[1] + np_bb[3]) / 2)
        arrow_tip_y = cy - 25
        tri = np.array(
            [[cx, arrow_tip_y], [cx - 12, arrow_tip_y - 22], [cx + 12, arrow_tip_y - 22]], np.int32
        )
        color, border = (0, 255, 0), (0, 0, 0)
        cv2.fillPoly(img, [tri], color)
        cv2.polylines(img, [tri], True, border, 2)

    def _draw_team_ball_control(self, frame, frame_idx, control_hist, team1_pct_list, team2_pct_list):
        # Unchanged
        h, w = frame.shape[:2]
        PW, PH = 380, 150
        PX, PY = w - PW - 40, h - PH - 40
        if self._hud_bg is None:
            roi = frame[PY : PY + PH, PX : PX + PW]
            self._hud_bg = cv2.addWeighted(
                cv2.GaussianBlur(roi, (0, 0), 15), 0.85, np.zeros_like(roi), 0.15, 0
            )
        frame[PY : PY + PH, PX : PX + PW] = self._hud_bg.copy()
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