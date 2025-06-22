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
        
        # First predict, then correct
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
        """Get the last known position (either from update or prediction)"""
        if not self.initialised:
            return (0.0, 0.0)
        return self.last_prediction or (0.0, 0.0)


class BallTracker:
    """Enhanced ball tracking with buffering and interpolation."""
    
    def __init__(self, max_lost_frames: int = 12, ball_radius: int = 16, buffer_size: int = 25):
        self.kalman = _Kalman2D()
        self.max_lost_frames = max_lost_frames
        self.ball_radius = ball_radius
        
        # Buffer to store recent ball detections
        self.detection_buffer = deque(maxlen=buffer_size)
        self.interpolation_cache: Dict[int, Dict] = {}
        
        # State tracking
        self.last_detection_frame = None
        self.frames_since_detection = 0
        self.last_detection_center = None
        
    def update(self, frame_idx: int, ball_bbox: Optional[List[float]] = None) -> Dict:
        """Update ball tracker with new frame data."""
        result = {}
        
        if ball_bbox is not None:
            # Real detection found
            cx, cy = get_center_of_bbox(ball_bbox)
            
            # Update Kalman filter
            self.kalman.update(cx, cy)
            
            # Fill gaps before adding new detection
            if (self.last_detection_frame is not None and 
                frame_idx - self.last_detection_frame > 1):
                self._fill_detection_gaps(frame_idx, (cx, cy))
            
            # Add to buffer
            self.detection_buffer.append({
                'frame': frame_idx,
                'bbox': ball_bbox,
                'center': (cx, cy)
            })
            
            # Update state
            self.last_detection_frame = frame_idx
            self.last_detection_center = (cx, cy)
            self.frames_since_detection = 0
            
            # Store result
            result[1] = {"bbox": ball_bbox}
            self.interpolation_cache[frame_idx] = result
            
        else:
            # No detection - try to interpolate
            self.frames_since_detection += 1
            
            if (self.frames_since_detection <= self.max_lost_frames and 
                self.last_detection_frame is not None and
                self.kalman.initialised):
                
                # Get prediction from Kalman filter
                pred_cx, pred_cy = self.kalman.predict()
                
                # Ensure prediction is valid (not 0,0 like before!)
                if pred_cx == 0 and pred_cy == 0 and self.last_detection_center:
                    # Fallback to last known position if prediction fails
                    pred_cx, pred_cy = self.last_detection_center
                
                # Create interpolated bbox
                r = self.ball_radius
                interp_bbox = [pred_cx - r, pred_cy - r, pred_cx + r, pred_cy + r]
                
                result[1] = {"bbox": interp_bbox, "interpolated": True}
                self.interpolation_cache[frame_idx] = result
            
            # Check if we have cached interpolation
            elif frame_idx in self.interpolation_cache:
                result = self.interpolation_cache[frame_idx]
        
        return result
    
    def _fill_detection_gaps(self, current_frame: int, current_center: Tuple[float, float]):
        """Fill gaps between the last detection and current detection."""
        if self.last_detection_frame is None or self.last_detection_center is None:
            return
            
        gap_size = current_frame - self.last_detection_frame - 1
        
        if gap_size > 0 and gap_size <= self.max_lost_frames:
            # Linear interpolation for gap frames
            last_center = self.last_detection_center
            
            for i in range(1, gap_size + 1):
                gap_frame = self.last_detection_frame + i
                alpha = i / (gap_size + 1)
                
                # Interpolate position
                interp_cx = last_center[0] * (1 - alpha) + current_center[0] * alpha
                interp_cy = last_center[1] * (1 - alpha) + current_center[1] * alpha
                
                # Create bbox
                r = self.ball_radius
                interp_bbox = [interp_cx - r, interp_cy - r, interp_cx + r, interp_cy + r]
                
                # Cache the interpolation
                self.interpolation_cache[gap_frame] = {
                    1: {"bbox": interp_bbox, "interpolated": True}
                }
    
    def get_recent_trajectory(self, num_frames: int = 10) -> List[Tuple[float, float]]:
        """Get recent ball positions for trajectory analysis."""
        trajectory = []
        for det in list(self.detection_buffer)[-num_frames:]:
            trajectory.append(det['center'])
        return trajectory
    
    def clear_old_cache(self, current_frame: int, keep_frames: int = 150):
        """Clear old interpolation cache to prevent memory buildup."""
        frames_to_remove = [f for f in self.interpolation_cache.keys() 
                          if f < current_frame - keep_frames]
        for frame in frames_to_remove:
            del self.interpolation_cache[frame]

 

class Tracker:
    def __init__(self, model_path: str, batch_size: int = 64):
        self.model_path, self.batch_size = model_path, batch_size
        self.model = YOLO(model_path)
        if model_path.endswith(".pt"):
            self.model.to("cuda", non_blocking=True).half()

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

        # Enhanced ball tracking with larger buffer for fast/missing balls
        self.ball_tracker = BallTracker(max_lost_frames=12, ball_radius=16, buffer_size=25) #tweak buffer_size

        self.font_big = self._load_font(24)
        self.font_small = self._load_font(22)
        self._hud_bg = None
        self.frame_idx = 0

    def track_batch(self, frames: List[np.ndarray], start_frame_idx: int = 0) -> List[Dict[str, Dict[int, Dict]]]:
        if not frames:
            return []

        results = self.model.predict(
            frames,
            batch=len(frames),
            conf=0.25,
            iou=0.45,
            device=0,
            half=self.model_path.endswith(".pt"),
            verbose=False,
            tracker=None
        )

        batch_tracks = []
        for i, res in enumerate(results):
            dets = sv.Detections.from_ultralytics(res)
            names, inv = res.names, {v: k for k, v in res.names.items()}

            for j, cid in enumerate(dets.class_id):
                if names[cid] == "goalkeeper":
                    dets.class_id[j] = inv["player"]

            bt = self.byte_tracker.update_with_detections(dets)
            frame_tracks = {"players": {}, "referees": {}, "ball": {}}
            current_raw_tids = set()
            frame_idx = start_frame_idx + i

            # Process player and referee detections
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

            # Enhanced ball detection and tracking
            ball_bbox = self._extract_ball_bbox(dets, inv)
            ball_tracks = self.ball_tracker.update(frame_idx, ball_bbox)
            frame_tracks["ball"] = ball_tracks

            # Periodic cache cleanup
            if frame_idx % 50 == 0:
                self.ball_tracker.clear_old_cache(frame_idx)

            batch_tracks.append(frame_tracks)

        self.frame_idx = start_frame_idx + len(frames)
        return batch_tracks

    def _get_stable_id(self, raw_tid: int, center: Tuple[float, float], frame_idx: int) -> int:
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
        lost_raw_tids = []
        for raw_tid, track_info in self.active_tracks.items():
            if raw_tid not in current_raw_tids:
                if frame_idx - track_info['last_seen'] > self.max_lost_frames:
                    lost_raw_tids.append(raw_tid)
        for raw_tid in lost_raw_tids:
            self.active_tracks.pop(raw_tid, None)

    #  DRAWING
    # ─────────────────────────────────────────────────────────────────────────
    def draw_annotations(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List[Dict[int, Dict]]],
        control_hist: np.ndarray,
        team1_pct_list: List[float],
        team2_pct_list: List[float],
    ) -> List[np.ndarray]:

        out = []
        for f_idx, frame in enumerate(frames):
            img, halo = frame.copy(), np.zeros_like(frame)

            # —— players ——
            for pid, info in tracks["players"][f_idx].items():
                col = info.get("team_color", (0, 0, 255))
                bb = info["bbox"]
                xc, _ = get_center_of_bbox(bb)
                y2 = int(bb[3])
                w = get_bbox_width(bb)
                # Clamp width to avoid drastic halo size changes
                w_clamped = max(32, min(w, 40))  # Adjust min/max bounds as needed
                axes = (int(w_clamped * 0.6), int(w_clamped * 0.22))

                # Soft white glow if has ball
                if info.get("has_ball", False):
                    # Draw white outer glow first
                    cv2.ellipse(halo, (xc, y2), (axes[0] + 12, axes[1] + 6), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.ellipse(img, (xc, y2), (axes[0] + 12, axes[1] + 6), 0, -45, 235, (255, 255, 255), 3, cv2.LINE_AA)
                    # Draw normal team color fill
                    cv2.ellipse(halo, (xc, y2), axes, 0, 0, 360, col, -1, cv2.LINE_AA)

                # Outer stroke ring (team color)
                cv2.ellipse(img, (xc, y2), axes, 0, -45, 235, col, 3, cv2.LINE_AA)

                # ID box
                txt = str(pid)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
                pad, bx, by = 6, xc - (tw + 2 * 6) // 2, y2 + 14
                cv2.rectangle(img, (bx, by), (bx + tw + 2 * pad, by + th + pad), col, -1)
                cv2.putText(img, txt, (bx + pad, by + th),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

            # —— referees ——
            for ref in tracks["referees"][f_idx].values():
                bb = ref["bbox"]
                xc, _ = get_center_of_bbox(bb)
                y2 = int(bb[3])
                # Fixed clean minimal size
                axes = (22, 8)
                cv2.ellipse(halo, (xc, y2), axes, 0, 0, 360, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.ellipse(img, (xc, y2), axes, 0, -45, 235, (0, 255, 255), 3, cv2.LINE_AA)

            img = cv2.addWeighted(halo, 0.35, img, 0.65, 0)

            # —— ball ——
            for ball_id, ball_info in tracks["ball"][f_idx].items():
                self._draw_ball_arrow(img, ball_info["bbox"])

            img = self._draw_team_ball_control(
                img,
                f_idx,
                control_hist[: f_idx + 1],
                team1_pct_list,
                team2_pct_list
            )
            out.append(img)

        return out




    def _load_font(self, size: int):
        p = "camera_movement_estimator/assets/fonts/Roboto-Bold.ttf"
        if not os.path.isfile(p):
            p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        return ImageFont.truetype(p, size)

    def _extract_ball_bbox(self, dets, inv):
        ball_c = inv["ball"]
        idxs = np.where(dets.class_id == ball_c)[0]
        if len(idxs):
            i = int(idxs[np.argmax(dets.confidence[idxs])])
            return dets.xyxy[i].tolist()
        return None
 
    def _draw_ball_arrow(self, img: np.ndarray, np_bb: List[float]):
        # Calculate the actual ball center
        cx = int((np_bb[0] + np_bb[2]) / 2)
        cy = int((np_bb[1] + np_bb[3]) / 2)
        
   
        arrow_tip_y = cy - 25  #   arrow distance from ball
        
        tri = np.array(
            [[cx, arrow_tip_y], [cx - 12, arrow_tip_y - 22], [cx + 12, arrow_tip_y - 22]], np.int32
        )
        color, border = (0, 255, 0), (0, 0, 0)
        cv2.fillPoly(img, [tri], color)
        cv2.polylines(img, [tri], True, border, 2)

    # —— HUD —— ----------------------------------------------------------------
    def _draw_team_ball_control(self, frame, frame_idx, control_hist, team1_pct_list, team2_pct_list):
        h, w = frame.shape[:2]
        PW, PH = 380, 150
        PX, PY = w - PW - 40, h - PH - 40
        if self._hud_bg is None:
            roi = frame[PY : PY + PH, PX : PX + PW]
            self._hud_bg = cv2.addWeighted(
                cv2.GaussianBlur(roi, (0, 0), 15), 0.85, np.zeros_like(roi), 0.15, 0
            )
        frame[PY : PY + PH, PX : PX + PW] = self._hud_bg.copy()

        local_idx = len(team1_pct_list) - 1  # Always take the latest value in the batch
        p1 = team1_pct_list[local_idx] / 100
        p2 = team2_pct_list[local_idx] / 100

        bx, by, ww, hh = PX + 50, PY + 70, 280, 22
        cv2.rectangle(frame, (bx, by), (bx + int(ww * p1), by + hh), (255, 255, 255), -1)
        cv2.rectangle(frame, (bx + int(ww * p1), by), (bx + ww, by + hh), (150, 150, 150), -1)
        cv2.rectangle(frame, (bx, by), (bx + ww, by + hh), (80, 80, 80), 1, cv2.LINE_AA)

        # Label text
        cv2.putText(frame, "Ball Control", (PX + 30, PY + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Team 1: {p1 * 100:5.1f}%", (PX + 30, PY + 110),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Team 2: {p2 * 100:5.1f}%", (PX + 200, PY + 110),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
