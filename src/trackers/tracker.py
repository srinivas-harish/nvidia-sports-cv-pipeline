# tracker.py – Fixed ID tracking with stable assignments
from __future__ import annotations
import os, sys
from typing import List, Dict, Tuple

import cv2, numpy as np, torch, supervision as sv
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# project-local helpers
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils import get_center_of_bbox, get_bbox_width, get_foot_position  # noqa: E402

torch.backends.cudnn.benchmark = True


class _Kalman2D:
    """tiny 2D constant-velocity Kalman filter for the ball. Doesn't work well yet."""
    def __init__(self):
        kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        kf.measurementMatrix[:]    = np.eye(2, 4, dtype=np.float32)
        kf.transitionMatrix[:]     = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov[:]      = np.eye(4,      dtype=np.float32) * 1e-2
        kf.measurementNoiseCov[:]  = np.eye(2,      dtype=np.float32) * 5e-1
        self.kf, self.initialised = kf, False

    def update(self, x: float, y: float):
        m = np.array([[x],[y]], dtype=np.float32)
        if not self.initialised:
            self.kf.statePre[:]  = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self.kf.statePost[:] = self.kf.statePre.copy()
            self.initialised = True
        self.kf.correct(m)

    def predict(self) -> Tuple[float,float]:
        p = self.kf.predict()
        return float(p[0]), float(p[1])


class Tracker:
    """batched YOLO→ByteTrack→Stable-ID→Overlay pipeline."""

    def __init__(self, model_path: str, batch_size: int = 64):
        # model
        self.model_path, self.batch_size = model_path, batch_size
        self.model = YOLO(model_path)
        if model_path.endswith(".pt"):
            self.model.to("cuda", non_blocking=True).half()
 
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,   
            lost_track_buffer=30,              
            minimum_matching_threshold=0.95,    # IMP value*
            frame_rate=30
        )
        
        # Stable ID tracking - NEW APPROACH
        self.active_tracks: Dict[int, Dict] = {}  # raw_tid -> {stable_id, center, last_seen}
        self.stable_id_history: Dict[int, Tuple[float, float]] = {}  # stable_id -> last_known_center
        self.next_stable_id = 1
        self.max_lost_frames = 300
        self.proximity_threshold = 100  

        # ball smoothing
        self.ball_kf          = _Kalman2D()
        self.last_ball_frame  = -999
        self.BALL_MAX_LOST    = 8
        self.BALL_R           = 16

        # HUD fonts
        self.font_big   = self._load_font(24)
        self.font_small = self._load_font(22)
        self._hud_bg    = None

        # frame index
        self.frame_idx = 0

    def track_batch(
        self,
        frames: List[np.ndarray],
        start_frame_idx: int = 0
    ) -> List[Dict[str, Dict[int,Dict]]]:
        """
        Process a batch *knowing* its starting frame index.
        """
        if not frames:
            return []

        #   frame index continuity
        self.frame_idx = start_frame_idx

        #   inference with consistent confidence 
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
        for res in results:
            dets = sv.Detections.from_ultralytics(res)
            names, inv = res.names, {v:k for k,v in res.names.items()}

            # merge GK into players
            for i, cid in enumerate(dets.class_id):
                if names[cid] == "goalkeeper":
                    dets.class_id[i] = inv["player"]

            # NO additional confidence filtering - using YOLO's threshold

            # ByteTrack update
            bt = self.byte_tracker.update_with_detections(dets)

            #   per-frame dict
            frame_tracks = {"players":{}, "referees":{}, "ball":{}}

            # NEW: Track current frame's raw track IDs
            current_raw_tids = set()

            # Process players & referees
            for d in bt:
                bb, cid, raw_tid = d[0].tolist(), d[3], int(d[4])
                center = get_foot_position(bb)

                if cid == inv["player"]:
                    current_raw_tids.add(raw_tid)
                    stable_id = self._get_stable_id(raw_tid, center)
                    frame_tracks["players"][stable_id] = {"bbox": bb}

                elif cid == inv["referee"]:
                    frame_tracks["referees"][raw_tid] = {"bbox": bb}

            # Clean up lost tracks
            self._cleanup_lost_tracks(current_raw_tids)

            # ball + Kalman
            bb_ball = self._extract_ball_bbox(dets, inv, self.frame_idx)
            if bb_ball is not None:
                frame_tracks["ball"][1] = {"bbox": bb_ball}

            batch_tracks.append(frame_tracks)
            self.frame_idx += 1

        return batch_tracks

    def _get_stable_id(self, raw_tid: int, center: Tuple[float, float]) -> int:
        """Get stable ID for a raw track ID, handling reuse scenarios."""
        
        # Update existing active track
        if raw_tid in self.active_tracks:
            track_info = self.active_tracks[raw_tid]
            track_info['center'] = center
            track_info['last_seen'] = self.frame_idx
            stable_id = track_info['stable_id']
            self.stable_id_history[stable_id] = center
            return stable_id
        
        # New raw track ID - try to match with recently lost stable ID
        best_stable_id = None
        best_distance = float('inf')
        
        # Look for recently lost stable IDs that might match this position
        for stable_id, last_center in list(self.stable_id_history.items()):
            # Skip if this stable ID is still active
            if any(track['stable_id'] == stable_id for track in self.active_tracks.values()):
                continue
                
            distance = np.linalg.norm(np.array(center) - np.array(last_center))
            if distance < self.proximity_threshold and distance < best_distance:
                best_distance = distance
                best_stable_id = stable_id
        
        # Use matched stable ID or create new one
        if best_stable_id is not None:
            stable_id = best_stable_id
        else:
            stable_id = self.next_stable_id
            self.next_stable_id += 1
        
        # Create new active track
        self.active_tracks[raw_tid] = {
            'stable_id': stable_id,
            'center': center,
            'last_seen': self.frame_idx
        }
        self.stable_id_history[stable_id] = center
        
        return stable_id

    def _cleanup_lost_tracks(self, current_raw_tids: set):
        """Remove tracks that are no longer active."""
        lost_raw_tids = []
        
        for raw_tid, track_info in self.active_tracks.items():
            if raw_tid not in current_raw_tids:
                # Check if track has been lost for too long
                frames_lost = self.frame_idx - track_info['last_seen']
                if frames_lost > self.max_lost_frames:
                    lost_raw_tids.append(raw_tid)
        
        # Remove lost tracks
        for raw_tid in lost_raw_tids:
            self.active_tracks.pop(raw_tid, None)
        
        # Clean up very old stable ID history (keep for potential matching)
        stable_ids_to_remove = []
        for stable_id, _ in self.stable_id_history.items():
            # Only remove if no active track uses this stable ID and it's very old
            if not any(track['stable_id'] == stable_id for track in self.active_tracks.values()):
                # Keep in history for a while to allow re-matching
                pass  # Don't remove immediately

    def draw_annotations(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List[Dict[int,Dict]]],
        control_hist: np.ndarray
    ) -> List[np.ndarray]:
        out = []
        for f_idx, frame in enumerate(frames):
            img, halo = frame.copy(), np.zeros_like(frame)

            # players
            for pid, info in tracks["players"][f_idx].items():
                col = info.get("team_color",(0,0,255))
                bb  = info["bbox"]
                xc,_ = get_center_of_bbox(bb)
                y2   = int(bb[3])
                w    = get_bbox_width(bb)
                axes = (int(w*0.6), int(w*0.22))
                cv2.ellipse(halo,(xc,y2),axes,0,0,360,col,-1,cv2.LINE_AA)
                cv2.ellipse(img,(xc,y2),axes,0,-45,235,col,3,cv2.LINE_AA)
                # ID plate
                txt = str(pid)
                (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX,0.65,2)
                pad, bx, by = 6, xc-(tw+2*6)//2, y2+14
                cv2.rectangle(img,(bx,by),(bx+tw+2*pad,by+th+pad),col,-1)
                cv2.putText(img,txt,(bx+pad,by+th),
                            cv2.FONT_HERSHEY_DUPLEX,0.65,(0,0,0),1,cv2.LINE_AA)

            # referees
            for ref in tracks["referees"][f_idx].values():
                bb = ref["bbox"]
                xc,_ = get_center_of_bbox(bb)
                y2   = int(bb[3])
                w    = get_bbox_width(bb)
                axes = (int(w*0.6), int(w*0.22))
                cv2.ellipse(halo,(xc,y2),axes,0,0,360,(0,255,255),-1,cv2.LINE_AA)
                cv2.ellipse(img,(xc,y2),axes,0,-45,235,(0,255,255),3,cv2.LINE_AA)

            # blend
            img = cv2.addWeighted(halo,0.35,img,0.65,0)

            # ball arrow
            for b in tracks["ball"][f_idx].values():
                self._draw_ball_arrow(img, b["bbox"])

            # HUD
            img = self._draw_team_ball_control(img, f_idx, control_hist[:f_idx+1])
            out.append(img)
        return out

    #  helpers ──────────────────────────────────────────────
    def _load_font(self, size:int):
        p = "camera_movement_estimator/assets/fonts/Roboto-Bold.ttf"
        if not os.path.isfile(p):
            p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        return ImageFont.truetype(p, size)

    def _extract_ball_bbox(self, dets, inv, frame_idx:int):
        ball_c = inv["ball"]
        idxs   = np.where(dets.class_id==ball_c)[0]
        if len(idxs):
            i  = int(idxs[np.argmax(dets.confidence[idxs])])
            bb = dets.xyxy[i].tolist()
            cx,cy = get_center_of_bbox(bb)
            self.ball_kf.update(cx,cy)
            self.last_ball_frame = frame_idx
            return bb
        # no detect → predict
        if frame_idx - self.last_ball_frame <= self.BALL_MAX_LOST and self.ball_kf.initialised:
            cx,cy = self.ball_kf.predict()
            return [cx-self.BALL_R, cy-self.BALL_R, cx+self.BALL_R, cy+self.BALL_R]
        return None

    def _draw_ball_arrow(self, img,np_bb):
        cx,top = int((np_bb[0]+np_bb[2])//2), int(np_bb[1])-10
        tri    = np.array([[cx,top],[cx-12,top-22],[cx+12,top-22]],np.int32)
        cv2.fillPoly(img,[tri],(0,255,0))
        cv2.polylines(img,[tri],True,(0,0,0),2)

    def _draw_team_ball_control(self, frame, idx, arr):
        h,w = frame.shape[:2]; PW,PH=380,150; PX,PY = w-PW-40,h-PH-40
        if self._hud_bg is None:
            roi = frame[PY:PY+PH, PX:PX+PW]
            self._hud_bg = cv2.addWeighted(cv2.GaussianBlur(roi,(0,0),15),
                                           0.85,np.zeros_like(roi),0.15,0)
        frame[PY:PY+PH, PX:PX+PW] = self._hud_bg.copy()
        valid = arr[arr!=0]
        p1 = np.sum(valid==1)/len(valid) if len(valid) else 0.5
        p2 = 1-p1
        bx,by,ww,hh = PX+50,PY+70,280,22
        cv2.rectangle(frame,(bx,by),(bx+int(ww*p1),by+hh),(255,255,255),-1)
        cv2.rectangle(frame,(bx+int(ww*p1),by),(bx+ww,by+hh),(150,150,150),-1)
        cv2.rectangle(frame,(bx,by),(bx+ww,by+hh),(80,80,80),1,cv2.LINE_AA)
        pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        d   = ImageDraw.Draw(pil)
        d.text((PX+30,PY+25),"Ball Control", font=self.font_big,  fill=(255,255,255,255))
        d.text((PX+30,PY+110),f"Team 1: {p1*100:5.1f}%",font=self.font_small,fill=(255,255,255,255))
        d.text((PX+200,PY+110),f"Team 2: {p2*100:5.1f}%",font=self.font_small,fill=(255,255,255,255))
        frame[:] = cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
        return frame