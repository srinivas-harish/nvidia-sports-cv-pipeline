# tracker.py – Improved ID tracking + fast overlay + Roboto HUD
from ultralytics import YOLO
import supervision as sv
import pickle, os, sys, cv2, torch
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

torch.backends.cudnn.benchmark = True

class Tracker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)
        if model_path.endswith('.pt'):
            self.model = self.model.to('cuda').half()
        self.batch_size = 64
        self.tracker = sv.ByteTrack()
        self.font_big = self.load_font(24)
        self.font_small = self.load_font(22)
        self._hud_bg = None
        self.max_lost_frames = 3
        self.id_map = {}
        self.id_last_seen = {}
        self.next_id = 1

    def load_font(self, size):
        font_path = "camera_movement_estimator/assets/fonts/Roboto-Bold.ttf"
        if not os.path.isfile(font_path):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        return ImageFont.truetype(font_path, size)

    def add_position_to_tracks(self, tracks):
        for obj, objs in tracks.items():
            for f, tr in enumerate(objs):
                for tid, info in tr.items():
                    pos = (get_center_of_bbox(info['bbox'])
                           if obj == 'ball' else get_foot_position(info['bbox']))
                    tracks[obj][f][tid]['position'] = pos

    def interpolate_ball_positions(self, bp):
        bp = [x.get(1, {}).get('bbox', []) for x in bp]
        df = pd.DataFrame(bp, columns=['x1', 'y1', 'x2', 'y2']).interpolate().bfill()
        return [{1: {'bbox': x}} for x in df.to_numpy().tolist()]

    def detect_frames(self, frames):
        # NOTE: device=0 already sends it to CUDA, half=True only valid for .pt
        return list(self.model.predict(
            frames, batch=self.batch_size, stream=True,
            device=0, half=self.model_path.endswith('.pt'), conf=0.1, verbose=False))

    def match_id(self, curr_pos, frame_idx):
        min_dist = float('inf')
        best_id = None
        for prev_id, (last_pos, last_frame) in self.id_last_seen.items():
            if frame_idx - last_frame > self.max_lost_frames:
                continue
            dist = np.linalg.norm(np.array(curr_pos) - np.array(last_pos))
            if dist < min_dist and dist < 50:
                min_dist = dist
                best_id = prev_id
        if best_id is not None:
            return best_id
        else:
            new_id = self.next_id
            self.next_id += 1
            return new_id

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            return pickle.load(open(stub_path, 'rb'))

        dets = self.detect_frames(frames)
        tracks = {'players': [], 'referees': [], 'ball': []}

        self.id_map = {}
        self.id_last_seen = {}
        self.next_id = 1

        for f, det in enumerate(dets):
            names, inv = det.names, {v: k for k, v in det.names.items()}
            dete = sv.Detections.from_ultralytics(det)

            for i, cid in enumerate(dete.class_id):
                if names[cid] == 'goalkeeper':
                    dete.class_id[i] = inv['player']

            trk = self.tracker.update_with_detections(dete)
            for k in tracks:
                tracks[k].append({})

            for d in trk:
                bb, cid, raw_tid = d[0].tolist(), d[3], d[4]
                center = get_foot_position(bb)

                if cid == inv['player']:
                    stable_id = self.match_id(center, f)
                    tracks['players'][f][stable_id] = {'bbox': bb}
                    self.id_last_seen[stable_id] = (center, f)

                elif cid == inv['referee']:
                    tracks['referees'][f][raw_tid] = {'bbox': bb}

            for d in dete:
                if d[3] == inv['ball']:
                    tracks['ball'][f][1] = {'bbox': d[0].tolist()}

        if stub_path:
            pickle.dump(tracks, open(stub_path, 'wb'))
        return tracks

    def _draw_ball_arrow(self, frame, bbox):
        cx = int((bbox[0]+bbox[2])//2); top = int(bbox[1])-10
        tri = np.array([[cx,top],[cx-12,top-22],[cx+12,top-22]],np.int32)
        cv2.fillPoly(frame,[tri],(0,255,0)); cv2.polylines(frame,[tri],True,(0,0,0),2)

    def draw_team_ball_control(self, frame, idx, arr):
        h,w = frame.shape[:2]; PW,PH = 380,150; PX,PY = w-PW-40, h-PH-40
        if self._hud_bg is None:
            roi = frame[PY:PY+PH, PX:PX+PW]
            self._hud_bg = cv2.addWeighted(cv2.GaussianBlur(roi,(0,0),15),0.85,np.zeros_like(roi),0.15,0)
        frame[PY:PY+PH, PX:PX+PW] = self._hud_bg.copy()

        valid_arr = arr[arr != 0]
        if len(valid_arr) == 0:
            p1, p2 = 0.5, 0.5
        else:
            t1 = np.sum(valid_arr == 1)
            t2 = np.sum(valid_arr == 2)
            tot = t1 + t2
            p1 = t1 / tot if tot > 0 else 0.5
            p2 = t2 / tot if tot > 0 else 0.5

        bar_x, bar_y = PX+50, PY+70; bar_w, bar_h = 280, 22
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+int(bar_w*p1),bar_y+bar_h),(255,255,255),-1)
        cv2.rectangle(frame,(bar_x+int(bar_w*p1),bar_y),(bar_x+bar_w,bar_y+bar_h),(150,150,150),-1)
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+bar_w,bar_y+bar_h),(80,80,80),1,cv2.LINE_AA)

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d   = ImageDraw.Draw(pil)
        d.text((PX+30,PY+25),"Ball Control", font=self.font_big, fill=(255,255,255,255))
        d.text((PX+30,PY+110), f"Team 1: {p1*100:5.1f}%", font=self.font_small, fill=(255,255,255,255))
        d.text((PX+200,PY+110),f"Team 2: {p2*100:5.1f}%", font=self.font_small, fill=(255,255,255,255))
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return frame

    def draw_annotations(self, frames, tracks, control):
        out = []
        for f, frame in enumerate(frames):
            img = frame.copy()
            halo = np.zeros_like(img)

            for tid, p in tracks['players'][f].items():
                col = p.get('team_color', (0, 0, 255))
                bb = p['bbox']
                xc, _ = get_center_of_bbox(bb)
                y2 = int(bb[3])
                w = get_bbox_width(bb)
                axes = (int(w * 0.6), int(w * 0.22))
                cv2.ellipse(halo, (xc, y2), axes, 0, 0, 360, col, -1, cv2.LINE_AA)
                cv2.ellipse(img, (xc, y2), axes, 0, -45, 235, col, 3, cv2.LINE_AA)
                if p.get('has_ball'):
                    cv2.ellipse(halo, (xc, y2), (axes[0] + 12, axes[1] + 6), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.ellipse(img, (xc, y2), (axes[0] + 12, axes[1] + 6), 0, -45, 235, (255, 255, 255), 3, cv2.LINE_AA)

                txt = f"{tid}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
                pad = 6
                bx, by = xc - (tw + pad * 2) // 2, y2 + 14
                cv2.rectangle(img, (bx, by), (bx + tw + pad * 2, by + th + pad), col, cv2.FILLED)
                cv2.putText(img, txt, (bx + pad, by + th),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

            for ref in tracks['referees'][f].values():
                bb = ref['bbox']
                xc, _ = get_center_of_bbox(bb)
                y2 = int(bb[3])
                w = get_bbox_width(bb)
                axes = (int(w * 0.6), int(w * 0.22))
                cv2.ellipse(halo, (xc, y2), axes, 0, 0, 360, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.ellipse(img, (xc, y2), axes, 0, -45, 235, (0, 255, 255), 3, cv2.LINE_AA)

            img = cv2.addWeighted(halo, 0.35, img, 0.65, 0)

            for b in tracks['ball'][f].values():
                self._draw_ball_arrow(img, b['bbox'])

            img = self.draw_team_ball_control(img, f, control if len(frames) == 1 else control[:f+1])
            out.append(img)
        return out
