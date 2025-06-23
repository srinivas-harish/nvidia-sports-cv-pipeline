import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque

class CameraMotionEstimator:
    def __init__(self, initial_frame):
        self.prev_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        h, w = self.prev_gray.shape
        
        #   feature mask  
        mask = np.zeros_like(self.prev_gray)
        mask[0:h // 4, 0:w // 6] = 1
        mask[0:h // 4, 5 * w // 6:w] = 1
        mask[3 * h // 4:h, 0:w // 6] = 1
        mask[3 * h // 4:h, 5 * w // 6:w] = 1
        mask[0:h // 8, w // 4:3 * w // 4] = 1
        mask[7 * h // 8:h, w // 4:3 * w // 4] = 1
        self.feature_mask = mask
         
        self.prev_pts = self._detect_features(self.prev_gray)
        
        # Optimized LK parameters
        self.lk_params = dict(
            winSize=(15, 15),   
            maxLevel=2,       
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),  # Relaxed
            flags=0,           
            minEigThreshold=1e-5   
        )
        
        # Feature refresh counter to avoid detecting features every frame
        self.frames_since_detection = 0
        self.detection_interval = 5   
        self.min_features = 20        
        
        # Simple outlier filtering instead of DBSCAN
        self.flow_history = []
        self.history_size = 3
        
        # Motion visualization parameters
        self.smoothing_buffer = deque(maxlen=3)
        self.max_display_magnitude = 30.0
         
        try:
            self.font_big = ImageFont.truetype("arial.ttf", 16)
            self.font_small = ImageFont.truetype("arial.ttf", 12)
        except:
          
            self.font_big = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def _detect_features(self, gray):
        return cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=80,      
            qualityLevel=0.05,  
            minDistance=15,     
            blockSize=5,        
            mask=self.feature_mask, 
            useHarrisDetector=False,  
            k=0.04
        )

    def _simple_outlier_filter(self, flow_vectors):
        """Fast outlier filtering using statistical methods instead of DBSCAN"""
        if len(flow_vectors) < 4:
            return flow_vectors
            
        # Calculate distances from median
        median_flow = np.median(flow_vectors, axis=0)
        distances = np.linalg.norm(flow_vectors - median_flow, axis=1)
        
        # Keep points within 2 standard deviations
        threshold = np.median(distances) + 2 * np.std(distances)
        mask = distances <= threshold
        
        return flow_vectors[mask] if np.any(mask) else flow_vectors

    def estimate(self, current_frame):
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Check if we need to refresh features
        need_refresh = (self.prev_pts is None or 
                       len(self.prev_pts) < self.min_features or
                       self.frames_since_detection >= self.detection_interval)
        
        if need_refresh:
            self.prev_pts = self._detect_features(self.prev_gray)
            self.frames_since_detection = 0
            if self.prev_pts is None or len(self.prev_pts) < 4:
                self.prev_gray = curr_gray
                return (0.0, 0.0)
        
        # Optical flow tracking
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None, **self.lk_params
        )
        
        if next_pts is None or st is None:
            self.prev_gray = curr_gray
            self.prev_pts = self._detect_features(curr_gray)
            self.frames_since_detection = 0
            return (0.0, 0.0)
        
        #   good matches with relaxed thresholds
        mask = (st.flatten() == 1) & (err.flatten() < 50)  # Relaxed error threshold
        good_old = self.prev_pts[mask]
        good_new = next_pts[mask]
        
        if len(good_old) < 4:
            self.prev_gray = curr_gray
            self.prev_pts = self._detect_features(curr_gray)
            self.frames_since_detection = 0
            return (0.0, 0.0)
        
        #   flow vectors
        flow_vectors = (good_new - good_old).reshape(-1, 2)
        
        #   outlier filtering
        filtered_flow = self._simple_outlier_filter(flow_vectors)
        
        #   median flow
        median_flow = np.median(filtered_flow, axis=0)
        
        # Update for next frame (keep existing good points)
        self.prev_gray = curr_gray
        self.prev_pts = good_new  # Reuse tracked points instead of detecting new ones
        self.frames_since_detection += 1
        
        return tuple(median_flow.tolist())
    
    def draw_camera_motion(self, frame, dx, dy):
        """Draw camera motion visualization widget on the frame"""
        WIDGET_W, WIDGET_H = 240, 240
        ORIGIN_X, ORIGIN_Y = 20, 20
        RING_R, RING_T = 70, 10

        img = frame.copy()
        
        # Add to smoothing buffer
        self.smoothing_buffer.append((dx, dy))
        if len(self.smoothing_buffer) >= 3:
            dx = float(np.mean([v[0] for v in self.smoothing_buffer]))
            dy = float(np.mean([v[1] for v in self.smoothing_buffer]))
        
        mag = float(np.hypot(dx, dy))

        # Create glass panel background (darker)
        panel = img[ORIGIN_Y:ORIGIN_Y+WIDGET_H, ORIGIN_X:ORIGIN_X+WIDGET_W].copy()
        blur = cv2.GaussianBlur(panel, (0, 0), 20)
        dark = cv2.addWeighted(blur, 0.9, np.zeros_like(panel), 0.1, 0)
        img[ORIGIN_Y:ORIGIN_Y+WIDGET_H, ORIGIN_X:ORIGIN_X+WIDGET_W] = dark

        # Draw ring & arrow on working copy
        working = img[ORIGIN_Y:ORIGIN_Y+WIDGET_H, ORIGIN_X:ORIGIN_X+WIDGET_W]
        center = (WIDGET_W // 2, WIDGET_H // 2 + 10)

        # Background ring
        cv2.circle(working, center, RING_R, (60, 60, 60), RING_T, cv2.LINE_AA)

        # Arc color by speed
        if mag < 5:
            col = (50, 220, 50)   # Green for slow
        elif mag < 12:
            col = (50, 220, 220)  # Yellow for medium
        else:
            col = (50, 50, 255)   # Red for fast

        # Draw speed arc
        sweep = np.clip(mag / self.max_display_magnitude * 360, 0, 360)
        cv2.ellipse(working, center, (RING_R, RING_R),
                    0, -90, -90 + sweep, col, RING_T, cv2.LINE_AA)

        # Draw direction arrow (negated for the camera movement direction)
        if mag > 0.8:
            u = np.array([-dx, -dy]) / (mag + 1e-6)   
            tip = (int(center[0] + u[0] * (RING_R - 20)),
                   int(center[1] + u[1] * (RING_R - 20)))
            cv2.arrowedLine(working, center, tip, col, 3,
                            tipLength=0.25, line_type=cv2.LINE_AA)
         
        cv2.circle(working, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
 
        pil = Image.fromarray(cv2.cvtColor(working, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
 
        draw.text((10, 8), "Camera Motion",
                  font=self.font_big, fill=(255, 255, 255, 255))
 
        txt = f"{mag:4.1f} px/frame"
        try:
           
            bbox = draw.textbbox((0, 0), txt, font=self.font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
           
            tw, th = draw.textsize(txt, font=self.font_small)
        
        draw.text(((WIDGET_W - tw) // 2, WIDGET_H - th - 10),
                  txt, font=self.font_small, fill=(255, 255, 255, 255))

       
        img[ORIGIN_Y:ORIGIN_Y+WIDGET_H,
            ORIGIN_X:ORIGIN_X+WIDGET_W] = cv2.cvtColor(
                np.array(pil), cv2.COLOR_RGB2BGR)

        return img
    
    def apply_adjustment(self, tracks: dict, frame_idx: int, dx: float, dy: float):
        """Apply camera motion adjustment to tracked objects"""
        for obj, obj_tracks in tracks.items():
            for tid, info in obj_tracks.items():
                x, y = info.get("position", (0, 0))
                info["position_adjusted"] = (x - dx, y - dy)