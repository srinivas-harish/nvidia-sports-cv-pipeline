import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque

class FrameBuffer:
    """Maintains a rolling buffer of previous frames for single-frame processing."""
    def __init__(self, maxlen=2):
        self.buffer = deque(maxlen=maxlen)

    def add(self, frame):
        self.buffer.append(frame)

    def get_previous(self):
        if len(self.buffer) < 2:
            return None
        return self.buffer[-2]

class CameraMotionEstimator:
    def __init__(self, initial_frame, calculation_interval=10):
        h, w = initial_frame.shape[:2]
        
        # Store original dimensions but work with smaller resolution for calculations
        self.original_size = (w, h)
        self.calc_scale = 0.5  # Process at half resolution for speed
        self.calc_size = (int(w * self.calc_scale), int(h * self.calc_scale))
         
        ch, cw = self.calc_size[1], self.calc_size[0]
        self.feature_mask = np.zeros((ch, cw), dtype=np.uint8)
         
        border = min(ch, cw) // 8
        self.feature_mask[0:border, 0:border] = 1  # Top-left
        self.feature_mask[0:border, cw-border:cw] = 1  # Top-right  
        self.feature_mask[ch-border:ch, 0:border] = 1  # Bottom-left
        self.feature_mask[ch-border:ch, cw-border:cw] = 1  # Bottom-right
        
        # Calculation interval and caching
        self.calculation_interval = calculation_interval
        self.frame_count = 0
        self.cached_motion = (0.0, 0.0)
        self.motion_history = deque(maxlen=5)  # For interpolation
        
        # Frame buffer for reduced resolution frames
        self.frame_buffer = FrameBuffer(maxlen=2)
        
        # Pre-allocate arrays to avoid memory allocation overhead
        self.prev_gray = None
        self.curr_gray = None
         
        self.lk_params = dict(
            winSize=(11, 11),  
            maxLevel=1,        
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05),   
            flags=0,
            minEigThreshold=1e-4  
        )
         
        self.smoothing_buffer = deque(maxlen=2)
        self.max_display_magnitude = 30.0
         
        self.WIDGET_W, self.WIDGET_H = 200, 200   
        self.ORIGIN_X, self.ORIGIN_Y = 20, 20
        self.RING_R, self.RING_T = 60, 8 
        self.widget_center = (self.WIDGET_W // 2, self.WIDGET_H // 2 + 10)
         
        self.widget_working = np.zeros((self.WIDGET_H, self.WIDGET_W, 3), dtype=np.uint8)
         
        try:
            self.font_big = ImageFont.truetype("arial.ttf", 14)
            self.font_small = ImageFont.truetype("arial.ttf", 10)
        except:
            self.font_big = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
             
        self._add_frame_to_buffer(initial_frame)

    def _add_frame_to_buffer(self, frame):
        """Add frame to buffer at calculation resolution"""
        if frame.shape[:2] != (self.calc_size[1], self.calc_size[0]):
            small_frame = cv2.resize(frame, self.calc_size, interpolation=cv2.INTER_LINEAR)
        else:
            small_frame = frame
        self.frame_buffer.add(small_frame)

    def _detect_features(self, gray):
        """Detect features with reduced parameters for speed"""
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=50,       
            qualityLevel=0.08,   
            minDistance=12,      
            blockSize=3,         
            mask=self.feature_mask,
            useHarrisDetector=False,
            k=0.04
        )

    def _simple_outlier_filter(self, flow_vectors):
        """Fast outlier filtering"""
        if len(flow_vectors) < 3:
            return flow_vectors
         
        median_flow = np.median(flow_vectors, axis=0)
        distances = np.sum(np.abs(flow_vectors - median_flow), axis=1)  # L1 norm is faster
        threshold = np.median(distances) * 2.5   
        mask = distances <= threshold
        return flow_vectors[mask] if np.any(mask) else flow_vectors

    def _interpolate_motion(self):
        """Interpolate motion between calculations"""
        if len(self.motion_history) < 2:
            return self.cached_motion
        
       
        pos = self.frame_count % self.calculation_interval
        factor = pos / self.calculation_interval
        
        if len(self.motion_history) >= 2:
            prev_motion = self.motion_history[-2]
            curr_motion = self.motion_history[-1]
            interpolated = (
                prev_motion[0] + (curr_motion[0] - prev_motion[0]) * factor,
                prev_motion[1] + (curr_motion[1] - prev_motion[1]) * factor
            )
            return interpolated
        
        return self.cached_motion

    def estimate(self, current_frame):
        """Estimate camera motion with frame skipping optimization"""
        self.frame_count += 1
        
        # Only calculate every N frames
        if self.frame_count % self.calculation_interval != 0:
            return self._interpolate_motion()
        
        # Add current frame to buffer
        self._add_frame_to_buffer(current_frame)
        
        prev_frame = self.frame_buffer.get_previous()
        if prev_frame is None:
            return (0.0, 0.0)

    
        if self.prev_gray is None or self.prev_gray.shape != prev_frame.shape[:2]:
            self.prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            self.curr_gray = cv2.cvtColor(self.frame_buffer.buffer[-1], cv2.COLOR_BGR2GRAY)
        else:
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY, dst=self.prev_gray)
            cv2.cvtColor(self.frame_buffer.buffer[-1], cv2.COLOR_BGR2GRAY, dst=self.curr_gray)

        # Detect features
        prev_pts = self._detect_features(self.prev_gray)
        if prev_pts is None or len(prev_pts) < 3:
            return self.cached_motion

        # Calculate optical flow
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, self.curr_gray, prev_pts, None, **self.lk_params
        )

        if next_pts is None or st is None:
            return self.cached_motion

        # Filter good points
        mask = (st.flatten() == 1) & (err.flatten() < 30)  # More lenient error threshold
        good_old = prev_pts[mask]
        good_new = next_pts[mask]

        if len(good_old) < 3:
            return self.cached_motion

        # Calculate motion
        flow_vectors = (good_new - good_old).reshape(-1, 2)
        filtered_flow = self._simple_outlier_filter(flow_vectors)
        median_flow = np.median(filtered_flow, axis=0)
        
        # Scale back to original resolution
        scaled_motion = (
            float(median_flow[0] / self.calc_scale),
            float(median_flow[1] / self.calc_scale)
        )
        
        # Update cache and history
        self.cached_motion = scaled_motion
        self.motion_history.append(scaled_motion)
        
        return scaled_motion

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