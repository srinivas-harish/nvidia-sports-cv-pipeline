import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from overlays.overlay_helper import draw_camera_motion_widget


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
        return draw_camera_motion_widget(
            frame=frame,
            dx=dx,
            dy=dy,
            widget_working=self.widget_working,
            widget_center=self.widget_center,
            ORIGIN_X=self.ORIGIN_X,
            ORIGIN_Y=self.ORIGIN_Y,
            WIDGET_W=self.WIDGET_W,
            WIDGET_H=self.WIDGET_H,
            RING_R=self.RING_R,
            RING_T=self.RING_T,
            max_display_magnitude=self.max_display_magnitude,
            font_big=self.font_big,
            font_small=self.font_small,
            smoothing_buffer=self.smoothing_buffer
        )


    def apply_adjustment(self, tracks: dict, frame_idx: int, dx: float, dy: float):
        """Apply motion adjustment to tracks"""
        for obj, obj_tracks in tracks.items():
            for tid, info in obj_tracks.items():
                x, y = info.get("position", (0, 0))
                info["position_adjusted"] = (x - dx, y - dy)