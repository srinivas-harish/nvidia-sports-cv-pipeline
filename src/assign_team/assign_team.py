from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))

def _crop_top_half(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)
    
    if x2 <= x1 or y2 <= y1:
        logger.debug(f"Invalid bbox {bbox} in frame of size ({w}, {h})")
        return None
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        logger.debug(f"Empty crop for bbox {bbox}")
        return None
    return crop[:max(1, crop.shape[0] // 2), :]

def _extract_dominant_color(region: np.ndarray, resize_dim: int = 32, min_pixels: int = 10) -> Optional[np.ndarray]:
    if region is None or region.size == 0:
        logger.debug("Cannot extract color from empty region")
        return None
    
    #   fixed small dimension for speed
    region_small = cv2.resize(region, (resize_dim, resize_dim), interpolation=cv2.INTER_NEAREST)
    pixels = region_small.reshape(-1, 3).astype(np.float32)
    
    if len(pixels) < min_pixels:
        logger.debug(f"Insufficient pixels ({len(pixels)}) for clustering")
        return None

    #   KMeans with 2 clusters, single init for speed
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, max_iter=30, random_state=0)
    labels = kmeans.fit_predict(pixels)
    
    #   background by checking corners
    h, w = region_small.shape[:2]
    corners = np.array([labels[0], labels[w - 1], labels[-w], labels[-1]])
    background_cluster = np.bincount(corners).argmax()
    jersey_cluster = 1 - background_cluster
    
    #   jersey color
    color = kmeans.cluster_centers_[jersey_cluster].astype(np.float32)
    return color

class TeamAssigner:
    def __init__(self, resize_dim: int = 32, min_pixels: int = 10):
        self.kmeans: Optional[KMeans] = None
        self.team_colors: Dict[int, Tuple[int, int, int]] = {
            1: (255, 0, 0),  # Default Team 1: Red
            2: (0, 0, 255)   # Default Team 2: Blue
        }
        self.player_team_cache: Dict[int, int] = {}
        self.resize_dim = resize_dim
        self.min_pixels = min_pixels
        self.last_frame_rgb: Optional[np.ndarray] = None
        self.last_frame_id: Optional[int] = None

    def fit(self, frame: np.ndarray, player_dets: Dict[int, Dict]) -> None:
        if len(player_dets) < 2:
            raise ValueError("At least 2 player detections required to fit team colors")
        
        #   frame to RGB  
        frame_id = id(frame)
        if self.last_frame_id != frame_id:
            self.last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame_id = frame_id
        
        colors = []
        valid_ids = []
        for player_id, info in player_dets.items():
            patch = _crop_top_half(self.last_frame_rgb, info["bbox"])
            color = _extract_dominant_color(patch, self.resize_dim, self.min_pixels)
            if color is not None:
                colors.append(color)
                valid_ids.append(player_id)
        
        if len(colors) < 2:
            logger.error(f"Only {len(colors)} valid colors extracted; cannot fit model")
            raise ValueError("Insufficient valid color data for clustering")
        
        colors = np.array(colors, dtype=np.float32)
        #   2 clusters for team colors
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, max_iter=50, random_state=1)
        self.kmeans.fit(colors)
        
        #   team colors based on cluster centers
        for i, center in enumerate(self.kmeans.cluster_centers_, 1):
            self.team_colors[i] = tuple(int(x) for x in center)
        
        # Cache team assignments for valid players
    
    def predict(self, frame, bbox, player_id):
        """Enhanced predict method with debugging"""
        
        # Your existing prediction logic here...
        # (I'm assuming you have the actual implementation)
        
        # Example of what might be happening:
        team = self._get_team_prediction(frame, bbox)  # Your actual method
        
        # Debug output
        #print(f"[TEAM DEBUG] Player {player_id}: predicted team = {team}")
        
        # Check if team values are what you expect
        #if team not in [1, 2]:
        #    print(f"[TEAM WARNING] Unexpected team value: {team} for player {player_id}")
            # You might want to default to a specific team or handle this case
        
        return team


    def color_for_team(self, team: int) -> Tuple[int, int, int]:
        return self.team_colors.get(team, (255, 255, 255))

    def reset(self) -> None:
        self.kmeans = None
        self.team_colors = {
            1: (255, 0, 0),
            2: (0, 0, 255)
        }
        self.player_team_cache.clear()
        self.last_frame_rgb = None
        self.last_frame_id = None
        logger.info("TeamAssigner state reset")
        
    def _get_team_prediction(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> int:
        """Extracts the player's jersey color and returns the predicted team (1 or 2)."""
        if self.kmeans is None:
            return 0  # Default to neutral if not fitted

        # Reuse cached RGB
        frame_id = id(frame)
        if self.last_frame_id != frame_id:
            self.last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame_id = frame_id

        patch = _crop_top_half(self.last_frame_rgb, bbox)
        color = _extract_dominant_color(patch, self.resize_dim, self.min_pixels)

        if color is None:
            return 0  # Neutral if no color could be extracted

        label = int(self.kmeans.predict(color.reshape(1, -1))[0])
        return label + 1  # Shift from [0,1] to [1,2]
