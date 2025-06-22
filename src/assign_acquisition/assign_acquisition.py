import sys
sys.path.append('../')

from utils import get_center_of_bbox, measure_distance

class BallAcquisition:
    def __init__(self, max_player_ball_distance: float = 70.0):
        self.max_dist = max_player_ball_distance

    def assign_ball_to_player(self, players: dict, ball_bbox: list | tuple) -> int:
        if not players or not ball_bbox:
            return -1
        
        ball_pos = get_center_of_bbox(ball_bbox)
        min_dist = float('inf')
        closest_player = -1

        for pid, pdata in players.items():
            pbbox = pdata["bbox"]
            left_foot = (pbbox[0], pbbox[3])
            right_foot = (pbbox[2], pbbox[3])
            dist = min(
                measure_distance(left_foot, ball_pos),
                measure_distance(right_foot, ball_pos)
            )
            if dist < self.max_dist and dist < min_dist:
                min_dist = dist
                closest_player = pid

        return closest_player
