
class PlayerPossessionTracker:
    def __init__(self):
        self.player_stats = {}  # {player_id: {'count': int, 'total_frames': int, 'current_streak': int}}
        self.current_possessor = None
        self.possession_start_frame = None
        
    def update_possession(self, assigned_player, frame_num):
        """Update possession tracking for current frame"""
        
        # If no player assigned or same player continues possession
        if assigned_player == -1:
            # No possession - end current streak if exists
            if self.current_possessor is not None:
                self._end_possession(frame_num)
            return
            
        # Initialize player stats if first time seeing this player
        if assigned_player not in self.player_stats:
            self.player_stats[assigned_player] = {
                'count': 0,
                'total_frames': 0, 
                'current_streak': 0
            }
        
        # Check if possession changed
        if self.current_possessor != assigned_player:
            # End previous possession
            if self.current_possessor is not None:
                self._end_possession(frame_num)
            
            # Start new possession
            self.current_possessor = assigned_player
            self.possession_start_frame = frame_num
            self.player_stats[assigned_player]['count'] += 1
            self.player_stats[assigned_player]['current_streak'] = 1
        else:
            # Continue current possession
            self.player_stats[assigned_player]['current_streak'] += 1
            
        # Always increment total frames for current possessor
        self.player_stats[assigned_player]['total_frames'] += 1
    
    def _end_possession(self, frame_num):
        """End current possession streak"""
        if self.current_possessor is not None:
            # The streak count is already updated in update_possession
            self.current_possessor = None
            self.possession_start_frame = None
    
    def print_stats(self, frame_num):
        """Print current possession statistics"""
        if not self.player_stats:
            print(f"[Frame {frame_num}] No possession data yet")
            return
            
        #print(f"\n[Frame {frame_num}] === POSSESSION STATS ===")
        
        # Sort players by total frames possessed (descending)
        sorted_players = sorted(self.player_stats.items(), 
                              key=lambda x: x[1]['total_frames'], 
                              reverse=True)
        
        for player_id, stats in sorted_players:
            current_indicator = " 🏈" if player_id == self.current_possessor else ""
            #print(f"Player {player_id}{current_indicator}: "
            #      f"{stats['count']} possessions, "
            #      f"{stats['total_frames']} total frames")
         
        
        #print("=" * 40)
    
    def get_top_possessors(self, n=3):
        """Get top N players by total possession time"""
        if not self.player_stats:
            return []
            
        sorted_players = sorted(self.player_stats.items(), 
                              key=lambda x: x[1]['total_frames'], 
                              reverse=True)
        return sorted_players[:n]
