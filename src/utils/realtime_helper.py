from assign_team.assign_team import TeamAssigner


def reshape_frame_tracks(frame_tracks):
    reshaped = {"players": [], "referees": [], "ball": []}
    for f in frame_tracks:
        for k in reshaped:
            reshaped[k].append(f[k])
    return reshaped

def annotate_teams(frame_idx, frames, frame_tracks, teamer):
    """Assigns team IDs and colors to players in a given frame."""
    if frame_idx == 0 and frame_tracks[0]["players"]:
        teamer.fit(frames[0], frame_tracks[0]["players"])
    if teamer.kmeans is None:
        return
    for pid, info in frame_tracks[frame_idx]["players"].items():
        if pid not in teamer.player_team_cache:
            team = teamer.predict(frames[frame_idx], info["bbox"], pid)
            teamer.player_team_cache[pid] = team
        else:
            team = teamer.player_team_cache[pid]
        info["team"] = team
        info["team_color"] = teamer.color_for_team(team)
