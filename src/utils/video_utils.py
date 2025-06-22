import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
 
 

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("No frames to save. output_video_frames is empty.")

    # Auto-create directory if missing
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    print(f"Saving video to: {output_video_path}")
    print(f"Resolution: {width}x{height}, Frames: {len(output_video_frames)}")

    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    if not out.isOpened():
        raise IOError(f"Failed to open VideoWriter with path: {output_video_path}")

    for frame in output_video_frames:
        out.write(frame)

    out.release()
