import cv2

from pathlib import Path
from datetime import datetime


def print_video_info(videos_path: Path):
    for video_path in videos_path.iterdir():
        if video_path.suffix not in {".avi", ".AVI", ".mp4", ".MP4"}:
            continue
        print(f"video name: {video_path.name}")
        starting_timestamp = video_path.stem.split("__")[-1]
        starting_datetime = datetime.fromisoformat(starting_timestamp)
        print(f"starting datetime: {starting_datetime}")
        cap = cv2.VideoCapture(str(video_path))
        print(f"frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")
        cap.release()


if __name__ == "__main__":
    folder_path = Path(
        "/Users/philipqueen/Documents/Humon Research Lab/Basler Stuff/calibration_attempt/"
    )
    raw_videos_path = folder_path / "raw_videos"
    synched_videos_path = folder_path / "synched_videos"

    print_video_info(synched_videos_path)
