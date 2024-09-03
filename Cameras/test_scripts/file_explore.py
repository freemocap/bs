import cv2

from pathlib import Path

def print_video_info(videos_path: Path):
    for video_path in videos_path.iterdir():
        print(f"video name: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        print(f"frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")
        cap.release()

if __name__ == "__main__":
    folder_path = Path("/Users/philipqueen/Documents/Humon Research Lab/Basler Stuff/calibration_attempt/")
    raw_videos_path = folder_path / "raw_videos"
    synched_videos_path = folder_path / "synched_videos"

    print_video_info(synched_videos_path)