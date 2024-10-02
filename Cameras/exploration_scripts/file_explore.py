import subprocess
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
        print(f"reported fps: {cap.get(cv2.CAP_PROP_FPS)}")
        ffprobe_fps = get_ffprobe_fps(
            video_path
        )
        print(f"ffprobe fps: {ffprobe_fps}")
        cap.release()


def get_ffprobe_fps(video_path: Path) -> float:
    duration_subprocess = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            f"{video_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = duration_subprocess.stdout
    print(f"duration from ffprobe: {float(duration)} seconds")

    frame_count_subprocess = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            f"{video_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    frame_count = int(frame_count_subprocess.stdout)

    print(f"frame count from ffprobe: {frame_count}")

    return frame_count / float(duration)


if __name__ == "__main__":
    # folder_path = Path(
    #     "/Users/philipqueen/Documents/Humon Research Lab/Basler Stuff/calibration_attempt/"
    # )
    folder_path = Path(
        "/home/scholl-lab/recordings/2_camera_test"
    )
    raw_videos_path = folder_path / "raw_videos"
    synched_videos_path = folder_path / "synched_videos"

    print_video_info(raw_videos_path)
