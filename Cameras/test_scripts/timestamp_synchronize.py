from datetime import datetime, timedelta
from typing import Dict
import cv2

from pathlib import Path


class TimestampSynchronize:
    def __init__(self, folder_path: Path):
        self.raw_videos_path = folder_path / "raw_videos"
        self.synched_videos_path = folder_path / "synched_videos"

        self.synched_videos_path.mkdir(parents=True, exist_ok=True)

    def synchronize(self):
        self.setup()
        target_framecount = (
            self.get_lowest_postoffset_frame_count() - 1
        )  # -1 accounts for rounding errors in calculating offset
        print(f"synchronizing videos to target framecount: {target_framecount}")
        for video_name, cap in self.capture_dict.items():
            print(f"synchronizing: {video_name}")
            current_framecount = 0
            offset = self.frame_offset_dict[video_name]
            while current_framecount < target_framecount:  # < to account for 0 indexing
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"{video_name} has no more frames.")
                if offset <= 0:
                    self.writer_dict[video_name].write(frame)
                    current_framecount += 1
                else:
                    offset -= 1
        self.close()
        print("Done synchronizing")

    def setup(self):
        print("Setting up for synchronization...")
        self.create_capture_dict()
        self.validate_fps()
        self.create_writer_dict()
        self.create_starting_timestamp_dict()
        self.create_frame_offset_dict()

    def create_capture_dict(self):
        self.capture_dict = {
            video_path.name: cv2.VideoCapture(str(video_path))
            for video_path in self.raw_videos_path.iterdir()
        }

    def create_writer_dict(self):
        self.writer_dict = {
            video_name: cv2.VideoWriter(
                str(self.synched_videos_path / (video_name.split(".")[0] + ".mp4")),
                cv2.VideoWriter.fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS),
                (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )
            for video_name, cap in self.capture_dict.items()
        }

    def create_starting_timestamp_dict(self):
        self.starting_timestamp_dict = {
            video_name: datetime.fromisoformat(video_name.split(".")[0].split("__")[-1])
            for video_name in self.capture_dict.keys()
        }

    def create_frame_offset_dict(self):
        latest_start = sorted(self.starting_timestamp_dict.values())[-1]
        frame_duration_seconds = 1 / self.fps

        self.frame_offset_dict: Dict[str, int] = {}

        for video_name, time in self.starting_timestamp_dict.items():
            offset_microseconds = (latest_start - time) / timedelta(microseconds=1)
            offset_frames = round(
                timedelta(microseconds=offset_microseconds)
                / timedelta(seconds=frame_duration_seconds)
            )
            print(f"{video_name} offset in microseconds: {offset_microseconds}")
            print(f"{video_name} offset in frames: {offset_frames}")
            print(
                f"{video_name} total frames: {int(self.capture_dict[video_name].get(cv2.CAP_PROP_FRAME_COUNT))}"
            )
            self.frame_offset_dict[video_name] = offset_frames

    def get_lowest_postoffset_frame_count(self) -> int:
        return int(
            min(
                cap.get(cv2.CAP_PROP_FRAME_COUNT) - self.frame_offset_dict[video_name]
                for video_name, cap in self.capture_dict.items()
            )
        )

    def validate_fps(self):
        fps = set(cap.get(cv2.CAP_PROP_FPS) for cap in self.capture_dict.values())

        if len(fps) > 1:
            print(f"set of video fps: {fps}")
            raise ValueError("Not all videos have the same fps")

        self.fps = fps.pop()

    def close(self):
        print("Closing all capture objects and writers")
        self.release_captures()
        self.release_writers()

    def release_captures(self):
        for cap in self.capture_dict.values():
            cap.release()

    def release_writers(self):
        for writer in self.writer_dict.values():
            writer.release()


if __name__ == "__main__":
    # folder_path = Path(
    #     "/Users/philipqueen/Documents/Humon Research Lab/Basler Stuff/calibration_attempt/"
    # )

    folder_path = Path(
        "/Users/philipqueen/Documents/Humon Research Lab/Basler Stuff/fabio_hand/"
    )

    timestamp_synchronize = TimestampSynchronize(folder_path)
    timestamp_synchronize.synchronize()
