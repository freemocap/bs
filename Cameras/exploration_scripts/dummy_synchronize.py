import cv2
import numpy as np

from pathlib import Path

class DummySynchronize:
    def __init__(self, folder_path: Path):
        self.raw_videos_path = folder_path / "raw_videos"
        self.synched_videos_path = folder_path / "synched_videos"

        self.synched_videos_path.mkdir(parents=True, exist_ok=True)

    def synchronize(self):
        self.setup()
        frame_count = self.get_lowest_frame_count() - 1
        for i in range(frame_count):
            for video_name, cap in self.capture_dict.items():
                ret, frame = cap.read()
                if ret:
                    self.writer_dict[video_name].write(frame)
                else:
                    raise ValueError(f"{video_name} has no more frames.")
        self.close()

    def setup(self):
        self.create_capture_dict()
        self.validate_fps()
        self.create_writer_dict()

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
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )
            for video_name, cap in self.capture_dict.items()
        }

    def get_lowest_frame_count(self) -> int:
        return int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in self.capture_dict.values()))
    
    def validate_fps(self):
        fps = set(cap.get(cv2.CAP_PROP_FPS) for cap in self.capture_dict.values())

        if len(fps) > 1:
            print(f"set of video fps: {fps}")
            raise ValueError("Not all videos have the same fps")
        
    def close(self):
        self.release_captures()
        self.release_writers()
        
    def release_captures(self):
        for cap in self.capture_dict.values():
            cap.release()

    def release_writers(self):
        for writer in self.writer_dict.values():
            writer.release()

if __name__ == "__main__":
    folder_path = Path("/Users/philipqueen/Documents/Humon Research Lab/Basler Stuff/calibration_attempt/")

    dummy_synchronize = DummySynchronize(folder_path)
    dummy_synchronize.synchronize()
