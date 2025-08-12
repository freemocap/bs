import shutil
from typing import Dict
import cv2
import numpy as np

from pathlib import Path
from python_code.cameras.diagnostics.skellycam_plots import timestamps_array_to_dictionary, calculate_camera_diagnostic_results
from python_code.cameras.intrinsics.intrinsics_corrector import IntrinsicsCorrector, get_calibrations_from_json

class TimestampSynchronize:
    def __init__(self, folder_path: Path, flip_videos: bool = False, correct_intrinsics: bool = True):
        self.flip_videos = flip_videos
        self.correct_intrinsics = correct_intrinsics
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        if not folder_path.exists:
            raise FileNotFoundError("Input folder path does not exist")

        raw_videos_path = folder_path / "raw_videos"
        if not raw_videos_path.exists():
            raw_videos_path = folder_path
        self.raw_videos_path = raw_videos_path
        if self.correct_intrinsics:
            self.synched_videos_path = folder_path / "synchronized_corrected_videos"
        else:
            self.synched_videos_path = folder_path / "synchronized_videos"

        self.timestamp_file_name = "timestamps.npy"
        timestamp_path = folder_path / self.timestamp_file_name
        if not timestamp_path.exists():
            timestamp_path = self.raw_videos_path / self.timestamp_file_name
            if not timestamp_path.exists():
                raise FileNotFoundError("Unable to find timestamps.npy file in main folder or raw_videos folder")
        self.timestamps = np.load(timestamp_path)

        self.timestamp_mapping_file_name = "timestamp_mapping.json"
        self.timestamp_mapping_path = raw_videos_path / self.timestamp_mapping_file_name

        self.synched_videos_path.mkdir(parents=True, exist_ok=True)

    def synchronize(self):
        self.setup()
        target_framecount = (
            self.get_lowest_postoffset_frame_count() - 1
        )  # -1 accounts for rounding errors in offset i.e. drop a frame off the end to be sure we don't overflow array
        print(f"synchronizing videos to target framecount: {target_framecount}")
        new_timestamps = np.zeros((self.timestamps.shape[0], target_framecount))
        for i, (video_name, cap) in enumerate(self.capture_dict.items()):
            print(f"synchronizing: {video_name}")
            current_framecount = 0
            offset = self.frame_offset_dict[video_name]
            new_timestamps[i] = self.timestamps[i, offset:target_framecount+offset]
            while current_framecount < target_framecount:  # < to account for 0 indexing
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"{video_name} has no more frames.")
                if offset <= 0:
                    if self.flip_videos:
                        frame = cv2.flip(frame, -1)
                    if self.correct_intrinsics:
                        frame = self.intrinsics_correctors[video_name].correct_frame(frame)
                    self.writer_dict[video_name].write(frame)
                    current_framecount += 1
                else:
                    offset -= 1
        print("Saving new timestamps file")
        np.save(self.synched_videos_path / self.timestamp_file_name, new_timestamps)
        if self.timestamp_mapping_path.exists():
            print("Copying timestamp mapping file")
            shutil.copyfile(self.timestamp_mapping_path, self.synched_videos_path / self.timestamp_mapping_file_name)

        self.close()
        print("Done synchronizing")

    def setup(self):
        print("Setting up for synchronization...")
        self.create_capture_dict()
        self.validate_fps()
        if self.correct_intrinsics:
            self.create_intrinsics_correctors()
        self.print_diagnostics()
        self.create_writer_dict()
        self.create_starting_timestamp_dict()
        self.create_frame_offset_dict()

    def print_diagnostics(self):
        try:
            timestamp_dictionary = timestamps_array_to_dictionary(self.timestamps)
            diagnostics = calculate_camera_diagnostic_results(timestamps_dictionary=timestamp_dictionary)
            print(f"Timestamp diagnostics: {diagnostics}")
        except Exception as e:
            print(f"Unable to print timestamp diagnostics due to error {e}")

    def create_capture_dict(self):
        self.capture_dict = {
            video_path.name: cv2.VideoCapture(str(video_path))
            for video_path in self.raw_videos_path.glob("*.mp4")
        }

    def create_intrinsics_correctors(self):
        calibrations = get_calibrations_from_json()

        self.intrinsics_correctors: dict[str, IntrinsicsCorrector] = {}

        for video_name, cap in self.capture_dict.items():
            name = video_name.split(".")[0] 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            camera_intrinsics = calibrations[name]
            self.intrinsics_correctors[video_name] = IntrinsicsCorrector.from_dict(camera_intrinsics, width, height)

        if len(self.intrinsics_correctors) != len(self.capture_dict):
            raise ValueError("Unable to find intrinsics for all videos")

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
            video_name: int(self.timestamps[i, 0])
            for i, video_name in enumerate(self.capture_dict.keys())
        }
        print(f"starting timestamp dict: {self.starting_timestamp_dict}")


    def create_frame_offset_dict(self):
        latest_start = sorted(self.starting_timestamp_dict.values())[-1]

        self.frame_offset_dict: Dict[str, int] = {}

        for i, (video_name, time) in enumerate(self.starting_timestamp_dict.items()):
            first_index_over_latest_start = int(np.searchsorted(self.timestamps[i, :], latest_start))
            self.frame_offset_dict[video_name] = first_index_over_latest_start

        print(f"Frame offset dict: { self.frame_offset_dict}")

    def get_lowest_postoffset_frame_count(self) -> int:
        return int(
            min(
                cap.get(cv2.CAP_PROP_FRAME_COUNT) - self.frame_offset_dict[video_name]
                for video_name, cap in self.capture_dict.items()
            )
        )

    def validate_fps(self):
        fps_dict = {cam: cap.get(cv2.CAP_PROP_FPS) for cam, cap in self.capture_dict.items()}
        fps = set(fps_dict.values())

        if len(fps) > 1:
            print(f"video fps: {fps_dict}")
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
    folder_path = Path("/home/scholl-lab/recordings/session_2025-07-11/calibration")

    timestamp_synchronize = TimestampSynchronize(folder_path, flip_videos=True)
    timestamp_synchronize.synchronize()
