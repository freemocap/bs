import shutil
from typing import Dict
import cv2
import numpy as np
import json

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

        self.index_to_serial_number_map_file_name = "index_to_serial_number_mapping.json"
        self.index_to_serial_number_map_path = raw_videos_path / self.index_to_serial_number_map_file_name
        if not self.index_to_serial_number_map_path.exists():
            raise FileExistsError("No index_to_serial_number_mapping.json file found - needed to match timestamps to video")
        
        with self.index_to_serial_number_map_path.open() as fp:
            self.index_to_serial_number_map = json.load(fp)


        self.synched_videos_path.mkdir(parents=True, exist_ok=True)

    def synchronize(self):
        self.setup()
        target_framecount = (
            self.get_lowest_postoffset_frame_count() - 1
        )  # -1 accounts for rounding errors in offset i.e. drop a frame off the end to be sure we don't overflow array
        print(f"synchronizing videos to target framecount: {target_framecount}")
        for video_name, cap in self.capture_dict.items():
            print(f"synchronizing: {video_name}")
            current_framecount = 0
            offset = self.frame_offset_dict[video_name.split(".")[0]]
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
        print("Saving new timestamps files")
        self.save_new_timestamps(target_framecount=target_framecount)
        if self.timestamp_mapping_path.exists():
            print("Copying timestamp mapping file")
            shutil.copyfile(self.timestamp_mapping_path, self.synched_videos_path / self.timestamp_mapping_file_name)
        # shutil.copyfile(self.index_to_serial_number_map_path, self.synched_videos_path / self.index_to_serial_number_map_file_name)

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
        file_suffix = "_synchronized_corrected" if self.correct_intrinsics else "_synchronized"
        self.writer_dict = {
            video_name: cv2.VideoWriter(
                str(self.synched_videos_path / (video_name.split(".")[0] + file_suffix + ".mp4")),
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
            video_name + ".mp4": int(self.timestamps[int(i), 0])
            for i, video_name in self.index_to_serial_number_map.items()
        }
        if not set(self.capture_dict.keys()).issubset(set(self.starting_timestamp_dict.keys())):
            raise ValueError(f"All video names ({self.capture_dict.keys()}) not found in timestamp dict ({self.starting_timestamp_dict.keys()})") 
        print(f"starting timestamp dict: {self.starting_timestamp_dict}")


    def create_frame_offset_dict(self):
        print(f"finding latest timestamp in {sorted(self.starting_timestamp_dict.values())}")
        latest_start = sorted(self.starting_timestamp_dict.values())[-1]
        print(f"lastest start is {latest_start}")
        self.frame_offset_dict: Dict[str, int] = {}

        for i, video_name in self.index_to_serial_number_map.items():
            closest_frame_number = int(np.argmin(np.abs(self.timestamps[int(i),:] - latest_start)))
            self.frame_offset_dict[video_name] = closest_frame_number

        self.frame_offset_dict = self.refine_frame_offsets(self.frame_offset_dict)
        for i, video_name in self.index_to_serial_number_map.items():
            print(f"starting time for cam {video_name} is {self.timestamps[int(i), self.frame_offset_dict[video_name]]}")
            print(f"start time off frame offset: {self.timestamps[int(i), self.frame_offset_dict[video_name]] - latest_start}")

        print(f"Frame offset dict: {self.frame_offset_dict}")

    def refine_frame_offsets(self, frame_offsets: dict):
        print("refining frame offsets to minimize frame spread")
        reference_camera = next((key for key, value in frame_offsets.items() if value == 0), None)
        if reference_camera is None:
            print("No camera with 0 offset found")
            return frame_offsets
        start_times = self._get_start_times_from_frame_offsets(frame_offsets=frame_offsets)
        best_spread = max(start_times.values()) - min(start_times.values())
        best_frame_offsets = frame_offsets

        print(f"starting offsets are {best_frame_offsets} with spread of {best_spread} ns")

        should_continue = True
        while should_continue:
            should_continue = False
            latest_camera = next(key for key, value in start_times.items() if value == max(start_times.values()))
            earliest_camera = next(key for key, value in start_times.items() if value == min(start_times.values()))

            # move latest back
            if latest_camera != reference_camera:
                trial_frame_offsets = best_frame_offsets.copy()
                trial_frame_offsets[latest_camera] -= 1
                trial_start_times = self._get_start_times_from_frame_offsets(trial_frame_offsets)
                trial_spread = max(trial_start_times.values()) - min(trial_start_times.values())
                # print(f"trial offsets are {trial_frame_offsets} with spread of {trial_spread} ns")
                if trial_spread < best_spread:
                    best_spread = trial_spread
                    best_frame_offsets = trial_frame_offsets
                    should_continue = True
                    continue

            # move earliest forward
            if earliest_camera != reference_camera:
                trial_frame_offsets = best_frame_offsets.copy()
                trial_frame_offsets[earliest_camera] += 1
                trial_start_times = self._get_start_times_from_frame_offsets(trial_frame_offsets)
                trial_spread = max(trial_start_times.values()) - min(trial_start_times.values())
                # print(f"trial offsets are {trial_frame_offsets} with spread of {trial_spread} ns")
                if trial_spread < best_spread:
                    best_spread = trial_spread
                    best_frame_offsets = trial_frame_offsets
                    should_continue = True
                    continue

            # move earliest forward and latest back
            if earliest_camera != reference_camera and latest_camera != reference_camera:
                trial_frame_offsets = best_frame_offsets.copy()
                trial_frame_offsets[latest_camera] -= 1
                trial_frame_offsets[earliest_camera] += 1
                trial_start_times = self._get_start_times_from_frame_offsets(trial_frame_offsets)
                trial_spread = max(trial_start_times.values()) - min(trial_start_times.values())
                # print(f"trial offsets are {trial_frame_offsets} with spread of {trial_spread} ns")
                if trial_spread < best_spread:
                    best_spread = trial_spread
                    best_frame_offsets = trial_frame_offsets
                    should_continue = True
                    continue

        print(f"best frame offsets found are {best_frame_offsets} with a spread of {best_spread} ns")

        return best_frame_offsets




    def _get_start_times_from_frame_offsets(self, frame_offsets: dict):
        return {video_name: self.timestamps[int(i), frame_offsets[video_name]] for i, video_name in self.index_to_serial_number_map.items()}

    def get_lowest_postoffset_frame_count(self) -> int:
        return int(
            min(
                cap.get(cv2.CAP_PROP_FRAME_COUNT) - self.frame_offset_dict[video_name.split(".")[0]]
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

    def save_new_timestamps(self, target_framecount: int):
        for i, video_name in self.index_to_serial_number_map.items():
            offset = self.frame_offset_dict[video_name]
            new_timestamps = self.timestamps[int(i), offset:target_framecount+offset]
            cam_name = video_name.split(".")[0]
            timestamps_name = f"{cam_name}_synchronized_timestamps_basler_time.npy"
            print(f"Cam {cam_name} starts at {new_timestamps[0]}, ends at {new_timestamps[-1]}")
            np.save(self.synched_videos_path / timestamps_name, new_timestamps)
        

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
    folder_path = Path("/home/scholl-lab/ferret_recordings/session_2025-10-11_ferret_402_E02/calibration")

    timestamp_synchronize = TimestampSynchronize(folder_path, flip_videos=False)
    # timestamp_synchronize.setup()
    timestamp_synchronize.synchronize()
