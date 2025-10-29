
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


class TimestampConverter:
    def __init__(self, folder_path: Path, include_eyes: bool = True):
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        if not folder_path.exists:
            raise FileNotFoundError("Input folder path does not exist")

        self.raw_videos_path = folder_path / "raw_videos"
        self.synched_videos_path = folder_path / "synchronized_corrected_videos"
        if not self.synched_videos_path.exists():
            self.synched_videos_path = folder_path / "synchronized_videos"

        self.basler_timestamp_mapping_file_name = "timestamp_mapping.json"
        basler_timestamp_mapping_file = (
                self.raw_videos_path / self.basler_timestamp_mapping_file_name
        )
        with open(basler_timestamp_mapping_file) as basler_timestamp_mapping_file:
            self.basler_timestamp_mapping = json.load(basler_timestamp_mapping_file)

        if include_eyes:
            self.pupil_path = folder_path / "pupil_output"
            self.pupil_eye0_video_path = self.pupil_path / "eye0.mp4"
            self.pupil_eye1_video_path = self.pupil_path / "eye1.mp4"

            self.pupil_timestamp_mapping_file_name = "info.player.json"
            pupil_timestamp_mapping_file = (
                    self.pupil_path / self.pupil_timestamp_mapping_file_name
            )
            with open(pupil_timestamp_mapping_file) as pupil_timestamp_mapping_file:
                self.pupil_timestamp_mapping = json.load(pupil_timestamp_mapping_file)

        self.synchronization_metadata = {}

        if include_eyes:
            self.load_and_convert_pupil_timestamps()
        self.load_and_convert_basler_timestamps()
        self.load_index_to_serial_number()
        self.verify_index_to_serial_number()

    def seconds_to_nanoseconds(self, seconds: float) -> int:
        return int(seconds * 1e9)

    def nanoseconds_to_seconds(self, nanoseconds: int) -> float:
        return nanoseconds / 1e9

    @property
    def pupil_start_time(self) -> int:
        return self.seconds_to_nanoseconds(
            self.pupil_timestamp_mapping["start_time_synced_s"]
        )

    @property
    def pupil_start_time_utc(self) -> int:
        return self.seconds_to_nanoseconds(
            self.pupil_timestamp_mapping["start_time_system_s"]
        )

    @property
    def basler_start_time(self) -> int:
        return self.basler_timestamp_mapping["starting_mapping"]["perf_counter_ns"]

    @property
    def basler_start_time_utc_ns(self) -> int:
        return self.basler_timestamp_mapping["starting_mapping"]["utc_time_ns"]

    @property
    def difference_in_start_times(self) -> int:
        return self.basler_start_time_utc_ns - self.pupil_start_time_utc

    @property
    def basler_camera_names(self) -> List[str]:
        return list(
            self.basler_timestamp_mapping["starting_mapping"][
                "camera_timestamps"
            ].keys()
        )

    def load_and_convert_pupil_timestamps(self):
        """
        Load pupil timestamps and convert to utc
        conversion method based on: https://github.com/pupil-labs/pupil/issues/1500#issuecomment-492067526
        """
        offset = self.pupil_start_time_utc - self.pupil_start_time

        pupil_eye0_timestamps_path = self.pupil_path / "eye0_timestamps.npy"
        pupil_eye0_timestamps = np.load(pupil_eye0_timestamps_path)
        pupil_eye0_timestamps *= 1e9  # convert to ns
        pupil_eye0_timestamps = pupil_eye0_timestamps.astype(int)  # cast to int
        self.pupil_eye0_timestamps_utc = (
            pupil_eye0_timestamps + offset
        )

        pupil_eye1_timestamps_path = self.pupil_path / "eye1_timestamps.npy"
        pupil_eye1_timestamps = np.load(pupil_eye1_timestamps_path)
        pupil_eye1_timestamps *= 1e9  # convert to ns
        pupil_eye1_timestamps = pupil_eye1_timestamps.astype(int)  # cast to int
        self.pupil_eye1_timestamps_utc = (
            pupil_eye1_timestamps + offset
        )

    def load_and_convert_basler_timestamps(self):
        """
        Load basler timestamps and convert to utc
        Basler timestamps are saved in ns since camera latch time, which is roughly equivalent to time since utc start
        """
        timestamp_paths = list(self.synched_videos_path.glob("*_synchronized_timestamps_basler_time.npy"))
        print(f"timestamp paths: {timestamp_paths}")

        self.synched_basler_timestamps = {}
        for timestamp_path in timestamp_paths:
            cam_name = timestamp_path.stem.split("_")[0]
            timestamp_array = np.load(timestamp_path)
            self.synched_basler_timestamps[cam_name] = timestamp_array
        # print(f"Raw timestamp array: {timestamp_array}")
        self.synched_basler_timestamps_utc = {
            cam_name: timestamps + self.basler_start_time_utc_ns 
            for cam_name, timestamps  in self.synched_basler_timestamps.items()
        }
        # print(f"synched timestamp array: {self.synched_basler_timestamps_utc}")

    def load_index_to_serial_number(self):
        index_to_serial_number_path = (
                self.raw_videos_path / "index_to_serial_number_mapping.json"
        )
        if not index_to_serial_number_path.exists():
            print(
                f"index_to_serial_number_path does not exist: {index_to_serial_number_path}")
            print("default mapping will be used instead, double check it for correctness")
            self.index_to_serial_number = {
                "0": "24908831",
                "1": "24908832",
                "2": "25000609",
                "3": "25006505"
            }
        else:
            with open(index_to_serial_number_path) as index_to_serial_number_file:
                self.index_to_serial_number = json.load(index_to_serial_number_file)

        self.synchronization_metadata["index_to_serial_number_mapping"] = self.index_to_serial_number

    def verify_index_to_serial_number(self):
        print("Check index to serial number mapping (smaller serial numbers come first):")
        for cam_name in self.basler_camera_names:
            print(f"\tcam {cam_name} serial number: {self.index_to_serial_number[cam_name]}")
    
    def get_closest_pupil_frame_to_basler_frame(self, basler_frame_number: int) -> tuple[int, int]:
        basler_utc = np.median(self.synched_basler_timestamps_utc[:, basler_frame_number])
        print(f"basler_utc is {basler_utc}")
        eye0_match = np.searchsorted(self.pupil_eye0_timestamps_utc, basler_utc, side="right")
        if (basler_utc - self.pupil_eye0_timestamps_utc[eye0_match-1]) < abs(basler_utc - self.pupil_eye0_timestamps_utc[eye0_match]):
            eye0_frame_number = eye0_match-1
        else: 
            eye0_frame_number = eye0_match

        eye1_match = np.searchsorted(self.pupil_eye0_timestamps_utc, basler_utc, side="right")
        if (basler_utc - self.pupil_eye1_timestamps_utc[eye1_match-1]) < abs(basler_utc - self.pupil_eye1_timestamps_utc[eye1_match]):
            eye1_frame_number = eye1_match-1
        else: 
            eye1_frame_number = eye1_match

        print(f"eye 0 match is frame {eye0_frame_number} at utc {self.pupil_eye0_timestamps_utc[eye0_frame_number]}")
        print(f"eye 1 match is frame {eye1_frame_number} at utc {self.pupil_eye1_timestamps_utc[eye1_frame_number]}")

        return eye0_frame_number, eye1_frame_number

    def save_basler_utc_timestamps(self):
        print(f"Saving Basler timestamps in UTC to {self.synched_videos_path}")
        for cam_name, timestamps in self.synched_basler_timestamps_utc.items():
            file_name = f"{cam_name}_synchronized_timestamps_utc.npy"
            np.save(self.synched_videos_path / file_name, timestamps)

    def save_pupil_utc_timestamps(self):
        print(f"Saving pupil timestamps in UTC to {self.pupil_path}")
        np.save(self.pupil_path / "eye0_timestamps_utc.npy", self.pupil_eye0_timestamps_utc)
        np.save(self.pupil_path / "eye1_timestamps_utc.npy", self.pupil_eye1_timestamps_utc)




if __name__ == "__main__":
    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5/base_data"
    )
    timestamp_converter = TimestampConverter(folder_path)

    utc_start_time_pupil = timestamp_converter.pupil_start_time_utc
    utc_start_time_basler = timestamp_converter.basler_start_time_utc_ns

    basler_start_timestamps = list(float(timestamps[0]) for timestamps in timestamp_converter.synched_basler_timestamps.values())

    print(f"first basler timestamps: {basler_start_timestamps}")
    print(f"difference between Basler starting times (ns) = {max(basler_start_timestamps) - min(basler_start_timestamps)}")

    basler_start_discrepancy = {cam: timestamps[0] - utc_start_time_basler for cam, timestamps in timestamp_converter.synched_basler_timestamps_utc.items()}
    print(f"Difference between Basler first start time and timestamp mapping creation:")
    for cam, time_offset in basler_start_discrepancy.items():
        print(f"\tcam {cam} delay: {time_offset} (ns)")

    print(f"Pupil start time in pupil time (ns): {timestamp_converter.pupil_start_time}")
    print(f"Pupil start time in pupil time (s): {timestamp_converter.nanoseconds_to_seconds(timestamp_converter.pupil_start_time)}")
    print(f"Pupil start time in utc (ns):  {utc_start_time_pupil}")
    print(f"Pupil start time in utc (s): {timestamp_converter.nanoseconds_to_seconds(utc_start_time_pupil)}")
    print(f"Basler start time in utc (ns): {utc_start_time_basler}")
    print(f"Basler start time in utc (s): {timestamp_converter.nanoseconds_to_seconds(utc_start_time_basler)}")
    print(
        f"Basler start time in Basler time (ns): {timestamp_converter.basler_start_time}"
    )

    print(
        f"Difference between start times (basler - pupil) in s: {timestamp_converter.difference_in_start_times / 1e9}"
    )

    print(f"Pupil start time as date time: {np.datetime64(utc_start_time_pupil, 'ns')}")
    print(
        f"Basler start time as date time: {np.datetime64(utc_start_time_basler, 'ns')}"
    )

    print(
        f"basler start times per camera: {timestamp_converter.basler_timestamp_mapping['starting_mapping']['camera_timestamps']}"
    )

    print(
        f"pupil timestamps shapes - eye0: {timestamp_converter.pupil_eye0_timestamps_utc.shape} eye1: {timestamp_converter.pupil_eye1_timestamps_utc.shape}"
    )
    print(f"pupil timestamps (eye0): {timestamp_converter.pupil_eye0_timestamps_utc}")
    print(f"pupil timestamps (eye1): {timestamp_converter.pupil_eye1_timestamps_utc}")
    first_basler_timestamps = {cam: int(timestamps[0]) for cam, timestamps in timestamp_converter.synched_basler_timestamps_utc.items()}
    print(f"basler timestamps utc (ns): {first_basler_timestamps}")

    # utc_timestamps.get_closest_pupil_frame_to_basler_frame(3377)
    # utc_timestamps.get_closest_pupil_frame_to_basler_frame(8754)

    timestamp_converter.save_basler_utc_timestamps()
    timestamp_converter.save_pupil_utc_timestamps()
