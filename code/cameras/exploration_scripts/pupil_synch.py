"""
General approach here is:
-> Load timestamp mapping from Basler data (raw_videos/timestamp_mapping.json)
    -> use "starting_mapping" to figure out camera_timestamp to utc mapping for each camera
    -> verify "starting_mapping" with "ending_mapping"
-> Load timestamp mapping from Pupil data (pupil_voutput/info.player.json)
    -> interested in "start_time_synced_s" (pupil timestamp format) and "start_time_system_s" (utc)
-> Get map from pupil timestamp to each Basler camera timestamp
-> Trim Pupil data to match Basler data
    -> for now, using synched Basler data (synchronized_videos/timestamps.npy) - may need to take a mean of Basler timestamps? or synch to 1 camera?
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np


class PupilSynchronize:
    """
    For now, this assumes the Basler data has already been synchronized, and that the pupil data starts before the Basler data and ends after.
    Which means, this is only set up to trim the pupil data, and leaves the Basler videos alone (for now).
    """

    def __init__(self, folder_path: Path):
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        if not folder_path.exists:
            raise FileNotFoundError("Input folder path does not exist")

        self.raw_videos_path = folder_path / "raw_videos"
        self.synched_videos_path = folder_path / "synchronized_videos"

        self.basler_timestamp_file_name = "timestamps.npy"
        synched_basler_timestamp_path = (
            self.synched_videos_path / self.basler_timestamp_file_name
        )
        self.synched_basler_timestamps = np.load(synched_basler_timestamp_path)

        self.basler_timestamp_mapping_file_name = "timestamp_mapping.json"
        basler_timestamp_mapping_file = (
            self.raw_videos_path / self.basler_timestamp_mapping_file_name
        )
        with open(basler_timestamp_mapping_file) as basler_timestamp_mapping_file:
            self.basler_timestamp_mapping = json.load(basler_timestamp_mapping_file)

        self.pupil_output_path = folder_path / "pupil_output"
        self.pupil_eye0_video_path = self.pupil_output_path / "eye0.mp4"
        self.pupil_eye1_video_path = self.pupil_output_path / "eye1.mp4"

        pupil_eye0_timestamps_path = self.pupil_output_path / "eye0_timestamps.npy"
        self.pupil_eye0_timestamps = np.load(pupil_eye0_timestamps_path)

        pupil_eye1_timestamps_path = self.pupil_output_path / "eye1_timestamps.npy"
        self.pupil_eye1_timestamps = np.load(pupil_eye1_timestamps_path)

        self.pupil_timestamp_mapping_file_name = "info.player.json"
        pupil_timestamp_mapping_file = (
            self.pupil_output_path / self.pupil_timestamp_mapping_file_name
        )
        with open(pupil_timestamp_mapping_file) as pupil_timestamp_mapping_file:
            self.pupil_timestamp_mapping = json.load(pupil_timestamp_mapping_file)

    def seconds_to_nanoseconds(self, seconds: float) -> int:
        return int(seconds * 1e9)
    
    def nanoseconds_to_seconds(self, nanoseconds: int) -> float:
        return nanoseconds / 1e9
    
    @property
    def pupil_start_time(self) -> int:
        return self.seconds_to_nanoseconds(self.pupil_timestamp_mapping["start_time_synced_s"])
    
    @property
    def pupil_start_time_utc(self) -> int:
        return self.seconds_to_nanoseconds(self.pupil_timestamp_mapping['start_time_system_s'])
    
    @property
    def basler_start_time(self) -> int:
        return self.basler_timestamp_mapping['starting_mapping']['perf_counter_ns']
    
    @property
    def basler_start_time_utc(self) -> int:
        return self.basler_timestamp_mapping['starting_mapping']['utc_time_ns']
    
    @property
    def basler_first_synched_timestamp(self) -> int:
        return int(np.min(pupil_synchronize.synched_basler_timestamps))
    
    @property
    def basler_first_synched_timestamp_utc(self) -> int:
        return self.basler_first_synched_timestamp + self.basler_start_time_utc
    
    @property
    def difference_in_start_times(self) -> int:
        return self.pupil_start_time_utc - self.basler_start_time_utc
    
    @property
    def basler_end_time(self) -> int:
        return self.basler_timestamp_mapping['ending_mapping']['perf_counter_ns']
    
    @property
    def basler_end_time_utc(self) -> int:
        return self.basler_timestamp_mapping['ending_mapping']['utc_time_ns']
    
    @property
    def basler_last_synched_timestamp(self) -> int:
        return int(np.max(pupil_synchronize.synched_basler_timestamps))
    
    @property
    def basler_last_synched_timestamp_utc(self) -> int:
        return self.basler_last_synched_timestamp + self.basler_start_time_utc

    def get_utc_timestamp_per_camera(self) -> Dict[int, int]:
        return {
            int(camera): (
                self.basler_start_time_utc - basler_timestamp
            )
            for camera, basler_timestamp in self.basler_timestamp_mapping[
                "starting_mapping"
            ]["camera_timestamps"].items()
        }

    def timestamp_from_pupil_to_utc(self, pupil_timestamp_ns: int) -> int:
        ns_since_start = pupil_timestamp_ns - self.pupil_start_time
        return self.pupil_start_time + ns_since_start
    
    def timestamp_from_basler_to_utc(self, basler_timestamp_ns: int) -> int:
        """Basler timestamps are stored in ns since start for each camera, so no extra calculation is needed"""
        return self.basler_start_time_utc + basler_timestamp_ns
    
    # TODO: use difference in start times to trim front of pupil timestamps and videos
    # TODO: use length of basler recordings to trim back of pupil timestamps and videos - need to use basler timestamps to calculatew this, not ending mapping




if __name__ == "__main__":
    folder_path = Path("/Users/philipqueen/basler_pupil_synch_test/")
    pupil_synchronize = PupilSynchronize(folder_path)

    # print(pupil_synchronize.basler_timestamp_mapping)
    # print(pupil_synchronize.pupil_timestamp_mapping)

    utc_timestamp_per_camera = pupil_synchronize.get_utc_timestamp_per_camera()
    utc_start_time_pupil = pupil_synchronize.pupil_start_time_utc
    utc_start_time_basler = pupil_synchronize.basler_start_time_utc


    print(f"Pupil start time in utc (ns):  {utc_start_time_pupil}")
    print(f"Basler start time in utc (ns): {utc_start_time_basler}")

    print(f"Difference between start times (pupil - basler) in s: {pupil_synchronize.difference_in_start_times / 1e9}")

    print(f"Pupil start time as date time: {np.datetime64(utc_start_time_pupil, 'ns')}")
    print(f"Basler start time as date time: {np.datetime64(utc_start_time_basler, 'ns')}")

    print(f"basler start times per camera: {pupil_synchronize.basler_timestamp_mapping['starting_mapping']['camera_timestamps']}")

    print("basler timestamps (in s since start):")
    print(f"{np.min(pupil_synchronize.synched_basler_timestamps) / 1e9}")
    print(f"{np.max(pupil_synchronize.synched_basler_timestamps) / 1e9}")
    print(f"{np.mean(pupil_synchronize.synched_basler_timestamps) / 1e9}")

    print(f"pupil timestamps shapes - eye0: {pupil_synchronize.pupil_eye0_timestamps.shape} eye1: {pupil_synchronize.pupil_eye1_timestamps.shape}")
    print(f"pupil timestamps (eye0): {pupil_synchronize.pupil_eye0_timestamps}")