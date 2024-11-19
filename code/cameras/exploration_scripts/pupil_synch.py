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

from datetime import datetime
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

    def get_utc_timestamp_per_camera(self) -> Dict[int, int]:
        return {
            int(camera): (
                self.basler_timestamp_mapping["starting_mapping"]["utc_time_ns"] - basler_timestamp
            )
            for camera, basler_timestamp in self.basler_timestamp_mapping[
                "starting_mapping"
            ]["camera_timestamps"].items()
        }

    def get_pupil_to_basler_timestamp_mapping(self):
        pupil_timestamp_start_ns = (
            self.pupil_timestamp_mapping["start_time_synced_s"] * 1e9
        )
        pupil_utc_start_ns = self.pupil_timestamp_mapping["start_time_system_s"] * 1e9


if __name__ == "__main__":
    folder_path = Path("/Users/philipqueen/basler_pupil_synch_test/")
    pupil_synchronize = PupilSynchronize(folder_path)

    print(pupil_synchronize.basler_timestamp_mapping)
    print(pupil_synchronize.pupil_timestamp_mapping)

    utc_timestamp_per_camera = pupil_synchronize.get_utc_timestamp_per_camera()
    utc_start_time_pupil = int(pupil_synchronize.pupil_timestamp_mapping['start_time_system_s'] * 1e9)
    utc_start_time_basler = pupil_synchronize.basler_timestamp_mapping['starting_mapping']['utc_time_ns']


    print(f"Pupil start time in ns:  {utc_start_time_pupil}")
    print(f"Basler start time in ns: {utc_start_time_basler}")

    print(f"Difference between start times in s: {(utc_start_time_pupil - utc_start_time_basler) / 1e9}")

    # print(f"Pupil start time as date time: {datetime.fromtimestamp(utc_start_time_pupil)}")
    # print(f"Basler start time as date time: {datetime.fromtimestamp(utc_start_time_basler)}")

    # for camera, utc_start_time in utc_timestamp_per_camera.items():
    #     print(f"Camera {camera} UTC start time: {utc_start_time / 1e9} - Pupil start time: {utc_start_time_pupil / 1e9}")
    #     # print utc start time as a date time
    #     print(f"Camera {camera} UTC start time: {np.datetime64(utc_start_time, 's')}")
    #     print(f"Pupil start time: {np.datetime64(utc_start_time_pupil, 's')}")
    #     print(f"Camera {camera} time difference: {(utc_start_time - utc_start_time_pupil) / 1e9} seconds")
