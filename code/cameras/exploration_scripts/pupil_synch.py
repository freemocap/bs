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

        self.pupil_timestamp_mapping_file_name = "info.player.json"
        pupil_timestamp_mapping_file = (
            self.pupil_output_path / self.pupil_timestamp_mapping_file_name
        )
        with open(pupil_timestamp_mapping_file) as pupil_timestamp_mapping_file:
            self.pupil_timestamp_mapping = json.load(pupil_timestamp_mapping_file)

        self.load_pupil_timestamps()

    def seconds_to_nanoseconds(self, seconds: float) -> int:
        return int(seconds * 1e9)

    def load_pupil_timestamps(self):
        # TODO: just convert all timestamps to utc
        """Load pupil timestamps and convert to ns since basler start time"""
        pupil_eye0_timestamps_path = self.pupil_output_path / "eye0_timestamps.npy"
        self.pupil_eye0_timestamps = np.load(pupil_eye0_timestamps_path)
        self.pupil_eye0_timestamps *= 1e9  # convert to ns
        self.pupil_eye0_timestamps = self.pupil_eye0_timestamps.astype(int)  # cast to int
        self.pupil_eye0_timestamps -= self.pupil_start_time  # convert to ns since pupil start time
        self.pupil_eye0_timestamps -= self.difference_in_start_times # correct pupil timestamps to ns since basler start time

        pupil_eye1_timestamps_path = self.pupil_output_path / "eye1_timestamps.npy"
        self.pupil_eye1_timestamps = np.load(pupil_eye1_timestamps_path)
        self.pupil_eye1_timestamps *= 1e9  # convert to ns
        self.pupil_eye1_timestamps = self.pupil_eye1_timestamps.astype(int)  # cast to int
        self.pupil_eye1_timestamps -= self.pupil_start_time  # convert to ns since pupil start time
        self.pupil_eye1_timestamps -= self.difference_in_start_times # correct pupil timestamps to ns since basler start time
    
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
        return self.basler_start_time_utc - self.pupil_start_time_utc 
    
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
    
    @property
    def length_of_basler_recording(self) -> int:
        return self.basler_last_synched_timestamp - self.basler_first_synched_timestamp

    def get_utc_timestamp_per_camera(self) -> Dict[int, int]:
        return {
            int(camera): (
                self.basler_start_time_utc - basler_timestamp
            )
            for camera, basler_timestamp in self.basler_timestamp_mapping[
                "starting_mapping"
            ]["camera_timestamps"].items()
        }
    
    def find_pupil_starting_offsets_in_frames(self) -> Dict[str, int]:
        # find pupil frame number where timestamp is >= the first basler frame
        starting_offsets_in_frames = {
            "eye0": np.where(self.pupil_eye0_timestamps >= self.basler_first_synched_timestamp)[0][0],
            "eye1": np.where(self.pupil_eye1_timestamps >= self.basler_first_synched_timestamp)[0][0]
        }
        print(starting_offsets_in_frames)
        return starting_offsets_in_frames
    
    def find_pupil_ending_offsets_in_frames(self, pupil_starting_offsets: Dict[str, int]) -> Dict[str, int]:
        ending_offsets_in_frames = {
            "eye0": np.where(self.pupil_eye0_timestamps >= self.basler_last_synched_timestamp)[0][0],
            "eye1": np.where(self.pupil_eye1_timestamps >= self.basler_last_synched_timestamp)[0][0]
        }
        print(ending_offsets_in_frames)
        return ending_offsets_in_frames

    
    def synchronize(self):
        pupil_starting_offsets = self.find_pupil_starting_offsets_in_frames()
        pupil_ending_offsets = self.find_pupil_ending_offsets_in_frames(pupil_starting_offsets=pupil_starting_offsets)

        corrected_pupil_timestamps = {
            "eye0": self.pupil_eye0_timestamps[pupil_starting_offsets["eye0"]:pupil_ending_offsets["eye0"]],
            "eye1": self.pupil_eye1_timestamps[pupil_starting_offsets["eye1"]:pupil_ending_offsets["eye1"]]
        }

        print(f"corrected timestamp shapes - eye0: {corrected_pupil_timestamps['eye0'].shape} eye1: {corrected_pupil_timestamps['eye1'].shape}")
        print(f"starting timestamps - eye0: {self.pupil_eye0_timestamps[pupil_starting_offsets['eye0']]} eye1: {self.pupil_eye0_timestamps[pupil_starting_offsets['eye1']]}")
        print(f"ending timestamps - eye0: {self.pupil_eye0_timestamps[pupil_ending_offsets['eye0']]} eye1: {self.pupil_eye0_timestamps[pupil_ending_offsets['eye1']]}")
        print(f"starting timestamp difference: {self.pupil_eye0_timestamps[pupil_starting_offsets['eye0']] - self.pupil_eye0_timestamps[pupil_starting_offsets['eye1']]}")
        print(f"ending timestamp difference: {self.pupil_eye0_timestamps[pupil_ending_offsets['eye0']] - self.pupil_eye0_timestamps[pupil_ending_offsets['eye1']]}")

        self.plot_timestamps(pupil_starting_offsets=pupil_starting_offsets, pupil_ending_offsets=pupil_ending_offsets)

    def plot_timestamps(self, pupil_starting_offsets: Dict[str, int], pupil_ending_offsets: Dict[str, int]):
        """plot some diagnostics to assess quality of camera sync"""

        # opportunistic load of matplotlib to avoid startup time costs
        from matplotlib import pyplot as plt

        plt.set_loglevel("warning")

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"Timestamps")

        ax1 = plt.subplot(
            title="(Raw) Camera Frame Timestamp vs Frame#\n(Lines should have same slope)",
            xlabel="Frame#",
            ylabel="Timestamp (ns)",
        )

        ax1.plot(self.pupil_eye0_timestamps, label="eye0")
        ax1.plot(self.pupil_eye1_timestamps, label="eye1")

        ax1.vlines([pupil_starting_offsets["eye0"], pupil_ending_offsets["eye0"]], ymin=-0.5e10, ymax=3e10, label="eye0 vlines", colors="purple")
        ax1.vlines([pupil_starting_offsets["eye1"], pupil_ending_offsets["eye1"]], ymin=-0.5e10, ymax=3e10, label="eye1 vlines", colors="red")
        ax1.legend()

        plt.tight_layout()

        plt.show()

    
    # TODO: use difference in start times to trim front of pupil timestamps and videos
    # TODO: use length of basler recordings to trim back of pupil timestamps and videos - need to use basler timestamps to calculate this, not ending mapping

    # TODO: check if pupil eye timestamps start synchronized

    # TODO: save some diagnostics from this, i.e. the actual start time in utc that basler and pupi




if __name__ == "__main__":
    folder_path = Path("/Users/philipqueen/basler_pupil_synch_test/")
    pupil_synchronize = PupilSynchronize(folder_path)

    # print(pupil_synchronize.basler_timestamp_mapping)
    # print(pupil_synchronize.pupil_timestamp_mapping)

    utc_timestamp_per_camera = pupil_synchronize.get_utc_timestamp_per_camera()
    utc_start_time_pupil = pupil_synchronize.pupil_start_time_utc
    utc_start_time_basler = pupil_synchronize.basler_start_time_utc


    print(f"Pupil start time in pupil time (ns): {pupil_synchronize.pupil_start_time}")
    print(f"Pupil start time in utc (ns):  {utc_start_time_pupil}")
    print(f"Basler start time in utc (ns): {utc_start_time_basler}")
    print(f"Basler start time in Basler time (ns): {pupil_synchronize.basler_start_time}")

    print(f"Difference between start times (basler - pupil) in s: {pupil_synchronize.difference_in_start_times / 1e9}")

    print(f"Pupil start time as date time: {np.datetime64(utc_start_time_pupil, 'ns')}")
    print(f"Basler start time as date time: {np.datetime64(utc_start_time_basler, 'ns')}")

    print(f"basler start times per camera: {pupil_synchronize.basler_timestamp_mapping['starting_mapping']['camera_timestamps']}")

    print("basler timestamps (in s since start):")
    print(f"{np.min(pupil_synchronize.synched_basler_timestamps) / 1e9}")
    print(f"{np.max(pupil_synchronize.synched_basler_timestamps) / 1e9}")
    print(f"{np.mean(pupil_synchronize.synched_basler_timestamps) / 1e9}")

    print(f"pupil timestamps shapes - eye0: {pupil_synchronize.pupil_eye0_timestamps.shape} eye1: {pupil_synchronize.pupil_eye1_timestamps.shape}")
    print(f"pupil timestamps (eye0): {pupil_synchronize.pupil_eye0_timestamps}")
    print(f"pupil timestamps (eye1): {pupil_synchronize.pupil_eye1_timestamps}")

    pupil_synchronize.synchronize()