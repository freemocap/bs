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
from typing import Dict, List, Tuple

import numpy as np


class PupilSynchronize:
    def __init__(self, folder_path: Path):
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        if not folder_path.exists:
            raise FileNotFoundError("Input folder path does not exist")

        self.raw_videos_path = folder_path / "raw_videos"
        self.synched_videos_path = folder_path / "synchronized_videos"

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
        self.load_basler_timestamps()

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
    def pupil_first_synched_timestamp_utc(self) -> int:
        return int(
            min(
                np.min(self.pupil_eye0_timestamps_utc),
                np.min(self.pupil_eye1_timestamps_utc),
            )
        )

    @property
    def pupil_last_synched_timestamp_utc(self) -> int:
        return int(
            min(
                np.max(self.pupil_eye0_timestamps_utc),
                np.max(self.pupil_eye1_timestamps_utc),
            )
        )

    @property
    def basler_start_time(self) -> int:
        return self.basler_timestamp_mapping["starting_mapping"]["perf_counter_ns"]

    @property
    def basler_start_time_utc(self) -> int:
        return self.basler_timestamp_mapping["starting_mapping"]["utc_time_ns"]

    @property
    def basler_first_synched_timestamp(self) -> int:
        return int(np.min(self.synched_basler_timestamps))

    @property
    def basler_first_synched_timestamp_utc(self) -> int:
        return int(np.min(self.synched_basler_timestamps_utc))

    @property
    def difference_in_start_times(self) -> int:
        return self.basler_start_time_utc - self.pupil_start_time_utc

    @property
    def basler_end_time(self) -> int:
        return self.basler_timestamp_mapping["ending_mapping"]["perf_counter_ns"]

    @property
    def basler_end_time_utc(self) -> int:
        return self.basler_timestamp_mapping["ending_mapping"]["utc_time_ns"]

    @property
    def basler_last_synched_timestamp(self) -> int:
        return int(np.min(self.synched_basler_timestamps[:, -1]))

    @property
    def basler_last_synched_timestamp_utc(self) -> int:
        return int(np.min(self.synched_basler_timestamps_utc[:, -1]))

    @property
    def length_of_basler_recording(self) -> int:
        return self.basler_last_synched_timestamp - self.basler_first_synched_timestamp

    @property
    def latest_synched_start_utc(self) -> int:
        return max(
            self.pupil_first_synched_timestamp_utc,
            self.basler_first_synched_timestamp_utc,
        )

    @property
    def earliest_synched_end_utc(self) -> int:
        return min(
            self.pupil_last_synched_timestamp_utc,
            self.basler_last_synched_timestamp_utc,
        )

    @property
    def basler_camera_names(self) -> List[str]:
        return list(
            self.basler_timestamp_mapping["starting_mapping"][
                "camera_timestamps"
            ].keys()
        )

    @property
    def pupil_camera_names(self) -> List[str]:
        return ["eye0", "eye1"]

    def load_pupil_timestamps(self):
        """Load pupil timestamps and convert to utc"""
        pupil_eye0_timestamps_path = self.pupil_output_path / "eye0_timestamps.npy"
        pupil_eye0_timestamps = np.load(pupil_eye0_timestamps_path)
        pupil_eye0_timestamps *= 1e9  # convert to ns
        pupil_eye0_timestamps = pupil_eye0_timestamps.astype(int)  # cast to int
        pupil_eye0_timestamps -= (
            self.pupil_start_time
        )  # convert to ns since pupil start time
        self.pupil_eye0_timestamps_utc = (
            pupil_eye0_timestamps + self.pupil_start_time_utc
        )

        pupil_eye1_timestamps_path = self.pupil_output_path / "eye1_timestamps.npy"
        pupil_eye1_timestamps = np.load(pupil_eye1_timestamps_path)
        pupil_eye1_timestamps *= 1e9  # convert to ns
        pupil_eye1_timestamps = pupil_eye1_timestamps.astype(int)  # cast to int
        pupil_eye1_timestamps -= (
            self.pupil_start_time
        )  # convert to ns since pupil start time
        self.pupil_eye1_timestamps_utc = (
            pupil_eye1_timestamps + self.pupil_start_time_utc
        )

    def load_basler_timestamps(self):
        """
        Load basler timestamps and convert to utc
        Basler timestamps are saved in ns since camera latch time, whcih is roughly equivalent to time since utc start
        """
        self.basler_timestamp_file_name = "timestamps.npy"
        synched_basler_timestamp_path = (
            self.synched_videos_path / self.basler_timestamp_file_name
        )
        self.synched_basler_timestamps = np.load(synched_basler_timestamp_path)
        self.synched_basler_timestamps_utc = (
            self.synched_basler_timestamps + self.basler_start_time_utc
        )

    def get_pupil_fps(self) -> Tuple[float, float]:
        eye0_time_elapsed_s = (
            self.pupil_eye0_timestamps_utc[-1] - self.pupil_eye0_timestamps_utc[0]
        ) / 1e9
        eye0_fps = self.pupil_eye0_timestamps_utc.shape[0] / eye0_time_elapsed_s

        eye1_time_elapsed_s = (
            self.pupil_eye1_timestamps_utc[-1] - self.pupil_eye1_timestamps_utc[0]
        ) / 1e9
        eye1_fps = self.pupil_eye1_timestamps_utc.shape[0] / eye1_time_elapsed_s

        print(
            f"pupil eye camera actual fps: eye0: {eye0_fps} fps, eye1: {eye1_fps} fps"
        )
        print(f"pupil eye camera average frame duration: eye0: {1 / eye0_fps * 1e9} ns, eye1: {1 / eye1_fps * 1e9} ns")
        print(f"difference in fps: {eye0_fps - eye1_fps}")

        return (eye0_fps, eye1_fps)
    
    def find_missing_pupil_frames(self):
        # TODO: find times where time gap between pupil eye camera timestamps changes
        smaller_length = min(self.pupil_eye0_timestamps_utc.shape[0], self.pupil_eye1_timestamps_utc.shape[0])
        difference = self.pupil_eye0_timestamps_utc[:smaller_length] - self.pupil_eye1_timestamps_utc[smaller_length]
        print(f"median difference in seconds: {np.median(np.abs(difference)) // 1e9}")

        change_in_difference = np.diff(difference)
        print(f"max difference: {np.max(np.abs(change_in_difference))}")
        print(f"median change in difference: {np.median(np.abs(change_in_difference))}")
        
        outliers = np.where(change_in_difference > 10000000)[0]
        print(outliers.shape)
        print(outliers)
        print(f"index of biggest change: {np.argmax(np.abs(change_in_difference))}")

    def get_utc_timestamp_per_camera(self) -> Dict[int, int]:
        return {
            int(camera): (self.basler_start_time_utc - basler_timestamp)
            for camera, basler_timestamp in self.basler_timestamp_mapping[
                "starting_mapping"
            ]["camera_timestamps"].items()
        }

    # def find_pupil_starting_offsets_in_frames(self) -> Dict[str, int]:
    #     # find pupil frame number where timestamp is >= the first basler frame
    #     starting_offsets_in_frames = {
    #         "eye0": np.where(
    #             self.pupil_eye0_timestamps_utc >= self.basler_first_synched_timestamp
    #         )[0][0],
    #         "eye1": np.where(
    #             self.pupil_eye1_timestamps_utc >= self.basler_first_synched_timestamp
    #         )[0][0],
    #     }
    #     print(starting_offsets_in_frames)
    #     return starting_offsets_in_frames

    # def find_pupil_ending_offsets_in_frames(
    #     self, pupil_starting_offsets: Dict[str, int]
    # ) -> Dict[str, int]:
    #     ending_offsets_in_frames = {
    #         "eye0": np.where(
    #             self.pupil_eye0_timestamps_utc >= self.basler_last_synched_timestamp
    #         )[0][0],
    #         "eye1": np.where(
    #             self.pupil_eye1_timestamps_utc >= self.basler_last_synched_timestamp
    #         )[0][0],
    #     }
    #     print(ending_offsets_in_frames)
    #     return ending_offsets_in_frames

    def find_starting_offsets_in_frames(self) -> Dict[str, int]:
        starting_offsets_in_frames = {
            cam_name: np.where(
                self.synched_basler_timestamps_utc[i, :] >= self.latest_synched_start_utc
            )[0][0]
            for i, cam_name in enumerate(self.basler_camera_names)
        }

        starting_offsets_in_frames["eye0"] = np.where(
            self.pupil_eye0_timestamps_utc >= self.latest_synched_start_utc
        )[0][0]
        starting_offsets_in_frames["eye1"] = np.where(
            self.pupil_eye1_timestamps_utc >= self.latest_synched_start_utc
        )[0][0]

        print(f"starting offsets in frames: {starting_offsets_in_frames}")
        return starting_offsets_in_frames
    
    def find_ending_offsets_in_frames(self) -> Dict[str, int]:
        ending_offsets_in_frames = {
            cam_name: np.where(
                self.synched_basler_timestamps_utc[i, :] >= self.earliest_synched_end_utc
            )[0][0]
            for i, cam_name in enumerate(self.basler_camera_names)
        }

        ending_offsets_in_frames["eye0"] = np.where(
            self.pupil_eye0_timestamps_utc >= self.earliest_synched_end_utc
        )[0][0]
        ending_offsets_in_frames["eye1"] = np.where(
            self.pupil_eye1_timestamps_utc >= self.earliest_synched_end_utc
        )[0][0]

        print(f"ending offsets in frames: {ending_offsets_in_frames}")
        return ending_offsets_in_frames

    def synchronize(self):
        print(f"latest synched start utc: {self.latest_synched_start_utc}")
        print(f"earliest synched end utc: {self.earliest_synched_end_utc}")
        starting_offsets_frames = self.find_starting_offsets_in_frames()
        ending_offsets_frames = self.find_ending_offsets_in_frames()

        self.corrected_timestamps = {
            cam_name: self.synched_basler_timestamps_utc[i, :][
                starting_offsets_frames[cam_name] : ending_offsets_frames[cam_name]
            ] for i, cam_name in enumerate(self.basler_camera_names)
        }

        self.corrected_timestamps["eye0"] = self.pupil_eye0_timestamps_utc[
            starting_offsets_frames["eye0"] : ending_offsets_frames["eye0"]
        ]
        self.corrected_timestamps["eye1"] = self.pupil_eye1_timestamps_utc[
            starting_offsets_frames["eye1"] : ending_offsets_frames["eye1"]
        ]

        for cam_name, timestamps in self.corrected_timestamps.items():
            print(f"cam {cam_name} timestamps shape: {timestamps.shape}")
            # TODO: getting different shapes for the corrected pupil timestamps??

        self.plot_timestamps(
            starting_offsets=starting_offsets_frames,
            ending_offsets=ending_offsets_frames
        )
        
        # if (
        #     self.basler_first_synched_timestamp_utc
        #     > self.pupil_first_synched_timestamp_utc
        # ):
        #     # pupil data starts before basler data
        #     pass
        # else:
        #     # basler data starts before pupil data
        #     pass

        # if (
        #     self.basler_last_synched_timestamp_utc
        #     < self.pupil_last_synched_timestamp_utc
        # ):
        #     # pupil data ends after basler data
        #     pass
        # else:
        #     # basler data ends after pupil data
        #     pass

    def old_synchronize(self):
        pupil_starting_offsets = self.find_pupil_starting_offsets_in_frames()
        pupil_ending_offsets = self.find_pupil_ending_offsets_in_frames(
            pupil_starting_offsets=pupil_starting_offsets
        )

        corrected_pupil_timestamps = {
            "eye0": self.pupil_eye0_timestamps_utc[
                pupil_starting_offsets["eye0"] : pupil_ending_offsets["eye0"]
            ],
            "eye1": self.pupil_eye1_timestamps_utc[
                pupil_starting_offsets["eye1"] : pupil_ending_offsets["eye1"]
            ],
        }

        print(
            f"corrected timestamp shapes - eye0: {corrected_pupil_timestamps['eye0'].shape} eye1: {corrected_pupil_timestamps['eye1'].shape}"
        )
        print(
            f"starting timestamps - eye0: {self.pupil_eye0_timestamps_utc[pupil_starting_offsets['eye0']]} eye1: {self.pupil_eye0_timestamps_utc[pupil_starting_offsets['eye1']]}"
        )
        print(
            f"ending timestamps - eye0: {self.pupil_eye0_timestamps_utc[pupil_ending_offsets['eye0']]} eye1: {self.pupil_eye0_timestamps_utc[pupil_ending_offsets['eye1']]}"
        )
        print(
            f"starting timestamp difference: {self.pupil_eye0_timestamps_utc[pupil_starting_offsets['eye0']] - self.pupil_eye0_timestamps_utc[pupil_starting_offsets['eye1']]}"
        )
        print(
            f"ending timestamp difference: {self.pupil_eye0_timestamps_utc[pupil_ending_offsets['eye0']] - self.pupil_eye0_timestamps_utc[pupil_ending_offsets['eye1']]}"
        )

    #     self.plot_timestamps(
    #         starting_offsets=pupil_starting_offsets,
    #         ending_offsets=pupil_ending_offsets,
    #     )

    def plot_timestamps(
        self,
        starting_offsets: Dict[str, int],
        ending_offsets: Dict[str, int],
    ):
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

        for i, cam_name in enumerate(self.basler_camera_names):
            ax1.plot(
                self.synched_basler_timestamps_utc[i, :],
                label=cam_name,
            )
        ax1.plot(self.pupil_eye0_timestamps_utc, label="eye0")
        ax1.plot(self.pupil_eye1_timestamps_utc, label="eye1")

        for name in starting_offsets.keys():
            ax1.vlines(
                [starting_offsets[name], ending_offsets[name]],
                ymin=self.latest_synched_start_utc,
                ymax=self.earliest_synched_end_utc,
                label=f"{name} vlines",
            )

        ax1.legend()
        ax1.set_ylim(self.latest_synched_start_utc, self.earliest_synched_end_utc)

        plt.tight_layout()

        plt.show()

    # TODO: use difference in start times to trim front of pupil timestamps and videos
    # TODO: use length of basler recordings to trim back of pupil timestamps and videos - need to use basler timestamps to calculate this, not ending mapping

    # TODO: check if pupil eye timestamps start synchronized

    # TODO: save some diagnostics from this, i.e. the actual start time in utc that basler and pupil


if __name__ == "__main__":
    folder_path = Path("/Users/philipqueen/Basler_plus_pupil_test_10min")
    pupil_synchronize = PupilSynchronize(folder_path)

    # print(pupil_synchronize.basler_timestamp_mapping)
    # print(pupil_synchronize.pupil_timestamp_mapping)

    utc_timestamp_per_camera = pupil_synchronize.get_utc_timestamp_per_camera()
    utc_start_time_pupil = pupil_synchronize.pupil_start_time_utc
    utc_start_time_basler = pupil_synchronize.basler_start_time_utc

    print(f"Pupil start time in pupil time (ns): {pupil_synchronize.pupil_start_time}")
    print(f"Pupil start time in utc (ns):  {utc_start_time_pupil}")
    print(f"Basler start time in utc (ns): {utc_start_time_basler}")
    print(
        f"Basler start time in Basler time (ns): {pupil_synchronize.basler_start_time}"
    )

    print(
        f"Difference between start times (basler - pupil) in s: {pupil_synchronize.difference_in_start_times / 1e9}"
    )

    print(f"Pupil start time as date time: {np.datetime64(utc_start_time_pupil, 'ns')}")
    print(
        f"Basler start time as date time: {np.datetime64(utc_start_time_basler, 'ns')}"
    )

    print(
        f"basler start times per camera: {pupil_synchronize.basler_timestamp_mapping['starting_mapping']['camera_timestamps']}"
    )

    # print("basler timestamps (in s since start):")
    # print(f"{np.min(pupil_synchronize.synched_basler_timestamps) / 1e9}")
    # print(f"{np.max(pupil_synchronize.synched_basler_timestamps) / 1e9}")
    # print(f"{np.mean(pupil_synchronize.synched_basler_timestamps) / 1e9}")

    print(
        f"pupil timestamps shapes - eye0: {pupil_synchronize.pupil_eye0_timestamps_utc.shape} eye1: {pupil_synchronize.pupil_eye1_timestamps_utc.shape}"
    )
    print(f"pupil timestamps (eye0): {pupil_synchronize.pupil_eye0_timestamps_utc}")
    print(f"pupil timestamps (eye1): {pupil_synchronize.pupil_eye1_timestamps_utc}")

    pupil_synchronize.get_pupil_fps()
    pupil_synchronize.find_missing_pupil_frames()

    pupil_synchronize.synchronize()
    # pupil_synchronize.old_synchronize()
