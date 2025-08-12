import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


class PupilSynchronize:
    def __init__(self, folder_path: Path):
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        if not folder_path.exists:
            raise FileNotFoundError("Input folder path does not exist")

        self.raw_videos_path = folder_path / "raw_videos"
        self.synched_videos_path = folder_path / "synchronized_corrected_videos"
        if not self.synched_videos_path.exists():
            self.synched_videos_path = folder_path / "synchronized_videos"
        self.output_path = folder_path / "basler_pupil_synchronized"

        self.basler_timestamp_mapping_file_name = "timestamp_mapping.json"
        basler_timestamp_mapping_file = (
                self.raw_videos_path / self.basler_timestamp_mapping_file_name
        )
        with open(basler_timestamp_mapping_file) as basler_timestamp_mapping_file:
            self.basler_timestamp_mapping = json.load(basler_timestamp_mapping_file)

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

        self.load_pupil_timestamps()
        self.load_basler_timestamps()
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
        pupil_eye0_timestamps_path = self.pupil_path / "eye0_timestamps.npy"
        pupil_eye0_timestamps = np.load(pupil_eye0_timestamps_path)
        pupil_eye0_timestamps *= 1e9  # convert to ns
        pupil_eye0_timestamps = pupil_eye0_timestamps.astype(int)  # cast to int
        pupil_eye0_timestamps -= (
            self.pupil_start_time
        )  # convert to ns since pupil start time
        self.pupil_eye0_timestamps_utc = (
                pupil_eye0_timestamps + self.pupil_start_time_utc
        )

        pupil_eye1_timestamps_path = self.pupil_path / "eye1_timestamps.npy"
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
        Basler timestamps are saved in ns since camera latch time, which is roughly equivalent to time since utc start
        """
        self.basler_timestamp_file_name = "timestamps.npy"
        synched_basler_timestamp_path = (
                self.synched_videos_path / self.basler_timestamp_file_name
        )
        self.synched_basler_timestamps = np.load(synched_basler_timestamp_path)
        self.synched_basler_timestamps_utc = (
                self.synched_basler_timestamps + self.basler_start_time_utc
        )

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

        self.synchronization_metadata["pupil_input_fps"] = {
            "eye0": eye0_fps,
            "eye1": eye1_fps
        }

        return eye0_fps, eye1_fps

    def get_pupil_median_fps(self) -> Tuple[float, float]:
        eye_0_time_difference = np.diff(self.pupil_eye0_timestamps_utc)
        eye_0_fps = 1e9 / np.median(eye_0_time_difference)

        eye_1_time_difference = np.diff(self.pupil_eye1_timestamps_utc)
        eye_1_fps = 1e9 / np.median(eye_1_time_difference)

        print(
            f"pupil eye camera median fps: eye0: {eye_0_fps} fps, eye1: {eye_1_fps} fps"
        )

        if eye_0_fps != eye_1_fps:
            print(f"WARNING: pupil median fps does not match for eye0 and eye1: {eye_0_fps} vs {eye_1_fps}")

        self.synchronization_metadata["pupil_median_fps"] = {
            "eye0": eye_0_fps,
            "eye1": eye_1_fps
        }

        return float(eye_0_fps), float(eye_1_fps)

    def get_utc_timestamp_per_camera(self) -> Dict[int, int]:
        return {
            int(camera): (self.basler_start_time_utc - basler_timestamp)
            for camera, basler_timestamp in self.basler_timestamp_mapping[
                "starting_mapping"
            ]["camera_timestamps"].items()
        }

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

    def save_synchronized_timestamps(self):
        if self.synchronized_timestamps is None:
            raise ValueError(
                "synchronized_timestamps is None, this method should only be called from synchronize(), it should not be called directly")
        for cam_name, timestamps in self.synchronized_timestamps.items():
            print(f"cam {cam_name} timestamps shape: {timestamps.shape}")
            np.save(f"{self.output_path}/cam_{cam_name}_synchronized_timestamps.npy", timestamps)

    def save_metadata(self):
        with open(f"{self.output_path}/metadata.json", "w") as f:
            json.dump(self.synchronization_metadata, f, indent=4)

    def trim_single_video(self,
                          start_frame: int,
                          end_frame: int,
                          input_video_pathstring: str,
                          output_video_pathstring: str,
                          ):
        frame_list = list(range(start_frame, end_frame + 1))
        cap = cv2.VideoCapture(input_video_pathstring)

        framerate = cap.get(cv2.CAP_PROP_FPS)
        framesize = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")

        video_writer_object = cv2.VideoWriter(
            output_video_pathstring, fourcc, framerate, framesize
        )

        print(f"saving synchronized video to {output_video_pathstring}")

        current_frame = 0
        written_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in frame_list:
                video_writer_object.write(frame)
                written_frames += 1

            if written_frames == len(frame_list):
                break

            current_frame += 1

        cap.release()
        video_writer_object.release()

    def trim_single_video_pupil(self,
                                        camera_name: str,
                                        input_video_pathstring: str,
                                        output_video_pathstring: str,
                                        ):
        cap = cv2.VideoCapture(input_video_pathstring)

        raw_timestamps = self.pupil_eye0_timestamps_utc.copy() if camera_name == "eye0" else self.pupil_eye1_timestamps_utc.copy()
        camera_median_fps = self.get_pupil_median_fps()[0] if camera_name == "eye0" else self.get_pupil_median_fps()[1]
        camera_median_fps = round(camera_median_fps, 2)
        print(f"camera median fps: {camera_median_fps}")
        median_duration = 1e9 / camera_median_fps

        framerate = cap.get(cv2.CAP_PROP_FPS)
        framesize = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # need to deal with higher frame rates

        video_writer_object = cv2.VideoWriter(
            output_video_pathstring, fourcc, camera_median_fps, framesize
        )

        print(f"saving synchronized video to {output_video_pathstring}")

        current_frame = 0
        written_frames = 0
        dropped_frames = 0
        skipped_frames = 0
        early_frames = 0
        synchronized_timestamps: List[int | None] = []
        previous_frame = np.zeros((framesize[1], framesize[0], 3), dtype=np.uint8)

        while True:
            reference_timestamp = self.latest_synched_start_utc + (written_frames * median_duration)
            if current_frame >= len(raw_timestamps):
                if reference_timestamp > self.earliest_synched_end_utc:
                    print("reached target ending timestamp, exiting")
                    break
                else:
                    print("ran out of frames")
                    print(
                        f"target final timestamp: {self.earliest_synched_end_utc}, actual final timestamp: {raw_timestamps[-1]}, final reference timestamp: {reference_timestamp}")
                    # TODO: We may want to fill in dummy frames here
                    break
            current_timestamp = raw_timestamps[current_frame]
            if reference_timestamp > self.earliest_synched_end_utc:
                # past the last synchronized time
                print("reached target ending timestamp, exiting")
                break
            elif current_timestamp < self.latest_synched_start_utc - (0.5 * median_duration):
                # before the first synchronized time
                early_frames += 1
                current_frame += 1
                continue

            # if we make it past the if/elif, the current timestamp is between the start and end times
            if current_timestamp > (reference_timestamp + (0.5 * median_duration)):
                # current frame is too late, don't read it and fill in a dummy frame instead
                frame = cv2.drawMarker(previous_frame, (20, 20), (0, 0, 255), cv2.MARKER_STAR, 30, 1)
                video_writer_object.write(frame)
                synchronized_timestamps.append(None)
                written_frames += 1
                dropped_frames += 1
            elif current_timestamp < (reference_timestamp - (0.5 * median_duration)):
                # current frame is too early, read it and move to the next frame
                ret, frame = cap.read()
                if not ret:
                    print(f"Unable to read frame {current_frame}")
                    raise ValueError("Unable to read frame")
                    break
                previous_frame = frame
                skipped_frames += 1
                current_frame += 1
            else:
                # current frame is in the correct time window
                ret, frame = cap.read()
                if not ret:
                    print(f"Unable to read frame {current_frame}")
                    raise ValueError("Unable to read frame")
                    break
                previous_frame = frame

                video_writer_object.write(frame)
                synchronized_timestamps.append(current_timestamp)

                written_frames += 1
                current_frame += 1

        print(f"Video {output_video_pathstring} saved with {written_frames} frames and {dropped_frames} dropped frames\n"
              f"\t(difference is {written_frames - dropped_frames})\n"
              f"\tearly frames: {early_frames}\n"
              f"\tskipped frames: {skipped_frames}")

        self.synchronized_timestamps[camera_name] = np.array(synchronized_timestamps)

        cap.release()
        video_writer_object.release()

    def trim_videos(self, starting_offsets_frames: Dict[str, int], ending_offsets_frames: Dict[str, int]):
        for cam_name in self.basler_camera_names:
            self.trim_single_video(
                starting_offsets_frames[cam_name],
                ending_offsets_frames[cam_name],
                str(self.synched_videos_path / f"{self.index_to_serial_number[cam_name]}.mp4"),
                str(self.output_path / f"{self.index_to_serial_number[cam_name]}.mp4"),
            )

        self.trim_single_video_pupil(
            camera_name="eye0",
            input_video_pathstring=str(self.pupil_eye0_video_path),
            output_video_pathstring=str(self.output_path / "eye0.mp4"),
        )

        self.trim_single_video_pupil(
            camera_name="eye1",
            input_video_pathstring=str(self.pupil_eye1_video_path),
            output_video_pathstring=str(self.output_path / "eye1.mp4"),
        )

    def verify_framecounts(self):
        for cam_name in self.synchronized_timestamps.keys():
            if cam_name.startswith("eye"):
                cap = cv2.VideoCapture(str(self.output_path / f"{cam_name}.mp4")) # THIS PATH IS WRONG
            else:
                cap = cv2.VideoCapture(str(self.output_path / f"{self.index_to_serial_number[cam_name]}.mp4"))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count != self.synchronized_timestamps[cam_name].shape[0]:
                print(
                    f"frame count mismatch for cam {cam_name}: video: {frame_count} vs timestamps: {self.synchronized_timestamps[cam_name].shape[0]}")
            else:
                print(
                    f"frame count match for cam {cam_name}: video: {frame_count} vs timestamps: {self.synchronized_timestamps[cam_name].shape[0]}")

            cap.release()

    def synchronize(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"latest synched start utc: {self.latest_synched_start_utc}")
        print(f"earliest synched end utc: {self.earliest_synched_end_utc}")
        starting_offsets_frames = self.find_starting_offsets_in_frames()
        ending_offsets_frames = self.find_ending_offsets_in_frames()

        self.synchronization_metadata = {
            "latest_synched_start_utc": self.latest_synched_start_utc,
            "earliest_synched_end_utc": self.earliest_synched_end_utc,
            "starting_offsets_frames": starting_offsets_frames,
            "ending_offsets_frames": ending_offsets_frames,
        }

        self.synchronized_timestamps = {
            cam_name: self.synched_basler_timestamps_utc[i, :][
                      starting_offsets_frames[cam_name]: ending_offsets_frames[cam_name] + 1
                      ] for i, cam_name in enumerate(self.basler_camera_names)
        }

        self.trim_videos(starting_offsets_frames=starting_offsets_frames, ending_offsets_frames=ending_offsets_frames)
        self.save_synchronized_timestamps()
        self.verify_framecounts()
        # TODO: verify all video files exist and are readable

        self.plot_raw_timestamps(
            starting_offsets=starting_offsets_frames,
            ending_offsets=ending_offsets_frames
        )

        self.plot_synchronized_timestamps()

    def plot_raw_timestamps(
            self,
            starting_offsets: Dict[str, int],
            ending_offsets: Dict[str, int],
    ):
        """plot some diagnostics to assess quality of camera sync"""
        # TODO: swap time and frame number, so x axis shows synching
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
                label=f"basler {cam_name}",
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

        # plt.show()
        plt.savefig(str(self.output_path / "raw_timestamps.png"))

    def plot_synchronized_timestamps(self):
        """plot some diagnostics to assess quality of camera sync"""
        # TODO: swap time and frame number, so x axis shows synching
        # opportunistic load of matplotlib to avoid startup time costs
        from matplotlib import pyplot as plt

        plt.set_loglevel("warning")

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"Timestamps")

        ax1 = plt.subplot(
            title="(Synchronized) Camera Frame Timestamp vs Frame#\n(Lines should have same slope)",
            xlabel="Frame#",
            ylabel="Timestamp (ns)",
        )

        for name, timestamps in self.synchronized_timestamps.items():
            ax1.plot(timestamps, label=name)

        ax1.legend()
        ax1.set_ylim(self.latest_synched_start_utc, self.earliest_synched_end_utc)

        plt.tight_layout()

        # plt.show()
        plt.savefig(str(self.output_path / "synchronized_timestamps.png"))


if __name__ == "__main__":
    folder_path = Path(
        "/Users/philipqueen/ferret_0776_P35_EO5/"
    )
    pupil_synchronize = PupilSynchronize(folder_path)

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

    print(
        f"pupil timestamps shapes - eye0: {pupil_synchronize.pupil_eye0_timestamps_utc.shape} eye1: {pupil_synchronize.pupil_eye1_timestamps_utc.shape}"
    )
    print(f"pupil timestamps (eye0): {pupil_synchronize.pupil_eye0_timestamps_utc}")
    print(f"pupil timestamps (eye1): {pupil_synchronize.pupil_eye1_timestamps_utc}")

    pupil_synchronize.synchronize()
