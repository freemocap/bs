from typing import Dict
import cv2
import numpy as np
from scipy import signal

from pathlib import Path

# TODO: should this just cross correlate the timestamps?

class TimestampSynchronize:
    def __init__(self, folder_path: Path):
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        if not folder_path.exists:
            raise FileNotFoundError("Input folder path does not exist")

        raw_videos_path = folder_path / "raw_videos"
        if not raw_videos_path.exists():
            raw_videos_path = folder_path
        self.raw_videos_path = raw_videos_path
        self.synched_videos_path = folder_path / "synched_videos"

        timestamp_file_name = "timestamps.npy"
        timestamp_path = folder_path / timestamp_file_name
        if not timestamp_path.exists():
            timestamp_path = self.raw_videos_path / timestamp_file_name
            if not timestamp_path.exists():
                raise FileNotFoundError("Unable to find timestamps.npy file in main folder or raw_videos folder")
        self.timestamps = np.load(timestamp_path)

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
        self.find_cross_correlation_lags(timestamps=self.timestamps)
        # self.create_starting_timestamp_dict()
        # self.create_frame_offset_dict()

    def create_capture_dict(self):
        self.capture_dict = {
            video_path.name: cv2.VideoCapture(str(video_path))
            for video_path in self.raw_videos_path.glob("*.mp4")
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
            video_name: self.timestamps[i, 0]
            for i, video_name in enumerate(self.capture_dict.keys())
        }
    
    def cross_correlate(self, reference: np.ndarray, comparison: np.ndarray) -> int:
        """Take two ndarrays, synchronize them using cross correlation, output a lag that can be used to synchronize them.
        Inputs are two arrays to be synchronized. Return the lag expressed in terms of the sample rate of the array.
        """

        # compute cross correlation with scipy correlate function, which gives the correlation of every different lag value
        # mode='full' makes sure every lag value possible between the two signals is used, and method='fft' uses the fast fourier transform to speed the process up
        correlation = signal.correlate(reference, comparison, mode="full", method="fft")
        # lags gives the amount of time shift used at each index, corresponding to the index of the correlate output list
        lags = signal.correlation_lags(reference.size, comparison.size, mode="full")
        # lag is the time shift used at the point of maximum correlation - this is the key value used for shifting our audio/video
        lag = lags[np.argmax(correlation)]

        return int(lag)

    def find_cross_correlation_lags(
        self, timestamps: np.ndarray
    ) -> Dict[str, float]:
        """Take an array of timestamps, as well as the sample rate of the audio, cross correlate the audio files, and output a lag dictionary.
        The lag dict is normalized so that the lag of the latest video to start in time is 0, and all other lags are positive.
        """

        lag_dict = {
            i: self.cross_correlate(reference = timestamps[0, :], comparison = timestamps[i, :])
            for i in range(timestamps.shape[0])
        }  # cross correlates all audio to the first audio file in the dict, and divides by the audio sample rate in order to get the lag in seconds

        # normalized_lag_dict = self.normalize_lag_dictionary(lag_dictionary=lag_dict)

        # print(
        #     f"original lag dict: {lag_dict} normalized lag dict: {normalized_lag_dict}"
        # )

        self.frame_offset_dict = lag_dict
        print(lag_dict)
        return lag_dict

        # return normalized_lag_dict

    def create_frame_offset_dict(self):
        # TODO: get the offsets of starting times from the last starting timestamps (offset <= 0), this will be used to trim 
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
    folder_path = Path(
        "/home/scholl-lab/recordings/mouse_zaber"
    )

    timestamp_synchronize = TimestampSynchronize(folder_path)
    timestamp_synchronize.setup()
    # timestamp_synchronize.synchronize()
