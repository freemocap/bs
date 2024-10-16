import subprocess
import cv2
import numpy as np

from pathlib import Path
from datetime import datetime

from skellycam_plots import create_timestamp_diagnostic_plots, timestamps_array_to_dictionary

Z_SCORE_95_CI = 1.96

# TODO: Make plots of timestamps for each camera

def print_video_info(folder_path: Path):
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)
    if not folder_path.exists:
        raise FileNotFoundError("Input folder path does not exist")

    raw_videos_path = folder_path / "raw_videos"
    if not raw_videos_path.exists():
        raw_videos_path = folder_path

    synched_videos_path = folder_path / "synchronized_videos"

    print("Raw Video Information:")
    print_basic_info(raw_videos_path)

    if synched_videos_path.exists():
        print("Synchronized Video Information:")
        print_basic_info(synched_videos_path)


    print_timestamp_info(raw_video_path = raw_videos_path, synched_video_path=synched_videos_path)

def print_basic_info(folder_path: Path):
    for video_path in folder_path.iterdir():
        if video_path.suffix not in {".avi", ".AVI", ".mp4", ".MP4"}:
            continue
        print(f"\tvideo name: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        print(f"\tframe count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        print(f"\treported fps: {cap.get(cv2.CAP_PROP_FPS)}")
        ffprobe_fps = get_ffprobe_fps(
            video_path
        )
        print(f"ffprobe fps: {ffprobe_fps}")
        cap.release()

def print_timestamp_info(raw_video_path: Path, synched_video_path: Path):
    timestamp_file_name = "timestamps.npy"
    raw_timestamps_path = raw_video_path / timestamp_file_name
    synched_timestamps_path = synched_video_path / timestamp_file_name
    if raw_timestamps_path.exists():
        raw_timestamps = np.load(raw_timestamps_path) 
        print_timestamp_statistics(timestamps=raw_timestamps)
        raw_timestamp_dict = timestamps_array_to_dictionary(raw_timestamps)

    if synched_timestamps_path.exists():
        synched_timestamps = np.load(synched_timestamps_path)
        print_timestamp_statistics(timestamps=synched_timestamps)
        synched_timestamp_dict = timestamps_array_to_dictionary(synched_timestamps)

    if raw_timestamps_path.exists():
        if not synched_timestamps_path.exists():
            synched_timestamp_dict = None
        print(f"Creating timestamp diagnostic plots, will save to: {synched_timestamps_path.parent / "timestamp_diagnostic_plot.png"}")
        create_timestamp_diagnostic_plots(
            raw_timestamp_dictionary=raw_timestamp_dict,
            synchronized_timestamp_dictionary=synched_timestamp_dict,
            path_to_save_plots_png=synched_timestamps_path.parent / "timestamp_diagnostic_plot.png"
        )

def print_timestamp_statistics(timestamps: np.ndarray):
    print(f"shape of timestamps: {timestamps.shape}")
    starting_time = np.min(timestamps)

    by_camera_fps = []
    by_camera_frame_duration = []

    for i in range(timestamps.shape[0]):
        num_samples = timestamps.shape[1]
        samples = (timestamps[i, :] - starting_time) / 1e9
        fps = num_samples / (samples[-1] - samples[0])
        mean_frame_duration = np.mean(np.diff(timestamps[i, :])) / 1e6
        by_camera_fps.append(fps)
        by_camera_frame_duration.append(mean_frame_duration)
        units = "seconds"
        print(f"cam {i} Descriptive Statistics:")
        print(f"\tEarliest Timestamp: {np.min(samples):.3f} {units}")
        print(f"\tLatest Timestamp: {np.max(samples):.3f} {units}")
        print(f"\tFPS: {fps}")
        print(f"\tMean Frame Duration for Camera: {mean_frame_duration} ms")

    print("Overall FPS and Mean Frame Duration")
    print(f"\tMean Overall FPS: {np.nanmean(by_camera_fps)}")
    print(f"\tMean Overall Mean Frame Duration: {np.nanmean(by_camera_frame_duration)}")

    for i in range(0, timestamps.shape[1]-1, 15):
        num_samples = timestamps.shape[0]
        samples = (timestamps[:, i] - starting_time) / 1e9
        units = "seconds"
        print(f"frame {i} Descriptive Statistics")
        print(f"\tNumber of Samples: {num_samples} {units}")
        print(f"\tMean: {np.nanmean(samples):.3f} {units}")
        print(f"\tMedian: {np.nanmedian(samples):.3f} {units}")
        print(f"\tStandard Deviation: {np.nanstd(samples):.3f} {units}")
        print(f"\tMedian Absolute Deviation: {np.nanmedian(np.abs(samples - np.nanmedian(samples))):.3f} {units}")
        print(f"\tInterquartile Range: {np.nanpercentile(samples, 75) - np.nanpercentile(samples, 25):.3f} {units}")
        print(f"\t95% Confidence Interval: {(Z_SCORE_95_CI * np.nanstd(samples) / (num_samples**0.5)):.3f} {units}")
        print(f"\tEarliest Timestamp: {np.min(samples):.3f}")
        print(f"\tLatest Timestamp: {np.max(samples):.3f}") 
        print(f"\tMean Frame Duration for Camera: {np.nanmean(timestamps[:, i+1] - timestamps[:, i]) / 1e6} ms")


def get_ffprobe_fps(video_path: Path) -> float:
    duration_subprocess = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            f"{video_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = duration_subprocess.stdout
    print(f"duration from ffprobe: {float(duration)} seconds")

    frame_count_subprocess = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            f"{video_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    frame_count = int(frame_count_subprocess.stdout)

    print(f"frame count from ffprobe: {frame_count}")

    return frame_count / float(duration)


if __name__ == "__main__":
    folder_path = Path(
        "/home/scholl-lab/recordings/test__3"
    )

    print_video_info(folder_path)

