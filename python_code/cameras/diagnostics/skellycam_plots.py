import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel
from scipy.stats import median_abs_deviation

logger = logging.getLogger(__name__)


class TimestampDiagnosticsDataClass(BaseModel):
    mean_framerates_per_camera: dict
    standard_deviation_framerates_per_camera: dict
    median_framerates_per_camera: dict
    median_absolute_deviation_per_camera: dict
    mean_mean_framerate: float
    mean_standard_deviation_framerates: float
    mean_median_framerates: float
    mean_median_absolute_deviation_per_camera: float


def create_timestamp_diagnostic_plots(
    path_to_save_plots_png: Union[str, Path],
    raw_timestamp_dictionary: Dict[int, np.ndarray],
    synchronized_timestamp_dictionary: Optional[Dict[int, np.ndarray]] = None,
):
    """plot some diagnostics to assess quality of camera sync"""

    # opportunistic load of matplotlib to avoid startup time costs
    from matplotlib import pyplot as plt

    plt.set_loglevel("warning")

    for timestamp_array in raw_timestamp_dictionary.values():
        timestamp_array /= 1e9

    if synchronized_timestamp_dictionary:
        for timestamp_array in  synchronized_timestamp_dictionary.values():
            timestamp_array /= 1e9

    max_frame_duration = 0.1
    fig = plt.figure(figsize=(18, 12))
    session_name = Path(path_to_save_plots_png).parent.parent.parent.stem
    recording_name = Path(path_to_save_plots_png).parent.parent.stem
    fig.suptitle(f"Timestamps of synchronized frames\nsession: {session_name}, recording: {recording_name}")

    ax1 = plt.subplot(
        231,
        title="(Raw) Camera Frame Timestamp vs Frame#\n(Lines should have same slope)",
        xlabel="Frame#",
        ylabel="Timestamp (sec)",
    )
    ax2 = plt.subplot(
        232,
        ylim=(0, max_frame_duration / 2),
        title="(Raw) Camera Frame Duration Trace",
        xlabel="Frame#",
        ylabel="Duration (sec)",
    )
    ax3 = plt.subplot(
        233,
        xlim=(0, max_frame_duration / 4),
        title="(Raw) Camera Frame Duration Histogram (count)",
        xlabel="Duration(s, 0.1ms bins)",
        ylabel="Probability",
    )
    ax4 = plt.subplot(
        234,
        title="(Synchronized) Camera Frame Timestamp vs Frame#\n(Lines should be on top of each other)",
        xlabel="Frame#",
        ylabel="Timestamp (sec)",
    )
    ax5 = plt.subplot(
        235,
        ylim=(0, max_frame_duration/2),
        title="(Synchronized) Camera Frame Duration Trace",
        xlabel="Frame#",
        ylabel="Duration (sec)",
    )
    ax6 = plt.subplot(
        236,
        xlim=(0, max_frame_duration / 4),
        title="(Synchronized) Camera Frame Duration Histogram (count)",
        xlabel="Duration(s, 0.1ms bins)",
        ylabel="Probability",
    )

    for camera_id, timestamps in raw_timestamp_dictionary.items():
        ax1.plot(timestamps, label=f"Camera# {str(camera_id)}")
        ax1.legend()
        ax2.plot(np.diff(timestamps), ".")
        ax3.hist(
            np.diff(timestamps),
            bins=np.arange(0, max_frame_duration, 0.00025),
            alpha=0.5,
        )

    if synchronized_timestamp_dictionary:
        for camera_id, timestamps in synchronized_timestamp_dictionary.items():
            ax4.plot(timestamps, label=f"Camera# {str(camera_id)}")
            ax4.legend()
            ax5.plot(np.diff(timestamps), ".")
            ax6.hist(
                np.diff(timestamps),
                bins=np.arange(0, max_frame_duration, 0.00025),
                alpha=0.5,
            )

    plt.tight_layout()

    fig_save_path = Path(path_to_save_plots_png)
    plt.savefig(str(fig_save_path))
    logger.info(f"Saving diagnostic figure tp: {fig_save_path}")

def timestamps_array_to_dictionary(timestamps_array: np.ndarray) -> Dict[int, np.ndarray]:
    return {i: timestamps_array[i, :] for i in range(timestamps_array.shape[0])}

def calculate_camera_diagnostic_results(
    timestamps_dictionary,
) -> TimestampDiagnosticsDataClass:
    mean_framerates_per_camera = {}
    standard_deviation_framerates_per_camera = {}
    median_framerates_per_camera = {}
    median_absolute_deviation_per_camera = {}

    for cam_id, timestamps in timestamps_dictionary.items():
        timestamps_formatted = (np.asarray(timestamps) - timestamps[0]) / 1e9
        frame_durations = np.diff(timestamps_formatted)
        if np.any(frame_durations == 0):
            print(f"zeroes found in frame durations for camera {cam_id}, replacing with NaN")
            frame_durations = np.where(frame_durations == 0, np.nan, frame_durations)
        framerate_per_frame = 1 / frame_durations
        mean_framerates_per_camera[cam_id] = np.nanmean(framerate_per_frame)
        median_framerates_per_camera[cam_id] = np.nanmedian(framerate_per_frame)
        standard_deviation_framerates_per_camera[cam_id] = np.nanstd(
            framerate_per_frame
        )
        median_absolute_deviation_per_camera[cam_id] = median_abs_deviation(
            framerate_per_frame
        )

    mean_mean_framerate = np.nanmean(list(mean_framerates_per_camera.values()))
    mean_standard_deviation_framerates = np.nanmean(
        list(standard_deviation_framerates_per_camera.values())
    )
    mean_median_framerates = np.nanmean(list(median_framerates_per_camera.values()))
    mean_median_absolute_deviation_per_camera = np.nanmean(
        list(median_absolute_deviation_per_camera.values())
    )

    return TimestampDiagnosticsDataClass(
        mean_framerates_per_camera=mean_framerates_per_camera,
        standard_deviation_framerates_per_camera=standard_deviation_framerates_per_camera,
        median_framerates_per_camera=median_framerates_per_camera,
        median_absolute_deviation_per_camera=median_absolute_deviation_per_camera,
        mean_mean_framerate=float(mean_mean_framerate),
        mean_standard_deviation_framerates=float(mean_standard_deviation_framerates),
        mean_median_framerates=float(mean_median_framerates),
        mean_median_absolute_deviation_per_camera=float(
            mean_median_absolute_deviation_per_camera
        ),
    )