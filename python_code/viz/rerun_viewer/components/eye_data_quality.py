from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from pydantic import BaseModel

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

COLOR_MEAN_CONFIDENCE       = [100, 149, 237]   # cornflower blue
COLOR_BLINK                 = [255, 165, 0]     # orange
COLOR_BLINK_HIGH            = [255, 100, 0]     # dark orange
COLOR_COMBINED_BLINK        = [255, 220, 100]   # yellow
COLOR_COMBINED_BLINK_HIGH   = [200, 160, 0]     # dark yellow
COLOR_CONFIDENCE            = [148, 103, 189]   # purple
COLOR_CONFIDENCE_HIGH       = [90, 50, 140]     # dark purple
COLOR_POSITION              = [44, 160, 44]     # green
COLOR_DENSITY               = [255, 80, 80]     # red
COLOR_GOOD_DATA_LOW         = [180, 180, 180]   # light gray
COLOR_GOOD_DATA_MEDIUM      = [255, 255, 255]   # white
COLOR_GOOD_DATA_HIGH        = [100, 220, 100]   # bright green

_QUALITY_SERIES: list[tuple[str, list[int]]] = [
    ("mean_confidence",               COLOR_MEAN_CONFIDENCE),
    ("confidence_threshold",          COLOR_CONFIDENCE),
    ("confidence_threshold_high",     COLOR_CONFIDENCE_HIGH),
    ("blink_threshold",               COLOR_BLINK),
    ("blink_threshold_high",          COLOR_BLINK_HIGH),
    ("combined_blink_threshold",      COLOR_COMBINED_BLINK),
    ("combined_blink_threshold_high", COLOR_COMBINED_BLINK_HIGH),
    ("eye_position_threshold",        COLOR_POSITION),
    ("density_threshold",             COLOR_DENSITY),
    ("good_data_low",                 COLOR_GOOD_DATA_LOW),
    ("good_data_medium",              COLOR_GOOD_DATA_MEDIUM),
    ("good_data_high",                COLOR_GOOD_DATA_HIGH),
]


class EyeDataQualityContext(BaseModel):
    eye_name: Literal["left", "right"]
    entity_path: str = "/"


def required_data_available(recording_folder: RecordingFolder, context: "EyeDataQualityContext") -> bool:
    timestamps_npy = (
        recording_folder.left_eye_timestamps_npy
        if context.eye_name == "left"
        else recording_folder.right_eye_timestamps_npy
    )
    return recording_folder.eye_mean_confidence is not None and timestamps_npy is not None


def get_eye_quality_view(
    eye_name: str,
    entity_path: str = "/",
    time_window_seconds: float = 5.0,
) -> rrb.TimeSeriesView:
    if not entity_path.endswith("/"):
        entity_path += "/"

    scrolling_time_range = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    return rrb.TimeSeriesView(
        name=f"{eye_name.capitalize()} Eye Data Quality",
        origin=f"{entity_path}quality/{eye_name}_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-0.1, 1.1)),
    )


def log_eye_quality_style(eye_name: str, entity_path: str = "/"):
    if not entity_path.endswith("/"):
        entity_path += "/"
    base = f"{entity_path}quality/{eye_name}_eye"

    for series_name, color in _QUALITY_SERIES:
        rr.log(f"{base}/{series_name}", rr.SeriesLines(widths=1.5, colors=[color]), static=True)
        rr.log(f"{base}/{series_name}", rr.SeriesPoints(marker_sizes=2.0, colors=[color]), static=True)


def plot_eye_quality(
    eye_name: str,
    camera_name: str,
    confidence_df: pd.DataFrame,
    all_timestamps: np.ndarray,
    entity_path: str = "/",
):
    """
    Plot eye data quality timeseries for one eye.

    Args:
        eye_name: "left" or "right" — used for entity path labelling
        camera_name: "eye0" or "eye1" — matches the camera column in confidence_df
        confidence_df: DataFrame from eye_model_v3_mean_confidence.csv
        all_timestamps: 1-D seconds array indexed by frame number (relative to recording start)
        entity_path: Rerun entity path prefix
    """
    if not entity_path.endswith("/"):
        entity_path += "/"
    base = f"{entity_path}quality/{eye_name}_eye"

    eye_df = confidence_df[confidence_df["camera"] == camera_name].copy()
    eye_df = eye_df.sort_values("frames").reset_index(drop=True)

    frame_indices = eye_df["frames"].to_numpy()
    timestamps = all_timestamps[frame_indices]
    time_column = rr.TimeColumn("time", duration=timestamps)

    for series_name, _ in _QUALITY_SERIES:
        if series_name not in eye_df.columns:
            print(f"Warning: column '{series_name}' not found in confidence DataFrame, skipping")
            continue
        values = eye_df[series_name].to_numpy(dtype=float)
        rr.send_columns(
            entity_path=f"{base}/{series_name}",
            indexes=[time_column],
            columns=rr.Scalars.columns(scalars=values),
        )


def plot_eye_data_quality(recording_folder: RecordingFolder, eye_name: str, entity_path: str = "/"):
    context = EyeDataQualityContext(eye_name=eye_name, entity_path=entity_path)
    if not required_data_available(recording_folder, context):
        print(f"Skipping eye data quality for {eye_name}: required data not found")
        return

    timestamps_npy = (
        recording_folder.left_eye_timestamps_npy
        if eye_name == "left"
        else recording_folder.right_eye_timestamps_npy
    )
    camera_name = (
        recording_folder.left_eye_name if eye_name == "left" else recording_folder.right_eye_name
    )

    confidence_df = pd.read_csv(recording_folder.eye_mean_confidence)
    all_timestamps = np.load(timestamps_npy)
    all_timestamps = all_timestamps - all_timestamps[0]

    plot_eye_quality(
        eye_name=eye_name,
        camera_name=camera_name,
        confidence_df=confidence_df,
        all_timestamps=all_timestamps,
        entity_path=entity_path,
    )


if __name__ == "__main__":
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2026-03-07_ferret_407_EO7/full_recording"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    eye_name, entity_path = "left", "/quality_plots"

    view = get_eye_quality_view(eye_name, entity_path)
    rr.send_blueprint(rrb.Horizontal(view))
    log_eye_quality_style(eye_name, entity_path)

    plot_eye_data_quality(recording_folder=recording_folder, eye_name=eye_name, entity_path=entity_path)
