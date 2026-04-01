import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

COLOR_MEAN_CONFIDENCE = [100, 149, 237]   # cornflower blue
COLOR_BLINK = [255, 165, 0]              # orange
COLOR_CONFIDENCE = [148, 103, 189]       # purple
COLOR_POSITION = [44, 160, 44]           # green
COLOR_DENSITY = [255, 80, 80]            # red
COLOR_GOOD_DATA = [255, 255, 255]        # white

_QUALITY_SERIES: list[tuple[str, list[int]]] = [
    ("mean_confidence", COLOR_MEAN_CONFIDENCE),
    ("blink_threshold", COLOR_BLINK),
    ("confidence_threshold", COLOR_CONFIDENCE),
    ("eye_position_threshold", COLOR_POSITION),
    ("density_threshold", COLOR_DENSITY),
    ("good_data", COLOR_GOOD_DATA),
]


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
        values = eye_df[series_name].to_numpy(dtype=float)
        rr.send_columns(
            entity_path=f"{base}/{series_name}",
            indexes=[time_column],
            columns=rr.Scalars.columns(scalars=values),
        )
