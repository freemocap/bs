"""
Reprojection Error Viewer
=========================

Visualizes skull solver reprojection error as time-series plots in Rerun.

Layout:
    Top:    Overall mean reprojection error (px)
    Middle: Per-camera mean reprojection error (one trace per camera)
    Bottom: Per-keypoint mean reprojection error (one trace per keypoint)

Prefers the resampled data in analyzable_output/ (zeroed timestamps, common
frame rate); falls back to the original data in solver_output/.

Usage:
    python -m python_code.rerun_viewer.plot_reprojection_error <recording_folder>
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import rerun as rr
import rerun.blueprint as rrb

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

# ── Camera display names ──────────────────────────────────────────────────────
CAMERA_LABELS = {
    "24676894": "topdown",
    "24908831": "side_0",
    "24908832": "side_1",
    "25000609": "side_2",
    "25006505": "side_3",
}

# ── Colour palettes ───────────────────────────────────────────────────────────
CAMERA_COLORS = {
    "24676894": [100, 149, 237],   # cornflower blue  (topdown)
    "24908831": [255,  80,  80],   # red              (side 0)
    "24908832": [ 80, 200,  80],   # green            (side 1)
    "25000609": [255, 165,   0],   # orange           (side 2)
    "25006505": [180,  80, 220],   # purple           (side 3)
}

KEYPOINT_COLORS = {
    "nose":          [255, 100, 180],  # pink
    "left_eye":      [  0, 220, 220],  # cyan
    "right_eye":     [255, 230,  50],  # yellow
    "left_ear":      [ 80, 160, 255],  # light blue
    "right_ear":     [100, 230, 100],  # light green
    "base":          [180, 180, 180],  # grey
    "left_cam_tip":  [255, 140,  60],  # orange-red
    "right_cam_tip": [255, 200, 120],  # peach
}

OVERALL_COLOR = [255, 255, 100]  # bright yellow


# ── Data loading ──────────────────────────────────────────────────────────────

def _find_csv_and_ts_col(recording_folder: RecordingFolder) -> tuple[Path, str]:
    """
    Locate the best available reprojection errors CSV.

    Prefers the resampled version in analyzable_output (timestamp_s, zeroed).
    Falls back to the original in solver_output (timestamp, UTC seconds).
    """
    resampled_csv = (
        recording_folder.analyzable_output / "reprojection_errors" / "reprojection_errors.csv"
        if recording_folder.analyzable_output
        else None
    )
    if resampled_csv and resampled_csv.exists():
        return resampled_csv, "timestamp_s"

    if recording_folder.mocap_solver_output:
        original_csv = recording_folder.mocap_solver_output / "reprojection_errors.csv"
        if original_csv.exists():
            return original_csv, "timestamp"

    raise FileNotFoundError(
        "No reprojection_errors.csv found. Run postprocess_recording first."
    )


def load_reprojection_data(recording_folder: RecordingFolder):
    """
    Load and pivot the reprojection error CSV into per-series numpy arrays.

    Returns a dict with keys:
        timestamps         : (n_frames,) float64 in seconds
        overall_mean       : (n_frames,) float64 — mean across all cameras + keypoints
        per_camera         : dict[camera_id, (n_frames,) float64] — mean across keypoints
        per_keypoint       : dict[keypoint_name, (n_frames,) float64] — mean across cameras
    """
    csv_path, ts_col = _find_csv_and_ts_col(recording_folder)
    print(f"Loading reprojection errors from: {csv_path}")

    df = pl.read_csv(csv_path)

    # ── Overall mean ────────────────────────────────────────────────────────
    overall_df = (
        df.filter((pl.col("trajectory") == "mean") & (pl.col("component") == "mean"))
        .sort("frame")
    )
    timestamps = overall_df[ts_col].to_numpy().astype(np.float64)
    # Zero timestamps if using UTC (solver output fallback)
    if ts_col == "timestamp":
        timestamps = timestamps - timestamps[0]
    overall_mean = overall_df["value"].to_numpy().astype(np.float64)

    n_frames = len(timestamps)

    # ── Per-camera mean (computed from per-camera/per-keypoint rows) ─────────
    camera_ids = [t for t in df["trajectory"].unique().to_list() if t != "mean"]
    per_camera: dict[str, np.ndarray] = {}
    for camera_id in sorted(camera_ids):
        cam_df = (
            df.filter(pl.col("trajectory") == camera_id)
            .group_by("frame")
            .agg(pl.col("value").mean().alias("mean_error"))
            .sort("frame")
        )
        values = np.full(n_frames, np.nan)
        for row in cam_df.iter_rows(named=True):
            if row["frame"] < n_frames:
                values[row["frame"]] = row["mean_error"]
        per_camera[camera_id] = values

    # ── Per-keypoint mean (from trajectory=="mean", component==keypoint) ─────
    keypoint_names = [
        c for c in df["component"].unique().to_list() if c != "mean"
    ]
    per_keypoint: dict[str, np.ndarray] = {}
    for kp_name in sorted(keypoint_names):
        kp_df = (
            df.filter(
                (pl.col("trajectory") == "mean") & (pl.col("component") == kp_name)
            )
            .sort("frame")
        )
        values = np.full(n_frames, np.nan)
        for row in kp_df.iter_rows(named=True):
            if row["frame"] < n_frames:
                values[row["frame"]] = row["value"]
        per_keypoint[kp_name] = values

    return {
        "timestamps": timestamps,
        "overall_mean": overall_mean,
        "per_camera": per_camera,
        "per_keypoint": per_keypoint,
    }


# ── Rerun logging ─────────────────────────────────────────────────────────────

def _log_scalar_series(
    entity_path: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    color: list[int],
    width: float = 1.5,
) -> None:
    """Log a scalar time series as both a line and point markers."""
    rr.log(entity_path, rr.SeriesLines(widths=width, colors=[color]), static=True)
    rr.log(entity_path, rr.SeriesPoints(marker_sizes=2.0, colors=[color]), static=True)
    rr.send_columns(
        entity_path=entity_path,
        indexes=[rr.TimeColumn("time", duration=timestamps)],
        columns=rr.Scalars.columns(scalars=values),
    )


def log_reprojection_error(data: dict, base: str = "/reprojection_error") -> None:
    """Log all reprojection error series to Rerun."""
    timestamps = data["timestamps"]

    # ── Overall mean ─────────────────────────────────────────────────────────
    _log_scalar_series(
        entity_path=f"{base}/overall/mean",
        timestamps=timestamps,
        values=data["overall_mean"],
        color=OVERALL_COLOR,
        width=2.5,
    )

    # ── Per-camera ───────────────────────────────────────────────────────────
    for camera_id, values in data["per_camera"].items():
        label = CAMERA_LABELS.get(camera_id, camera_id)
        color = CAMERA_COLORS.get(camera_id, [200, 200, 200])
        _log_scalar_series(
            entity_path=f"{base}/per_camera/{label}",
            timestamps=timestamps,
            values=values,
            color=color,
        )

    # ── Per-keypoint ─────────────────────────────────────────────────────────
    for kp_name, values in data["per_keypoint"].items():
        color = KEYPOINT_COLORS.get(kp_name, [200, 200, 200])
        _log_scalar_series(
            entity_path=f"{base}/per_keypoint/{kp_name}",
            timestamps=timestamps,
            values=values,
            color=color,
        )


# ── Blueprint ─────────────────────────────────────────────────────────────────

def _make_time_series_view(name: str, origin: str, y_label: str = "px") -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
        name=name,
        origin=origin,
        plot_legend=rrb.PlotLegend(visible=True),
        axis_y=rrb.ScalarAxis(lock_range_during_zoom=True),
    )


def make_blueprint(base: str = "/reprojection_error") -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            _make_time_series_view(
                "Overall Mean Reprojection Error (px)",
                f"{base}/overall",
            ),
            _make_time_series_view(
                "Per-Camera Mean Reprojection Error (px)",
                f"{base}/per_camera",
            ),
            _make_time_series_view(
                "Per-Keypoint Mean Reprojection Error (px)",
                f"{base}/per_keypoint",
            ),
        )
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def plot_reprojection_error(recording_folder: RecordingFolder) -> None:
    data = load_reprojection_data(recording_folder)

    timestamps = data["timestamps"]
    overall = data["overall_mean"]
    valid = overall[~np.isnan(overall)]
    if len(valid) > 0:
        print(f"Overall mean reprojection error: {np.mean(valid):.2f} px  "
              f"(median {np.median(valid):.2f} px,  max {np.max(valid):.2f} px)")

    recording_string = (
        f"reprojection_error_{recording_folder.recording_name}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)
    rr.send_blueprint(make_blueprint())
    log_reprojection_error(data)


if __name__ == "__main__":
    from python_code.utilities.folder_utilities.recording_folder import PipelineStep

    if len(sys.argv) < 2:
        print("Usage: python -m python_code.rerun_viewer.plot_reprojection_error <recording_folder>")
        sys.exit(1)

    folder = RecordingFolder.from_folder_path(
        sys.argv[1],
        expected_processing_step=PipelineStep.SKULL_POST_PROCESSED,
    )
    plot_reprojection_error(folder)
