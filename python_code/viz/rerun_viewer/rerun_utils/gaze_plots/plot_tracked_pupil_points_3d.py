from pathlib import Path

import numpy as np
import polars as pl
import rerun as rr
from numpy.typing import NDArray

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

PUPIL_CENTER_COLOR: tuple[int, int, int] = (255, 215, 0)  # gold
PUPIL_BOUNDARY_COLOR: tuple[int, int, int] = (255, 140, 0)  # dark orange

PUPIL_BOUNDARY_POINT_NAMES: list[str] = [f"p{i + 1}" for i in range(8)]


def load_tracked_pupil_points_from_parquet(
    parquet_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Load actual tracked pupil points (world space) from a {side}_gaze_kinematics.parquet file.

    These are the `tracked_pupil__*` trajectories added alongside the canonical
    `keypoint__pupil_center` - real per-frame detections rather than idealized
    geometry. They are parquet-only (not written to the CSV).

    Returns:
        timestamps: (N,)
        pupil_center_world: (N, 3)
        pupil_points_world: (N, 8, 3) in p1..p8 order
    """
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(pl.col("trajectory").cast(pl.Utf8))

    center_df = df.filter(pl.col("trajectory") == "tracked_pupil__pupil_center")
    if center_df.is_empty():
        raise ValueError(
            f"No 'tracked_pupil__pupil_center' trajectory found in {parquet_path} - "
            "was this recording processed after tracked pupil world projection was added?"
        )

    timestamps_df = center_df.select(["frame", "timestamp_s"]).unique().sort("frame")
    n_frames = timestamps_df.height
    timestamps = timestamps_df["timestamp_s"].to_numpy().astype(np.float64)

    def _pivot_xyz(trajectory_name: str) -> NDArray[np.float64]:
        pivoted = (
            df.filter(pl.col("trajectory") == trajectory_name)
            .select(["frame", "component", "value"])
            .pivot(on="component", index="frame", values="value")
            .sort("frame")
        )
        return pivoted.select(["x", "y", "z"]).to_numpy().astype(np.float64)

    pupil_center_world = _pivot_xyz("tracked_pupil__pupil_center")

    pupil_points_world = np.zeros((n_frames, len(PUPIL_BOUNDARY_POINT_NAMES), 3), dtype=np.float64)
    for point_index, point_name in enumerate(PUPIL_BOUNDARY_POINT_NAMES):
        pupil_points_world[:, point_index, :] = _pivot_xyz(f"tracked_pupil__{point_name}")

    return timestamps, pupil_center_world, pupil_points_world


def log_tracked_pupil_points_3d_style(
    eye_name: str,
    entity_path: str = "/",
) -> None:
    if not entity_path.endswith("/"):
        entity_path += "/"

    rr.log(
        f"{entity_path}skeleton/tracked_pupil_{eye_name}/center",
        rr.Points3D.from_fields(radii=3.0, colors=PUPIL_CENTER_COLOR),
        static=True,
    )
    rr.log(
        f"{entity_path}skeleton/tracked_pupil_{eye_name}/boundary",
        rr.Points3D.from_fields(radii=1.5, colors=PUPIL_BOUNDARY_COLOR),
        static=True,
    )


def send_tracked_pupil_points_3d(
    timestamps: NDArray[np.float64],
    pupil_center_world: NDArray[np.float64],
    pupil_points_world: NDArray[np.float64],
    eye_name: str,
    entity_path: str = "/",
) -> None:
    """
    Send the actual tracked pupil center + 8 boundary points (world space) to Rerun,
    animated over time. Logs under `{entity_path}skeleton/tracked_pupil_{eye_name}/...`
    so it shows up in the existing "3D Skeleton" view (which shows everything under
    `skeleton/`) without needing a new view.
    """
    if not entity_path.endswith("/"):
        entity_path += "/"

    t0 = timestamps[0]
    time_column = rr.TimeColumn("time", duration=timestamps - t0)

    rr.send_columns(
        f"{entity_path}skeleton/tracked_pupil_{eye_name}/center",
        indexes=[time_column],
        columns=[*rr.Points3D.columns(positions=pupil_center_world)],
    )
    rr.send_columns(
        f"{entity_path}skeleton/tracked_pupil_{eye_name}/boundary",
        indexes=[time_column],
        columns=[*rr.Points3D.columns(positions=pupil_points_world)],
    )


def plot_tracked_pupil_points_3d(
    eye_name: str,
    recording_folder: RecordingFolder,
    entity_path: str = "/",
) -> None:
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")

    if recording_folder.gaze_kinematics is None:
        print("  Gaze kinematics not found, skipping tracked pupil points.")
        return

    parquet_path = recording_folder.gaze_kinematics / f"{eye_name}_gaze_kinematics.parquet"
    if not parquet_path.exists():
        print(f"  {parquet_path.name} not found, skipping tracked pupil points.")
        return

    timestamps, pupil_center_world, pupil_points_world = load_tracked_pupil_points_from_parquet(
        parquet_path
    )
    print(f"  Loaded {eye_name} tracked pupil points: {len(timestamps)} frames")

    send_tracked_pupil_points_3d(
        timestamps=timestamps,
        pupil_center_world=pupil_center_world,
        pupil_points_world=pupil_points_world,
        eye_name=eye_name,
        entity_path=entity_path,
    )


if __name__ == "__main__":
    import rerun.blueprint as rrb
    from datetime import datetime

    from python_code.viz.rerun_viewer.rerun_utils.gaze_plots.plot_ferret_skull_and_spine_3d import (
        get_ferret_skull_and_spine_3d_view,
    )

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    entity_path = "/"

    rr.init(recording_string, spawn=True)

    view = get_ferret_skull_and_spine_3d_view(entity_path=entity_path)
    blueprint = rrb.Horizontal(view)
    rr.send_blueprint(blueprint)

    for side in ["left", "right"]:
        log_tracked_pupil_points_3d_style(eye_name=side, entity_path=entity_path)
        plot_tracked_pupil_points_3d(eye_name=side, recording_folder=recording_folder, entity_path=entity_path)
