"""
Serialization for FerretEyeKinematics.

Saves and loads FerretEyeKinematics to/from disk using:
- {name}_reference_geometry.json: eyeball reference geometry
- {name}_kinematics.csv: tidy-format CSV with essential kinematics data

The CSV format is tidy (one row per observation) with columns:
    frame, timestamp_s, trajectory, component, value, units

Saved trajectories:
    - orientation (quaternion wxyz)
    - angular_velocity_global / angular_velocity_local
    - angular_acceleration_global / angular_acceleration_local
    - socket_landmark__tear_duct / socket_landmark__outer_eye
    - tracked_pupil__center / tracked_pupil__p1-p8

NOT saved (can be recomputed from quaternions + reference geometry):
    - position (always [0,0,0] for eye)
    - linear velocity/acceleration (always [0,0,0])
    - canonical keypoints (rotated reference geometry)
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import (
    FerretEyeKinematics,
    SocketLandmarks,
    TrackedPupil,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eyeball_reference_geometry import (
    NUM_PUPIL_POINTS,
)
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics

logger = logging.getLogger(__name__)


def _build_vector_chunk(
    frame_indices: NDArray[np.int64],
    timestamps: NDArray[np.float64],
    values: NDArray[np.float64],
    trajectory_name: str,
    component_names: list[str],
    units: str,
) -> pl.DataFrame:
    """Build a tidy DataFrame chunk for a vector quantity."""
    n_frames = len(frame_indices)
    n_components = len(component_names)

    repeated_frames = np.repeat(frame_indices, n_components)
    repeated_timestamps = np.repeat(timestamps, n_components)
    tiled_components = np.tile(component_names, n_frames)
    flattened_values = values.ravel()

    return pl.DataFrame({
        "frame": repeated_frames,
        "timestamp_s": repeated_timestamps,
        "component": tiled_components,
        "value": flattened_values,
    }).with_columns(
        pl.lit(trajectory_name).alias("trajectory").cast(pl.Categorical),
        pl.col("component").cast(pl.Categorical),
        pl.lit(units).alias("units").cast(pl.Categorical),
    ).select(["frame", "timestamp_s", "trajectory", "component", "value", "units"])


def ferret_eye_kinematics_to_tidy_dataframe(
    kinematics: FerretEyeKinematics,
) -> pl.DataFrame:
    """
    Convert FerretEyeKinematics to a tidy-format polars DataFrame.

    Only saves essential data - canonical keypoints are NOT saved since they
    can be recomputed from quaternions + reference geometry.
    """
    n_frames = kinematics.n_frames
    frame_indices = np.arange(n_frames, dtype=np.int64)
    timestamps = kinematics.timestamps

    chunks: list[pl.DataFrame] = []

    # Orientation quaternion (wxyz)
    chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.quaternions_wxyz,
        trajectory_name="orientation",
        component_names=["w", "x", "y", "z"],
        units="quaternion",
    ))

    # Angular velocity global
    chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.angular_velocity_global,
        trajectory_name="angular_velocity_global",
        component_names=["x", "y", "z"],
        units="rad_s",
    ))

    # Angular velocity local
    chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.angular_velocity_local,
        trajectory_name="angular_velocity_local",
        component_names=["x", "y", "z"],
        units="rad_s",
    ))

    # Angular acceleration global
    chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.angular_acceleration_global,
        trajectory_name="angular_acceleration_global",
        component_names=["x", "y", "z"],
        units="rad_s2",
    ))

    # Angular acceleration local
    chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.angular_acceleration_local,
        trajectory_name="angular_acceleration_local",
        component_names=["x", "y", "z"],
        units="rad_s2",
    ))

    # Socket landmarks
    chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.socket_landmarks.tear_duct_mm,
        trajectory_name="socket_landmark__tear_duct",
        component_names=["x", "y", "z"],
        units="mm",
    ))

    chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.socket_landmarks.outer_eye_mm,
        trajectory_name="socket_landmark__outer_eye",
        component_names=["x", "y", "z"],
        units="mm",
    ))

    # Tracked pupil center
    chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.tracked_pupil.pupil_center_mm,
        trajectory_name="tracked_pupil__center",
        component_names=["x", "y", "z"],
        units="mm",
    ))

    # Tracked pupil boundary points p1-p8
    for i in range(NUM_PUPIL_POINTS):
        chunks.append(_build_vector_chunk(
            frame_indices=frame_indices,
            timestamps=timestamps,
            values=kinematics.tracked_pupil.pupil_points_mm[:, i, :],
            trajectory_name=f"tracked_pupil__p{i + 1}",
            component_names=["x", "y", "z"],
            units="mm",
        ))

    df = pl.concat(chunks)
    df = df.sort(by=["frame"])
    return df


def save_ferret_eye_kinematics(
    kinematics: FerretEyeKinematics,
    output_directory: Path,
) -> tuple[Path, Path]:
    """
    Save FerretEyeKinematics to disk.

    Creates two files:
        {name}_reference_geometry.json - Eyeball reference geometry
        {name}_kinematics.csv - Tidy-format kinematic data
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    eye_name = kinematics.name

    if eye_name not in ['left_eye', 'right_eye']:
        raise ValueError(
            f"Unexpected FerretEyeKinematics name '{eye_name}'. "
            f"Expected 'left_eye' or 'right_eye'."
        )

    # Save reference geometry JSON
    reference_geometry_path = output_directory / f"{eye_name}_reference_geometry.json"
    kinematics.eyeball.reference_geometry.to_json_file(path=reference_geometry_path)

    # Save kinematics CSV
    kinematics_csv_path = output_directory / f"{eye_name}_kinematics.csv"
    df = ferret_eye_kinematics_to_tidy_dataframe(kinematics=kinematics)
    df.write_csv(file=kinematics_csv_path)

    logger.info(f"Saved FerretEyeKinematics '{eye_name}' to {output_directory}")
    return reference_geometry_path, kinematics_csv_path


def load_ferret_eye_kinematics(
    reference_geometry_path: Path,
    kinematics_csv_path: Path,
) -> FerretEyeKinematics:
    """Load FerretEyeKinematics from disk."""
    eye_name = "left_eye" if "left_eye" in kinematics_csv_path.name else "right_eye"

    # Load reference geometry
    reference_geometry = ReferenceGeometry.from_json_file(path=reference_geometry_path)

    # Load kinematics CSV
    df = pl.read_csv(kinematics_csv_path)

    # Extract timestamps
    timestamps = _extract_timestamps(df)
    n_frames = len(timestamps)

    # Extract orientation quaternions
    quaternions_wxyz = _extract_quaternions(df, n_frames)

    # Reconstruct eyeball RigidBodyKinematics
    position_xyz = np.zeros((n_frames, 3), dtype=np.float64)
    eyeball = RigidBodyKinematics.from_pose_arrays(
        name=eye_name,
        reference_geometry=reference_geometry,
        timestamps=timestamps,
        position_xyz=position_xyz,
        quaternions_wxyz=quaternions_wxyz,
    )

    # Extract socket landmarks
    tear_duct_mm = _extract_vector_trajectory(df, "socket_landmark__tear_duct", n_frames)
    outer_eye_mm = _extract_vector_trajectory(df, "socket_landmark__outer_eye", n_frames)

    socket_landmarks = SocketLandmarks(
        timestamps=timestamps,
        tear_duct_mm=tear_duct_mm,
        outer_eye_mm=outer_eye_mm,
    )

    # Extract tracked pupil data
    pupil_center_mm = _extract_vector_trajectory(df, "tracked_pupil__center", n_frames)

    pupil_points_mm = np.zeros((n_frames, NUM_PUPIL_POINTS, 3), dtype=np.float64)
    for i in range(NUM_PUPIL_POINTS):
        pupil_points_mm[:, i, :] = _extract_vector_trajectory(
            df, f"tracked_pupil__p{i + 1}", n_frames
        )

    tracked_pupil = TrackedPupil(
        timestamps=timestamps,
        pupil_center_mm=pupil_center_mm,
        pupil_points_mm=pupil_points_mm,
    )

    return FerretEyeKinematics(
        name=eye_name,
        eyeball=eyeball,
        socket_landmarks=socket_landmarks,
        tracked_pupil=tracked_pupil,
    )


def load_ferret_eye_kinematics_from_directory(
    input_directory: Path,
    eye_name: str,
) -> FerretEyeKinematics:
    """Load FerretEyeKinematics from a directory using standard naming convention."""
    if eye_name not in ['left_eye', 'right_eye']:
        raise ValueError(
            f"Unexpected eye_name '{eye_name}'. Expected 'left_eye' or 'right_eye'."
        )
    reference_geometry_path = input_directory / f"{eye_name}_reference_geometry.json"
    kinematics_csv_path = input_directory / f"{eye_name}_kinematics.csv"

    return load_ferret_eye_kinematics(
        reference_geometry_path=reference_geometry_path,
        kinematics_csv_path=kinematics_csv_path,
    )


def _extract_timestamps(df: pl.DataFrame) -> NDArray[np.float64]:
    """Extract unique timestamps from tidy dataframe."""
    frame_timestamps = (
        df.select(["frame", "timestamp_s"])
        .unique()
        .sort("frame")
    )
    return frame_timestamps["timestamp_s"].to_numpy().astype(np.float64)


def _extract_quaternions(df: pl.DataFrame, n_frames: int) -> NDArray[np.float64]:
    """Extract orientation quaternions from tidy dataframe."""
    orientation_df = df.filter(pl.col("trajectory") == "orientation")
    quaternions = np.zeros((n_frames, 4), dtype=np.float64)

    for i, component in enumerate(["w", "x", "y", "z"]):
        component_df = (
            orientation_df
            .filter(pl.col("component") == component)
            .sort("frame")
        )
        quaternions[:, i] = component_df["value"].to_numpy().astype(np.float64)

    return quaternions


def _extract_vector_trajectory(
    df: pl.DataFrame,
    trajectory_name: str,
    n_frames: int,
) -> NDArray[np.float64]:
    """Extract a 3D vector trajectory from tidy dataframe."""
    trajectory_df = df.filter(pl.col("trajectory") == trajectory_name)

    if len(trajectory_df) == 0:
        raise ValueError(f"Trajectory '{trajectory_name}' not found in dataframe")

    values = np.zeros((n_frames, 3), dtype=np.float64)

    for i, component in enumerate(["x", "y", "z"]):
        component_df = (
            trajectory_df
            .filter(pl.col("component") == component)
            .sort("frame")
        )
        values[:, i] = component_df["value"].to_numpy().astype(np.float64)

    return values