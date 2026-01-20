"""
Serialization for FerretEyeKinematics.

Saves and loads FerretEyeKinematics to/from disk using:
- {name}_metadata.json: name, eye_side, paths, calibration matrices
- {name}_reference_geometry.json: eyeball reference geometry (via ReferenceGeometry.to_json_file)
- {name}_kinematics.csv: tidy-format CSV with eyeball kinematics + socket landmarks

The CSV format is tidy (one row per observation) with columns:
    frame, timestamp_s, trajectory, component, value, units
"""

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import (
    FerretEyeKinematics,
    SocketLandmarks,
)
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.kinematics_serialization import (
    kinematics_to_tidy_dataframe,
    _build_vector_chunk,
)

logger = logging.getLogger(__name__)


def ferret_eye_kinematics_to_tidy_dataframe(
    kinematics: FerretEyeKinematics,
) -> pl.DataFrame:
    """
    Convert FerretEyeKinematics to a tidy-format polars DataFrame.

    This extends the base RigidBodyKinematics serialization with socket landmarks.

    Trajectories included:
        - position (eyeball origin, always [0,0,0])
        - orientation (eyeball quaternion wxyz)
        - linear_velocity (eyeball, always [0,0,0])
        - angular_velocity_global (eyeball)
        - angular_velocity_local (eyeball)
        - keypoint__* (eyeball keypoints: eyeball_center, pupil_center, p1-p8)
        - socket_landmark__tear_duct (socket, does NOT rotate with eyeball)
        - socket_landmark__outer_eye (socket, does NOT rotate with eyeball)

    Args:
        kinematics: The FerretEyeKinematics to convert

    Returns:
        Tidy-format polars DataFrame
    """
    # Get base eyeball kinematics dataframe
    eyeball_df = kinematics_to_tidy_dataframe(kinematics=kinematics.eyeball)

    # Build socket landmark chunks
    n_frames = kinematics.n_frames
    frame_indices = np.arange(n_frames, dtype=np.int64)
    timestamps = kinematics.timestamps

    socket_chunks: list[pl.DataFrame] = []

    # Tear duct trajectory
    socket_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.socket_landmarks.tear_duct_mm,
        trajectory_name="socket_landmark__tear_duct",
        component_names=["x", "y", "z"],
        units="mm",
    ))

    # Outer eye trajectory
    socket_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.socket_landmarks.outer_eye_mm,
        trajectory_name="socket_landmark__outer_eye",
        component_names=["x", "y", "z"],
        units="mm",
    ))



    # Concatenate all
    df = pl.concat([eyeball_df] + socket_chunks)
    df = df.sort(by=["frame"])
    return df


def save_ferret_eye_kinematics(
    kinematics: FerretEyeKinematics,
    output_directory: Path,
) -> tuple[ Path, Path]:
    """
    Save FerretEyeKinematics to disk.

    Creates three files:
        {name}_reference_geometry.json - Eyeball reference geometry
        {name}_kinematics.csv - Tidy-format kinematic data

    Args:
        kinematics: The FerretEyeKinematics to save
        output_directory: Directory to save files (created if needed)

    Returns:
        Tuple of ( reference_geometry_path, kinematics_csv_path)
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
    return  reference_geometry_path, kinematics_csv_path


def load_ferret_eye_kinematics(
    reference_geometry_path: Path,
    kinematics_csv_path: Path,
) -> FerretEyeKinematics:
    """
    Load FerretEyeKinematics from disk.

    Args:
        reference_geometry_path: Path to {name}_reference_geometry.json
        kinematics_csv_path: Path to {name}_kinematics.csv

    Returns:
        Reconstructed FerretEyeKinematics
    """

    eye_name ="left_eye" if "left_eye" in kinematics_csv_path.name else "right_eye"

    # Load reference geometry
    reference_geometry = ReferenceGeometry.from_json_file(path=reference_geometry_path)

    # Load kinematics CSV
    df = pl.read_csv(kinematics_csv_path)

    # Extract timestamps (from any trajectory, they're all the same)
    timestamps = _extract_timestamps(df=df)
    n_frames = len(timestamps)

    # Extract orientation quaternions
    quaternions_wxyz = _extract_quaternions(df=df, n_frames=n_frames)

    # Extract position (should be all zeros, but extract anyway for completeness)
    position_xyz = _extract_vector_trajectory(
        df=df,
        trajectory_name="position",
        n_frames=n_frames,
    )
    # Reconstruct eyeball RigidBodyKinematics
    eyeball = RigidBodyKinematics.from_pose_arrays(
        name=eye_name,
        reference_geometry=reference_geometry,
        timestamps=timestamps,
        position_xyz=position_xyz,
        quaternions_wxyz=quaternions_wxyz,
    )

    # Extract socket landmarks
    tear_duct_mm = _extract_vector_trajectory(
        df=df,
        trajectory_name="socket_landmark__tear_duct",
        n_frames=n_frames,
    )
    outer_eye_mm = _extract_vector_trajectory(
        df=df,
        trajectory_name="socket_landmark__outer_eye",
        n_frames=n_frames,
    )

    socket_landmarks = SocketLandmarks(
        timestamps=timestamps,
        tear_duct_mm=tear_duct_mm,
        outer_eye_mm=outer_eye_mm,
    )

    return FerretEyeKinematics(
        name=eye_name,
        eyeball=eyeball,
        socket_landmarks=socket_landmarks,
    )

def load_ferret_eye_kinematics_from_directory(
    input_directory: Path,
    eye_name: str,
) -> FerretEyeKinematics:
    """
    Load FerretEyeKinematics from a directory using the standard naming convention.

    Args:
        directory: Directory containing the serialized files
        eye_name: Base name used when saving (e.g., "ferret_right_eye")

    Returns:
        Reconstructed FerretEyeKinematics
    """
    if eye_name not in ['left_eye', 'right_eye']:
        raise ValueError(
            f"Unexpected FerretEyeKinematics name '{eye_name}'. "
            f"Expected 'left_eye' or 'right_eye'."
        )
    reference_geometry_path = input_directory / f"{eye_name}_reference_geometry.json"
    kinematics_csv_path = input_directory / f"{eye_name}_kinematics.csv"

    return load_ferret_eye_kinematics(
        reference_geometry_path=reference_geometry_path,
        kinematics_csv_path=kinematics_csv_path,
    )


def _extract_timestamps(df: pl.DataFrame) -> NDArray[np.float64]:
    """Extract unique timestamps from tidy dataframe."""
    # Get unique frame/timestamp pairs, sorted by frame
    frame_timestamps = (
        df.select(["frame", "timestamp_s"])
        .unique()
        .sort("frame")
    )
    return frame_timestamps["timestamp_s"].to_numpy().astype(np.float64)


def _extract_quaternions(
    df: pl.DataFrame,
    n_frames: int,
) -> NDArray[np.float64]:
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
