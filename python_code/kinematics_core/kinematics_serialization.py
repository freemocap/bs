"""
Serialization methods for RigidBodyKinematics.

Outputs:
- Reference geometry: JSON (via ReferenceGeometry.to_json_file)
- Kinematics: Tidy-format CSV
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numpy.typing import NDArray

from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

if TYPE_CHECKING:
    from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics

logger = logging.getLogger(__name__)


def kinematics_to_tidy_dataframe(
    kinematics: "RigidBodyKinematics",
) -> pl.DataFrame:
    """
    Convert RigidBodyKinematics to a tidy-format polars DataFrame.

    Tidy format: one row per (frame, trajectory, component) observation.

    Columns:
        - frame: int — frame index
        - timestamp_s: float — timestamp in seconds
        - trajectory: str — what is being measured (position, orientation, etc.)
        - component: str — vector/quaternion component (x, y, z, w, roll, pitch, yaw)
        - value: float — the measurement value
        - unit: str — unit of measurement (mm, mm_s, mm_s2, rad_s, rad_s2, quaternion)

    Args:
        kinematics: The kinematics data to convert

    Returns:
        Tidy-format polars DataFrame
    """
    number_of_frames = kinematics.n_frames
    frame_indices = np.arange(number_of_frames, dtype=np.int64)
    timestamps = kinematics.timestamps

    # Build list of DataFrames to concatenate at the end
    dataframe_chunks: list[pl.DataFrame] = []

    # Position (world frame) - 3 components
    dataframe_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.position_xyz,
        trajectory_name="position",
        component_names=["x", "y", "z"],
        units="mm",
    ))

    # Orientation quaternion (world frame) - 4 components
    logger.info(f" Quaternion values shape: {kinematics.orientations.quaternions_wxyz.shape}")
    dataframe_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.orientations.quaternions_wxyz,
        trajectory_name="orientation",
        component_names=["w", "x", "y", "z"],
        units="quaternion",
    ))

    # Linear velocity (world frame) - 3 components
    logger.info(f" Linear velocity shape: {kinematics.velocity_xyz.shape}")
    dataframe_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.velocity_xyz,
        trajectory_name="linear_velocity",
        component_names=["x", "y", "z"],
        units="mm_s",
    ))

    # Linear acceleration (world frame) - 3 components
    logger.info(f" Linear acceleration shape: {kinematics.acceleration_xyz.shape}")
    dataframe_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.acceleration_xyz,
        trajectory_name="linear_acceleration",
        component_names=["x", "y", "z"],
        units="mm_s2",
    ))

    # Angular velocity global (world frame) - 3 components
    logger.info(f" Angular velocity global shape: {kinematics.angular_velocity_global.shape}")
    dataframe_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.angular_velocity_global,
        trajectory_name="angular_velocity_global",
        component_names=["roll", "pitch", "yaw"],
        units="rad_s",
    ))

    # Angular velocity local (body frame) - 3 components
    logger.info(f" Angular velocity local shape: {kinematics.angular_velocity_local.shape}")
    dataframe_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.angular_velocity_local,
        trajectory_name="angular_velocity_local",
        component_names=["roll", "pitch", "yaw"],
        units="rad_s",
    ))

    # Angular acceleration global (world frame) - 3 components
    logger.info(f" Angular acceleration global shape: {kinematics.angular_acceleration_global.shape}")
    dataframe_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.angular_acceleration_global,
        trajectory_name="angular_acceleration_global",
        component_names=["roll", "pitch", "yaw"],
        units="rad_s2",
    ))

    # Angular acceleration local (body frame) - 3 components
    logger.info(f" Angular acceleration local shape: {kinematics.angular_acceleration_local.shape}")
    dataframe_chunks.append(_build_vector_chunk(
        frame_indices=frame_indices,
        timestamps=timestamps,
        values=kinematics.angular_acceleration_local,
        trajectory_name="angular_acceleration_local",
        component_names=["roll", "pitch", "yaw"],
        units="rad_s2",
    ))

    # Keypoint trajectories (world frame)
    logger.info(f" Keypoint names: {kinematics.keypoint_names}, and shape: {kinematics.keypoint_trajectories.trajectories_fr_id_xyz.shape}")

    for keypoint_name in kinematics.keypoint_names:
        trajectory = kinematics.keypoint_trajectories[keypoint_name]  # (n_frames, 3)
        logger.info(f" Keypoint '{keypoint_name}' trajectory shape: {trajectory.shape}")

        dataframe_chunks.append(_build_vector_chunk(
            frame_indices=frame_indices,
            timestamps=timestamps,
            values=trajectory,
            trajectory_name=f"keypoint__{keypoint_name}",
            component_names=["x", "y", "z"],
            units="mm",
        ))

    df = pl.concat(dataframe_chunks)

    # sort by frame
    df = df.sort(by=["frame"])
    return df


def _build_vector_chunk(
    frame_indices: NDArray[np.int64],
    timestamps: NDArray[np.float64],
    values: NDArray[np.float64],
    trajectory_name: str,
    component_names: list[str],
    units: str,
) -> pl.DataFrame:
    """
    Build a tidy DataFrame chunk for a vector quantity using vectorized operations.

    Args:
        frame_indices: (N,) array of frame indices
        timestamps: (N,) array of timestamps
        values: (N, C) array where C is number of components
        trajectory_name: Name for the trajectory column
        component_names: List of component names (length C)
        units: Unit string

    Returns:
        Tidy polars DataFrame with N * C rows
    """
    number_of_frames = len(frame_indices)
    number_of_components = len(component_names)

    # Repeat frame indices and timestamps for each component
    # [0,0,0, 1,1,1, 2,2,2, ...] for each component
    repeated_frame_indices = np.repeat(frame_indices, number_of_components)
    repeated_timestamps = np.repeat(timestamps, number_of_components)

    # Tile component names for each frame
    # ["x","y","z", "x","y","z", "x","y","z", ...]
    tiled_component_names = np.tile(component_names, number_of_frames)

    # Flatten values in row-major order (all components for frame 0, then frame 1, etc.)
    flattened_values = values.ravel()

    return pl.DataFrame({
        "frame": repeated_frame_indices,
        "timestamp_s": repeated_timestamps,
        "component": tiled_component_names,
        "value": flattened_values,
    }).with_columns(
        pl.lit(trajectory_name).alias("trajectory").cast(pl.Categorical),
        pl.col("component").cast(pl.Categorical),
        pl.lit(units).alias("units").cast(pl.Categorical),
    ).select(["frame", "timestamp_s", "trajectory", "component", "value", "units"])


def save_kinematics(
    kinematics: "RigidBodyKinematics",
    output_directory: Path,
) -> tuple[Path, Path]:
    """
    Save RigidBodyKinematics to disk.

    Creates two files:
        {name}_reference_geometry.json - Static geometry
        {name}_kinematics.csv - Tidy-format kinematic data

    Args:
        kinematics: The kinematics data to save
        output_directory: Directory to save files (created if needed)

    Returns:
        Tuple of (reference_geometry_path, kinematics_csv_path)
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    name = kinematics.name

    # Save reference geometry as JSON using its built-in Pydantic method
    reference_geometry_path = output_directory / f"{name}_reference_geometry.json"
    kinematics.reference_geometry.to_json_file(path=reference_geometry_path)

    # Save kinematics as tidy CSV
    kinematics_csv_path = output_directory / f"{name}_kinematics.csv"
    dataframe = kinematics_to_tidy_dataframe(kinematics=kinematics)
    dataframe.write_csv(file=kinematics_csv_path)

    return reference_geometry_path, kinematics_csv_path


def load_kinematics(
    reference_geometry_path: Path,
    kinematics_csv_path: Path,
) -> "RigidBodyKinematics":
    """
    Load RigidBodyKinematics from disk.

    Reads the files created by save_kinematics and reconstructs the
    RigidBodyKinematics object. Derived quantities (velocity, acceleration, etc.)
    will be recomputed lazily from the position and orientation data.

    Args:
        reference_geometry_path: Path to the reference geometry JSON file
        kinematics_csv_path: Path to the tidy-format kinematics CSV file

    Returns:
        Reconstructed RigidBodyKinematics instance

    Raises:
        FileNotFoundError: If either file does not exist
        ValueError: If the CSV is missing required trajectories or has invalid data
    """
    # Import here to avoid circular dependency at module level
    from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics

    # Load reference geometry
    reference_geometry = ReferenceGeometry.from_json_file(path=reference_geometry_path)

    # Extract name from filename (remove _reference_geometry.json suffix)
    name = reference_geometry_path.stem.removesuffix("_reference_geometry")

    # Load kinematics CSV
    df = pl.read_csv(kinematics_csv_path)

    # Extract timestamps and position data
    timestamps = _extract_timestamps(df=df)
    position_xyz = _extract_trajectory_array(
        df=df,
        trajectory_name="position",
        component_names=["x", "y", "z"],
    )
    quaternions_wxyz = _extract_trajectory_array(
        df=df,
        trajectory_name="orientation",
        component_names=["w", "x", "y", "z"],
    )

    return RigidBodyKinematics.from_pose_arrays(
        name=name,
        reference_geometry=reference_geometry,
        timestamps=timestamps,
        position_xyz=position_xyz,
        quaternions_wxyz=quaternions_wxyz,
    )


def _extract_timestamps(df: pl.DataFrame) -> NDArray[np.float64]:
    """
    Extract unique timestamps from the tidy dataframe, sorted by frame.

    Args:
        df: Tidy-format kinematics DataFrame

    Returns:
        (N,) array of timestamps in seconds
    """
    timestamps_df = (
        df
        .select(["frame", "timestamp_s"])
        .unique()
        .sort(by="frame")
    )
    return timestamps_df["timestamp_s"].to_numpy().astype(np.float64)


def _extract_trajectory_array(
    df: pl.DataFrame,
    trajectory_name: str,
    component_names: list[str],
) -> NDArray[np.float64]:
    """
    Extract a trajectory from the tidy dataframe and reshape to (N, C) array.

    Args:
        df: Tidy-format kinematics DataFrame
        trajectory_name: Name of the trajectory to extract (e.g., "position", "orientation")
        component_names: Ordered list of component names (e.g., ["x", "y", "z"])

    Returns:
        (N, C) array where N is number of frames and C is number of components

    Raises:
        ValueError: If the trajectory or any component is missing from the dataframe
    """
    # Filter to the trajectory we want
    trajectory_df = df.filter(pl.col("trajectory") == trajectory_name)

    if trajectory_df.is_empty():
        raise ValueError(f"Trajectory '{trajectory_name}' not found in dataframe")

    # Pivot to wide format: rows are frames, columns are components
    pivoted = (
        trajectory_df
        .select(["frame", "component", "value"])
        .pivot(
            on="component",
            index="frame",
            values="value",
        )
        .sort(by="frame")
    )

    # Verify all components are present
    missing_components = set(component_names) - set(pivoted.columns)
    if missing_components:
        raise ValueError(
            f"Missing components {missing_components} for trajectory '{trajectory_name}'"
        )

    # Extract columns in the correct order
    values = pivoted.select(component_names).to_numpy().astype(np.float64)

    return values