"""
Ferret Gaze Kinematics Analysis

Combines skull kinematics and eye kinematics to compute gaze vectors in world frame.

Coordinate conventions:
- Left eye at rest (0, 0) points +Y in skull frame
- Right eye at rest (0, 0) points -Y in skull frame
- Eye angle X: medial(-) / lateral(+) -> maps to skull X axis (toward/away from nose)
- Eye angle Y: superior(+) / inferior(-) -> maps to skull Z axis (up/down)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_code.ferret_gaze.kinematics_calculators.ferret_eye_kinematics import EyeKinematics, load_eye_data
from python_code.ferret_gaze.kinematics_calculators.ferret_skull_kinematics import SkullKinematics, load_skull_pose, \
    compute_skull_kinematics
from python_code.ferret_gaze.quaternion_helper import Quaternion, resample_quaternions

GAZE_VECTOR_LENGTH_MM: float = 100.0  # 10 cm


def compute_median_sample_rate(timestamps: NDArray[np.float64]) -> float:
    """Compute median sample rate from timestamps.

    Args:
        timestamps: Array of timestamps in seconds

    Returns:
        Median sample rate in Hz
    """
    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to compute sample rate")
    frame_durations = np.diff(timestamps)
    median_duration = float(np.median(frame_durations))
    if median_duration <= 0:
        raise ValueError(f"Invalid median frame duration: {median_duration}")
    return 1.0 / median_duration


def compute_mean_eye_framerate(left_eye: EyeKinematics, right_eye: EyeKinematics) -> float:
    left_rate = compute_median_sample_rate(left_eye.timestamps)
    right_rate = compute_median_sample_rate(right_eye.timestamps)
    mean_rate = (left_rate + right_rate) / 2.0
    print(f"  Eye sample rates: left={left_rate:.2f}Hz, right={right_rate:.2f}Hz, mean={mean_rate:.2f}Hz")
    return mean_rate


@dataclass
class GazeKinematics:
    """Gaze kinematics data for both eyes in world frame."""

    timestamps: NDArray[np.float64]  # (N,) seconds
    # Left eye
    left_eyeball_center_xyz_mm: NDArray[np.float64]  # (N, 3) eyeball center in world frame
    left_gaze_azimuth_rad: NDArray[np.float64]  # (N,) azimuth in world frame
    left_gaze_elevation_rad: NDArray[np.float64]  # (N,) elevation in world frame
    left_gaze_endpoint_mm: NDArray[np.float64]  # (N, 3) gaze vector endpoint in world frame
    left_gaze_direction: NDArray[np.float64]  # (N, 3) unit gaze direction in world frame
    # Right eye
    right_eyeball_center_xyz_mm: NDArray[np.float64]  # (N, 3) eyeball center in world frame
    right_gaze_azimuth_rad: NDArray[np.float64]  # (N,) azimuth in world frame
    right_gaze_elevation_rad: NDArray[np.float64]  # (N,) elevation in world frame
    right_gaze_endpoint_mm: NDArray[np.float64]  # (N, 3) gaze vector endpoint in world frame
    right_gaze_direction: NDArray[np.float64]  # (N, 3) unit gaze direction in world frame

    def to_dataframe(self) -> pd.DataFrame:
        """Convert GazeKinematics to a pandas DataFrame."""
        return pd.DataFrame({
            "timestamp": self.timestamps,
            # Left eye
            "left_eye_center_x_mm": self.left_eyeball_center_xyz_mm[:, 0],
            "left_eye_center_y_mm": self.left_eyeball_center_xyz_mm[:, 1],
            "left_eye_center_z_mm": self.left_eyeball_center_xyz_mm[:, 2],
            "left_gaze_azimuth_rad": self.left_gaze_azimuth_rad,
            "left_gaze_elevation_rad": self.left_gaze_elevation_rad,
            "left_gaze_endpoint_x_mm": self.left_gaze_endpoint_mm[:, 0],
            "left_gaze_endpoint_y_mm": self.left_gaze_endpoint_mm[:, 1],
            "left_gaze_endpoint_z_mm": self.left_gaze_endpoint_mm[:, 2],
            "left_gaze_direction_x": self.left_gaze_direction[:, 0],
            "left_gaze_direction_y": self.left_gaze_direction[:, 1],
            "left_gaze_direction_z": self.left_gaze_direction[:, 2],
            # Right eye
            "right_eye_center_x_mm": self.right_eyeball_center_xyz_mm[:, 0],
            "right_eye_center_y_mm": self.right_eyeball_center_xyz_mm[:, 1],
            "right_eye_center_z_mm": self.right_eyeball_center_xyz_mm[:, 2],
            "right_gaze_azimuth_rad": self.right_gaze_azimuth_rad,
            "right_gaze_elevation_rad": self.right_gaze_elevation_rad,
            "right_gaze_endpoint_x_mm": self.right_gaze_endpoint_mm[:, 0],
            "right_gaze_endpoint_y_mm": self.right_gaze_endpoint_mm[:, 1],
            "right_gaze_endpoint_z_mm": self.right_gaze_endpoint_mm[:, 2],
            "right_gaze_direction_x": self.right_gaze_direction[:, 0],
            "right_gaze_direction_y": self.right_gaze_direction[:, 1],
            "right_gaze_direction_z": self.right_gaze_direction[:, 2],
        })


def compute_gaze_direction_skull_frame(
    eyeball_angle_azimuth_rad: float,
    eyeball_angle_elevation_rad: float,
    is_left_eye: bool,
) -> NDArray[np.float64]:
    """Compute gaze direction in skull frame from eye angles.

    Args:
        eyeball_angle_azimuth_rad: Medial(-)/lateral(+) angle in radians
        eyeball_angle_elevation_rad: Superior(+)/inferior(-) angle in radians
        is_left_eye: True for left eye (+Y rest), False for right eye (-Y rest)

    Returns:
        Unit gaze direction vector in skull frame [x, y, z]
    """
    # Eye angle X: medial(-) = toward nose (+X_skull), lateral(+) = away from nose (-X_skull)
    # So skull_x_component = -eye_angle_x
    # Eye angle Y: superior(+) = up (+Z_skull), inferior(-) = down (-Z_skull)
    # So skull_z_component = eye_angle_y

    # NOTE - this calculation wrong, i think

    skull_x = -eyeball_angle_azimuth_rad
    skull_z = eyeball_angle_elevation_rad

    if is_left_eye:
        # Left eye at rest points +Y
        gaze_skull = np.array([skull_x, 1.0, skull_z], dtype=np.float64)
    else:
        # Right eye at rest points -Y
        gaze_skull = np.array([skull_x, -1.0, skull_z], dtype=np.float64)

    # Normalize to unit vector
    norm = np.linalg.norm(gaze_skull)
    if norm < 1e-10:
        raise ValueError("Gaze direction is zero")
    return gaze_skull / norm


def transform_to_world_frame(
    vector_skull: NDArray[np.float64],
    basis_x: NDArray[np.float64],
    basis_y: NDArray[np.float64],
    basis_z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform a vector from skull frame to world frame.

    Args:
        vector_skull: Vector in skull frame [x, y, z]
        basis_x: Skull X axis in world frame (unit vector)
        basis_y: Skull Y axis in world frame (unit vector)
        basis_z: Skull Z axis in world frame (unit vector)

    Returns:
        Vector in world frame [x, y, z]
    """
    return (
        vector_skull[0] * basis_x +
        vector_skull[1] * basis_y +
        vector_skull[2] * basis_z
    )


def compute_azimuth_elevation(direction: NDArray[np.float64]) -> tuple[float, float]:
    """Compute azimuth and elevation from a direction vector.

    Args:
        direction: Unit direction vector in world frame [x, y, z]

    Returns:
        (azimuth_rad, elevation_rad) where:
        - azimuth: angle in XY plane from +X axis (positive toward +Y)
        - elevation: angle above XY plane (positive toward +Z)
    """
    azimuth = np.arctan2(direction[1], direction[0])
    elevation = np.arcsin(np.clip(direction[2], -1.0, 1.0))
    return float(azimuth), float(elevation)




def load_body_trajectory_data(trajectory_csv_path: Path) -> dict[str, NDArray[np.float64]]:
    """Load trajectory data from CSV into arrays of shape (n_frames, 3).

    Args:
        trajectory_csv_path: Path to tidy_trajectory_data.csv

    Returns:
        Dictionary mapping marker name to position array of shape (n_frames, 3),
        plus a "timestamps" key with shape (n_frames,)
    """
    df = pd.read_csv(trajectory_csv_path)

    frame_indices = df["frame"].unique()
    n_frames = len(frame_indices)
    if n_frames == 0:
        raise ValueError(f"No frames found in {trajectory_csv_path}")

    # Verify frames are contiguous starting from 0
    expected_frames = np.arange(n_frames)
    if not np.array_equal(np.sort(frame_indices), expected_frames):
        raise ValueError(f"Frame indices must be contiguous from 0 to {n_frames - 1}, got: {sorted(frame_indices)}")

    marker_names = df["marker"].unique()
    if len(marker_names) == 0:
        raise ValueError(f"No markers found in {trajectory_csv_path}")

    # Prefer "optimized" data, fall back to "original"
    optimized_df = df[df["data_type"] == "optimized"]
    original_df = df[df["data_type"] == "original"]

    body_trajectories: dict[str, NDArray[np.float64]] = {}

    # Extract timestamps per frame
    timestamps = np.zeros(n_frames, dtype=np.float64)
    for frame_idx in range(n_frames):
        frame_rows = df[df["frame"] == frame_idx]
        if frame_rows.empty:
            raise ValueError(f"No data found for frame {frame_idx}")
        unique_timestamps = frame_rows["timestamp"].unique()
        if len(unique_timestamps) != 1:
            raise ValueError(f"Frame {frame_idx} has multiple timestamps: {unique_timestamps}")
        timestamps[frame_idx] = float(unique_timestamps[0])
    body_trajectories["timestamps"] = timestamps

    for marker in marker_names:
        positions = np.zeros((n_frames, 3), dtype=np.float64)

        for frame_idx in range(n_frames):
            # Try optimized first
            marker_frame_data = optimized_df[
                (optimized_df["marker"] == marker) & (optimized_df["frame"] == frame_idx)
            ]
            if marker_frame_data.empty:
                # Fall back to original
                marker_frame_data = original_df[
                    (original_df["marker"] == marker) & (original_df["frame"] == frame_idx)
                ]
            if marker_frame_data.empty:
                raise ValueError(f"No data found for marker '{marker}' at frame {frame_idx}")
            if len(marker_frame_data) > 1:
                raise ValueError(f"Multiple entries for marker '{marker}' at frame {frame_idx}")

            row = marker_frame_data.iloc[0]
            positions[frame_idx, 0] = float(row["x"])
            positions[frame_idx, 1] = float(row["y"])
            positions[frame_idx, 2] = float(row["z"])

        body_trajectories[str(marker)] = positions

    return body_trajectories


def resample_trajectory_data(
    trajectory_data: dict[str, NDArray[np.float64]],
    original_timestamps: NDArray[np.float64],
    target_timestamps: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:

    # implement linear interpolation for each marker across frames

    resampled_data = {}

    for marker_name, marker_og_xyz  in trajectory_data.items():
        if marker_og_xyz.shape != (len(original_timestamps), 3):
            raise ValueError(f"Marker {marker_name} has shape {marker_og_xyz.shape}, expected ({len(original_timestamps)}, 3)")
        resampled_data[marker_name] = np.interp(target_timestamps, original_timestamps, marker_og_xyz, axis=0)

    return resampled_data






def resample_eye_kinematics(
    eye: EyeKinematics,
    target_timestamps: NDArray[np.float64],
) -> EyeKinematics:

    # Interpolate left eye
    eye_x = np.interp(target_timestamps, eye.timestamps, eye.eye_angle_x_rad)
    eye_y = np.interp(target_timestamps, eye.timestamps, eye.eye_angle_y_rad)


    return EyeKinematics(
        timestamps=target_timestamps,
        eye_angle_x_rad=eye_x,
        eye_angle_y_rad=eye_y,
    )


def validate_timestamp_alignment(
    skull_timestamps: NDArray[np.float64],
    left_eye_timestamps: NDArray[np.float64],
    right_eye_timestamps: NDArray[np.float64],
    tolerance_s: float,
) -> None:
    """Validate that skull and eye data cover the same time period.

    Args:
        skull_timestamps: Skull timestamps
        left_eye_timestamps: Left eye timestamps
        right_eye_timestamps: Right eye timestamps
        tolerance_s: Maximum allowed difference in seconds (typically 1 frame duration)

    Raises:
        ValueError: If start or end times differ by more than tolerance
    """
    skull_start = skull_timestamps[0]
    skull_end = skull_timestamps[-1]

    # Check both eyes against skull timestamps
    for eye_name, eye_timestamps in [
        ("left", left_eye_timestamps),
        ("right", right_eye_timestamps),
    ]:
        eye_start = eye_timestamps[0]
        eye_end = eye_timestamps[-1]

        start_diff = abs(skull_start - eye_start)
        end_diff = abs(skull_end - eye_end)

        if start_diff > tolerance_s:
            raise ValueError(
                f"Skull and {eye_name} eye data start times differ by {start_diff * 1000:.2f} ms "
                f"(tolerance: {tolerance_s * 1000:.2f} ms). "
                f"Skull starts at {skull_start:.4f}s, {eye_name} eye starts at {eye_start:.4f}s"
            )

        if end_diff > tolerance_s:
            raise ValueError(
                f"Skull and {eye_name} eye data end times differ by {end_diff * 1000:.2f} ms "
                f"(tolerance: {tolerance_s * 1000:.2f} ms). "
                f"Skull ends at {skull_end:.4f}s, {eye_name} eye ends at {eye_end:.4f}s"
            )


def create_uniform_timestamps(
    start_time: float,
    end_time: float,
    sample_rate_hz: float,
) -> NDArray[np.float64]:
    """Create uniformly spaced timestamps at the target sample rate.

    Args:
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        sample_rate_hz: Target sample rate in Hz

    Returns:
        Array of uniformly spaced timestamps
    """
    frame_duration = 1.0 / sample_rate_hz
    n_frames = int(np.floor((end_time - start_time) / frame_duration)) + 1
    end_time = start_time + (n_frames - 1) * frame_duration
    return np.linspace(start_time, end_time, n_frames)


def resample_skull_kinematics(
    skull: SkullKinematics,
    target_timestamps: NDArray[np.float64],
) -> SkullKinematics:
    """Resample skull kinematics to target timestamps.

    Uses SLERP for quaternion interpolation to properly interpolate rotations
    along the geodesic on the quaternion hypersphere. Basis vectors and euler
    angles are derived from the interpolated quaternions.

    Args:
        skull: Original skull kinematics
        target_timestamps: Target timestamps to resample to

    Returns:
        Resampled SkullKinematics
    """
    n_frames = len(target_timestamps)

    # Interpolate position (linear is fine for positions)
    position = np.zeros((n_frames, 3), dtype=np.float64)
    for axis in range(3):
        position[:, axis] = np.interp(target_timestamps, skull.timestamps, skull.position[:, axis])

    # SLERP interpolate quaternions - this is the correct way to interpolate rotations
    orientation_quaternions = resample_quaternions(
        quaternions=skull.orientation_quaternions,
        original_timestamps=skull.timestamps,
        target_timestamps=target_timestamps,
    )

    # Derive euler angles and basis vectors from interpolated quaternions
    euler_angles_deg = np.zeros((n_frames, 3), dtype=np.float64)
    basis_x = np.zeros((n_frames, 3), dtype=np.float64)
    basis_y = np.zeros((n_frames, 3), dtype=np.float64)
    basis_z = np.zeros((n_frames, 3), dtype=np.float64)

    canonical_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    canonical_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    canonical_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for i, q in enumerate(orientation_quaternions):
        # Compute euler angles from quaternion
        roll, pitch, yaw = q.to_euler_xyz()
        euler_angles_deg[i] = np.rad2deg([roll, pitch, yaw])

        # Compute basis vectors by rotating canonical basis
        basis_x[i] = q.rotate_vector(canonical_x)
        basis_y[i] = q.rotate_vector(canonical_y)
        basis_z[i] = q.rotate_vector(canonical_z)

    # Interpolate angular velocities (linear is acceptable for velocities)
    angular_velocity_world_deg_s = np.zeros((n_frames, 3), dtype=np.float64)
    angular_velocity_local_deg_s = np.zeros((n_frames, 3), dtype=np.float64)
    for axis in range(3):
        angular_velocity_world_deg_s[:, axis] = np.interp(
            target_timestamps, skull.timestamps, skull.angular_velocity_world_deg_s[:, axis]
        )
        angular_velocity_local_deg_s[:, axis] = np.interp(
            target_timestamps, skull.timestamps, skull.angular_velocity_local_deg_s[:, axis]
        )

    return SkullKinematics(
        timestamps=target_timestamps,
        position=position,
        orientation_quaternions=orientation_quaternions,
        euler_angles_deg=euler_angles_deg,
        angular_velocity_world_deg_s=angular_velocity_world_deg_s,
        angular_velocity_local_deg_s=angular_velocity_local_deg_s,
        basis_x=basis_x,
        basis_y=basis_y,
        basis_z=basis_z,
    )


def resample_data(
    skull: SkullKinematics,
    left_eye: EyeKinematics,
    right_eye: EyeKinematics,
    trajectory_data:dict[str, NDArray[np.float64]],
    target_framerate_strategy: str = "mean_eye_framerate",
) -> tuple[SkullKinematics, EyeKinematics,EyeKinematics, dict[str, NDArray[np.float64]]]:

    if not target_framerate_strategy == "mean_eye_framerate":
        raise NotImplementedError(f"Unsupported target framerate strategy: {target_framerate_strategy} - use mean_eye_framerate for now")

    # Compute target sample rate from mean of left and right eye median sample rates
    target_sample_rate_hz = compute_mean_eye_framerate(left_eye=left_eye, right_eye=right_eye)
    frame_duration_s = 1.0 / target_sample_rate_hz

    # Validate timestamp alignment (tolerance = 1 frame duration)
    validate_timestamp_alignment(
        skull_timestamps=skull.timestamps,
        left_eye_timestamps=left_eye.timestamps,
        right_eye_timestamps=right_eye.timestamps,
        tolerance_s=frame_duration_s*2,
    )

    # Create uniform timestamps at target sample rate
    # Use the later start time and earlier end time to ensure all streams have data
    start_time = max(
        skull.timestamps[0],
        left_eye.timestamps[0],
        left_eye.timestamps[0],
    )
    end_time = min(
        skull.timestamps[-1],
        left_eye.timestamps[-1],
        left_eye.timestamps[-1],
    )
    uniform_timestamps = create_uniform_timestamps(
        start_time=start_time,
        end_time=end_time,
        sample_rate_hz=target_sample_rate_hz,
    )

    print(f"  Resampling to {target_sample_rate_hz:.2f}Hz: {len(uniform_timestamps)} frames")
    print(f"    Time range: {start_time:.4f}s to {end_time:.4f}s ({end_time - start_time:.3f}s duration)")

    # Resample skull and eye kinematics to uniform timestamps
    skull_resampled = resample_skull_kinematics(skull, uniform_timestamps)
    left_eye_resampled = resample_eye_kinematics(left_eye, uniform_timestamps)
    right_eye_resampled = resample_eye_kinematics(right_eye, uniform_timestamps)


    trajectories_resampled = resample_trajectory_data(
        trajectory_data=trajectory_data,
        original_timestamps=skull.timestamps,
        target_timestamps=uniform_timestamps,
    )
    return skull_resampled, left_eye_resampled, right_eye_resampled, trajectories_resampled




def compute_gaze_kinematics(
    skull: SkullKinematics,
    left_eye: EyeKinematics,
    right_eye: EyeKinematics,
    trajectory_data:dict[str, NDArray[np.float64]],) -> GazeKinematics:
    """Compute gaze kinematics from skull and eye data.

    Validates that skull and eye data cover the same time period, then resamples
    both to a uniform sample rate (mean of left/right eye median sample rates)
    before computing gaze vectors.

    Args:
        skull: Skull kinematics (orientation and basis vectors)
        left_eye: Eye kinematics (eye-in-skull angles) for left eye
        right_eye: Eye kinematics (eye-in-skull angles) for right eye
        trajectory_data: Trajectory data head and body markers

    Returns:
        GazeKinematics

    Raises:
        ValueError: If skull and eye data start/end times differ by more than one frame
    """


    n_frames = len(skull.timestamps)

    # Extract eye centers from trajectory
    left_eye_center = trajectory_data.get("left_eye_center")
    right_eye_center = trajectory_data.get("right_eye_center")

    # Allocate output arrays
    left_gaze_azimuth = np.zeros(n_frames, dtype=np.float64)
    left_gaze_elevation = np.zeros(n_frames, dtype=np.float64)
    left_gaze_endpoint = np.zeros((n_frames, 3), dtype=np.float64)
    left_gaze_direction = np.zeros((n_frames, 3), dtype=np.float64)

    right_gaze_azimuth = np.zeros(n_frames, dtype=np.float64)
    right_gaze_elevation = np.zeros(n_frames, dtype=np.float64)
    right_gaze_endpoint = np.zeros((n_frames, 3), dtype=np.float64)
    right_gaze_direction = np.zeros((n_frames, 3), dtype=np.float64)

    for i in range(n_frames):
        # Get skull basis vectors for this frame
        basis_x = skull.basis_x[i]
        basis_y = skull.basis_y[i]
        basis_z = skull.basis_z[i]

        # Left eye
        gaze_skull_left = compute_gaze_direction_skull_frame(
            eyeball_angle_azimuth_rad=left_eye.eye_angle_x_rad[i],
            eyeball_angle_elevation_rad=left_eye.eye_angle_y_rad[i],
            is_left_eye=True,
        )
        gaze_world_left = transform_to_world_frame(gaze_skull_left, basis_x, basis_y, basis_z)
        left_gaze_direction[i] = gaze_world_left
        left_gaze_azimuth[i], left_gaze_elevation[i] = compute_azimuth_elevation(gaze_world_left)
        left_gaze_endpoint[i] = left_eye_center[i] + gaze_world_left * GAZE_VECTOR_LENGTH_MM

        # Right eye
        gaze_skull_right = compute_gaze_direction_skull_frame(
            eyeball_angle_azimuth_rad=right_eye.eye_angle_x_rad[i],
            eyeball_angle_elevation_rad=right_eye.eye_angle_y_rad[i],
            is_left_eye=False,
        )
        gaze_world_right = transform_to_world_frame(gaze_skull_right, basis_x, basis_y, basis_z)
        right_gaze_direction[i] = gaze_world_right
        right_gaze_azimuth[i], right_gaze_elevation[i] = compute_azimuth_elevation(gaze_world_right)
        right_gaze_endpoint[i] = right_eye_center[i] + gaze_world_right * GAZE_VECTOR_LENGTH_MM

    return GazeKinematics(
        timestamps=skull.timestamps,
        left_eyeball_center_xyz_mm=left_eye_center,
        left_gaze_azimuth_rad=left_gaze_azimuth,
        left_gaze_elevation_rad=left_gaze_elevation,
        left_gaze_endpoint_mm=left_gaze_endpoint,
        left_gaze_direction=left_gaze_direction,

        right_eyeball_center_xyz_mm=right_eye_center,
        right_gaze_azimuth_rad=right_gaze_azimuth,
        right_gaze_elevation_rad=right_gaze_elevation,
        right_gaze_endpoint_mm=right_gaze_endpoint,
        right_gaze_direction=right_gaze_direction,
    )