"""
Ferret Gaze Kinematics Analysis

Combines head kinematics and eye kinematics to compute gaze vectors in world frame.

Coordinate conventions:
- Left eye at rest (0, 0) points +Y in head frame
- Right eye at rest (0, 0) points -Y in head frame
- Eye angle X: medial(-) / lateral(+) -> maps to head X axis (toward/away from nose)
- Eye angle Y: superior(+) / inferior(-) -> maps to head Z axis (up/down)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ferret_eye_kinematics import EyeKinematics, load_eye_data
from ferret_head_kinematics import HeadKinematics, load_skull_pose, compute_head_kinematics


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


def compute_target_sample_rate(ek: EyeKinematics) -> float:
    """Compute target sample rate as mean of left and right eye median sample rates.

    Args:
        ek: EyeKinematics with separate timestamps for each eye

    Returns:
        Mean of left and right eye median sample rates in Hz
    """
    left_rate = compute_median_sample_rate(ek.left_eye_timestamps)
    right_rate = compute_median_sample_rate(ek.right_eye_timestamps)
    mean_rate = (left_rate + right_rate) / 2.0
    print(f"  Eye sample rates: left={left_rate:.2f}Hz, right={right_rate:.2f}Hz, mean={mean_rate:.2f}Hz")
    return mean_rate


@dataclass
class GazeKinematics:
    """Gaze kinematics data for both eyes in world frame."""

    timestamps: NDArray[np.float64]  # (N,) seconds
    # Left eye
    left_eye_center_mm: NDArray[np.float64]  # (N, 3) eyeball center in world frame
    left_gaze_azimuth_rad: NDArray[np.float64]  # (N,) azimuth in world frame
    left_gaze_elevation_rad: NDArray[np.float64]  # (N,) elevation in world frame
    left_gaze_endpoint_mm: NDArray[np.float64]  # (N, 3) gaze vector endpoint in world frame
    left_gaze_direction: NDArray[np.float64]  # (N, 3) unit gaze direction in world frame
    # Right eye
    right_eye_center_mm: NDArray[np.float64]  # (N, 3) eyeball center in world frame
    right_gaze_azimuth_rad: NDArray[np.float64]  # (N,) azimuth in world frame
    right_gaze_elevation_rad: NDArray[np.float64]  # (N,) elevation in world frame
    right_gaze_endpoint_mm: NDArray[np.float64]  # (N, 3) gaze vector endpoint in world frame
    right_gaze_direction: NDArray[np.float64]  # (N, 3) unit gaze direction in world frame

    def to_dataframe(self) -> pd.DataFrame:
        """Convert GazeKinematics to a pandas DataFrame."""
        return pd.DataFrame({
            "timestamp": self.timestamps,
            # Left eye
            "left_eye_center_x_mm": self.left_eye_center_mm[:, 0],
            "left_eye_center_y_mm": self.left_eye_center_mm[:, 1],
            "left_eye_center_z_mm": self.left_eye_center_mm[:, 2],
            "left_gaze_azimuth_rad": self.left_gaze_azimuth_rad,
            "left_gaze_elevation_rad": self.left_gaze_elevation_rad,
            "left_gaze_endpoint_x_mm": self.left_gaze_endpoint_mm[:, 0],
            "left_gaze_endpoint_y_mm": self.left_gaze_endpoint_mm[:, 1],
            "left_gaze_endpoint_z_mm": self.left_gaze_endpoint_mm[:, 2],
            "left_gaze_direction_x": self.left_gaze_direction[:, 0],
            "left_gaze_direction_y": self.left_gaze_direction[:, 1],
            "left_gaze_direction_z": self.left_gaze_direction[:, 2],
            # Right eye
            "right_eye_center_x_mm": self.right_eye_center_mm[:, 0],
            "right_eye_center_y_mm": self.right_eye_center_mm[:, 1],
            "right_eye_center_z_mm": self.right_eye_center_mm[:, 2],
            "right_gaze_azimuth_rad": self.right_gaze_azimuth_rad,
            "right_gaze_elevation_rad": self.right_gaze_elevation_rad,
            "right_gaze_endpoint_x_mm": self.right_gaze_endpoint_mm[:, 0],
            "right_gaze_endpoint_y_mm": self.right_gaze_endpoint_mm[:, 1],
            "right_gaze_endpoint_z_mm": self.right_gaze_endpoint_mm[:, 2],
            "right_gaze_direction_x": self.right_gaze_direction[:, 0],
            "right_gaze_direction_y": self.right_gaze_direction[:, 1],
            "right_gaze_direction_z": self.right_gaze_direction[:, 2],
        })


def compute_gaze_direction_head_frame(
    eye_angle_x_rad: float,
    eye_angle_y_rad: float,
    is_left_eye: bool,
) -> NDArray[np.float64]:
    """Compute gaze direction in head frame from eye angles.

    Args:
        eye_angle_x_rad: Medial(-)/lateral(+) angle in radians
        eye_angle_y_rad: Superior(+)/inferior(-) angle in radians
        is_left_eye: True for left eye (+Y rest), False for right eye (-Y rest)

    Returns:
        Unit gaze direction vector in head frame [x, y, z]
    """
    # Eye angle X: medial(-) = toward nose (+X_head), lateral(+) = away from nose (-X_head)
    # So head_x_component = -eye_angle_x
    # Eye angle Y: superior(+) = up (+Z_head), inferior(-) = down (-Z_head)
    # So head_z_component = eye_angle_y

    head_x = -eye_angle_x_rad
    head_z = eye_angle_y_rad

    if is_left_eye:
        # Left eye at rest points +Y
        gaze_head = np.array([head_x, 1.0, head_z], dtype=np.float64)
    else:
        # Right eye at rest points -Y
        gaze_head = np.array([head_x, -1.0, head_z], dtype=np.float64)

    # Normalize to unit vector
    norm = np.linalg.norm(gaze_head)
    if norm < 1e-10:
        raise ValueError("Gaze direction is zero")
    return gaze_head / norm


def transform_to_world_frame(
    vector_head: NDArray[np.float64],
    basis_x: NDArray[np.float64],
    basis_y: NDArray[np.float64],
    basis_z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform a vector from head frame to world frame.

    Args:
        vector_head: Vector in head frame [x, y, z]
        basis_x: Head X axis in world frame (unit vector)
        basis_y: Head Y axis in world frame (unit vector)
        basis_z: Head Z axis in world frame (unit vector)

    Returns:
        Vector in world frame [x, y, z]
    """
    return (
        vector_head[0] * basis_x +
        vector_head[1] * basis_y +
        vector_head[2] * basis_z
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


def load_trajectory_data(trajectory_csv_path: Path) -> dict[int, dict[str, NDArray[np.float64]]]:
    """Load trajectory data from CSV and organize by frame.

    Args:
        trajectory_csv_path: Path to tidy_trajectory_data.csv

    Returns:
        Dictionary mapping frame index to marker positions
    """
    df = pd.read_csv(trajectory_csv_path)
    frames: dict[int, dict[str, NDArray[np.float64]]] = {}

    for frame_idx in df["frame"].unique():
        frame_data = df[df["frame"] == frame_idx]
        frames[int(frame_idx)] = {}

        for _, row in frame_data.iterrows():
            marker = str(row["marker"])
            data_type = str(row["data_type"])
            if data_type == "optimized":
                frames[int(frame_idx)][marker] = np.array(
                    [row["x"], row["y"], row["z"]], dtype=np.float64
                )
            elif marker not in frames[int(frame_idx)]:
                frames[int(frame_idx)][marker] = np.array(
                    [row["x"], row["y"], row["z"]], dtype=np.float64
                )

    return frames


def extract_eye_centers(
    trajectory_data: dict[int, dict[str, NDArray[np.float64]]],
    n_frames: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract left and right eye center positions from trajectory data.

    Args:
        trajectory_data: Dictionary mapping frame index to marker positions
        n_frames: Number of frames expected

    Returns:
        (left_eye_centers, right_eye_centers) each (N, 3) arrays in mm
    """
    left_eye_centers = np.zeros((n_frames, 3), dtype=np.float64)
    right_eye_centers = np.zeros((n_frames, 3), dtype=np.float64)

    available_frames = sorted(trajectory_data.keys())

    for i in range(n_frames):
        # Find closest frame
        closest_frames = [f for f in available_frames if f <= i]
        closest_frame = max(closest_frames) if closest_frames else available_frames[0]
        frame_data = trajectory_data[closest_frame]

        if "left_eye" in frame_data:
            left_eye_centers[i] = frame_data["left_eye"]
        if "right_eye" in frame_data:
            right_eye_centers[i] = frame_data["right_eye"]

    return left_eye_centers, right_eye_centers


def interpolate_eye_kinematics(
    ek: EyeKinematics,
    target_timestamps: NDArray[np.float64],
) -> EyeKinematics:
    """Interpolate eye kinematics to match target timestamps.

    Each eye is interpolated independently using its own source timestamps.

    Args:
        ek: Original eye kinematics with separate timestamps per eye
        target_timestamps: Timestamps to interpolate to

    Returns:
        Interpolated EyeKinematics with both eyes on the same target timestamps
    """
    # Interpolate left eye
    left_x = np.interp(target_timestamps, ek.left_eye_timestamps, ek.left_eye_angle_x_rad)
    left_y = np.interp(target_timestamps, ek.left_eye_timestamps, ek.left_eye_angle_y_rad)

    # Interpolate right eye
    right_x = np.interp(target_timestamps, ek.right_eye_timestamps, ek.right_eye_angle_x_rad)
    right_y = np.interp(target_timestamps, ek.right_eye_timestamps, ek.right_eye_angle_y_rad)

    return EyeKinematics(
        left_eye_timestamps=target_timestamps,
        left_eye_angle_x_rad=left_x,
        left_eye_angle_y_rad=left_y,
        right_eye_timestamps=target_timestamps,
        right_eye_angle_x_rad=right_x,
        right_eye_angle_y_rad=right_y,
    )


def validate_timestamp_alignment(
    head_timestamps: NDArray[np.float64],
    ek: EyeKinematics,
    tolerance_s: float,
) -> None:
    """Validate that head and eye data cover the same time period.

    Args:
        head_timestamps: Head kinematics timestamps
        ek: Eye kinematics with separate timestamps per eye
        tolerance_s: Maximum allowed difference in seconds (typically 1 frame duration)

    Raises:
        ValueError: If start or end times differ by more than tolerance
    """
    head_start = head_timestamps[0]
    head_end = head_timestamps[-1]

    # Check both eyes against head timestamps
    for eye_name, eye_timestamps in [
        ("left", ek.left_eye_timestamps),
        ("right", ek.right_eye_timestamps),
    ]:
        eye_start = eye_timestamps[0]
        eye_end = eye_timestamps[-1]

        start_diff = abs(head_start - eye_start)
        end_diff = abs(head_end - eye_end)

        if start_diff > tolerance_s:
            raise ValueError(
                f"Head and {eye_name} eye data start times differ by {start_diff * 1000:.2f} ms "
                f"(tolerance: {tolerance_s * 1000:.2f} ms). "
                f"Head starts at {head_start:.4f}s, {eye_name} eye starts at {eye_start:.4f}s"
            )

        if end_diff > tolerance_s:
            raise ValueError(
                f"Head and {eye_name} eye data end times differ by {end_diff * 1000:.2f} ms "
                f"(tolerance: {tolerance_s * 1000:.2f} ms). "
                f"Head ends at {head_end:.4f}s, {eye_name} eye ends at {eye_end:.4f}s"
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
    return np.linspace(start_time, start_time + (n_frames - 1) * frame_duration, n_frames)


def resample_head_kinematics(
    hk: HeadKinematics,
    target_timestamps: NDArray[np.float64],
) -> HeadKinematics:
    """Resample head kinematics to target timestamps using linear interpolation.

    Args:
        hk: Original head kinematics
        target_timestamps: Target timestamps to resample to

    Returns:
        Resampled HeadKinematics
    """
    n_frames = len(target_timestamps)

    # Interpolate position
    position = np.zeros((n_frames, 3), dtype=np.float64)
    for axis in range(3):
        position[:, axis] = np.interp(target_timestamps, hk.timestamps, hk.position[:, axis])

    # Interpolate euler angles (simple linear interp - may have issues at ±180)
    euler_angles_deg = np.zeros((n_frames, 3), dtype=np.float64)
    for axis in range(3):
        euler_angles_deg[:, axis] = np.interp(
            target_timestamps, hk.timestamps, hk.euler_angles_deg[:, axis]
        )

    # Interpolate angular velocities
    angular_velocity_world_deg_s = np.zeros((n_frames, 3), dtype=np.float64)
    angular_velocity_local_deg_s = np.zeros((n_frames, 3), dtype=np.float64)
    for axis in range(3):
        angular_velocity_world_deg_s[:, axis] = np.interp(
            target_timestamps, hk.timestamps, hk.angular_velocity_world_deg_s[:, axis]
        )
        angular_velocity_local_deg_s[:, axis] = np.interp(
            target_timestamps, hk.timestamps, hk.angular_velocity_local_deg_s[:, axis]
        )

    # Interpolate basis vectors and renormalize
    basis_x = np.zeros((n_frames, 3), dtype=np.float64)
    basis_y = np.zeros((n_frames, 3), dtype=np.float64)
    basis_z = np.zeros((n_frames, 3), dtype=np.float64)
    for axis in range(3):
        basis_x[:, axis] = np.interp(target_timestamps, hk.timestamps, hk.basis_x[:, axis])
        basis_y[:, axis] = np.interp(target_timestamps, hk.timestamps, hk.basis_y[:, axis])
        basis_z[:, axis] = np.interp(target_timestamps, hk.timestamps, hk.basis_z[:, axis])

    # Renormalize basis vectors after interpolation
    for i in range(n_frames):
        basis_x[i] = basis_x[i] / np.linalg.norm(basis_x[i])
        basis_y[i] = basis_y[i] / np.linalg.norm(basis_y[i])
        basis_z[i] = basis_z[i] / np.linalg.norm(basis_z[i])

    return HeadKinematics(
        timestamps=target_timestamps,
        position=position,
        euler_angles_deg=euler_angles_deg,
        angular_velocity_world_deg_s=angular_velocity_world_deg_s,
        angular_velocity_local_deg_s=angular_velocity_local_deg_s,
        basis_x=basis_x,
        basis_y=basis_y,
        basis_z=basis_z,
    )


def resample_eye_kinematics(
    ek: EyeKinematics,
    target_timestamps: NDArray[np.float64],
) -> EyeKinematics:
    """Resample eye kinematics to target timestamps using linear interpolation.

    Args:
        ek: Original eye kinematics
        target_timestamps: Target timestamps to resample to

    Returns:
        Resampled EyeKinematics
    """
    return interpolate_eye_kinematics(ek, target_timestamps)


def compute_gaze_kinematics(
    hk: HeadKinematics,
    ek: EyeKinematics,
    trajectory_data: dict[int, dict[str, NDArray[np.float64]]],
) -> tuple[GazeKinematics, HeadKinematics, EyeKinematics]:
    """Compute gaze kinematics from head and eye data.

    Validates that head and eye data cover the same time period, then resamples
    both to a uniform sample rate (mean of left/right eye median sample rates)
    before computing gaze vectors.

    Args:
        hk: Head kinematics (orientation and basis vectors)
        ek: Eye kinematics (eye-in-head angles, with separate timestamps per eye)
        trajectory_data: Trajectory data containing left_eye and right_eye markers

    Returns:
        Tuple of (GazeKinematics, resampled HeadKinematics, resampled EyeKinematics)
        All at uniform sample rate derived from mean of eye sampling rates.

    Raises:
        ValueError: If head and eye data start/end times differ by more than one frame
    """
    # Compute target sample rate from mean of left and right eye median sample rates
    target_sample_rate_hz = compute_target_sample_rate(ek)
    frame_duration_s = 1.0 / target_sample_rate_hz

    # Validate timestamp alignment (tolerance = 1 frame duration)
    validate_timestamp_alignment(
        head_timestamps=hk.timestamps,
        ek=ek,
        tolerance_s=frame_duration_s*2,
    )

    # Create uniform timestamps at target sample rate
    # Use the later start time and earlier end time to ensure all streams have data
    start_time = max(
        hk.timestamps[0],
        ek.left_eye_timestamps[0],
        ek.right_eye_timestamps[0],
    )
    end_time = min(
        hk.timestamps[-1],
        ek.left_eye_timestamps[-1],
        ek.right_eye_timestamps[-1],
    )
    uniform_timestamps = create_uniform_timestamps(
        start_time=start_time,
        end_time=end_time,
        sample_rate_hz=target_sample_rate_hz,
    )

    print(f"  Resampling to {target_sample_rate_hz:.2f}Hz: {len(uniform_timestamps)} frames")
    print(f"    Time range: {start_time:.4f}s to {end_time:.4f}s ({end_time - start_time:.3f}s duration)")

    # Resample head and eye kinematics to uniform timestamps
    hk_resampled = resample_head_kinematics(hk, uniform_timestamps)
    ek_resampled = resample_eye_kinematics(ek, uniform_timestamps)

    n_frames = len(uniform_timestamps)

    # Extract eye centers from trajectory
    left_eye_centers, right_eye_centers = extract_eye_centers(trajectory_data, n_frames)

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
        # Get head basis vectors for this frame
        basis_x = hk_resampled.basis_x[i]
        basis_y = hk_resampled.basis_y[i]
        basis_z = hk_resampled.basis_z[i]

        # Left eye
        gaze_head_left = compute_gaze_direction_head_frame(
            eye_angle_x_rad=ek_resampled.left_eye_angle_x_rad[i],
            eye_angle_y_rad=ek_resampled.left_eye_angle_y_rad[i],
            is_left_eye=True,
        )
        gaze_world_left = transform_to_world_frame(gaze_head_left, basis_x, basis_y, basis_z)
        left_gaze_direction[i] = gaze_world_left
        left_gaze_azimuth[i], left_gaze_elevation[i] = compute_azimuth_elevation(gaze_world_left)
        left_gaze_endpoint[i] = left_eye_centers[i] + gaze_world_left * GAZE_VECTOR_LENGTH_MM

        # Right eye
        gaze_head_right = compute_gaze_direction_head_frame(
            eye_angle_x_rad=ek_resampled.right_eye_angle_x_rad[i],
            eye_angle_y_rad=ek_resampled.right_eye_angle_y_rad[i],
            is_left_eye=False,
        )
        gaze_world_right = transform_to_world_frame(gaze_head_right, basis_x, basis_y, basis_z)
        right_gaze_direction[i] = gaze_world_right
        right_gaze_azimuth[i], right_gaze_elevation[i] = compute_azimuth_elevation(gaze_world_right)
        right_gaze_endpoint[i] = right_eye_centers[i] + gaze_world_right * GAZE_VECTOR_LENGTH_MM

    gk = GazeKinematics(
        timestamps=uniform_timestamps,
        left_eye_center_mm=left_eye_centers,
        left_gaze_azimuth_rad=left_gaze_azimuth,
        left_gaze_elevation_rad=left_gaze_elevation,
        left_gaze_endpoint_mm=left_gaze_endpoint,
        left_gaze_direction=left_gaze_direction,
        right_eye_center_mm=right_eye_centers,
        right_gaze_azimuth_rad=right_gaze_azimuth,
        right_gaze_elevation_rad=right_gaze_elevation,
        right_gaze_endpoint_mm=right_gaze_endpoint,
        right_gaze_direction=right_gaze_direction,
    )

    return gk, hk_resampled, ek_resampled


if __name__ == "__main__":
    # Paths - edit these
    skull_pose_csv = Path(r"D:\bs\ferret_recordings\example\rotation_translation_data.csv")
    trajectory_csv = Path(r"D:\bs\ferret_recordings\example\tidy_trajectory_data.csv")
    eye_data_csv = Path(r"D:\bs\ferret_recordings\example\eye_data.csv")

    print(f"Loading skull pose data from {skull_pose_csv}...")
    timestamps, positions, quaternions = load_skull_pose(skull_pose_csv)
    print(f"  Loaded {len(timestamps)} frames")

    print("Computing head kinematics...")
    hk = compute_head_kinematics(
        timestamps=timestamps,
        positions=positions,
        quaternions=quaternions,
    )

    print(f"Loading eye data from {eye_data_csv}...")
    ek = load_eye_data(eye_data_csv)
    print(f"  Loaded {len(ek.timestamps)} frames")

    print(f"Loading trajectory data from {trajectory_csv}...")
    trajectory_data = load_trajectory_data(trajectory_csv)
    print(f"  Loaded {len(trajectory_data)} frames")

    print("Computing gaze kinematics...")
    gk, hk_resampled, ek_resampled = compute_gaze_kinematics(
        hk=hk,
        ek=ek,
        trajectory_data=trajectory_data,
    )

    # Save CSVs
    output_path = skull_pose_csv.parent / "gaze_kinematics.csv"
    df = gk.to_dataframe()
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Save resampled head kinematics
    head_resampled_path = skull_pose_csv.parent / "head_kinematics_resampled.csv"
    hk_resampled.to_dataframe().to_csv(head_resampled_path, index=False)
    print(f"Saved: {head_resampled_path}")

    # Save resampled eye kinematics
    eye_resampled_path = skull_pose_csv.parent / "eye_kinematics_resampled.csv"
    ek_resampled.to_dataframe().to_csv(eye_resampled_path, index=False)
    print(f"Saved: {eye_resampled_path}")