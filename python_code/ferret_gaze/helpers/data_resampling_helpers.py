import numpy as np
from numpy._typing import NDArray

from python_code.ferret_gaze.kinematics_core.quaternion_model import resample_quaternions
from python_code.ferret_gaze.kinematics_calculators.ferret_eye_kinematics import EyeballKinematics
from python_code.ferret_gaze.kinematics_calculators.ferret_skull_kinematics import SkullKinematics


def resample_trajectory_data(
    trajectory_data: dict[str, NDArray[np.float64]],
    original_timestamps: NDArray[np.float64],
    target_timestamps: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Resample trajectory data to new timestamps using linear interpolation.

    Args:
        trajectory_data: Dictionary mapping marker names to (n_frames, 3) arrays
        original_timestamps: Original timestamps, shape (n_original,)
        target_timestamps: Target timestamps, shape (n_target,)

    Returns:
        Dictionary with resampled trajectories at target timestamps

    Raises:
        ValueError: If array shapes are inconsistent
    """
    n_original = len(original_timestamps)
    n_target = len(target_timestamps)

    if original_timestamps.ndim != 1:
        raise ValueError(f"original_timestamps must be 1D, got shape {original_timestamps.shape}")
    if target_timestamps.ndim != 1:
        raise ValueError(f"target_timestamps must be 1D, got shape {target_timestamps.shape}")

    resampled_data: dict[str, NDArray[np.float64]] = {}

    # Add resampled timestamps
    resampled_data["timestamps"] = target_timestamps.copy()

    for marker_name, marker_xyz in trajectory_data.items():
        if marker_name == "timestamps":
            continue  # Already handled above

        # Validate input shape
        if marker_xyz.shape != (n_original, 3):
            raise ValueError(
                f"Marker '{marker_name}' has shape {marker_xyz.shape}, "
                f"expected ({n_original}, 3)"
            )

        # Interpolate each axis separately (np.interp only works on 1D)
        resampled_xyz = np.zeros((n_target, 3), dtype=np.float64)
        for axis in range(3):
            resampled_xyz[:, axis] = np.interp(
                target_timestamps,
                original_timestamps,
                marker_xyz[:, axis],
            )

        resampled_data[marker_name] = resampled_xyz

    return resampled_data


def resample_eye_kinematics(
    eye: EyeballKinematics,
    target_timestamps: NDArray[np.float64],
) -> EyeballKinematics:
    """Resample eye kinematics to target timestamps using linear interpolation.

    Args:
        eye: Original eye kinematics
        target_timestamps: Target timestamps, shape (n_target,)

    Returns:
        Resampled EyeKinematics
    """
    if target_timestamps.ndim != 1:
        raise ValueError(f"target_timestamps must be 1D, got shape {target_timestamps.shape}")

    eye_x = np.interp(target_timestamps, eye.timestamps, eye.eyeball_angle_azimuth_rad)
    eye_y = np.interp(target_timestamps, eye.timestamps, eye.eyeball_angle_elevation_rad)

    return EyeballKinematics(
        timestamps=target_timestamps.copy(),
        eyeball_angle_azimuth_rad=eye_x,
        eyeball_angle_elevation_rad=eye_y,
    )


def validate_timestamp_alignment(
    skull_timestamps: NDArray[np.float64],
    left_eye_timestamps: NDArray[np.float64],
    right_eye_timestamps: NDArray[np.float64],
    tolerance_s: float,
) -> None:
    """Validate that skull and eye data cover approximately the same time period.

    Args:
        skull_timestamps: Skull timestamps, shape (n_skull,)
        left_eye_timestamps: Left eye timestamps, shape (n_left,)
        right_eye_timestamps: Right eye timestamps, shape (n_right,)
        tolerance_s: Maximum allowed difference in seconds (typically 1-2 frame durations)

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
        Array of uniformly spaced timestamps, shape (n_frames,)

    Raises:
        ValueError: If parameters are invalid
    """
    if end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

    frame_duration = 1.0 / sample_rate_hz
    n_frames = int(np.floor((end_time - start_time) / frame_duration)) + 1
    if n_frames < 2:
        raise ValueError(f"Time range too short for sample rate: only {n_frames} frame(s)")
    actual_end_time = start_time + (n_frames - 1) * frame_duration
    return np.linspace(start_time, actual_end_time, n_frames)


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
        target_timestamps: Target timestamps to resample to, shape (n_target,)

    Returns:
        Resampled SkullKinematics
    """
    if target_timestamps.ndim != 1:
        raise ValueError(f"target_timestamps must be 1D, got shape {target_timestamps.shape}")

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
        timestamps=target_timestamps.copy(),
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
    left_eye: EyeballKinematics,
    right_eye: EyeballKinematics,
    trajectory_data: dict[str, NDArray[np.float64]],
    target_framerate_strategy: str = "mean_eye_framerate",
) -> tuple[SkullKinematics, EyeballKinematics, EyeballKinematics, dict[str, NDArray[np.float64]]]:
    """Resample all kinematic data to a common uniform timestamp grid.

    Args:
        skull: Skull kinematics
        left_eye: Left eye kinematics
        right_eye: Right eye kinematics
        trajectory_data: Body trajectory data
        target_framerate_strategy: Strategy for choosing target framerate

    Returns:
        Tuple of (skull_resampled, left_eye_resampled, right_eye_resampled, trajectories_resampled)

    Raises:
        NotImplementedError: If unknown framerate strategy
        ValueError: If timestamps are misaligned
    """
    if target_framerate_strategy != "mean_eye_framerate":
        raise NotImplementedError(
            f"Unsupported target framerate strategy: {target_framerate_strategy} - "
            f"use 'mean_eye_framerate' for now"
        )

    # Compute target sample rate from mean of left and right eye median sample rates
    target_sample_rate_hz = compute_mean_eye_framerate(left_eye=left_eye, right_eye=right_eye)
    frame_duration_s = 1.0 / target_sample_rate_hz

    # Validate timestamp alignment (tolerance = 2 frame durations)
    validate_timestamp_alignment(
        skull_timestamps=skull.timestamps,
        left_eye_timestamps=left_eye.timestamps,
        right_eye_timestamps=right_eye.timestamps,
        tolerance_s=frame_duration_s * 2,
    )

    # Create uniform timestamps at target sample rate
    # Use the later start time and earlier end time to ensure all streams have data
    # BUG FIX: Was using left_eye.timestamps for right_eye bounds
    start_time = max(
        skull.timestamps[0],
        left_eye.timestamps[0],
        right_eye.timestamps[0],  # FIXED: was left_eye.timestamps[0]
    )
    end_time = min(
        skull.timestamps[-1],
        left_eye.timestamps[-1],
        right_eye.timestamps[-1],  # FIXED: was left_eye.timestamps[-1]
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

    # Get original timestamps from trajectory data for resampling
    trajectory_timestamps = trajectory_data.get("timestamps")
    if trajectory_timestamps is None:
        raise ValueError("trajectory_data must contain 'timestamps' key")

    trajectories_resampled = resample_trajectory_data(
        trajectory_data=trajectory_data,
        original_timestamps=trajectory_timestamps,
        target_timestamps=uniform_timestamps,
    )
    return skull_resampled, left_eye_resampled, right_eye_resampled, trajectories_resampled


def compute_median_sample_rate(timestamps: NDArray[np.float64]) -> float:
    """Compute median sample rate from timestamps.

    Args:
        timestamps: Array of timestamps in seconds, shape (N,)

    Returns:
        Median sample rate in Hz

    Raises:
        ValueError: If fewer than 2 timestamps or invalid duration
    """
    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be 1D, got shape {timestamps.shape}")
    if len(timestamps) < 2:
        raise ValueError(f"Need at least 2 timestamps to compute sample rate, got {len(timestamps)}")
    frame_durations = np.diff(timestamps)
    median_duration = float(np.median(frame_durations))
    if median_duration <= 0:
        raise ValueError(f"Invalid median frame duration: {median_duration}")
    return 1.0 / median_duration


def compute_mean_eye_framerate(left_eye: EyeballKinematics, right_eye: EyeballKinematics) -> float:
    """Compute mean framerate from left and right eye data.

    Args:
        left_eye: Left eye kinematics
        right_eye: Right eye kinematics

    Returns:
        Mean sample rate in Hz
    """
    left_rate = compute_median_sample_rate(left_eye.timestamps)
    right_rate = compute_median_sample_rate(right_eye.timestamps)
    mean_rate = (left_rate + right_rate) / 2.0
    print(f"  Eye sample rates: left={left_rate:.2f}Hz, right={right_rate:.2f}Hz, mean={mean_rate:.2f}Hz")
    return mean_rate
