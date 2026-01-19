"""
Resampling methods for RigidBodyKinematics and KeypointTrajectories.

Properly handles quaternion interpolation via SLERP and recomputes derived
quantities (velocities, angular velocities) from the resampled base data.
"""

import numpy as np
from numpy.typing import NDArray

from python_code.kinematics_core.quaternion_model import Quaternion
from python_code.kinematics_core.keypoint_trajectories import KeypointTrajectories
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry


def resample_kinematics(
    kinematics: RigidBodyKinematics,
    target_timestamps: NDArray[np.float64],
) -> RigidBodyKinematics:
    """
    Resample RigidBodyKinematics to new target timestamps.

    Position is linearly interpolated. Quaternions are interpolated using SLERP
    (spherical linear interpolation) to maintain proper rotation geometry.

    Velocities and angular velocities are RECOMPUTED from the resampled positions
    and orientations rather than interpolated directly - this ensures kinematic
    consistency (derivatives match the resampled trajectories).

    Args:
        kinematics: The original kinematics to resample
        target_timestamps: (M,) array of target timestamps in seconds.
            Must be monotonically increasing. Values outside the original
            time range will be clamped to boundary values.

    Returns:
        New RigidBodyKinematics at the target timestamps

    Raises:
        ValueError: If target_timestamps is not monotonically increasing
        ValueError: If fewer than 2 target timestamps provided
    """
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)

    if len(target_timestamps) < 2:
        raise ValueError(
            f"Need at least 2 target timestamps to resample, got {len(target_timestamps)}"
        )

    if not np.all(np.diff(target_timestamps) > 0):
        raise ValueError("target_timestamps must be strictly increasing")

    original_timestamps = kinematics.timestamps

    # Resample positions via linear interpolation (component-wise)
    resampled_position_xyz = _resample_positions(
        original_timestamps=original_timestamps,
        original_positions=kinematics.position_xyz,
        target_timestamps=target_timestamps,
    )

    # Resample quaternions via SLERP
    resampled_quaternions_wxyz = _resample_quaternions_slerp(
        original_timestamps=original_timestamps,
        original_quaternions=kinematics.orientations.quaternions,
        target_timestamps=target_timestamps,
    )

    # Use from_pose_arrays to recompute everything else correctly
    # This ensures velocities and angular velocities are derived from
    # the resampled trajectories, not interpolated independently
    return RigidBodyKinematics.from_pose_arrays(
        name=kinematics.name,
        reference_geometry=kinematics.reference_geometry,
        timestamps=target_timestamps,
        position_xyz=resampled_position_xyz,
        quaternions_wxyz=resampled_quaternions_wxyz,
    )


def resample_kinematics_to_uniform_rate(
    kinematics: RigidBodyKinematics,
    target_fps: float,
) -> RigidBodyKinematics:
    """
    Resample RigidBodyKinematics to a uniform frame rate.

    Convenience wrapper around resample_kinematics that generates uniform
    timestamps at the specified frame rate.

    Args:
        kinematics: The original kinematics to resample
        target_fps: Target frames per second (must be positive)

    Returns:
        New RigidBodyKinematics at uniform timestamps

    Raises:
        ValueError: If target_fps <= 0
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    t_start = kinematics.timestamps[0]
    t_end = kinematics.timestamps[-1]
    dt = 1.0 / target_fps

    # Generate uniform timestamps covering the original range
    n_frames = int(np.floor((t_end - t_start) / dt)) + 1
    target_timestamps = t_start + np.arange(n_frames, dtype=np.float64) * dt

    return resample_kinematics(
        kinematics=kinematics,
        target_timestamps=target_timestamps,
    )


def _resample_positions(
    original_timestamps: NDArray[np.float64],
    original_positions: NDArray[np.float64],
    target_timestamps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Resample positions using linear interpolation.

    Args:
        original_timestamps: (N,) original timestamps
        original_positions: (N, 3) original positions
        target_timestamps: (M,) target timestamps

    Returns:
        (M, 3) resampled positions
    """
    n_targets = len(target_timestamps)
    resampled = np.empty((n_targets, 3), dtype=np.float64)

    # Interpolate each component independently
    for i in range(3):
        resampled[:, i] = np.interp(
            x=target_timestamps,
            xp=original_timestamps,
            fp=original_positions[:, i],
        )

    return resampled


def _resample_quaternions_slerp(
    original_timestamps: NDArray[np.float64],
    original_quaternions: list[Quaternion],
    target_timestamps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Resample quaternions using SLERP (Spherical Linear Interpolation).

    SLERP is essential for proper quaternion interpolation because:
    1. It maintains unit norm (stays on the unit hypersphere)
    2. It produces constant angular velocity interpolation
    3. It takes the shortest path between orientations

    Args:
        original_timestamps: (N,) original timestamps
        original_quaternions: List of N Quaternion objects
        target_timestamps: (M,) target timestamps

    Returns:
        (M, 4) array of resampled quaternions as [w, x, y, z]
    """
    n_targets = len(target_timestamps)
    n_original = len(original_quaternions)

    # Find interpolation indices for all target timestamps
    # np.searchsorted gives the index where target would be inserted
    indices = np.searchsorted(original_timestamps, target_timestamps)

    resampled_wxyz = np.empty((n_targets, 4), dtype=np.float64)

    for i in range(n_targets):
        target_t = target_timestamps[i]
        idx = indices[i]

        # Handle boundary cases: clamp to first/last quaternion
        if idx == 0:
            q = original_quaternions[0]
            resampled_wxyz[i] = [q.w, q.x, q.y, q.z]
            continue
        if idx >= n_original:
            q = original_quaternions[-1]
            resampled_wxyz[i] = [q.w, q.x, q.y, q.z]
            continue

        # Get bracketing quaternions and timestamps
        t0 = original_timestamps[idx - 1]
        t1 = original_timestamps[idx]
        q0 = original_quaternions[idx - 1]
        q1 = original_quaternions[idx]

        # Compute interpolation parameter
        dt = t1 - t0
        if dt < 1e-10:
            # Timestamps effectively identical, use q0
            resampled_wxyz[i] = [q0.w, q0.x, q0.y, q0.z]
            continue

        t = float((target_t - t0) / dt)
        t = np.clip(t, 0.0, 1.0)

        # SLERP interpolate
        q_interp = _slerp(q0=q0, q1=q1, t=t)
        resampled_wxyz[i] = [q_interp.w, q_interp.x, q_interp.y, q_interp.z]

    return resampled_wxyz


def _slerp(q0: Quaternion, q1: Quaternion, t: float) -> Quaternion:
    """
    Spherical linear interpolation between two quaternions.

    Computes the quaternion at parameter t along the shortest great arc
    between q0 (t=0) and q1 (t=1) on the unit quaternion hypersphere.

    Args:
        q0: Start quaternion
        q1: End quaternion
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated unit quaternion
    """
    # Compute cosine of angle between quaternions (4D dot product)
    dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z

    # If dot < 0, negate q1 to take the shorter path
    # (q and -q represent the same rotation)
    q1_w, q1_x, q1_y, q1_z = q1.w, q1.x, q1.y, q1.z
    if dot < 0.0:
        q1_w, q1_x, q1_y, q1_z = -q1_w, -q1_x, -q1_y, -q1_z
        dot = -dot

    # Clamp dot to valid range for arccos
    dot = min(dot, 1.0)

    # If quaternions are very close, use linear interpolation
    # to avoid numerical issues with small angles
    if dot > 0.9995:
        w = q0.w + t * (q1_w - q0.w)
        x = q0.x + t * (q1_x - q0.x)
        y = q0.y + t * (q1_y - q0.y)
        z = q0.z + t * (q1_z - q0.z)
        # Quaternion.__post_init__ normalizes
        return Quaternion(w=w, x=x, y=y, z=z)

    # Standard SLERP formula
    theta_0 = np.arccos(dot)  # Angle between quaternions
    theta = theta_0 * t  # Angle at interpolation point

    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    w = s0 * q0.w + s1 * q1_w
    x = s0 * q0.x + s1 * q1_x
    y = s0 * q0.y + s1 * q1_y
    z = s0 * q0.z + s1 * q1_z

    return Quaternion(w=float(w), x=float(x), y=float(y), z=float(z))


# =============================================================================
# KeypointTrajectories Resampling
# =============================================================================


def resample_keypoint_trajectories(
    keypoint_trajectories: KeypointTrajectories,
    target_timestamps: NDArray[np.float64],
) -> KeypointTrajectories:
    """
    Resample KeypointTrajectories to new target timestamps.

    All keypoint positions are linearly interpolated independently.

    Args:
        keypoint_trajectories: The original keypoint trajectories to resample
        target_timestamps: (M,) array of target timestamps in seconds.
            Must be strictly increasing. Values outside the original
            time range will be clamped to boundary values.

    Returns:
        New KeypointTrajectories at the target timestamps

    Raises:
        ValueError: If target_timestamps is not strictly increasing
        ValueError: If fewer than 2 target timestamps provided
    """
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)

    if len(target_timestamps) < 2:
        raise ValueError(
            f"Need at least 2 target timestamps to resample, got {len(target_timestamps)}"
        )

    if not np.all(np.diff(target_timestamps) > 0):
        raise ValueError("target_timestamps must be strictly increasing")

    original_timestamps = keypoint_trajectories.timestamps
    original_trajectories = keypoint_trajectories.trajectories_fr_id_xyz  # (N, M, 3)

    n_targets = len(target_timestamps)
    n_keypoints = keypoint_trajectories.n_keypoints

    # Resample each keypoint's trajectory via linear interpolation
    resampled_trajectories = np.empty((n_targets, n_keypoints, 3), dtype=np.float64)

    for kp_idx in range(n_keypoints):
        for coord_idx in range(3):
            resampled_trajectories[:, kp_idx, coord_idx] = np.interp(
                x=target_timestamps,
                xp=original_timestamps,
                fp=original_trajectories[:, kp_idx, coord_idx],
            )

    return KeypointTrajectories(
        keypoint_names=keypoint_trajectories.keypoint_names,
        timestamps=target_timestamps,
        trajectories_fr_id_xyz=resampled_trajectories,
    )


def resample_keypoint_trajectories_to_uniform_rate(
    keypoint_trajectories: KeypointTrajectories,
    target_fps: float,
) -> KeypointTrajectories:
    """
    Resample KeypointTrajectories to a uniform frame rate.

    Convenience wrapper around resample_keypoint_trajectories that generates
    uniform timestamps at the specified frame rate.

    Args:
        keypoint_trajectories: The original keypoint trajectories to resample
        target_fps: Target frames per second (must be positive)

    Returns:
        New KeypointTrajectories at uniform timestamps

    Raises:
        ValueError: If target_fps <= 0
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    t_start = keypoint_trajectories.timestamps[0]
    t_end = keypoint_trajectories.timestamps[-1]
    dt = 1.0 / target_fps

    # Generate uniform timestamps covering the original range
    n_frames = int(np.floor((t_end - t_start) / dt)) + 1
    target_timestamps = t_start + np.arange(n_frames, dtype=np.float64) * dt

    return resample_keypoint_trajectories(
        keypoint_trajectories=keypoint_trajectories,
        target_timestamps=target_timestamps,
    )