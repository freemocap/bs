"""Helper functions for computing kinematic derivatives."""
import numpy as np
from numpy.typing import NDArray

from python_code.kinematics_core.quaternion_model import Quaternion


def compute_velocity(
    position_xyz: NDArray[np.float64],
    timestamps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute linear velocity using forward differences.

    First frame velocity is copied from second frame (forward difference not available).

    Args:
        position_xyz: (N, 3) array of positions in mm
        timestamps: (N,) array of timestamps in seconds

    Returns:
        (N, 3) array of velocities in mm/s

    Raises:
        ValueError: If any consecutive timestamps have zero or negative difference
    """
    number_of_frames = len(timestamps)
    if number_of_frames < 2:
        raise ValueError(f"Need at least 2 frames to compute velocity, got {number_of_frames}")

    # Compute time differences between consecutive frames
    time_differences = np.diff(timestamps)

    # Check for zero or negative time differences
    invalid_indices = np.where(time_differences <= 1e-10)[0]
    if len(invalid_indices) > 0:
        first_invalid = invalid_indices[0]
        raise ValueError(
            f"Invalid timestamp difference at frame {first_invalid} -> {first_invalid + 1}: "
            f"dt = {time_differences[first_invalid]:.2e} seconds. "
            f"Timestamps must be strictly increasing."
        )

    # Compute position differences (N-1, 3)
    position_differences = np.diff(position_xyz, axis=0)

    # Pre-allocate velocity array and compute directly into it
    velocity = np.zeros((number_of_frames, 3), dtype=np.float64)
    velocity[1:] = position_differences / time_differences[:, np.newaxis]
    velocity[0] = velocity[1]

    return velocity


def compute_angular_velocity(
    orientations: list[Quaternion],
    timestamps: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute global and local angular velocity from quaternion sequence.

    Uses central differences for interior frames, forward/backward for endpoints.

    Args:
        orientations: List of N quaternions
        timestamps: (N,) array of timestamps in seconds

    Returns:
        Tuple of (global_xyz, local_xyz) each (N, 3) arrays in rad/s

    Raises:
        ValueError: If any timestamp differences used in computation are zero or negative
    """
    number_of_frames = len(orientations)
    if number_of_frames < 2:
        raise ValueError(f"Need at least 2 frames to compute angular velocity, got {number_of_frames}")

    # Pre-check all timestamp differences
    time_differences = np.diff(timestamps)
    invalid_indices = np.where(time_differences <= 1e-10)[0]
    if len(invalid_indices) > 0:
        first_invalid = invalid_indices[0]
        raise ValueError(
            f"Invalid timestamp difference at frame {first_invalid} -> {first_invalid + 1}: "
            f"dt = {time_differences[first_invalid]:.2e} seconds. "
            f"Timestamps must be strictly increasing."
        )

    # Extract quaternion components into arrays (N, 4) as [w, x, y, z]
    quaternion_array = np.array(
        [[quaternion.w, quaternion.x, quaternion.y, quaternion.z] for quaternion in orientations],
        dtype=np.float64,
    )

    # Build index arrays for "current" and "next" quaternions at each frame
    # Frame 0: forward difference (curr=0, next=1)
    # Frame i (interior): central difference (curr=i-1, next=i+1)
    # Frame N-1: backward difference (curr=N-2, next=N-1)
    current_indices = np.concatenate([[0], np.arange(0, number_of_frames - 2), [number_of_frames - 2]])
    next_indices = np.concatenate([[1], np.arange(2, number_of_frames), [number_of_frames - 1]])

    # Build time deltas for each frame
    # Frame 0: t[1] - t[0]
    # Frame i (interior): t[i+1] - t[i-1]
    # Frame N-1: t[N-1] - t[N-2]
    time_deltas = np.empty(number_of_frames, dtype=np.float64)
    time_deltas[0] = timestamps[1] - timestamps[0]
    time_deltas[1:-1] = timestamps[2:] - timestamps[:-2]
    time_deltas[-1] = timestamps[-1] - timestamps[-2]

    # Get quaternion arrays for current and next
    quaternions_current = quaternion_array[current_indices]  # (N, 4)
    quaternions_next = quaternion_array[next_indices]  # (N, 4)

    # Compute relative quaternion: q_relative = q_next * q_current^-1
    # For unit quaternions, inverse is conjugate: [w, -x, -y, -z]
    quaternions_current_conjugate = quaternions_current * np.array([1, -1, -1, -1])

    # Quaternion multiplication: q_next * q_current_conjugate
    quaternions_relative = _batch_quaternion_multiply(quaternions_next, quaternions_current_conjugate)

    # Extract axis-angle from relative quaternions
    axes, angles = _batch_quaternion_to_axis_angle(quaternions_relative)

    # Global angular velocity: omega = axis * (angle / dt)
    global_xyz = axes * (angles / time_deltas)[:, np.newaxis]

    # Local angular velocity: R^T @ omega for each frame
    # R is rotation matrix of orientations[i] (not q_current used for differentiation)
    rotation_matrices = _batch_quaternion_to_rotation_matrix(quaternion_array)  # (N, 3, 3)
    local_xyz = np.einsum("nij,nj->ni", rotation_matrices.transpose(0, 2, 1), global_xyz)

    return global_xyz, local_xyz


def _batch_quaternion_multiply(
    quaternions_a: NDArray[np.float64],
    quaternions_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Batch quaternion multiplication (Hamilton product).

    Args:
        quaternions_a: (N, 4) array of quaternions [w, x, y, z]
        quaternions_b: (N, 4) array of quaternions [w, x, y, z]

    Returns:
        (N, 4) array of product quaternions
    """
    w1, x1, y1, z1 = quaternions_a[:, 0], quaternions_a[:, 1], quaternions_a[:, 2], quaternions_a[:, 3]
    w2, x2, y2, z2 = quaternions_b[:, 0], quaternions_b[:, 1], quaternions_b[:, 2], quaternions_b[:, 3]

    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _batch_quaternion_to_axis_angle(
    quaternions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Batch convert quaternions to axis-angle representation.

    Args:
        quaternions: (N, 4) array of quaternions [w, x, y, z]

    Returns:
        Tuple of (axes, angles) where axes is (N, 3) and angles is (N,)
    """
    w = quaternions[:, 0]
    xyz = quaternions[:, 1:4]

    # Clamp w to [-1, 1] for numerical stability
    w_clamped = np.clip(w, -1.0, 1.0)

    # angle = 2 * arccos(|w|)
    angles = 2.0 * np.arccos(np.abs(w_clamped))

    # sin(angle/2) = sqrt(1 - w^2)
    sin_half_angle = np.sqrt(1.0 - w_clamped ** 2)

    # axis = xyz / sin(angle/2), with fallback for small angles
    axes = np.zeros_like(xyz)
    valid_mask = sin_half_angle > 1e-10
    axes[valid_mask] = xyz[valid_mask] / sin_half_angle[valid_mask, np.newaxis]
    axes[~valid_mask] = [1.0, 0.0, 0.0]  # arbitrary axis for zero rotation

    # Flip axis if w < 0 (to stay in positive hemisphere)
    negative_w_mask = w < 0
    axes[negative_w_mask] *= -1

    return axes, angles


def _batch_quaternion_to_rotation_matrix(
    quaternions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Batch convert quaternions to rotation matrices.

    Args:
        quaternions: (N, 4) array of quaternions [w, x, y, z]

    Returns:
        (N, 3, 3) array of rotation matrices
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Pre-compute repeated terms
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    number_of_quaternions = len(quaternions)
    rotation_matrices = np.empty((number_of_quaternions, 3, 3), dtype=np.float64)

    rotation_matrices[:, 0, 0] = 1 - 2 * (yy + zz)
    rotation_matrices[:, 0, 1] = 2 * (xy - wz)
    rotation_matrices[:, 0, 2] = 2 * (xz + wy)

    rotation_matrices[:, 1, 0] = 2 * (xy + wz)
    rotation_matrices[:, 1, 1] = 1 - 2 * (xx + zz)
    rotation_matrices[:, 1, 2] = 2 * (yz - wx)

    rotation_matrices[:, 2, 0] = 2 * (xz - wy)
    rotation_matrices[:, 2, 1] = 2 * (yz + wx)
    rotation_matrices[:, 2, 2] = 1 - 2 * (xx + yy)

    return rotation_matrices