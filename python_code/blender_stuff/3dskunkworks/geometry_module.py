"""
Geometry and Rotation Utilities for Rigid Body Tracking
========================================================

Core utilities for:
- Geometry generation
- Rotation mathematics (JAX and NumPy)
- Rotation unwrapping and smoothing
- Signal filtering

Author: AI Assistant
Date: 2025
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import butter, filtfilt
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def generate_cube_vertices(*, size: float = 0.5) -> np.ndarray:
    """
    Generate 8 vertices of a cube centered at origin.

    Args:
        size: Half-width of the cube

    Returns:
        Array of shape (8, 3) containing vertex coordinates
    """
    s = size
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ])
    return vertices


# =============================================================================
# ROTATION UTILITIES
# =============================================================================

@jit
def rotation_matrix_from_rotvec_jax(*, rotvec: jnp.ndarray) -> jnp.ndarray:
    """
    Convert rotation vector to rotation matrix using Rodrigues' formula.

    JAX-compatible for automatic differentiation.

    Args:
        rotvec: Rotation vector (axis * angle) of shape (3,)

    Returns:
        Rotation matrix of shape (3, 3)
    """
    angle = jnp.linalg.norm(rotvec)
    small_angle = angle < 1e-8

    # Normalized axis (with safe fallback for zero rotation)
    axis = jnp.where(small_angle, jnp.array([0., 0., 1.]), rotvec / angle)

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    K = jnp.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    I = jnp.eye(3)
    R = I + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)

    return jnp.where(small_angle, I, R)


def rotation_matrix_from_axis_angle(
    *, 
    axis: np.ndarray, 
    angle: float
) -> np.ndarray:
    """
    Create rotation matrix from axis-angle representation (NumPy version).

    Args:
        axis: Rotation axis (will be normalized)
        angle: Rotation angle in radians

    Returns:
        Rotation matrix of shape (3, 3)
    """
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def rotation_error_angle(*, R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute rotation error as geodesic angle in degrees.

    Args:
        R1: First rotation matrix (3, 3)
        R2: Second rotation matrix (3, 3)

    Returns:
        Angle difference in degrees
    """
    R_error = R1.T @ R2
    trace = np.trace(R_error)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(angle_rad)


def unwrap_rotation_vectors_advanced(*, rotvecs: np.ndarray) -> np.ndarray:
    """
    Advanced rotation vector unwrapping with multi-pass consistency checking.

    This is CRITICAL for preventing spinning artifacts!

    Uses three passes:
    1. Forward pass: Unwrap from start to end
    2. Backward pass: Smooth remaining discontinuities
    3. Quaternion consistency: Ensure same hemisphere

    Args:
        rotvecs: Array of rotation vectors, shape (n_frames, 3)

    Returns:
        Unwrapped rotation vectors with minimal discontinuities
    """
    logger.info("Advanced rotation unwrapping (3-pass)...")

    unwrapped = rotvecs.copy()
    n_frames = len(rotvecs)

    # === FORWARD PASS ===
    for i in range(1, n_frames):
        current_rotvec = unwrapped[i]
        prev_rotvec = unwrapped[i - 1]

        # Check multiple representations of same rotation
        candidates = [
            current_rotvec,
            -current_rotvec,  # Sign flip (same rotation)
        ]

        # Check 2π wraparound
        angle = np.linalg.norm(current_rotvec)
        if angle > np.pi:
            axis = current_rotvec / (angle + 1e-10)
            new_angle = 2 * np.pi - angle
            candidates.append(-axis * new_angle)

        # Choose candidate with minimum distance to previous
        distances = [np.linalg.norm(cand - prev_rotvec) for cand in candidates]
        best_idx = np.argmin(distances)
        unwrapped[i] = candidates[best_idx]

    # === BACKWARD PASS ===
    for i in range(n_frames - 2, -1, -1):
        current_rotvec = unwrapped[i]
        next_rotvec = unwrapped[i + 1]

        dist_normal = np.linalg.norm(current_rotvec - next_rotvec)
        dist_flipped = np.linalg.norm(-current_rotvec - next_rotvec)

        if dist_flipped < dist_normal:
            unwrapped[i] = -current_rotvec

    # === QUATERNION CONSISTENCY CHECK ===
    unwrapped_quats = np.array([
        Rotation.from_rotvec(rv).as_quat() for rv in unwrapped
    ])

    for i in range(1, n_frames):
        # Ensure quaternions stay in same hemisphere
        if np.dot(unwrapped_quats[i], unwrapped_quats[i - 1]) < 0:
            unwrapped_quats[i] = -unwrapped_quats[i]
            unwrapped[i] = Rotation.from_quat(unwrapped_quats[i]).as_rotvec()

    # Log improvement
    original_max_jump = np.max(np.linalg.norm(np.diff(rotvecs, axis=0), axis=1))
    unwrapped_max_jump = np.max(np.linalg.norm(np.diff(unwrapped, axis=0), axis=1))

    logger.info(f"  Max discontinuity: {original_max_jump:.3f} → {unwrapped_max_jump:.3f}")

    return unwrapped


# =============================================================================
# SMOOTHING UTILITIES
# =============================================================================

def apply_butterworth_filter(
    *,
    data: np.ndarray,
    cutoff_freq: float = 0.1,
    sampling_rate: float = 1.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-lag Butterworth lowpass filter.

    Good for translation smoothing, but use SLERP for rotations!

    Args:
        data: Time series data of shape (n_frames, n_dims)
        cutoff_freq: Cutoff frequency (0-1, relative to Nyquist)
        sampling_rate: Sampling rate of the data
        order: Filter order (higher = sharper cutoff)

    Returns:
        Filtered data of same shape
    """
    n_frames, n_dims = data.shape

    if n_frames < 2 * order:
        logger.warning(f"Not enough frames ({n_frames}) for filtering")
        return data

    nyquist = sampling_rate / 2
    normalized_cutoff = np.clip(cutoff_freq / nyquist, 0.001, 0.999)

    b, a = butter(N=order, Wn=normalized_cutoff, btype='low', analog=False)

    filtered = np.zeros_like(data)
    for dim in range(n_dims):
        # filtfilt applies filter forwards then backwards for zero phase shift
        filtered[:, dim] = filtfilt(b=b, a=a, x=data[:, dim])

    return filtered


def slerp_smooth_rotations(
    *,
    rotations: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Smooth rotations using SLERP (Spherical Linear Interpolation).

    This respects the SO(3) manifold structure - much better than
    filtering rotation vectors!

    Args:
        rotations: Array of rotation matrices, shape (n_frames, 3, 3)
        window_size: Size of smoothing window (must be odd)

    Returns:
        Smoothed rotation matrices of same shape
    """
    logger.info(f"SLERP smoothing (window={window_size})...")

    n_frames = len(rotations)
    rot_objects = [Rotation.from_matrix(R) for R in rotations]
    smoothed_rots: list[Rotation] = []

    for i in range(n_frames):
        # Symmetric window around current frame
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_frames, i + window_size // 2 + 1)

        if end_idx - start_idx <= 1:
            smoothed_rots.append(rot_objects[i])
            continue

        # Create SLERP interpolator for window
        times = np.array(range(start_idx, end_idx))
        window_rots = rot_objects[start_idx:end_idx]

        try:
            slerp = Slerp(
                times=times,
                rotations=Rotation.concatenate(window_rots)
            )
            smoothed_rots.append(slerp([i])[0])
        except Exception as e:
            logger.warning(f"SLERP failed at frame {i}: {e}")
            smoothed_rots.append(rot_objects[i])

    smoothed_matrices = np.array([r.as_matrix() for r in smoothed_rots])
    return smoothed_matrices
