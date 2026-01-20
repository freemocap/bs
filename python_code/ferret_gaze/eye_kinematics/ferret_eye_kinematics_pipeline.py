"""
Ferret Eye Kinematics Pipeline
==============================

This module provides the processing pipeline for converting camera-frame
eye tracking data into eye-centered kinematics with quaternion orientations.

The pipeline:
1. Computes rest gaze direction from median gaze
2. Builds camera-to-eye rotation matrix using socket landmarks
3. Transforms gaze directions to eye-centered frame
4. Computes quaternions representing eye orientation
5. Transforms socket landmarks to eye-centered frame
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def compute_gaze_quaternions(
    gaze_directions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute quaternions that rotate [1, 0, 0] to each gaze direction.

    Uses the half-angle formula for rotation between two vectors:
        q = [cos(θ/2), sin(θ/2) * axis]

    where axis = normalize(source × target) and θ = angle between vectors.

    For the special case of source = [1, 0, 0]:
        axis = normalize([0, -target_z, target_y])  (simplified cross product)
        cos(θ) = target_x  (dot product with [1,0,0])

    Args:
        gaze_directions: (N, 3) array of unit gaze vectors

    Returns:
        (N, 4) array of quaternions [w, x, y, z]
    """
    n = len(gaze_directions)

    # Rest gaze is [1, 0, 0]
    # dot product with [1, 0, 0] = gaze_x
    cos_theta = gaze_directions[:, 0]  # (N,)

    # Handle edge cases
    quaternions = np.zeros((n, 4), dtype=np.float64)

    # Case 1: Nearly parallel (cos_theta > 0.999999) -> identity quaternion
    parallel_mask = cos_theta > 0.999999
    quaternions[parallel_mask, 0] = 1.0  # w = 1, xyz = 0

    # Case 2: Nearly anti-parallel (cos_theta < -0.999999) -> 180° rotation
    antiparallel_mask = cos_theta < -0.999999
    # Rotate 180° around Y axis (or Z, both work)
    quaternions[antiparallel_mask, 0] = 0.0  # w = cos(90°) = 0
    quaternions[antiparallel_mask, 2] = 1.0  # y = sin(90°) * 1 = 1

    # Case 3: General case
    general_mask = ~parallel_mask & ~antiparallel_mask

    if np.any(general_mask):
        gaze_gen = gaze_directions[general_mask]
        cos_gen = cos_theta[general_mask]

        # Cross product: [1,0,0] × gaze = [0, -gaze_z, gaze_y]
        # This gives the rotation axis (unnormalized)
        axis_x = np.zeros(len(gaze_gen), dtype=np.float64)
        axis_y = -gaze_gen[:, 2]
        axis_z = gaze_gen[:, 1]

        # sin(θ) = |source × target| = sqrt(gaze_y² + gaze_z²)
        sin_theta = np.sqrt(axis_y ** 2 + axis_z ** 2)

        # Half-angle formulas:
        # cos(θ/2) = sqrt((1 + cos_theta) / 2)
        # sin(θ/2) = sqrt((1 - cos_theta) / 2)
        cos_half = np.sqrt((1.0 + cos_gen) / 2.0)
        sin_half = np.sqrt((1.0 - cos_gen) / 2.0)

        # Quaternion: [cos(θ/2), sin(θ/2) * axis_normalized]
        # axis_normalized = [0, -gaze_z, gaze_y] / sin_theta
        # So sin(θ/2) * axis_normalized = sin(θ/2) / sin_theta * [0, -gaze_z, gaze_y]

        # Avoid division by zero (sin_theta = 0 only when gaze = [1,0,0] which we handled)
        scale = np.where(sin_theta > 1e-10, sin_half / sin_theta, 0.0)

        quaternions[general_mask, 0] = cos_half
        quaternions[general_mask, 1] = axis_x * scale  # always 0
        quaternions[general_mask, 2] = axis_y * scale
        quaternions[general_mask, 3] = axis_z * scale

    # Normalize (should already be unit, but ensure numerical stability)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / np.maximum(norms, 1e-10)

    return quaternions


def compute_camera_to_eye_rotation(
    rest_gaze_camera: NDArray[np.float64],
    y_approx_camera: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Build rotation matrix from camera frame to eye-centered frame.

    Eye frame:
        +X = rest gaze direction
        +Y = subject's left (orthogonalized from y_approx)
        +Z = superior (up) via right-hand rule

    Args:
        rest_gaze_camera: (3,) rest gaze unit vector in camera frame
        y_approx_camera: (3,) approximate Y direction in camera frame

    Returns:
        (3, 3) rotation matrix R such that v_eye = R @ v_camera
    """
    # X = exact gaze direction
    x_camera = rest_gaze_camera / np.linalg.norm(rest_gaze_camera)

    # Y = orthogonalize y_approx against X
    y_camera = y_approx_camera - np.dot(y_approx_camera, x_camera) * x_camera
    y_camera = y_camera / np.linalg.norm(y_camera)

    # Z = X × Y (right-hand rule)
    z_camera = np.cross(x_camera, y_camera)
    z_camera = z_camera / np.linalg.norm(z_camera)

    # R takes camera coords to eye coords
    # Columns of R^T are x_camera, y_camera, z_camera
    # So R = [x_camera, y_camera, z_camera]^T
    return np.vstack([x_camera, y_camera, z_camera])


def transform_points_batch(
    points_camera: NDArray[np.float64],
    R_camera_to_eye: NDArray[np.float64],
    origin_camera: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform points from camera frame to eye-centered frame (vectorized).

    Args:
        points_camera: (N, 3) or (N, M, 3) points in camera frame
        R_camera_to_eye: (3, 3) rotation matrix
        origin_camera: (3,) origin position in camera frame

    Returns:
        Points in eye-centered frame (same shape as input)
    """
    # Subtract origin, then rotate
    centered = points_camera - origin_camera

    if centered.ndim == 2:
        # (N, 3) case
        return (R_camera_to_eye @ centered.T).T
    else:
        # (N, M, 3) case - reshape, transform, reshape back
        original_shape = centered.shape
        flat = centered.reshape(-1, 3)
        transformed = (R_camera_to_eye @ flat.T).T
        return transformed.reshape(original_shape)


def process_ferret_eye_data(
    name: str,
    source_path: str,
    eye_side: Literal["left", "right"],
    timestamps: NDArray[np.float64],
    gaze_directions_camera: NDArray[np.float64],
    pupil_centers_camera: NDArray[np.float64],
    pupil_points_camera: NDArray[np.float64],
    eye_centers_camera: NDArray[np.float64],
    tear_duct_camera: NDArray[np.float64],
    outer_eye_camera: NDArray[np.float64],
    eye_radius_mm: float = 3.5,
    pupil_radius_mm: float = 0.5,
    pupil_eccentricity: float = 0.8,
) -> tuple[
    NDArray[np.float64],  # timestamps
    NDArray[np.float64],  # quaternions_wxyz (N, 4)
    NDArray[np.float64],  # tear_duct_eye (N, 3)
    NDArray[np.float64],  # outer_eye_eye (N, 3)
    NDArray[np.float64],  # rest_gaze_camera (3,)
    NDArray[np.float64],  # R_camera_to_eye (3, 3)
]:
    """
    Process camera-frame eye tracking data into eye-centered kinematics.

    This function:
    1. Computes rest gaze direction (median of all gaze directions)
    2. Builds camera-to-eye rotation using socket landmarks for Y-axis
    3. Transforms gaze to eye frame and computes quaternions
    4. Transforms socket landmarks to eye frame

    Args:
        name: Identifier for this recording
        source_path: Path to source data
        eye_side: "left" or "right"
        timestamps: (N,) timestamps in seconds
        gaze_directions_camera: (N, 3) gaze unit vectors in camera frame
        pupil_centers_camera: (N, 3) pupil center positions in camera frame
        pupil_points_camera: (N, 8, 3) pupil boundary points in camera frame
        eye_centers_camera: (N, 3) eyeball center positions in camera frame
        tear_duct_camera: (N, 3) tear duct positions in camera frame
        outer_eye_camera: (N, 3) outer eye positions in camera frame
        eye_radius_mm: Eyeball radius
        pupil_radius_mm: Pupil ellipse semi-major axis
        pupil_eccentricity: Pupil ellipse eccentricity

    Returns:
        Tuple of (timestamps, quaternions_wxyz, tear_duct_eye, outer_eye_eye,
                  rest_gaze_camera, R_camera_to_eye)
    """
    # Step 1: Compute rest gaze direction (median)
    rest_gaze_camera = np.median(gaze_directions_camera, axis=0)
    rest_gaze_camera = rest_gaze_camera / np.linalg.norm(rest_gaze_camera)

    # Step 2: Compute Y-axis direction from socket landmarks
    mean_tear_duct = np.mean(tear_duct_camera, axis=0)
    mean_outer_eye = np.mean(outer_eye_camera, axis=0)

    if eye_side == "right":
        y_approx_camera = mean_tear_duct - mean_outer_eye
    else:
        y_approx_camera = mean_outer_eye - mean_tear_duct

    # Step 3: Build rotation matrix
    R_camera_to_eye = compute_camera_to_eye_rotation(rest_gaze_camera, y_approx_camera)

    # Step 4: Transform gaze directions to eye frame
    gaze_directions_eye = (R_camera_to_eye @ gaze_directions_camera.T).T

    # Normalize
    gaze_norms = np.linalg.norm(gaze_directions_eye, axis=1, keepdims=True)
    gaze_directions_eye = gaze_directions_eye / gaze_norms

    # Step 5: Compute quaternions
    quaternions_wxyz = compute_gaze_quaternions(gaze_directions_eye)

    # Step 6: Transform socket landmarks
    mean_eye_center_camera = np.mean(eye_centers_camera, axis=0)

    tear_duct_eye = transform_points_batch(
        tear_duct_camera, R_camera_to_eye, mean_eye_center_camera
    )
    outer_eye_eye = transform_points_batch(
        outer_eye_camera, R_camera_to_eye, mean_eye_center_camera
    )

    return (
        timestamps,
        quaternions_wxyz,
        tear_duct_eye,
        outer_eye_eye,
        rest_gaze_camera,
        R_camera_to_eye,
    )


def create_eye_kinematics(
    name: str,
    source_path: str,
    eye_side: Literal["left", "right"],
    timestamps: NDArray[np.float64],
    gaze_directions_camera: NDArray[np.float64],
    pupil_centers_camera: NDArray[np.float64],
    pupil_points_camera: NDArray[np.float64],
    eye_centers_camera: NDArray[np.float64],
    tear_duct_camera: NDArray[np.float64],
    outer_eye_camera: NDArray[np.float64],
    eye_radius_mm: float = 3.5,
    pupil_radius_mm: float = 0.5,
    pupil_eccentricity: float = 0.8,
):
    """
    Create FerretEyeKinematics from camera-frame tracking data.

    This is a convenience function that processes the data and constructs
    the FerretEyeKinematics object.

    Args:
        name: Identifier for this recording
        source_path: Path to source data
        eye_side: "left" or "right"
        timestamps: (N,) timestamps in seconds
        gaze_directions_camera: (N, 3) gaze unit vectors in camera frame
        pupil_centers_camera: (N, 3) pupil center positions in camera frame
        pupil_points_camera: (N, 8, 3) pupil boundary points in camera frame
        eye_centers_camera: (N, 3) eyeball center positions in camera frame
        tear_duct_camera: (N, 3) tear duct positions in camera frame
        outer_eye_camera: (N, 3) outer eye positions in camera frame
        eye_radius_mm: Eyeball radius
        pupil_radius_mm: Pupil ellipse semi-major axis
        pupil_eccentricity: Pupil ellipse eccentricity

    Returns:
        FerretEyeKinematics instance
    """
    from python_code.ferret_gaze.eye_kinematics.eye_kinematics_model import FerretEyeKinematics

    # Process data
    (
        timestamps,
        quaternions_wxyz,
        tear_duct_eye,
        outer_eye_eye,
        rest_gaze_camera,
        R_camera_to_eye,
    ) = process_ferret_eye_data(
        name=name,
        source_path=source_path,
        eye_side=eye_side,
        timestamps=timestamps,
        gaze_directions_camera=gaze_directions_camera,
        pupil_centers_camera=pupil_centers_camera,
        pupil_points_camera=pupil_points_camera,
        eye_centers_camera=eye_centers_camera,
        tear_duct_camera=tear_duct_camera,
        outer_eye_camera=outer_eye_camera,
        eye_radius_mm=eye_radius_mm,
        pupil_radius_mm=pupil_radius_mm,
        pupil_eccentricity=pupil_eccentricity,
    )

    # Create FerretEyeKinematics
    return FerretEyeKinematics.from_pose_data(
        name=name,
        source_path=source_path,
        eye_side=eye_side,
        timestamps=timestamps,
        quaternions_wxyz=quaternions_wxyz,
        tear_duct_mm=tear_duct_eye,
        outer_eye_mm=outer_eye_eye,
        rest_gaze_direction_camera=rest_gaze_camera,
        camera_to_eye_rotation=R_camera_to_eye,
        eye_radius_mm=eye_radius_mm,
        pupil_radius_mm=pupil_radius_mm,
        pupil_eccentricity=pupil_eccentricity,
    )