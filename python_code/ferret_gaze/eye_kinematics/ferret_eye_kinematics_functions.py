"""
Ferret Eye Kinematics Functions
===============================

Processing functions for ferret eye tracking data.

Coordinate System (at rest, right-handed):
    +Z = gaze direction (toward pupil) - "north pole"
    +Y = superior (up)
    +X = subject's left (computed via Y × Z)
"""
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame

from python_code.ferret_gaze.eye_kinematics.ferret_eyeball_reference_geometry import (
    PUPIL_KEYPOINT_NAMES,
    FERRET_EYE_RADIUS_MM,
)
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

EYE_SIDE_TO_VIDEO: dict[str, str] = {
    "left": "eye0",
    "right": "eye1",
}


def pixels_to_camera_3d(
    points_px: NDArray[np.float64],
    eye_center_px: NDArray[np.float64],
    camera_distance_mm: float,
    px_to_mm_scale: float,
    eyeball_radius_mm: float = FERRET_EYE_RADIUS_MM,
) -> NDArray[np.float64]:
    """
    Convert 2D pixel coordinates to 3D camera-frame coordinates.

    Projects pixel coordinates onto a sphere centered at the camera distance.

    IMPORTANT - Image coordinate flips:
    - X is flipped: camera sees a mirror image (subject's left is on right of image)
    - Y is flipped: Y increases downward in images, but +Y is "up" in 3D space
    """
    original_shape = points_px.shape
    points_flat = points_px.reshape(-1, 2)

    centered_px = points_flat - eye_center_px
    centered_mm = centered_px * px_to_mm_scale

    # FLIP X: Camera sees mirror image (subject's left appears on right side of image)
    x_cam = -centered_mm[:, 0]
    # FLIP Y: In image coords Y increases downward, but in 3D +Y is up
    y_cam = -centered_mm[:, 1]

    r_squared = x_cam ** 2 + y_cam ** 2
    r_squared = np.minimum(r_squared, eyeball_radius_mm ** 2 * 0.99)
    z_offset = np.sqrt(eyeball_radius_mm ** 2 - r_squared)

    points_3d = np.column_stack([
        x_cam,
        y_cam,
        camera_distance_mm - z_offset,
    ])

    return points_3d.reshape(original_shape[:-1] + (3,))


def extract_frame_data(
    df: pl.DataFrame,
) -> tuple[
    NDArray[np.float64],  # timestamps
    NDArray[np.float64],  # pupil_centers_px (N, 2)
    NDArray[np.float64],  # pupil_points_px (N, 8, 2)
    NDArray[np.float64],  # tear_duct_px (N, 2)
    NDArray[np.float64],  # outer_eye_px (N, 2)
]:
    """Extract per-frame keypoint positions from a polars DataFrame."""
    required_keypoints = set(PUPIL_KEYPOINT_NAMES) | {"tear_duct", "outer_eye"}
    df_filtered = df.filter(pl.col("keypoint").is_in(required_keypoints))

    all_frames = df_filtered.group_by("frame").agg(
        pl.col("keypoint").n_unique().alias("n_keypoints"),
        pl.col("timestamp").first().alias("timestamp"),
    )

    if len(all_frames.filter(pl.col("n_keypoints") == len(required_keypoints))) != len(all_frames):
        raise ValueError("All frames must have complete keypoints for processing.")

    frames_numpy = all_frames["frame"].to_numpy()
    n_frames = len(frames_numpy)

    df_valid = df_filtered.filter(pl.col("frame").is_in(frames_numpy)).sort(["frame", "keypoint"])
    timestamps = all_frames.sort("frame")["timestamp"].to_numpy().astype(np.float64)

    pupil_df = df_valid.filter(pl.col("keypoint").is_in(PUPIL_KEYPOINT_NAMES))
    pupil_points = np.zeros((n_frames, 8, 2), dtype=np.float64)

    for i, kp_name in enumerate(PUPIL_KEYPOINT_NAMES):
        kp_df = pupil_df.filter(pl.col("keypoint") == kp_name).sort("frame")
        pupil_points[:, i, 0] = kp_df["x"].to_numpy().astype(np.float64)
        pupil_points[:, i, 1] = kp_df["y"].to_numpy().astype(np.float64)

    pupil_centers = np.mean(pupil_points, axis=1)

    tear_duct_df = df_valid.filter(pl.col("keypoint") == "tear_duct").sort("frame")
    outer_eye_df = df_valid.filter(pl.col("keypoint") == "outer_eye").sort("frame")

    tear_duct = np.column_stack([
        tear_duct_df["x"].to_numpy().astype(np.float64),
        tear_duct_df["y"].to_numpy().astype(np.float64),
    ])
    outer_eye = np.column_stack([
        outer_eye_df["x"].to_numpy().astype(np.float64),
        outer_eye_df["y"].to_numpy().astype(np.float64),
    ])

    return timestamps, pupil_centers, pupil_points, tear_duct, outer_eye


def compute_gaze_quaternions(
    gaze_directions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute quaternions that rotate [0, 0, 1] to each gaze direction.

    The eye frame has gaze along +Z (north pole), so at rest the gaze vector is [0, 0, 1].
    This function computes the quaternion that rotates from rest to the actual gaze.

    Uses the half-angle formula for rotation between two vectors:
        q = [cos(θ/2), sin(θ/2) * axis]
    where axis = normalize(source × target) and θ = angle between vectors.

    For source = [0, 0, 1]:
        axis = normalize([0, 0, 1] × target) = normalize([-target_y, target_x, 0])
        cos(θ) = target_z

    Args:
        gaze_directions: (N, 3) array of unit gaze vectors

    Returns:
        (N, 4) array of quaternions [w, x, y, z]
    """
    n = len(gaze_directions)
    cos_theta = gaze_directions[:, 2]  # dot product with [0, 0, 1]
    quaternions = np.zeros((n, 4), dtype=np.float64)

    # Case 1: Nearly parallel (cos_theta > 0.999999) -> identity quaternion
    parallel_mask = cos_theta > 0.999999
    quaternions[parallel_mask, 0] = 1.0

    # Case 2: Nearly anti-parallel -> 180° rotation around X or Y axis
    antiparallel_mask = cos_theta < -0.999999
    quaternions[antiparallel_mask, 0] = 0.0
    quaternions[antiparallel_mask, 1] = 1.0  # Rotate 180° around X

    # Case 3: General case
    general_mask = ~parallel_mask & ~antiparallel_mask

    if np.any(general_mask):
        gaze_gen = gaze_directions[general_mask]
        cos_gen = cos_theta[general_mask]

        # Cross product: [0, 0, 1] × gaze = [-gaze_y, gaze_x, 0]
        axis_x = -gaze_gen[:, 1]
        axis_y = gaze_gen[:, 0]

        # sin(θ) = |[0,0,1] × gaze| = sqrt(gx² + gy²)
        sin_theta = np.sqrt(axis_x ** 2 + axis_y ** 2)

        # Half-angle formulas
        cos_half = np.sqrt((1.0 + cos_gen) / 2.0)
        sin_half = np.sqrt((1.0 - cos_gen) / 2.0)

        # Quaternion components
        scale = np.where(sin_theta > 1e-10, sin_half / sin_theta, 0.0)

        quaternions[general_mask, 0] = cos_half
        quaternions[general_mask, 1] = axis_x * scale
        quaternions[general_mask, 2] = axis_y * scale
        quaternions[general_mask, 3] = 0.0  # axis_z is always 0

    # Normalize
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / np.maximum(norms, 1e-10)

    return quaternions


def compute_camera_to_eye_rotation(
    rest_gaze_camera: NDArray[np.float64],
    y_approx_camera: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Build rotation matrix from camera frame to eye-centered frame.

    Eye-centered frame (Z+ = gaze convention):
        +Z = rest gaze direction (anterior, toward pupil)
        +Y = superior (up), orthogonalized from y_approx
        +X = subject's left, computed via Y × Z for right-handed system

    Args:
        rest_gaze_camera: (3,) rest gaze unit vector in camera frame
        y_approx_camera: (3,) approximate Y direction (up) in camera frame

    Returns:
        (3, 3) rotation matrix R such that v_eye = R @ v_camera
    """
    # Z = exact gaze direction
    z_camera = rest_gaze_camera / np.linalg.norm(rest_gaze_camera)

    # Y = orthogonalize y_approx against Z (Gram-Schmidt)
    y_camera = y_approx_camera - np.dot(y_approx_camera, z_camera) * z_camera
    y_norm = np.linalg.norm(y_camera)
    if y_norm < 1e-10:
        raise ValueError("y_approx_camera is parallel to rest_gaze_camera.")
    y_camera = y_camera / y_norm

    # X = Y × Z (right-hand rule) → points to subject's left
    x_camera = np.cross(y_camera, z_camera)
    x_camera = x_camera / np.linalg.norm(x_camera)

    # R takes camera coords to eye coords
    # Rows are the basis vectors expressed in camera frame
    return np.vstack([x_camera, y_camera, z_camera])


def transform_points_batch(
    points_camera: NDArray[np.float64],
    R_camera_to_eye: NDArray[np.float64],
    origin_camera: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform points from camera frame to eye-centered frame."""
    centered = points_camera - origin_camera

    if centered.ndim == 2:
        return (R_camera_to_eye @ centered.T).T
    else:
        original_shape = centered.shape
        flat = centered.reshape(-1, 3)
        transformed = (R_camera_to_eye @ flat.T).T
        return transformed.reshape(original_shape)


def pixels_to_tangent_plane_3d(
    points_px: NDArray[np.float64],
    eye_center_px: NDArray[np.float64],
    camera_distance_mm: float,
    px_to_mm_scale: float,
    eyeball_radius_mm: float = FERRET_EYE_RADIUS_MM,
) -> NDArray[np.float64]:
    """
    Convert 2D pixel coordinates to 3D camera-frame coordinates on the tangent plane.

    Projects pixel coordinates onto a plane at the front of the eye sphere
    (at Z = camera_distance - eyeball_radius), NOT onto the sphere surface.
    This is appropriate for socket landmarks (tear_duct, outer_eye) which are
    on the eyelids/socket, not on the eyeball itself.

    IMPORTANT - Image coordinate flips:
    - X is flipped: camera sees a mirror image (subject's left is on right of image)
    - Y is flipped: Y increases downward in images, but +Y is "up" in 3D space
    """
    original_shape = points_px.shape
    points_flat = points_px.reshape(-1, 2)

    centered_px = points_flat - eye_center_px
    centered_mm = centered_px * px_to_mm_scale

    # FLIP X: Camera sees mirror image (subject's left appears on right side of image)
    x_cam = -centered_mm[:, 0]
    # FLIP Y: In image coords Y increases downward, but in 3D +Y is up
    y_cam = -centered_mm[:, 1]

    # Place on tangent plane at front of eye (Z = camera_distance - R)
    z_cam = np.full_like(x_cam, camera_distance_mm - eyeball_radius_mm)

    points_3d = np.column_stack([x_cam, y_cam, z_cam])
    return points_3d.reshape(original_shape[:-1] + (3,))


def get_camera_centered_positions(
    df: DataFrame,
    eye_camera_distance_mm: float,
) -> tuple[
    NDArray[np.float64],  # eye_centers_cam
    NDArray[np.float64],  # gaze_directions_cam
    NDArray[np.float64],  # pupil_center_cam (N, 3)
    NDArray[np.float64],  # pupil_points_cam (N, 8, 3)
    NDArray[np.float64],  # outer_eye_cam
    NDArray[np.float64],  # tear_duct_cam
    NDArray[np.float64],  # timestamps
]:
    """Extract 3D positions in camera frame from 2D pixel data."""
    timestamps, pupil_centers_px, pupil_points_px, tear_duct_px, outer_eye_px = extract_frame_data(df)

    n_frames = len(timestamps)

    mean_tear_duct_px = np.mean(tear_duct_px, axis=0)
    mean_outer_eye_px = np.mean(outer_eye_px, axis=0)
    eye_center_px = (mean_tear_duct_px + mean_outer_eye_px) / 2.0
    tear_duct_to_eye_outer_px = np.linalg.norm(mean_outer_eye_px - mean_tear_duct_px)

    # Convert pixels to mm: tear_duct_to_outer_eye distance ≈ eye diameter (2 * radius)
    # px_to_mm_scale has units mm/px, so we need: physical_distance_mm / pixel_distance_px
    eye_width_mm = 2 * FERRET_EYE_RADIUS_MM
    px_to_mm_scale = eye_width_mm / tear_duct_to_eye_outer_px

    eye_radius_px = tear_duct_to_eye_outer_px / 2.0

    if eye_radius_px < 1.0:
        raise ValueError(f"Eye width too small: {tear_duct_to_eye_outer_px:.1f} pixels")

    # Pupil center is ON the eyeball surface - project onto sphere
    pupil_centers_cam = pixels_to_camera_3d(
        points_px=pupil_centers_px,
        camera_distance_mm=eye_camera_distance_mm,
        eye_center_px=eye_center_px,
        px_to_mm_scale=px_to_mm_scale,
    )

    # Pupil boundary points p1-p8 are ALSO on the eyeball surface - project onto sphere
    # pupil_points_px is (N, 8, 2), we need to reshape for pixels_to_camera_3d
    pupil_points_cam = pixels_to_camera_3d(
        points_px=pupil_points_px,
        camera_distance_mm=eye_camera_distance_mm,
        eye_center_px=eye_center_px,
        px_to_mm_scale=px_to_mm_scale,
    )  # Returns (N, 8, 3)

    # Socket landmarks (tear_duct, outer_eye) are NOT on the eyeball - they're on the
    # eyelids/socket at approximately the tangent plane at the front of the eye
    tear_duct_cam = pixels_to_tangent_plane_3d(
        points_px=tear_duct_px,
        camera_distance_mm=eye_camera_distance_mm,
        eye_center_px=eye_center_px,
        px_to_mm_scale=px_to_mm_scale,
    )
    outer_eye_cam = pixels_to_tangent_plane_3d(
        points_px=outer_eye_px,
        camera_distance_mm=eye_camera_distance_mm,
        eye_center_px=eye_center_px,
        px_to_mm_scale=px_to_mm_scale,
    )

    eye_centers_cam = np.zeros((n_frames, 3), dtype=np.float64)
    eye_centers_cam[:, 2] = eye_camera_distance_mm

    gaze_directions_cam = pupil_centers_cam - eye_centers_cam
    gaze_norms = np.linalg.norm(gaze_directions_cam, axis=1, keepdims=True)
    gaze_directions_cam = gaze_directions_cam / gaze_norms

    return (
        eye_centers_cam,
        gaze_directions_cam,
        pupil_centers_cam,
        pupil_points_cam,
        outer_eye_cam,
        tear_duct_cam,
        timestamps,
    )


def load_eye_trajectories_csv(
    csv_path: Path,
    eye_side: Literal["left", "right"],
    processing_level: str = "cleaned",
) -> pl.DataFrame:
    """Load and filter eye tracking CSV data using polars."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Eye data CSV not found: {csv_path}")

    video_name = EYE_SIDE_TO_VIDEO[eye_side]

    df = pl.read_csv(csv_path)
    df = df.filter(
        (pl.col("processing_level") == processing_level) &
        (pl.col("video") == video_name)
    )

    if len(df) == 0:
        raise ValueError(
            f"No data with processing_level='{processing_level}' and video='{video_name}' "
            f"(eye_side='{eye_side}') in {csv_path}"
        )

    return df


def process_ferret_eye_data(
    eye_trajectories_csv_path: str | Path,
    eye_name: Literal["left_eye", "right_eye"],
    eye_camera_distance_mm: float,
) -> tuple[
    NDArray[np.float64],  # timestamps
    NDArray[np.float64],  # quaternions_wxyz
    NDArray[np.float64],  # pupil_center_eye (N, 3)
    NDArray[np.float64],  # pupil_points_eye (N, 8, 3)
    NDArray[np.float64],  # tear_duct_eye
    NDArray[np.float64],  # outer_eye_eye
    NDArray[np.float64],  # rest_gaze_camera
    NDArray[np.float64],  # R_camera_to_eye
]:
    """
    Process camera-frame eye tracking data into eye-centered kinematics.

    Eye frame convention (Z+ = gaze):
        +Z = rest gaze direction (anterior)
        +Y = superior (up)
        +X = subject's left (computed via Gram-Schmidt)
    """
    eye_side: Literal["left", "right"] = "left" if eye_name == "left_eye" else "right"
    df = load_eye_trajectories_csv(
        csv_path=Path(eye_trajectories_csv_path),
        eye_side=eye_side,
    )

    (eye_centers_camera,
     gaze_directions_camera,
     pupil_centers_camera,
     pupil_points_camera,
     outer_eye_camera,
     tear_duct_camera,
     timestamps) = get_camera_centered_positions(
        df=df,
        eye_camera_distance_mm=eye_camera_distance_mm,
    )

    # Compute rest gaze direction (median)
    rest_gaze_camera = np.median(gaze_directions_camera, axis=0)
    rest_gaze_camera = rest_gaze_camera / np.linalg.norm(rest_gaze_camera)

    # Compute Y-axis direction (up) from socket landmarks
    mean_tear_duct = np.mean(tear_duct_camera, axis=0)
    mean_outer_eye = np.mean(outer_eye_camera, axis=0)

    # Eye opening direction
    eye_opening_dir = mean_outer_eye - mean_tear_duct
    if eye_side == "right":
        eye_opening_dir = -eye_opening_dir

    # Y_approx = gaze × eye_opening (perpendicular to both)
    y_approx_camera = np.cross(rest_gaze_camera, eye_opening_dir)
    y_norm = np.linalg.norm(y_approx_camera)
    if y_norm < 1e-10:
        y_approx_camera = np.array([0.0, -1.0, 0.0])
    else:
        y_approx_camera = y_approx_camera / y_norm

    # Build rotation matrix
    R_camera_to_eye = compute_camera_to_eye_rotation(rest_gaze_camera, y_approx_camera)

    # Transform gaze directions to eye frame
    gaze_directions_eye = (R_camera_to_eye @ gaze_directions_camera.T).T
    gaze_norms = np.linalg.norm(gaze_directions_eye, axis=1, keepdims=True)
    gaze_directions_eye = gaze_directions_eye / gaze_norms

    # Compute quaternions (rotating [0, 0, 1] to gaze)
    quaternions_wxyz = compute_gaze_quaternions(gaze_directions_eye)

    # Transform all points to eye frame
    mean_eye_center_camera = np.mean(eye_centers_camera, axis=0)

    # Pupil center (N, 3)
    pupil_center_eye = transform_points_batch(
        pupil_centers_camera, R_camera_to_eye, mean_eye_center_camera
    )

    # Pupil boundary points (N, 8, 3)
    pupil_points_eye = transform_points_batch(
        pupil_points_camera, R_camera_to_eye, mean_eye_center_camera
    )

    # Socket landmarks
    tear_duct_eye = transform_points_batch(
        tear_duct_camera, R_camera_to_eye, mean_eye_center_camera
    )
    outer_eye_eye = transform_points_batch(
        outer_eye_camera, R_camera_to_eye, mean_eye_center_camera
    )

    return (
        timestamps,
        quaternions_wxyz,
        pupil_center_eye,
        pupil_points_eye,
        tear_duct_eye,
        outer_eye_eye,
        rest_gaze_camera,
        R_camera_to_eye,
    )


def eye_camera_distance_from_skull_geometry(
    skull_reference_geometry: ReferenceGeometry,
    eye_side: Literal["left", "right"],
) -> float:
    """Get eye-camera distance from skull geometry landmarks."""
    eye_landmark_name = f"{eye_side}_eye"
    eye_camera_landmark_name = f"{eye_side}_cam_tip"
    eye_position = skull_reference_geometry.get_keypoint_position(eye_landmark_name)
    eye_camera_position = skull_reference_geometry.get_keypoint_position(eye_camera_landmark_name)
    distance_mm = np.linalg.norm(eye_camera_position - eye_position)
    return float(distance_mm)