from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars import DataFrame

from python_code.ferret_gaze.eye_kinematics.ferret_eyeball_reference_geometry import PUPIL_KEYPOINT_NAMES, \
    FERRET_EYE_RADIUS_MM
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

EYE_SIDE_TO_VIDEO: dict[str, str] = {
    "left": "eye0",  # Note L/R switched from standard Pupil Core convention
    "right": "eye1",
}


def pixels_to_camera_3d(
        points_px: NDArray[np.float64],
        eye_center_px: NDArray[np.float64],
        camera_distance_mm: float,
        px_to_mm_scale:float,
        eyeball_radius_mm: float = FERRET_EYE_RADIUS_MM,
) -> NDArray[np.float64]:
    """
    Convert 2D pixel coordinates to 3D camera-frame coordinates.

    Projects pixel coordinates onto a sphere centered at the camera distance.

    Args:
        points_px: (..., 2) pixel coordinates
        eye_center_px: (2,) eye center in pixels
        px_to_mm_scale: Scale factor from pixels to mm

    Returns:
        (..., 3) coordinates in camera frame (mm)
    """
    original_shape = points_px.shape
    points_flat = points_px.reshape(-1, 2)


    # Center and scale
    centered_px = points_flat - eye_center_px
    centered_mm = centered_px * px_to_mm_scale

    x_cam = centered_mm[:, 0]
    y_cam = centered_mm[:, 1]

    # Project onto sphere
    r_squared = x_cam ** 2 + y_cam ** 2
    r_squared = np.minimum(r_squared, eyeball_radius_mm ** 2 * 0.99)
    z_offset = np.sqrt(eyeball_radius_mm ** 2 - r_squared)

    # 3D position
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
    """
    Extract per-frame keypoint positions from a polars DataFrame.

    Uses vectorized polars operations for efficient extraction.

    Args:
        df: Polars DataFrame with columns: frame, timestamp, keypoint, x, y

    Returns:
        Tuple of (timestamps, pupil_centers, pupil_points, tear_duct, outer_eye)
    """
    # Get all required keypoints
    required_keypoints = set(PUPIL_KEYPOINT_NAMES) | {"tear_duct", "outer_eye"}

    # Filter to only required keypoints
    df_filtered = df.filter(pl.col("keypoint").is_in(required_keypoints))

    # Group by frame and check completeness
    all_frames = df_filtered.group_by("frame").agg(
        pl.col("keypoint").n_unique().alias("n_keypoints"),
        pl.col("timestamp").first().alias("timestamp"),
    )

    if len(all_frames.filter(pl.col("n_keypoints") == len(required_keypoints))) != len(all_frames):
        raise ValueError("All frames must have complete keypoints for processing.")

    frames_numpy = all_frames["frame"].to_numpy()
    n_frames = len(frames_numpy)

    # Filter to complete frames and sort
    df_valid = df_filtered.filter(pl.col("frame").is_in(frames_numpy)).sort(["frame", "keypoint"])

    # Extract timestamps (one per frame)
    timestamps = all_frames.sort("frame")["timestamp"].to_numpy().astype(np.float64)

    # Extract pupil points - pivot to wide format
    pupil_df = df_valid.filter(pl.col("keypoint").is_in(PUPIL_KEYPOINT_NAMES))

    # Create arrays for each keypoint
    pupil_points = np.zeros((n_frames, 8, 2), dtype=np.float64)

    for i, kp_name in enumerate(PUPIL_KEYPOINT_NAMES):
        kp_df = pupil_df.filter(pl.col("keypoint") == kp_name).sort("frame")
        pupil_points[:, i, 0] = kp_df["x"].to_numpy().astype(np.float64)
        pupil_points[:, i, 1] = kp_df["y"].to_numpy().astype(np.float64)

    # Pupil center = centroid of p1-p8
    pupil_centers = np.mean(pupil_points, axis=1)

    # Extract socket landmarks
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

    Eye-centered frame:
        +Z = rest gaze direction
        +X = subject's left (orthogonalized from y_approx)
        +Y = superior (up) via right-hand rule

    Args:
        rest_gaze_camera: (3,) rest gaze unit vector in camera frame
        y_approx_camera: (3,) approximate Y direction in camera frame

    Returns:
        (3, 3) rotation matrix R such that v_eye = R @ v_camera
    """

    #TODO - Change eye reference frame so pupil center is a Z+ (i.e. north pole)
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


def get_camera_centered_positions(df: DataFrame,
                                  eye_camera_distance_mm: float):
    # Extract frame data (vectorized)
    timestamps, pupil_centers_px, pupil_points_px, tear_duct_px, outer_eye_px = \
        extract_frame_data(df)

    n_frames = len(timestamps)

    # Estimate eye parameters from socket landmarks
    mean_tear_duct_px = np.mean(tear_duct_px, axis=0)
    mean_outer_eye_px = np.mean(outer_eye_px, axis=0)
    eye_center_px = (mean_tear_duct_px + mean_outer_eye_px) / 2.0
    tear_duct_to_eye_outer_px = np.linalg.norm(mean_outer_eye_px - mean_tear_duct_px)
    px_to_mm_scale = tear_duct_to_eye_outer_px / FERRET_EYE_RADIUS_MM
    eye_radius_px = tear_duct_to_eye_outer_px / 2.0

    if eye_radius_px < 1.0:
        raise ValueError(f"Eye width too small: {tear_duct_to_eye_outer_px:.1f} pixels")

    # Convert all points to 3D (vectorized)
    pupil_centers_cam = pixels_to_camera_3d(
        points_px=pupil_centers_px,
        camera_distance_mm=eye_camera_distance_mm,
        eye_center_px=eye_center_px,
        px_to_mm_scale=px_to_mm_scale,
    )
    tear_duct_cam = pixels_to_camera_3d(
        points_px=tear_duct_px,
        camera_distance_mm=eye_camera_distance_mm,
        eye_center_px=eye_center_px,
        px_to_mm_scale=px_to_mm_scale,
    )
    outer_eye_cam = pixels_to_camera_3d(
        points_px=outer_eye_px,
        camera_distance_mm=eye_camera_distance_mm,
        eye_center_px=eye_center_px,
        px_to_mm_scale=px_to_mm_scale,
    )

    # Eye centers
    eye_centers_cam = np.zeros((n_frames, 3), dtype=np.float64)
    eye_centers_cam[:, 2] = eye_camera_distance_mm

    # Gaze directions
    gaze_directions_cam = pupil_centers_cam - eye_centers_cam
    gaze_norms = np.linalg.norm(gaze_directions_cam, axis=1, keepdims=True)
    gaze_directions_cam = gaze_directions_cam / gaze_norms
    return eye_centers_cam, gaze_directions_cam, outer_eye_cam, tear_duct_cam, timestamps


def load_eye_trajectories_csv(
        csv_path: Path,
        eye_side: Literal["left", "right"],
        processing_level: str = "cleaned",
) -> pl.DataFrame:
    """
    Load and filter eye tracking CSV data using polars.

    Args:
        csv_path: Path to eye_data.csv
        eye_side: Which eye to load ("left" maps to "eye0", "right" maps to "eye1")
        processing_level: Filter to this processing level

    Returns:
        Filtered polars DataFrame
    """
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
        eye_camera_distance_mm: float) -> tuple[np.ndarray, ...]:
    """
    Process camera-frame eye tracking data into eye-centered kinematics.

    This function:
    1. Computes rest gaze direction (median of all gaze directions)
    2. Builds camera-to-eye rotation using socket landmarks for Y-axis
    3. Transforms gaze to eye frame and computes quaternions
    4. Transforms socket landmarks to eye frame

    Args:
        eye_trajectories_csv_path: Path to eye trajectories CSV
        eye_name: "left_eye" or "right_eye"
        eye_camera_distance_mm: Distance from camera to eye center in mm

    Returns:
        Tuple of (timestamps, quaternions_wxyz, tear_duct_eye, outer_eye_eye,
                  rest_gaze_camera, R_camera_to_eye)
    """
    eye_side: Literal["left", "right"] = "left" if eye_name == "left_eye" else "right"
    df = load_eye_trajectories_csv(csv_path=Path(eye_trajectories_csv_path),
                                   eye_side=eye_side)

    (eye_centers_camera,
     gaze_directions_camera,
     outer_eye_camera,
     tear_duct_camera,
     timestamps) = get_camera_centered_positions(df=df,
                                                 eye_camera_distance_mm=eye_camera_distance_mm)

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


def eye_camera_distance_from_skull_geometry(skull_reference_geometry: ReferenceGeometry,
                                            eye_side: Literal["left", "right"]) -> float:
    """
    Measure distance of `left/right_eye` and `left/right_eye_camera_tip` landmarks to get eye-camera distance in mm.
    :param skull_reference_geometry:
    :param eye_side:
    :return:
    """
    eye_landmark_name = f"{eye_side}_eye"
    eye_camera_landmark_name = f"{eye_side}_cam_tip"
    eye_position = skull_reference_geometry.get_keypoint_position(eye_landmark_name)
    eye_camera_position = skull_reference_geometry.get_keypoint_position(eye_camera_landmark_name)
    distance_mm = np.linalg.norm(eye_camera_position - eye_position)
    return float(distance_mm)
