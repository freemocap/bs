"""
Eye Data Loading
================

This module provides functions for loading and processing eye tracking data
from CSV files. It uses polars for efficient data extraction and converts
2D pixel coordinates to 3D camera-frame coordinates.

Main entry point: `load_ferret_eye_kinematics()` loads a CSV and returns
a FerretEyeKinematics object ready for analysis and visualization.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_pipeline import (
    process_ferret_eye_data,
)

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_model import FerretEyeKinematics

PUPIL_KEYPOINT_NAMES: tuple[str, ...] = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")

EYE_SIDE_TO_VIDEO: dict[str, str] = {
    "left": "eye0",
    "right": "eye1",
}


def load_eye_csv(
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
    frame_counts = df_filtered.group_by("frame").agg(
        pl.col("keypoint").n_unique().alias("n_keypoints"),
        pl.col("timestamp").first().alias("timestamp"),
    )

    # Keep only complete frames
    complete_frames = frame_counts.filter(pl.col("n_keypoints") == len(required_keypoints))

    if len(complete_frames) == 0:
        raise ValueError("No valid frames with all required keypoints")

    valid_frames = complete_frames["frame"].to_numpy()
    n_frames = len(valid_frames)

    # Filter to complete frames and sort
    df_valid = df_filtered.filter(pl.col("frame").is_in(valid_frames)).sort(["frame", "keypoint"])

    # Extract timestamps (one per frame)
    timestamps = complete_frames.sort("frame")["timestamp"].to_numpy().astype(np.float64)

    # Extract pupil points - pivot to wide format
    pupil_df = df_valid.filter(pl.col("keypoint").is_in(PUPIL_KEYPOINT_NAMES))

    # Create arrays for each keypoint
    pupil_points = np.zeros((n_frames, 8, 2), dtype=np.float64)
    pupil_centers = np.zeros((n_frames, 2), dtype=np.float64)

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


def pixels_to_camera_3d(
    points_px: NDArray[np.float64],
    eye_center_px: NDArray[np.float64],
    eye_radius_px: float,
    eye_radius_mm: float,
    camera_distance_mm: float,
) -> NDArray[np.float64]:
    """
    Convert 2D pixel coordinates to 3D camera-frame coordinates.

    Projects pixel coordinates onto a sphere centered at the camera distance.

    Args:
        points_px: (..., 2) pixel coordinates
        eye_center_px: (2,) eye center in pixels
        eye_radius_px: Eye radius in pixels
        eye_radius_mm: Eye radius in mm
        camera_distance_mm: Distance from camera to eye center

    Returns:
        (..., 3) coordinates in camera frame (mm)
    """
    original_shape = points_px.shape
    points_flat = points_px.reshape(-1, 2)

    # Scale factor
    mm_per_px = eye_radius_mm / eye_radius_px

    # Center and scale
    centered_px = points_flat - eye_center_px
    centered_mm = centered_px * mm_per_px

    x_cam = centered_mm[:, 0]
    y_cam = centered_mm[:, 1]

    # Project onto sphere
    r_squared = x_cam ** 2 + y_cam ** 2
    r_squared = np.minimum(r_squared, eye_radius_mm ** 2 * 0.99)
    z_offset = np.sqrt(eye_radius_mm ** 2 - r_squared)

    # 3D position
    points_3d = np.column_stack([
        x_cam,
        y_cam,
        camera_distance_mm - z_offset,
    ])

    return points_3d.reshape(original_shape[:-1] + (3,))


def load_ferret_eye_kinematics(
    eye_trajectories_csv_path: Path,
    eye_side: Literal["left", "right"],
    camera_distance_mm: float,
    eye_radius_mm: float = 3.5,
    pupil_radius_mm: float = 0.5,
    pupil_eccentricity: float = 0.8,
    processing_level: str = "cleaned",
):
    """
    Load eye tracking CSV and compute kinematics.

    This is the main entry point for loading eye data. It:
    1. Loads the CSV and extracts frame data
    2. Estimates eye geometry from socket landmarks
    3. Converts 2D pixels to 3D camera coordinates
    4. Computes quaternion orientations
    5. Returns a FerretEyeKinematics object

    Args:
        eye_trajectories_csv_path: Path to eye_data.csv file
        eye_side: "left" or "right" eye ("left" maps to "eye0", "right" maps to "eye1")
        camera_distance_mm: Distance from camera to eye center in mm
        eye_radius_mm: Eyeball radius in mm (default: 3.5)
        pupil_radius_mm: Pupil ellipse semi-major axis in mm (default: 0.5)
        pupil_eccentricity: Pupil ellipse eccentricity (default: 0.8)
        processing_level: Filter CSV to this processing level (default: "cleaned")

    Returns:
        FerretEyeKinematics object ready for analysis
    """

    # Load data with polars (now filters by eye_side)
    df = load_eye_csv(csv_path=eye_trajectories_csv_path, eye_side=eye_side, processing_level=processing_level)

    # Extract frame data (vectorized)
    timestamps, pupil_centers_px, pupil_points_px, tear_duct_px, outer_eye_px = \
        extract_frame_data(df)

    n_frames = len(timestamps)

    # Estimate eye parameters from socket landmarks
    mean_tear_duct_px = np.mean(tear_duct_px, axis=0)
    mean_outer_eye_px = np.mean(outer_eye_px, axis=0)
    eye_center_px = (mean_tear_duct_px + mean_outer_eye_px) / 2.0
    eye_width_px = np.linalg.norm(mean_outer_eye_px - mean_tear_duct_px)
    eye_radius_px = eye_width_px / 2.0

    if eye_radius_px < 1.0:
        raise ValueError(f"Eye width too small: {eye_width_px:.1f} pixels")

    # Convert all points to 3D (vectorized)
    pupil_centers_cam = pixels_to_camera_3d(
        pupil_centers_px, eye_center_px, eye_radius_px, eye_radius_mm, camera_distance_mm
    )
    pupil_points_cam = pixels_to_camera_3d(
        pupil_points_px, eye_center_px, eye_radius_px, eye_radius_mm, camera_distance_mm
    )
    tear_duct_cam = pixels_to_camera_3d(
        tear_duct_px, eye_center_px, eye_radius_px, eye_radius_mm, camera_distance_mm
    )
    outer_eye_cam = pixels_to_camera_3d(
        outer_eye_px, eye_center_px, eye_radius_px, eye_radius_mm, camera_distance_mm
    )

    # Eye centers
    eye_centers_cam = np.zeros((n_frames, 3), dtype=np.float64)
    eye_centers_cam[:, 2] = camera_distance_mm

    # Gaze directions
    gaze_directions_cam = pupil_centers_cam - eye_centers_cam
    gaze_norms = np.linalg.norm(gaze_directions_cam, axis=1, keepdims=True)
    gaze_directions_cam = gaze_directions_cam / gaze_norms

    # Process to get quaternions and eye-frame coordinates
    (
        timestamps,
        quaternions_wxyz,
        tear_duct_eye,
        outer_eye_eye,
        rest_gaze_camera,
        R_camera_to_eye,
    ) = process_ferret_eye_data(
        name=eye_trajectories_csv_path.stem,
        source_path=str(eye_trajectories_csv_path),
        eye_side=eye_side,
        timestamps=timestamps,
        gaze_directions_camera=gaze_directions_cam,
        pupil_centers_camera=pupil_centers_cam,
        pupil_points_camera=pupil_points_cam,
        eye_centers_camera=eye_centers_cam,
        tear_duct_camera=tear_duct_cam,
        outer_eye_camera=outer_eye_cam,
        eye_radius_mm=eye_radius_mm,
        pupil_radius_mm=pupil_radius_mm,
        pupil_eccentricity=pupil_eccentricity,
    )

    # Create FerretEyeKinematics
    return FerretEyeKinematics.from_pose_data(
        name=eye_trajectories_csv_path.stem,
        source_path=str(eye_trajectories_csv_path),
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


if __name__ == "__main__":
    from python_code.ferret_gaze.eye_kinematics.eyeball_viewer import (
        create_animated_eye_figure,
        create_gaze_timeseries_figure,
    )

    # =========================================================================
    # CONFIGURATION - Edit these values
    # =========================================================================
    eye_trajectories_csv_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_trajectories.csv")
    eye_side: Literal["left", "right"] = "right"
    camera_distance_mm = 50.0
    # =========================================================================

    print(f"Loading eye data from {eye_trajectories_csv_path}...")
    print(f"Eye side: {eye_side} (video: {EYE_SIDE_TO_VIDEO[eye_side]})")
    kinematics = load_ferret_eye_kinematics(
        eye_trajectories_csv_path=eye_trajectories_csv_path,
        eye_side=eye_side,
        camera_distance_mm=camera_distance_mm,
    )

    print(f"Loaded {kinematics.n_frames} frames, {kinematics.duration_seconds:.2f}s duration")
    print(f"Eye side: {kinematics.eye_side}")

    # Create and show visualizations
    print("Creating animated 3D visualization...")
    fig_3d = create_animated_eye_figure(kinematics, frame_step=5)
    fig_3d.show()

    print("Creating gaze timeseries plot...")
    fig_ts = create_gaze_timeseries_figure(kinematics)
    fig_ts.show()