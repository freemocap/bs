"""
Load Eye Data from CSV and Compute FerretEyeKinematics

Converts 2D pixel-space eye tracking data to 3D eye-centered kinematics
using a pinhole camera model and spherical eye geometry.

Camera distance is computed from skull reference geometry (eye to camera tip distance).
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_model import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_pipeline import process_ferret_eye_data
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry


PUPIL_KEYPOINT_NAMES: tuple[str, ...] = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")


def load_skull_reference_geometry(json_path: Path) -> ReferenceGeometry:
    """Load skull reference geometry from JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Skull reference geometry not found: {json_path}")
    return ReferenceGeometry.from_json_file(json_path)


def compute_camera_distances_from_skull(
    skull_geometry: ReferenceGeometry,
) -> tuple[float, float]:
    """
    Compute camera distances for left and right eyes from skull geometry.

    The camera distance is the distance from the eye marker to the camera tip marker.

    Args:
        skull_geometry: Skull reference geometry with eye and camera markers

    Returns:
        (left_camera_distance_mm, right_camera_distance_mm)
    """
    markers = skull_geometry.get_marker_positions()

    required_markers = ["left_eye", "right_eye", "left_cam_tip", "right_cam_tip"]
    missing = [m for m in required_markers if m not in markers]
    if missing:
        raise ValueError(
            f"Skull geometry missing required markers: {missing}. "
            f"Available: {sorted(markers.keys())}"
        )

    left_eye = markers["left_eye"]
    right_eye = markers["right_eye"]
    left_cam_tip = markers["left_cam_tip"]
    right_cam_tip = markers["right_cam_tip"]

    left_distance = float(np.linalg.norm(left_cam_tip - left_eye))
    right_distance = float(np.linalg.norm(right_cam_tip - right_eye))

    return left_distance, right_distance


def load_eye_csv(
    csv_path: Path,
    processing_level: str = "cleaned",
) -> pd.DataFrame:
    """
    Load and filter eye tracking CSV data.

    Args:
        csv_path: Path to eye_data.csv
        processing_level: Filter to this processing level

    Returns:
        Filtered DataFrame with columns: frame, timestamp, keypoint, x, y
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Eye data CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["processing_level"] == processing_level]

    if len(df) == 0:
        raise ValueError(f"No data with processing_level='{processing_level}' in {csv_path}")

    return df


def extract_frame_data(
    df: pd.DataFrame,
) -> tuple[
    NDArray[np.float64],  # timestamps
    NDArray[np.float64],  # pupil_centers_px (N, 2)
    NDArray[np.float64],  # pupil_points_px (N, 8, 2)
    NDArray[np.float64],  # tear_duct_px (N, 2)
    NDArray[np.float64],  # outer_eye_px (N, 2)
]:
    """
    Extract per-frame keypoint positions from DataFrame.

    Returns only frames where all required keypoints are present.
    """
    frames = sorted(df["frame"].unique())

    timestamps: list[float] = []
    pupil_centers: list[NDArray[np.float64]] = []
    pupil_points: list[NDArray[np.float64]] = []
    tear_ducts: list[NDArray[np.float64]] = []
    outer_eyes: list[NDArray[np.float64]] = []

    for frame in frames:
        frame_df = df[df["frame"] == frame]
        timestamp = float(frame_df["timestamp"].iloc[0])

        # Get pupil keypoints
        pupil_df = frame_df[frame_df["keypoint"].isin(PUPIL_KEYPOINT_NAMES)]
        if len(pupil_df) < len(PUPIL_KEYPOINT_NAMES):
            continue

        # Get socket landmarks
        tear_duct_df = frame_df[frame_df["keypoint"] == "tear_duct"]
        outer_eye_df = frame_df[frame_df["keypoint"] == "outer_eye"]
        if len(tear_duct_df) == 0 or len(outer_eye_df) == 0:
            continue

        # Extract pupil points in order p1-p8
        frame_pupil_points = np.zeros((8, 2), dtype=np.float64)
        for i, name in enumerate(PUPIL_KEYPOINT_NAMES):
            row = pupil_df[pupil_df["keypoint"] == name].iloc[0]
            frame_pupil_points[i] = [row["x"], row["y"]]

        # Pupil center = centroid of p1-p8
        pupil_center = np.mean(frame_pupil_points, axis=0)

        # Socket landmarks
        tear_duct = np.array([
            tear_duct_df["x"].iloc[0],
            tear_duct_df["y"].iloc[0],
        ], dtype=np.float64)
        outer_eye = np.array([
            outer_eye_df["x"].iloc[0],
            outer_eye_df["y"].iloc[0],
        ], dtype=np.float64)

        timestamps.append(timestamp)
        pupil_centers.append(pupil_center)
        pupil_points.append(frame_pupil_points)
        tear_ducts.append(tear_duct)
        outer_eyes.append(outer_eye)

    if len(timestamps) == 0:
        raise ValueError("No valid frames with all required keypoints")

    return (
        np.array(timestamps, dtype=np.float64),
        np.stack(pupil_centers),
        np.stack(pupil_points),
        np.stack(tear_ducts),
        np.stack(outer_eyes),
    )


def pixels_to_camera_3d(
    points_px: NDArray[np.float64],
    eye_center_px: NDArray[np.float64],
    eye_radius_px: float,
    eye_radius_mm: float,
    camera_distance_mm: float,
) -> NDArray[np.float64]:
    """
    Convert 2D pixel coordinates to 3D camera-frame coordinates.

    Uses a simple model:
    - Eye center is at (0, 0, camera_distance_mm) in camera frame
    - Points on eye surface are projected onto a sphere
    - Camera frame: +X = right, +Y = down, +Z = into scene

    Args:
        points_px: (..., 2) pixel coordinates
        eye_center_px: (2,) eye center in pixels
        eye_radius_px: Eye radius in pixels (half of eye width)
        eye_radius_mm: Eye radius in mm
        camera_distance_mm: Distance from camera to eye center

    Returns:
        (..., 3) coordinates in camera frame (mm)
    """
    # Scale factor: mm per pixel
    mm_per_px = eye_radius_mm / eye_radius_px

    # Convert to mm, centered on eye
    original_shape = points_px.shape
    points_flat = points_px.reshape(-1, 2)
    centered_px = points_flat - eye_center_px
    centered_mm = centered_px * mm_per_px

    # Project onto sphere surface
    # x_cam = centered_mm[:, 0] (right in image)
    # y_cam = centered_mm[:, 1] (down in image)
    # z_cam = sqrt(R^2 - x^2 - y^2) + distance (forward, on sphere surface)

    x_cam = centered_mm[:, 0]
    y_cam = centered_mm[:, 1]

    # Clamp to sphere radius to avoid sqrt of negative
    r_squared = x_cam**2 + y_cam**2
    r_squared = np.minimum(r_squared, eye_radius_mm**2 * 0.99)

    # Z offset from eye center (positive = toward camera)
    z_offset = np.sqrt(eye_radius_mm**2 - r_squared)

    # Full 3D position in camera frame
    points_3d = np.zeros((len(points_flat), 3), dtype=np.float64)
    points_3d[:, 0] = x_cam
    points_3d[:, 1] = y_cam
    points_3d[:, 2] = camera_distance_mm - z_offset  # Closer to camera than eye center

    # Reshape to original
    return points_3d.reshape(original_shape[:-1] + (3,))


def load_ferret_eye_kinematics(
    csv_path: Path,
    eye_side: Literal["left", "right"],
    camera_distance_mm: float,
    eye_radius_mm: float = 3.5,
    pupil_radius_mm: float = 0.5,
    pupil_eccentricity: float = 0.8,
    processing_level: str = "cleaned",
) -> FerretEyeKinematics:
    """
    Load eye tracking CSV and compute full FerretEyeKinematics.

    Args:
        csv_path: Path to eye_data.csv
        eye_side: "left" or "right"
        camera_distance_mm: Distance from camera to eye center (from skull geometry)
        eye_radius_mm: Eyeball radius in mm
        pupil_radius_mm: Pupil ellipse semi-major axis in mm
        pupil_eccentricity: Pupil ellipse eccentricity (minor/major)
        processing_level: CSV processing level filter

    Returns:
        FerretEyeKinematics with computed orientations and angular velocities
    """
    # Load and extract data
    df = load_eye_csv(csv_path=csv_path, processing_level=processing_level)
    timestamps, pupil_centers_px, pupil_points_px, tear_duct_px, outer_eye_px = extract_frame_data(df)

    n_frames = len(timestamps)

    # Estimate eye center and radius in pixels from socket landmarks
    mean_tear_duct_px = np.mean(tear_duct_px, axis=0)
    mean_outer_eye_px = np.mean(outer_eye_px, axis=0)
    eye_center_px = (mean_tear_duct_px + mean_outer_eye_px) / 2.0
    eye_width_px = np.linalg.norm(mean_outer_eye_px - mean_tear_duct_px)
    eye_radius_px = eye_width_px / 2.0

    if eye_radius_px < 1.0:
        raise ValueError(f"Eye width too small: {eye_width_px:.1f} pixels")

    # Convert all points to 3D camera frame
    pupil_centers_cam = pixels_to_camera_3d(
        points_px=pupil_centers_px,
        eye_center_px=eye_center_px,
        eye_radius_px=eye_radius_px,
        eye_radius_mm=eye_radius_mm,
        camera_distance_mm=camera_distance_mm,
    )

    pupil_points_cam = pixels_to_camera_3d(
        points_px=pupil_points_px,
        eye_center_px=eye_center_px,
        eye_radius_px=eye_radius_px,
        eye_radius_mm=eye_radius_mm,
        camera_distance_mm=camera_distance_mm,
    )

    tear_duct_cam = pixels_to_camera_3d(
        points_px=tear_duct_px,
        eye_center_px=eye_center_px,
        eye_radius_px=eye_radius_px,
        eye_radius_mm=eye_radius_mm,
        camera_distance_mm=camera_distance_mm,
    )

    outer_eye_cam = pixels_to_camera_3d(
        points_px=outer_eye_px,
        eye_center_px=eye_center_px,
        eye_radius_px=eye_radius_px,
        eye_radius_mm=eye_radius_mm,
        camera_distance_mm=camera_distance_mm,
    )

    # Eye centers in camera frame (fixed at origin + distance)
    eye_centers_cam = np.zeros((n_frames, 3), dtype=np.float64)
    eye_centers_cam[:, 2] = camera_distance_mm

    # Compute gaze directions (from eye center toward pupil center)
    gaze_directions_cam = pupil_centers_cam - eye_centers_cam
    gaze_norms = np.linalg.norm(gaze_directions_cam, axis=1, keepdims=True)
    gaze_directions_cam = gaze_directions_cam / gaze_norms

    # Build kinematics
    return process_ferret_eye_data(
        name=csv_path.stem,
        source_path=str(csv_path),
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


def load_left_right_eye_kinematics(
    left_csv_path: Path,
    right_csv_path: Path,
    skull_reference_geometry_path: Path,
    eye_radius_mm: float = 3.5,
    pupil_radius_mm: float = 0.5,
    pupil_eccentricity: float = 0.8,
    processing_level: str = "cleaned",
) -> tuple[FerretEyeKinematics, FerretEyeKinematics]:
    """
    Load both left and right eye data using camera distances from skull geometry.

    Args:
        left_csv_path: Path to left eye CSV
        right_csv_path: Path to right eye CSV
        skull_reference_geometry_path: Path to skull_reference_geometry.json
        eye_radius_mm: Eyeball radius in mm
        pupil_radius_mm: Pupil ellipse semi-major axis
        pupil_eccentricity: Pupil ellipse eccentricity
        processing_level: CSV processing level filter

    Returns:
        (left_kinematics, right_kinematics)
    """
    # Load skull geometry and compute camera distances
    skull_geometry = load_skull_reference_geometry(skull_reference_geometry_path)
    left_camera_distance, right_camera_distance = compute_camera_distances_from_skull(skull_geometry)

    print(f"Camera distances from skull geometry:")
    print(f"  Left eye:  {left_camera_distance:.2f} mm")
    print(f"  Right eye: {right_camera_distance:.2f} mm")

    left = load_ferret_eye_kinematics(
        csv_path=left_csv_path,
        eye_side="left",
        camera_distance_mm=left_camera_distance,
        eye_radius_mm=eye_radius_mm,
        pupil_radius_mm=pupil_radius_mm,
        pupil_eccentricity=pupil_eccentricity,
        processing_level=processing_level,
    )

    right = load_ferret_eye_kinematics(
        csv_path=right_csv_path,
        eye_side="right",
        camera_distance_mm=right_camera_distance,
        eye_radius_mm=eye_radius_mm,
        pupil_radius_mm=pupil_radius_mm,
        pupil_eccentricity=pupil_eccentricity,
        processing_level=processing_level,
    )

    return left, right


def create_dual_eye_figure(
    left_kinematics: FerretEyeKinematics,
    right_kinematics: FerretEyeKinematics,
    frame_step: int = 1,
) -> go.Figure:
    """
    Create animated figure showing both eyes side-by-side with synchronized time.

    Layout:
    - Left column: Left eye 3D view
    - Right column: Right eye 3D view
    - Bottom row: Shared timeseries (adduction, elevation, torsion for both eyes)

    Args:
        left_kinematics: Left eye kinematics
        right_kinematics: Right eye kinematics
        frame_step: Step between animation frames

    Returns:
        Plotly Figure with synchronized animation
    """
    # Align timestamps - find common time range
    left_t = left_kinematics.timestamps
    right_t = right_kinematics.timestamps

    t_start = max(left_t[0], right_t[0])
    t_end = min(left_t[-1], right_t[-1])

    # Create common timestamp array
    # Use the timestamps from whichever eye has more frames in the overlap
    left_mask = (left_t >= t_start) & (left_t <= t_end)
    right_mask = (right_t >= t_start) & (right_t <= t_end)

    left_t_overlap = left_t[left_mask]
    right_t_overlap = right_t[right_mask]

    # Use the eye with more frames as reference
    if len(left_t_overlap) >= len(right_t_overlap):
        common_t = left_t_overlap
    else:
        common_t = right_t_overlap

    # Get indices for each eye at common timestamps
    left_indices = np.searchsorted(left_t, common_t).clip(0, len(left_t) - 1)
    right_indices = np.searchsorted(right_t, common_t).clip(0, len(right_t) - 1)

    n_frames = len(common_t)
    frame_indices = list(range(0, n_frames, frame_step))

    # Get geometry parameters (assume same for both eyes)
    R = left_kinematics.eyeball.reference_geometry.markers["pupil_center"].x
    axis_length = R * 1.5

    # Create figure with subplots
    # Layout: 2 3D scenes on top, 3 timeseries on bottom
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.4, 0.2, 0.2, 0.2],
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
        ],
        subplot_titles=(
            "Left Eye", "Right Eye",
            "Adduction (+toward nose) / Abduction (-away)",
            "Elevation (+up) / Depression (-down)",
            "Extorsion (+top out) / Intorsion (-top in)",
        ),
        horizontal_spacing=0.05,
        vertical_spacing=0.06,
    )

    # =========================================================================
    # TIMESERIES PLOTS (both eyes overlaid)
    # =========================================================================

    # Get angles in degrees
    left_add = np.degrees(left_kinematics.adduction_angle.values)
    left_elev = np.degrees(left_kinematics.elevation_angle.values)
    left_tors = np.degrees(left_kinematics.torsion_angle.values)

    right_add = np.degrees(right_kinematics.adduction_angle.values)
    right_elev = np.degrees(right_kinematics.elevation_angle.values)
    right_tors = np.degrees(right_kinematics.torsion_angle.values)

    # Adduction
    fig.add_trace(go.Scatter(
        x=left_t, y=left_add, mode='lines', name='Left',
        line=dict(color='blue', width=1.5), legendgroup='left', showlegend=True,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=right_t, y=right_add, mode='lines', name='Right',
        line=dict(color='red', width=1.5), legendgroup='right', showlegend=True,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="lightgray", row=2, col=1)

    # Elevation
    fig.add_trace(go.Scatter(
        x=left_t, y=left_elev, mode='lines', name='Left',
        line=dict(color='blue', width=1.5), legendgroup='left', showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=right_t, y=right_elev, mode='lines', name='Right',
        line=dict(color='red', width=1.5), legendgroup='right', showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="lightgray", row=3, col=1)

    # Torsion
    fig.add_trace(go.Scatter(
        x=left_t, y=left_tors, mode='lines', name='Left',
        line=dict(color='blue', width=1.5), legendgroup='left', showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=right_t, y=right_tors, mode='lines', name='Right',
        line=dict(color='red', width=1.5), legendgroup='right', showlegend=False,
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="lightgray", row=4, col=1)

    # Compute y-axis ranges
    all_add = np.concatenate([left_add, right_add])
    all_elev = np.concatenate([left_elev, right_elev])
    all_tors = np.concatenate([left_tors, right_tors])

    add_range = [np.nanmin(all_add) - 2, np.nanmax(all_add) + 2]
    elev_range = [np.nanmin(all_elev) - 2, np.nanmax(all_elev) + 2]
    tors_range = [np.nanmin(all_tors) - 2, np.nanmax(all_tors) + 2]

    # Time indicator lines
    t0 = common_t[frame_indices[0]]
    for row, y_range in [(2, add_range), (3, elev_range), (4, tors_range)]:
        fig.add_trace(go.Scatter(
            x=[t0, t0], y=y_range, mode='lines',
            line=dict(color='black', width=2), showlegend=False,
        ), row=row, col=1)

    # =========================================================================
    # 3D EYE VIEWS - Initial frame
    # =========================================================================

    def generate_sphere_mesh(radius: float, resolution: int = 20):
        theta = np.linspace(0, np.pi, resolution)
        phi = np.linspace(0, 2 * np.pi, resolution * 2)
        theta, phi = np.meshgrid(theta, phi)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return x, y, z

    def quaternion_to_rotation_matrix(q_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
        w, x, y, z = q_wxyz
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    # Add initial 3D traces for both eyes
    for col, kin, indices, color in [
        (1, left_kinematics, left_indices, 'blue'),
        (2, right_kinematics, right_indices, 'red'),
    ]:
        idx = indices[frame_indices[0]]
        q = kin.quaternions_wxyz[idx]
        R_mat = quaternion_to_rotation_matrix(q)

        pupil_center = kin.pupil_center_trajectory[idx]
        pupil_points = kin.pupil_points_trajectories[idx]

        # Eyeball sphere
        sx, sy, sz = generate_sphere_mesh(R)
        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz, opacity=0.15,
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            showscale=False, hoverinfo='skip',
        ), row=1, col=col)

        # World axes (dotted)
        for axis_vec, ax_color in [([1,0,0], 'darkred'), ([0,1,0], 'darkgreen'), ([0,0,1], 'darkblue')]:
            tip = np.array(axis_vec) * axis_length
            fig.add_trace(go.Scatter3d(
                x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
                mode='lines', line=dict(color=ax_color, width=3, dash='dot'),
                showlegend=False, hoverinfo='skip',
            ), row=1, col=col)

        # Eye axes (solid, rotating)
        for i, ax_color in enumerate(['red', 'limegreen', 'dodgerblue']):
            rest_vec = np.zeros(3)
            rest_vec[i] = axis_length
            rot_vec = R_mat @ rest_vec
            fig.add_trace(go.Scatter3d(
                x=[0, rot_vec[0]], y=[0, rot_vec[1]], z=[0, rot_vec[2]],
                mode='lines', line=dict(color=ax_color, width=5),
                showlegend=False,
            ), row=1, col=col)

        # Pupil center
        fig.add_trace(go.Scatter3d(
            x=[pupil_center[0]], y=[pupil_center[1]], z=[pupil_center[2]],
            mode='markers', marker=dict(size=6, color='black'),
            showlegend=False,
        ), row=1, col=col)

        # Pupil boundary
        pp_closed = np.vstack([pupil_points, pupil_points[0]])
        fig.add_trace(go.Scatter3d(
            x=pp_closed[:, 0], y=pp_closed[:, 1], z=pp_closed[:, 2],
            mode='lines', line=dict(color='darkblue', width=4),
            showlegend=False,
        ), row=1, col=col)

    # =========================================================================
    # ANIMATION FRAMES
    # =========================================================================

    frames = []
    # Count traces: per eye scene: sphere(1) + world_axes(3) + eye_axes(3) + pupil_center(1) + pupil_boundary(1) = 9
    # timeseries: 2 lines per plot × 3 plots = 6, plus 3 time indicators = 9
    # Total initial traces before animation = 9*2 (eyes) + 6 (timeseries lines) + 3 (time indicators) = 27

    n_ts_traces = 6  # timeseries data traces
    n_time_indicators = 3
    n_eye_traces = 9  # per eye

    for frame_i, common_idx in enumerate(frame_indices):
        frame_data = []
        t_current = common_t[common_idx]

        # Left eye traces (indices 0-8 after timeseries)
        left_idx = left_indices[common_idx]
        left_q = left_kinematics.quaternions_wxyz[left_idx]
        left_R = quaternion_to_rotation_matrix(left_q)
        left_pupil = left_kinematics.pupil_center_trajectory[left_idx]
        left_pp = left_kinematics.pupil_points_trajectories[left_idx]

        # Right eye traces
        right_idx = right_indices[common_idx]
        right_q = right_kinematics.quaternions_wxyz[right_idx]
        right_R = quaternion_to_rotation_matrix(right_q)
        right_pupil = right_kinematics.pupil_center_trajectory[right_idx]
        right_pp = right_kinematics.pupil_points_trajectories[right_idx]

        for kin, idx, R_mat, pupil_center, pupil_points in [
            (left_kinematics, left_idx, left_R, left_pupil, left_pp),
            (right_kinematics, right_idx, right_R, right_pupil, right_pp),
        ]:
            # Sphere (unchanged)
            sx, sy, sz = generate_sphere_mesh(R)
            frame_data.append(go.Surface(x=sx, y=sy, z=sz, opacity=0.15,
                colorscale=[[0, 'lightgray'], [1, 'lightgray']], showscale=False))

            # World axes (unchanged)
            for axis_vec, ax_color in [([1,0,0], 'darkred'), ([0,1,0], 'darkgreen'), ([0,0,1], 'darkblue')]:
                tip = np.array(axis_vec) * axis_length
                frame_data.append(go.Scatter3d(
                    x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
                    mode='lines', line=dict(color=ax_color, width=3, dash='dot'),
                ))

            # Eye axes (rotating)
            for i, ax_color in enumerate(['red', 'limegreen', 'dodgerblue']):
                rest_vec = np.zeros(3)
                rest_vec[i] = axis_length
                rot_vec = R_mat @ rest_vec
                frame_data.append(go.Scatter3d(
                    x=[0, rot_vec[0]], y=[0, rot_vec[1]], z=[0, rot_vec[2]],
                    mode='lines', line=dict(color=ax_color, width=5),
                ))

            # Pupil center
            frame_data.append(go.Scatter3d(
                x=[pupil_center[0]], y=[pupil_center[1]], z=[pupil_center[2]],
                mode='markers', marker=dict(size=6, color='black'),
            ))

            # Pupil boundary
            pp_closed = np.vstack([pupil_points, pupil_points[0]])
            frame_data.append(go.Scatter3d(
                x=pp_closed[:, 0], y=pp_closed[:, 1], z=pp_closed[:, 2],
                mode='lines', line=dict(color='darkblue', width=4),
            ))

        # Time indicators
        for y_range in [add_range, elev_range, tors_range]:
            frame_data.append(go.Scatter(x=[t_current, t_current], y=y_range,
                mode='lines', line=dict(color='black', width=2)))

        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_i),
            traces=list(range(n_ts_traces, n_ts_traces + 2*n_eye_traces + n_time_indicators)),
        ))

    fig.frames = frames

    # =========================================================================
    # LAYOUT
    # =========================================================================

    scene_config = dict(
        aspectmode='data',
        xaxis=dict(title='X', range=[-R*2, R*2], showbackground=False),
        yaxis=dict(title='Y', range=[-R*2, R*2], showbackground=False),
        zaxis=dict(title='Z', range=[-R*2, R*2], showbackground=False),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
    )

    fig.update_layout(
        title=dict(text="Binocular Eye Kinematics (Synchronized)", x=0.5),
        height=1000,
        width=1200,
        scene=scene_config,
        scene2=scene_config,
        legend=dict(x=1.02, y=0.5, orientation='v'),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True, mode='immediate')]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate')]),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top", xanchor="left",
                currentvalue=dict(font=dict(size=12), prefix="Frame: ", visible=True),
                transition=dict(duration=0),
                pad=dict(b=10, t=50),
                len=0.4, x=0.1, y=0,
                steps=[
                    dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                         label=str(frame_indices[i]), method="animate")
                    for i in range(len(frame_indices))
                ],
            ),
        ],
    )

    # Update timeseries axes
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=2, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=3, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=4, col=1)
    fig.update_yaxes(title_text="°", range=add_range, row=2, col=1)
    fig.update_yaxes(title_text="°", range=elev_range, row=3, col=1)
    fig.update_yaxes(title_text="°", range=tors_range, row=4, col=1)

    return fig


if __name__ == "__main__":
    # Example paths - edit as needed
    base_dir = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s"
    )

    left_eye_csv = base_dir / "eye_data" / "output_data" / "eye0_data.csv"
    right_eye_csv = base_dir / "eye_data" / "output_data" / "eye1_data.csv"
    skull_geometry_json = base_dir / "mocap_data" / "output_data"/ "solver_output" / "skull_reference_geometry.json"

    print(f"Loading skull geometry from {skull_geometry_json}...")
    print(f"Loading left eye from {left_eye_csv}...")
    print(f"Loading right eye from {right_eye_csv}...")

    left_kin, right_kin = load_left_right_eye_kinematics(
        left_csv_path=left_eye_csv,
        right_csv_path=right_eye_csv,
        skull_reference_geometry_path=skull_geometry_json,
    )

    print(f"\n--- Left Eye ---")
    print(f"  Frames: {left_kin.n_frames}")
    print(f"  Duration: {left_kin.duration_seconds:.2f}s")
    print(f"  Adduction range: {np.degrees(left_kin.adduction_angle.values.min()):.1f}° to {np.degrees(left_kin.adduction_angle.values.max()):.1f}°")
    print(f"  Elevation range: {np.degrees(left_kin.elevation_angle.values.min()):.1f}° to {np.degrees(left_kin.elevation_angle.values.max()):.1f}°")
    print(f"  Mean angular speed: {np.mean(left_kin.angular_speed.values):.3f} rad/s")

    print(f"\n--- Right Eye ---")
    print(f"  Frames: {right_kin.n_frames}")
    print(f"  Duration: {right_kin.duration_seconds:.2f}s")
    print(f"  Adduction range: {np.degrees(right_kin.adduction_angle.values.min()):.1f}° to {np.degrees(right_kin.adduction_angle.values.max()):.1f}°")
    print(f"  Elevation range: {np.degrees(right_kin.elevation_angle.values.min()):.1f}° to {np.degrees(right_kin.elevation_angle.values.max()):.1f}°")
    print(f"  Mean angular speed: {np.mean(right_kin.angular_speed.values):.3f} rad/s")

    # Socket landmarks should show minimal movement (just tracking noise)
    print(f"\n--- Socket Landmark Stability ---")
    print(f"  Left tear_duct std: {np.std(left_kin.tear_duct_mm, axis=0)}")
    print(f"  Right tear_duct std: {np.std(right_kin.tear_duct_mm, axis=0)}")

    # Create and show synchronized dual-eye visualization
    print("\nCreating synchronized dual-eye visualization...")
    fig = create_dual_eye_figure(
        left_kinematics=left_kin,
        right_kinematics=right_kin,
        frame_step=2,
    )
    fig.show()