"""
Ferret Head and Gaze Kinematics Visualization (Rerun)

Displays:
- 3D skeleton viewer with animation
- Head origin point (mean of eyes and ears, white dot)
- Head basis vectors (x, y, z axes of head coordinate frame, 100mm arrows)
- Head position (mm)
- Head orientation (roll, pitch, yaw in degrees)
- Angular velocity in world frame (x, y, z in deg/s)
- Angular velocity in head-local frame (roll, pitch, yaw rates in deg/s)
- Eyeball spheres at left_eye and right_eye positions
- Gaze vectors originating from eyeball centers
- Eye-local 3D views showing each eye's gaze in its reference frame
- Gaze endpoint tracers with 2-second history
- Eye-in-head angle time series (horizontal/vertical for each eye)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

from python_code.ferret_gaze.kinematics_calculators.ferret_eye_kinematics import EyeballKinematics
from python_code.ferret_gaze.kinematics_calculators.ferret_gaze_kinematics import GazeKinematics
from python_code.ferret_gaze.kinematics_calculators.ferret_skull_kinematics import SkullKinematics
from python_code.ferret_gaze.kinematics_core.quaternion_model import Quaternion

# =============================================================================
# TOPOLOGY DEFINITION
# =============================================================================
SKULL_MARKER_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "base",
]
EYE_CAM_MARKER_NAMES: list[str] = ["left_cam_tip", "right_cam_tip"]
BODY_MARKER_NAMES: list[str] = ["spine_t1", "sacrum", "tail_tip"]
MARKER_NAMES: list[str] = SKULL_MARKER_NAMES + EYE_CAM_MARKER_NAMES + BODY_MARKER_NAMES

DISPLAY_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5),
    (5, 6), (5, 7), (3, 8), (4, 8), (8, 9), (9, 10),
]

MARKER_COLORS: dict[str, tuple[int, int, int]] = {
    "nose": (255, 107, 107), "left_eye": (78, 205, 196), "right_eye": (78, 205, 196),
    "left_ear": (149, 225, 211), "right_ear": (149, 225, 211), "base": (255, 230, 109),
    "left_cam_tip": (168, 230, 207), "right_cam_tip": (168, 230, 207),
    "spine_t1": (221, 160, 221), "sacrum": (221, 160, 221), "tail_tip": (221, 160, 221),
}

AXIS_COLORS: dict[str, tuple[int, int, int]] = {
    "roll": (255, 107, 107), "pitch": (78, 205, 196), "yaw": (255, 230, 109),
    "x": (255, 107, 107), "y": (78, 205, 196), "z": (255, 230, 109),
}

ENCLOSURE_SIZE_MM: float = 1000.0
EYEBALL_RADIUS_MM: float = 5.0
EYEBALL_COLOR_LEFT: tuple[int, int, int, int] = (100, 200, 255, 180)
EYEBALL_COLOR_RIGHT: tuple[int, int, int, int] = (255, 150, 100, 180)
GAZE_VECTOR_COLOR_LEFT: tuple[int, int, int] = (0, 150, 255)
GAZE_VECTOR_COLOR_RIGHT: tuple[int, int, int] = (255, 100, 0)
GAZE_VECTOR_RADIUS_MM: float = 1.5
GAZE_TRACER_RADIUS_MM: float = 2.0
GAZE_TRACER_HISTORY_SECONDS: float = 2.0
EYE_LOCAL_GAZE_LENGTH_MM: float = 50.0


# =============================================================================
# DATA LOADING FROM CSV (FAST VERSIONS)
# =============================================================================
def load_trajectory_data(trajectory_csv_path: Path) -> dict[str, NDArray[np.float64]]:
    """Load trajectory data from CSV into arrays of shape (n_frames, 3). FAST version."""
    trajectory_csv_path = Path(trajectory_csv_path)
    if not trajectory_csv_path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {trajectory_csv_path}")

    df = pd.read_csv(trajectory_csv_path)
    if len(df) == 0:
        raise ValueError(f"Empty CSV file: {trajectory_csv_path}")

    optimized_df = df[df["data_type"] == "optimized"]
    if len(optimized_df) == 0:
        raise ValueError(f"No 'optimized' data_type found in {trajectory_csv_path}")

    n_frames = optimized_df["frame"].nunique()
    frame_indices = optimized_df["frame"].unique()
    expected_frames = np.arange(n_frames)
    if not np.array_equal(np.sort(frame_indices), expected_frames):
        raise ValueError(f"Frame indices must be contiguous from 0 to {n_frames - 1}")

    marker_names = optimized_df["marker"].unique()
    timestamp_df = optimized_df.groupby("frame")["timestamp"].first().sort_index()
    timestamps = timestamp_df.values.astype(np.float64)
    optimized_df = optimized_df.sort_values("frame")
    body_trajectories: dict[str, NDArray[np.float64]] = {"timestamps": timestamps}
    grouped = optimized_df.groupby("marker")

    for marker_name in marker_names:
        marker_df = grouped.get_group(marker_name).sort_values("frame")
        if len(marker_df) != n_frames:
            raise ValueError(f"Marker '{marker_name}' has {len(marker_df)} rows, expected {n_frames}")
        positions = marker_df[["x", "y", "z"]].values.astype(np.float64)
        body_trajectories[str(marker_name)] = positions

    return body_trajectories


def load_skull_kinematics_from_csv(csv_path: Path) -> SkullKinematics:
    """Load SkullKinematics from a saved CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Skull kinematics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"Empty CSV file: {csv_path}")

    timestamps = df["timestamp"].values.astype(np.float64)
    position = df[["position_x_mm", "position_y_mm", "position_z_mm"]].values.astype(np.float64)
    quaternions = [Quaternion(w=r["quaternion_w"], x=r["quaternion_x"], y=r["quaternion_y"], z=r["quaternion_z"]) for _, r in df.iterrows()]
    euler_angles_deg = df[["roll_deg", "pitch_deg", "yaw_deg"]].values.astype(np.float64)
    angular_velocity_world_deg_s = df[["omega_world_x_deg_s", "omega_world_y_deg_s", "omega_world_z_deg_s"]].values.astype(np.float64)
    angular_velocity_local_deg_s = df[["omega_local_roll_deg_s", "omega_local_pitch_deg_s", "omega_local_yaw_deg_s"]].values.astype(np.float64)
    basis_x = df[["basis_x_x", "basis_x_y", "basis_x_z"]].values.astype(np.float64)
    basis_y = df[["basis_y_x", "basis_y_y", "basis_y_z"]].values.astype(np.float64)
    basis_z = df[["basis_z_x", "basis_z_y", "basis_z_z"]].values.astype(np.float64)

    return SkullKinematics(
        timestamps=timestamps, position=position, orientation_quaternions=quaternions,
        euler_angles_deg=euler_angles_deg, angular_velocity_world_deg_s=angular_velocity_world_deg_s,
        angular_velocity_local_deg_s=angular_velocity_local_deg_s, basis_x=basis_x, basis_y=basis_y, basis_z=basis_z,
    )


def load_eye_kinematics_from_csv(csv_path: Path) -> EyeballKinematics:
    """Load EyeKinematics from a saved CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Eye kinematics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"Empty CSV file: {csv_path}")

    return EyeballKinematics(
        timestamps=df["timestamp"].values.astype(np.float64),
        eyeball_angle_azimuth_rad=df["angle_x_rad"].values.astype(np.float64),
        eyeball_angle_elevation_rad=df["angle_y_rad"].values.astype(np.float64),
    )


def load_gaze_kinematics_from_csv(csv_path: Path) -> GazeKinematics:
    """Load GazeKinematics from a saved CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Gaze kinematics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"Empty CSV file: {csv_path}")

    return GazeKinematics(
        timestamps=df["timestamp"].values.astype(np.float64),
        left_eyeball_center_xyz_mm=df[["left_eyeball_center_x_mm", "left_eyeball_center_y_mm", "left_eyeball_center_z_mm"]].values.astype(np.float64),
        left_gaze_azimuth_rad=df["left_gaze_azimuth_rad"].values.astype(np.float64),
        left_gaze_elevation_rad=df["left_gaze_elevation_rad"].values.astype(np.float64),
        left_gaze_endpoint_mm=df[["left_gaze_endpoint_x_mm", "left_gaze_endpoint_y_mm", "left_gaze_endpoint_z_mm"]].values.astype(np.float64),
        left_gaze_direction=df[["left_gaze_direction_x", "left_gaze_direction_y", "left_gaze_direction_z"]].values.astype(np.float64),
        right_eyeball_center_xyz_mm=df[["right_eyeball_center_x_mm", "right_eyeball_center_y_mm", "right_eyeball_center_z_mm"]].values.astype(np.float64),
        right_gaze_azimuth_rad=df["right_gaze_azimuth_rad"].values.astype(np.float64),
        right_gaze_elevation_rad=df["right_gaze_elevation_rad"].values.astype(np.float64),
        right_gaze_endpoint_mm=df[["right_gaze_endpoint_x_mm", "right_gaze_endpoint_y_mm", "right_gaze_endpoint_z_mm"]].values.astype(np.float64),
        right_gaze_direction=df[["right_gaze_direction_x", "right_gaze_direction_y", "right_gaze_direction_z"]].values.astype(np.float64),
    )


# =============================================================================
# SKELETON VISUALIZATION
# =============================================================================
def send_skeleton_data(trajectory_data: dict[str, NDArray[np.float64]], trajectory_timestamps: NDArray[np.float64]) -> None:
    """Send skeleton marker and edge data to Rerun."""
    n_frames = len(trajectory_timestamps)
    if n_frames == 0:
        return
    t0 = trajectory_timestamps[0]
    marker_times, all_positions, marker_partition_lengths, all_colors = [], [], [], []
    edge_times, all_edge_strips, edge_partition_lengths = [], [], []

    for frame_idx in range(n_frames):
        time_val = float(trajectory_timestamps[frame_idx] - t0)
        frame_positions, frame_colors = [], []
        for name in MARKER_NAMES:
            if name in trajectory_data:
                pos = trajectory_data[name][frame_idx]
                if not np.any(np.isnan(pos)):
                    frame_positions.append(pos)
                    frame_colors.append(np.array(MARKER_COLORS.get(name, (255, 255, 255)), dtype=np.uint8))
        if frame_positions:
            marker_times.append(time_val)
            all_positions.extend(frame_positions)
            all_colors.extend(frame_colors)
            marker_partition_lengths.append(len(frame_positions))

        frame_strips = []
        for idx_i, idx_j in DISPLAY_EDGES:
            name_i, name_j = MARKER_NAMES[idx_i], MARKER_NAMES[idx_j]
            if name_i in trajectory_data and name_j in trajectory_data:
                pos_i, pos_j = trajectory_data[name_i][frame_idx], trajectory_data[name_j][frame_idx]
                if not np.any(np.isnan(pos_i)) and not np.any(np.isnan(pos_j)):
                    frame_strips.append(np.array([pos_i, pos_j], dtype=np.float64))
        if frame_strips:
            edge_times.append(time_val)
            all_edge_strips.extend(frame_strips)
            edge_partition_lengths.append(len(frame_strips))

    if marker_times and all_positions:
        rr.send_columns("skeleton/markers", indexes=[rr.TimeColumn("time", duration=marker_times)],
            columns=[*rr.Points3D.columns(positions=np.array(all_positions)).partition(lengths=marker_partition_lengths),
                     *rr.Points3D.columns(colors=np.array(all_colors), radii=[4.0]*len(all_positions)).partition(lengths=marker_partition_lengths)])
    if edge_times and all_edge_strips:
        rr.send_columns("skeleton/edges", indexes=[rr.TimeColumn("time", duration=edge_times)],
            columns=[*rr.LineStrips3D.columns(strips=all_edge_strips, colors=[(0,200,200)]*len(all_edge_strips), radii=[1.0]*len(all_edge_strips)).partition(lengths=edge_partition_lengths)])


def send_enclosure() -> None:
    """Send a 1m³ enclosure as wireframe box."""
    half = ENCLOSURE_SIZE_MM / 2.0
    corners = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0],[-half,-half,ENCLOSURE_SIZE_MM],[half,-half,ENCLOSURE_SIZE_MM],[half,half,ENCLOSURE_SIZE_MM],[-half,half,ENCLOSURE_SIZE_MM]], dtype=np.float64)
    edge_indices = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    strips = [np.array([corners[i], corners[j]]) for i, j in edge_indices]
    rr.log("skeleton/enclosure", rr.LineStrips3D(strips=strips, colors=[(200,200,200,100)]*len(strips), radii=[2]*len(strips)), static=True)


# =============================================================================
# TIME SERIES LOGGING
# =============================================================================
def send_head_kinematics(hk: SkullKinematics) -> None:
    """Send head kinematics time series data to Rerun."""
    t0, times = hk.timestamps[0], hk.timestamps - hk.timestamps[0]
    for i, name in enumerate(["x", "y", "z"]):
        rr.send_columns(f"position/{name}", indexes=[rr.TimeColumn("time", duration=times)], columns=rr.Scalars.columns(scalars=hk.position[:, i]))
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        rr.send_columns(f"orientation/{name}", indexes=[rr.TimeColumn("time", duration=times)], columns=rr.Scalars.columns(scalars=hk.euler_angles_deg[:, i]))
    for i, name in enumerate(["x", "y", "z"]):
        rr.send_columns(f"omega_world/{name}", indexes=[rr.TimeColumn("time", duration=times)], columns=rr.Scalars.columns(scalars=hk.angular_velocity_world_deg_s[:, i]))
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        rr.send_columns(f"omega_local/{name}", indexes=[rr.TimeColumn("time", duration=times)], columns=rr.Scalars.columns(scalars=hk.angular_velocity_local_deg_s[:, i]))


def send_head_origin(head_origins: NDArray[np.float64], timestamps: NDArray[np.float64]) -> None:
    t0, times = timestamps[0], timestamps - timestamps[0]
    rr.send_columns("skeleton/head_origin", indexes=[rr.TimeColumn("time", duration=times)], columns=[*rr.Points3D.columns(positions=head_origins)])


def send_head_basis_vectors(hk: SkullKinematics, head_origins: NDArray[np.float64], scale: float = 100.0) -> None:
    t0, n_frames, times = hk.timestamps[0], len(hk.timestamps), hk.timestamps - hk.timestamps[0]
    colors = {"x": (255, 107, 107), "y": (78, 255, 96), "z": (55, 20, 255)}
    basis_data = {"x": hk.basis_x, "y": hk.basis_y, "z": hk.basis_z}
    for axis_name, basis in basis_data.items():
        vectors = basis * scale
        color_array = np.tile(np.array(colors[axis_name], dtype=np.uint8), (n_frames, 1))
        rr.send_columns(f"skeleton/head_basis/{axis_name}", indexes=[rr.TimeColumn("time", duration=times)],
            columns=[*rr.Arrows3D.columns(origins=head_origins, vectors=vectors, colors=color_array).partition(lengths=[1]*n_frames)])


# =============================================================================
# GAZE VISUALIZATION
# =============================================================================
def send_gaze_eyeballs(gk: GazeKinematics) -> None:
    t0, times, n_frames = gk.timestamps[0], gk.timestamps - gk.timestamps[0], len(gk.timestamps)
    half_sizes = np.tile(np.array([[EYEBALL_RADIUS_MM]*3]), (n_frames, 1))
    left_colors = np.tile(np.array(EYEBALL_COLOR_LEFT, dtype=np.uint8), (n_frames, 1))
    right_colors = np.tile(np.array(EYEBALL_COLOR_RIGHT, dtype=np.uint8), (n_frames, 1))
    rr.send_columns("skeleton/gaze/left_eyeball", indexes=[rr.TimeColumn("time", duration=times)],
        columns=[*rr.Ellipsoids3D.columns(centers=gk.left_eyeball_center_xyz_mm, half_sizes=half_sizes, colors=left_colors).partition(lengths=[1]*n_frames)])
    rr.send_columns("skeleton/gaze/right_eyeball", indexes=[rr.TimeColumn("time", duration=times)],
        columns=[*rr.Ellipsoids3D.columns(centers=gk.right_eyeball_center_xyz_mm, half_sizes=half_sizes, colors=right_colors).partition(lengths=[1]*n_frames)])


def send_gaze_vectors(gk: GazeKinematics) -> None:
    t0, times, n_frames = gk.timestamps[0], gk.timestamps - gk.timestamps[0], len(gk.timestamps)
    left_vectors = gk.left_gaze_endpoint_mm - gk.left_eyeball_center_xyz_mm
    right_vectors = gk.right_gaze_endpoint_mm - gk.right_eyeball_center_xyz_mm
    left_colors = np.tile(np.array(GAZE_VECTOR_COLOR_LEFT, dtype=np.uint8), (n_frames, 1))
    right_colors = np.tile(np.array(GAZE_VECTOR_COLOR_RIGHT, dtype=np.uint8), (n_frames, 1))
    rr.send_columns("skeleton/gaze/left_gaze_vector", indexes=[rr.TimeColumn("time", duration=times)],
        columns=[*rr.Arrows3D.columns(origins=gk.left_eyeball_center_xyz_mm, vectors=left_vectors, colors=left_colors).partition(lengths=[1]*n_frames)])
    rr.send_columns("skeleton/gaze/right_gaze_vector", indexes=[rr.TimeColumn("time", duration=times)],
        columns=[*rr.Arrows3D.columns(origins=gk.right_eyeball_center_xyz_mm, vectors=right_vectors, colors=right_colors).partition(lengths=[1]*n_frames)])


def send_gaze_tracers(gk: GazeKinematics, trail_duration_s: float = GAZE_TRACER_HISTORY_SECONDS) -> None:
    t0, times, n_frames = gk.timestamps[0], gk.timestamps - gk.timestamps[0], len(gk.timestamps)
    if n_frames < 2:
        return
    median_dt = float(np.median(np.diff(gk.timestamps)))
    trail_frames = max(1, int(trail_duration_s / median_dt))

    for i in range(n_frames):
        rr.set_time("time", duration=float(times[i]))
        start_idx = max(0, i - trail_frames)
        n_trail = i - start_idx + 1
        alphas = np.linspace(50, 255, n_trail).astype(np.uint8)
        radii = np.linspace(GAZE_TRACER_RADIUS_MM * 0.5, GAZE_TRACER_RADIUS_MM, n_trail)

        left_colors = np.array([[GAZE_VECTOR_COLOR_LEFT[0], GAZE_VECTOR_COLOR_LEFT[1], GAZE_VECTOR_COLOR_LEFT[2], a] for a in alphas], dtype=np.uint8)
        rr.log("skeleton/gaze/left_tracer", rr.Points3D(positions=gk.left_gaze_endpoint_mm[start_idx:i+1], colors=left_colors, radii=radii))

        right_colors = np.array([[GAZE_VECTOR_COLOR_RIGHT[0], GAZE_VECTOR_COLOR_RIGHT[1], GAZE_VECTOR_COLOR_RIGHT[2], a] for a in alphas], dtype=np.uint8)
        rr.log("skeleton/gaze/right_tracer", rr.Points3D(positions=gk.right_gaze_endpoint_mm[start_idx:i+1], colors=right_colors, radii=radii))


def send_gaze_kinematics_timeseries(gk: GazeKinematics) -> None:
    t0, times = gk.timestamps[0], gk.timestamps - gk.timestamps[0]
    rr.send_columns("gaze/left_azimuth", indexes=[rr.TimeColumn("time", duration=times)], columns=rr.Scalars.columns(scalars=np.rad2deg(gk.left_gaze_azimuth_rad)))
    rr.send_columns("gaze/left_elevation", indexes=[rr.TimeColumn("time", duration=times)], columns=rr.Scalars.columns(scalars=np.rad2deg(gk.left_gaze_elevation_rad)))
    rr.send_columns("gaze/right_azimuth", indexes=[rr.TimeColumn("time", duration=times)], columns=rr.Scalars.columns(scalars=np.rad2deg(gk.right_gaze_azimuth_rad)))
    rr.send_columns("gaze/right_elevation", indexes=[rr.TimeColumn("time", duration=times)], columns=rr.Scalars.columns(scalars=np.rad2deg(gk.right_gaze_elevation_rad)))


def send_eye_in_head_timeseries(left_eye: EyeballKinematics, right_eye: EyeballKinematics) -> None:
    t0 = left_eye.timestamps[0]
    times_l, times_r = left_eye.timestamps - t0, right_eye.timestamps - t0
    rr.send_columns("eye_in_head/left/horizontal", indexes=[rr.TimeColumn("time", duration=times_l)], columns=rr.Scalars.columns(scalars=np.rad2deg(left_eye.eyeball_angle_azimuth_rad)))
    rr.send_columns("eye_in_head/left/vertical", indexes=[rr.TimeColumn("time", duration=times_l)], columns=rr.Scalars.columns(scalars=np.rad2deg(left_eye.eyeball_angle_elevation_rad)))
    rr.send_columns("eye_in_head/right/horizontal", indexes=[rr.TimeColumn("time", duration=times_r)], columns=rr.Scalars.columns(scalars=np.rad2deg(right_eye.eyeball_angle_azimuth_rad)))
    rr.send_columns("eye_in_head/right/vertical", indexes=[rr.TimeColumn("time", duration=times_r)], columns=rr.Scalars.columns(scalars=np.rad2deg(right_eye.eyeball_angle_elevation_rad)))


def send_eye_local_views(gk: GazeKinematics, left_eye: EyeballKinematics, right_eye: EyeballKinematics) -> None:
    t0, times, n_frames = gk.timestamps[0], gk.timestamps - gk.timestamps[0], len(gk.timestamps)
    half_size = np.array([EYEBALL_RADIUS_MM]*3)
    rr.log("eye_local/left/eyeball", rr.Ellipsoids3D(centers=[[0,0,0]], half_sizes=[half_size], colors=[EYEBALL_COLOR_LEFT], fill_mode="solid"), static=True)
    rr.log("eye_local/right/eyeball", rr.Ellipsoids3D(centers=[[0,0,0]], half_sizes=[half_size], colors=[EYEBALL_COLOR_RIGHT], fill_mode="solid"), static=True)
    if n_frames < 2:
        return

    median_dt = float(np.median(np.diff(gk.timestamps)))
    trail_frames = max(1, int(GAZE_TRACER_HISTORY_SECONDS / median_dt))
    left_dirs, right_dirs, origins = np.zeros((n_frames, 3)), np.zeros((n_frames, 3)), np.zeros((n_frames, 3))

    for i in range(n_frames):
        left_dir = np.array([-left_eye.eyeball_angle_azimuth_rad[i], 1.0, left_eye.eyeball_angle_elevation_rad[i]])
        left_dirs[i] = left_dir / np.linalg.norm(left_dir) * EYE_LOCAL_GAZE_LENGTH_MM
        right_dir = np.array([-right_eye.eyeball_angle_azimuth_rad[i], -1.0, right_eye.eyeball_angle_elevation_rad[i]])
        right_dirs[i] = right_dir / np.linalg.norm(right_dir) * EYE_LOCAL_GAZE_LENGTH_MM

    left_colors = np.tile(np.array(GAZE_VECTOR_COLOR_LEFT, dtype=np.uint8), (n_frames, 1))
    right_colors = np.tile(np.array(GAZE_VECTOR_COLOR_RIGHT, dtype=np.uint8), (n_frames, 1))
    rr.send_columns("eye_local/left/gaze_vector", indexes=[rr.TimeColumn("time", duration=times)],
        columns=[*rr.Arrows3D.columns(origins=origins, vectors=left_dirs, colors=left_colors).partition(lengths=[1]*n_frames)])
    rr.send_columns("eye_local/right/gaze_vector", indexes=[rr.TimeColumn("time", duration=times)],
        columns=[*rr.Arrows3D.columns(origins=origins, vectors=right_dirs, colors=right_colors).partition(lengths=[1]*n_frames)])

    for i in range(n_frames):
        rr.set_time("time", duration=float(times[i]))
        start_idx = max(0, i - trail_frames)
        n_trail = i - start_idx + 1
        alphas = np.linspace(50, 255, n_trail).astype(np.uint8)
        radii = np.linspace(GAZE_TRACER_RADIUS_MM * 0.5, GAZE_TRACER_RADIUS_MM, n_trail)
        left_trail_colors = np.array([[GAZE_VECTOR_COLOR_LEFT[0], GAZE_VECTOR_COLOR_LEFT[1], GAZE_VECTOR_COLOR_LEFT[2], a] for a in alphas], dtype=np.uint8)
        right_trail_colors = np.array([[GAZE_VECTOR_COLOR_RIGHT[0], GAZE_VECTOR_COLOR_RIGHT[1], GAZE_VECTOR_COLOR_RIGHT[2], a] for a in alphas], dtype=np.uint8)
        rr.log("eye_local/left/tracer", rr.Points3D(positions=left_dirs[start_idx:i+1], colors=left_trail_colors, radii=radii))
        rr.log("eye_local/right/tracer", rr.Points3D(positions=right_dirs[start_idx:i+1], colors=right_trail_colors, radii=radii))


# =============================================================================
# PLOT STYLING & BLUEPRINT
# =============================================================================
def setup_plot_styling(include_gaze: bool = False, include_eye_in_head: bool = False) -> None:
    # FIX: SeriesLines uses plural parameter names: colors, names, widths (not color, name, width)
    for name in ["x", "y", "z"]:
        rr.log(f"position/{name}", rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name]), static=True)
    for name in ["roll", "pitch", "yaw"]:
        rr.log(f"orientation/{name}", rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name]), static=True)
    for name in ["x", "y", "z"]:
        rr.log(f"omega_world/{name}", rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name]), static=True)
    for name in ["roll", "pitch", "yaw"]:
        rr.log(f"omega_local/{name}", rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name]), static=True)
    rr.log("skeleton/head_origin", rr.Points3D.from_fields(radii=6.0, colors=(255, 255, 255)), static=True)
    if include_gaze:
        rr.log("gaze/left_azimuth", rr.SeriesLines(colors=[GAZE_VECTOR_COLOR_LEFT], names=["L azimuth"]), static=True)
        rr.log("gaze/left_elevation", rr.SeriesLines(colors=[(0, 100, 200)], names=["L elevation"]), static=True)
        rr.log("gaze/right_azimuth", rr.SeriesLines(colors=[GAZE_VECTOR_COLOR_RIGHT], names=["R azimuth"]), static=True)
        rr.log("gaze/right_elevation", rr.SeriesLines(colors=[(200, 70, 0)], names=["R elevation"]), static=True)
    if include_eye_in_head:
        rr.log("eye_in_head/left/horizontal", rr.SeriesLines(colors=[GAZE_VECTOR_COLOR_LEFT], names=["horizontal"]), static=True)
        rr.log("eye_in_head/left/vertical", rr.SeriesLines(colors=[(0, 100, 200)], names=["vertical"]), static=True)
        rr.log("eye_in_head/right/horizontal", rr.SeriesLines(colors=[GAZE_VECTOR_COLOR_RIGHT], names=["horizontal"]), static=True)
        rr.log("eye_in_head/right/vertical", rr.SeriesLines(colors=[(200, 70, 0)], names=["vertical"]), static=True)


def create_blueprint(include_gaze: bool = False, include_eye_in_head: bool = False) -> rrb.Blueprint:
    time_series_panels = [
        rrb.TimeSeriesView(name="Position (mm)", origin="position", plot_legend=rrb.PlotLegend(visible=True), axis_y=rrb.ScalarAxis(range=(-500.0, 500.0))),
        rrb.TimeSeriesView(name="Orientation (deg)", origin="orientation", plot_legend=rrb.PlotLegend(visible=True), axis_y=rrb.ScalarAxis(range=(-180.0, 180.0))),
        rrb.TimeSeriesView(name="Angular Velocity - World Frame (deg/s)", origin="omega_world", plot_legend=rrb.PlotLegend(visible=True), axis_y=rrb.ScalarAxis(range=(-800.0, 800.0))),
        rrb.TimeSeriesView(name="Angular Velocity - Head Local (deg/s)", origin="omega_local", plot_legend=rrb.PlotLegend(visible=True), axis_y=rrb.ScalarAxis(range=(-800.0, 800.0))),
    ]
    if include_gaze:
        time_series_panels.append(rrb.TimeSeriesView(name="Gaze Direction (deg)", origin="gaze", plot_legend=rrb.PlotLegend(visible=True), axis_y=rrb.ScalarAxis(range=(-180.0, 180.0))))

    spatial_3d_view = rrb.Spatial3DView(name="3D Skeleton", origin="skeleton", contents=["+ skeleton/**"],
        eye_controls=rrb.EyeControls3D(position=(0.0, -2000.0, 500.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 0.0, 1.0)),
        line_grid=rrb.LineGrid3D(visible=True, spacing=100.0, plane=rr.components.Plane3D.XY, color=[100, 100, 100, 128]))

    eye_local_views, eye_in_head_panels = [], []
    if include_eye_in_head:
        eye_local_views = [
            rrb.Spatial3DView(name="Left Eye (Local)", origin="eye_local/left", eye_controls=rrb.EyeControls3D(position=(0.0, -150.0, 50.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 0.0, 1.0))),
            rrb.Spatial3DView(name="Right Eye (Local)", origin="eye_local/right", eye_controls=rrb.EyeControls3D(position=(0.0, 150.0, 50.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 0.0, 1.0))),
        ]
        eye_in_head_panels = [
            rrb.TimeSeriesView(name="Left Eye-in-Head (deg)", origin="eye_in_head/left", plot_legend=rrb.PlotLegend(visible=True),  axis_y=rrb.ScalarAxis(range=(-45.0, 45.0))),
            rrb.TimeSeriesView(name="Right Eye-in-Head (deg)", origin="eye_in_head/right", plot_legend=rrb.PlotLegend(visible=True),  axis_y=rrb.ScalarAxis(range=(-45.0, 45.0))),
        ]

    if include_eye_in_head:
        left_column = rrb.Vertical(spatial_3d_view, rrb.Horizontal(*eye_local_views), row_shares=[2, 1])
        right_column = rrb.Vertical(*time_series_panels, *eye_in_head_panels)
        layout = rrb.Horizontal(left_column, right_column, column_shares=[1, 1])
    else:
        layout = rrb.Horizontal(spatial_3d_view, rrb.Vertical(*time_series_panels), column_shares=[1, 1])
    return rrb.Blueprint(layout, collapse_panels=True)


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================
def run_visualization(
    skull: SkullKinematics,
    trajectory_data: dict[str, NDArray[np.float64]],
    trajectory_timestamps: NDArray[np.float64],
    gaze: GazeKinematics | None = None,
    left_eye_resampled: EyeballKinematics | None = None,
    right_eye_resampled: EyeballKinematics | None = None,
    application_id: str = "ferret_head_kinematics",
    spawn: bool = True,
) -> None:
    """Run the Rerun visualization for head and gaze kinematics."""
    include_gaze = gaze is not None
    include_eye_in_head = left_eye_resampled is not None and right_eye_resampled is not None and gaze is not None

    rr.init(application_id)
    blueprint = create_blueprint(include_gaze=include_gaze, include_eye_in_head=include_eye_in_head)
    if spawn:
        rr.spawn()
    rr.send_blueprint(blueprint)
    setup_plot_styling(include_gaze=include_gaze, include_eye_in_head=include_eye_in_head)

    print(f"Logging {len(skull.timestamps)} frames...")
    print("  Sending enclosure...")
    send_enclosure()
    print("  Sending head kinematics...")
    send_head_kinematics(skull)
    print("  Sending head origin...")
    send_head_origin(skull.position, skull.timestamps)
    print("  Sending head basis vectors...")
    send_head_basis_vectors(skull, head_origins=skull.position, scale=100.0)
    print("  Sending skeleton data...")
    send_skeleton_data(trajectory_data=trajectory_data, trajectory_timestamps=trajectory_timestamps)

    if gaze is not None:
        print("  Sending gaze eyeballs...")
        send_gaze_eyeballs(gaze)
        print("  Sending gaze vectors...")
        send_gaze_vectors(gaze)
        print("  Sending gaze tracers...")
        send_gaze_tracers(gaze)
        print("  Sending gaze time series...")
        send_gaze_kinematics_timeseries(gaze)

    if left_eye_resampled is not None and right_eye_resampled is not None:
        print("  Sending eye-in-head time series...")
        send_eye_in_head_timeseries(left_eye=left_eye_resampled, right_eye=right_eye_resampled)
        if gaze is not None:
            print("  Sending eye-local views...")
            send_eye_local_views(gk=gaze, left_eye=left_eye_resampled, right_eye=right_eye_resampled)
    print("Done!")


def run_visualization_from_disk(clip_path: Path) -> None:
    """Load all data from disk and run visualization."""
    clip_path = Path(clip_path)
    analyzable_output = clip_path / "analyzable_output"
    if not analyzable_output.exists():
        raise FileNotFoundError(f"analyzable_output folder not found: {analyzable_output}")

    print("Loading data from disk...")
    skull = load_skull_kinematics_from_csv(analyzable_output / "skull_kinematics_resampled.csv")
    print(f"  Skull: {len(skull.timestamps)} frames")
    gaze = load_gaze_kinematics_from_csv(analyzable_output / "gaze_kinematics.csv")
    print(f"  Gaze: {len(gaze.timestamps)} frames")
    left_eye = load_eye_kinematics_from_csv(analyzable_output / "left_eye_kinematics_resampled.csv")
    print(f"  Left eye: {len(left_eye.timestamps)} frames")
    right_eye = load_eye_kinematics_from_csv(analyzable_output / "right_eye_kinematics_resampled.csv")
    print(f"  Right eye: {len(right_eye.timestamps)} frames")

    trajectory_csv = clip_path / "mocap_data" / "output_data" / "solver_output" / "skull_and_spine_trajectory_data.csv"
    trajectory_data = load_trajectory_data(trajectory_csv)
    print(f"  Trajectory: {len(trajectory_data['timestamps'])} frames")

    print("\nLaunching visualization...")
    run_visualization(skull=skull, trajectory_data=trajectory_data, trajectory_timestamps=trajectory_data["timestamps"],
                      gaze=gaze, left_eye_resampled=left_eye, right_eye_resampled=right_eye, application_id="ferret_gaze_kinematics", spawn=True)


if __name__ == "__main__":
    _clip_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s")
    run_visualization_from_disk(_clip_path)