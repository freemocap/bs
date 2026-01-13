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

from python_code.ferret_gaze.kinematics_calculators.ferret_eye_kinematics import EyeKinematics
from python_code.ferret_gaze.kinematics_calculators.ferret_gaze_kinematics import GazeKinematics
from python_code.ferret_gaze.kinematics_calculators.ferret_skull_kinematics import SkullKinematics

# =============================================================================
# TOPOLOGY DEFINITION
# =============================================================================
SKULL_MARKER_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "base",
]
EYE_CAM_MARKER_NAMES: list[str] = [
    "left_cam_tip",
    "right_cam_tip",
]
BODY_MARKER_NAMES: list[str] = [
    "spine_t1",
    "sacrum",
    "tail_tip",
]

MARKER_NAMES: list[str] = SKULL_MARKER_NAMES + EYE_CAM_MARKER_NAMES + BODY_MARKER_NAMES

DISPLAY_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5),
    (5, 6), (5, 7), (3, 8), (4, 8), (8, 9), (9, 10),
]

# RGB colors (0-255)
MARKER_COLORS: dict[str, tuple[int, int, int]] = {
    "nose": (255, 107, 107),
    "left_eye": (78, 205, 196),
    "right_eye": (78, 205, 196),
    "left_ear": (149, 225, 211),
    "right_ear": (149, 225, 211),
    "base": (255, 230, 109),
    "left_cam_tip": (168, 230, 207),
    "right_cam_tip": (168, 230, 207),
    "spine_t1": (221, 160, 221),
    "sacrum": (221, 160, 221),
    "tail_tip": (221, 160, 221),
}

AXIS_COLORS: dict[str, tuple[int, int, int]] = {
    "roll": (255, 107, 107),   # Red - X axis
    "pitch": (78, 205, 196),   # Cyan - Y axis
    "yaw": (255, 230, 109),    # Yellow - Z axis
    "x": (255, 107, 107),
    "y": (78, 205, 196),
    "z": (255, 230, 109),
}

ENCLOSURE_SIZE_MM: float = 1000.0

# Gaze visualization constants
EYEBALL_RADIUS_MM: float = 5.0  # Approximate ferret eyeball radius
EYEBALL_COLOR_LEFT: tuple[int, int, int, int] = (100, 200, 255, 180)  # Light blue, semi-transparent
EYEBALL_COLOR_RIGHT: tuple[int, int, int, int] = (255, 150, 100, 180)  # Light orange, semi-transparent
GAZE_VECTOR_COLOR_LEFT: tuple[int, int, int] = (0, 150, 255)  # Blue
GAZE_VECTOR_COLOR_RIGHT: tuple[int, int, int] = (255, 100, 0)  # Orange
GAZE_VECTOR_RADIUS_MM: float = 1.5
GAZE_TRACER_RADIUS_MM: float = 2.0
GAZE_TRACER_HISTORY_SECONDS: float = 2.0

# Eye-local view constants
EYE_LOCAL_GAZE_LENGTH_MM: float = 50.0  # Length of gaze vector in eye-local view


# =============================================================================
# DATA LOADING
# =============================================================================
def load_trajectory_data(trajectory_csv_path: Path) -> dict[str, NDArray[np.float64]]:
    """Load trajectory data from CSV into arrays of shape (n_frames, 3).

    Args:
        trajectory_csv_path: Path to tidy_trajectory_data.csv

    Returns:
        Dictionary mapping marker name to position array of shape (n_frames, 3),
        plus a "timestamps" key with shape (n_frames,)
    """
    df = pd.read_csv(trajectory_csv_path)

    frame_indices = df["frame"].unique()
    n_frames = len(frame_indices)
    if n_frames == 0:
        raise ValueError(f"No frames found in {trajectory_csv_path}")

    # Verify frames are contiguous starting from 0
    expected_frames = np.arange(n_frames)
    if not np.array_equal(np.sort(frame_indices), expected_frames):
        raise ValueError(f"Frame indices must be contiguous from 0 to {n_frames - 1}, got: {sorted(frame_indices)}")

    marker_names = df["marker"].unique()
    if len(marker_names) == 0:
        raise ValueError(f"No markers found in {trajectory_csv_path}")

    # Prefer "optimized" data, fall back to "original"
    optimized_df = df[df["data_type"] == "optimized"]
    original_df = df[df["data_type"] == "original"]

    body_trajectories: dict[str, NDArray[np.float64]] = {}

    # Extract timestamps per frame
    timestamps = np.zeros(n_frames, dtype=np.float64)
    for frame_idx in range(n_frames):
        frame_rows = df[df["frame"] == frame_idx]
        if frame_rows.empty:
            raise ValueError(f"No data found for frame {frame_idx}")
        unique_timestamps = frame_rows["timestamp"].unique()
        if len(unique_timestamps) != 1:
            raise ValueError(f"Frame {frame_idx} has multiple timestamps: {unique_timestamps}")
        timestamps[frame_idx] = float(unique_timestamps[0])
    body_trajectories["timestamps"] = timestamps

    for marker in marker_names:
        positions = np.zeros((n_frames, 3), dtype=np.float64)

        for frame_idx in range(n_frames):
            # Try optimized first
            marker_frame_data = optimized_df[
                (optimized_df["marker"] == marker) & (optimized_df["frame"] == frame_idx)
            ]
            if marker_frame_data.empty:
                # Fall back to original
                marker_frame_data = original_df[
                    (original_df["marker"] == marker) & (original_df["frame"] == frame_idx)
                ]
            if marker_frame_data.empty:
                raise ValueError(f"No data found for marker '{marker}' at frame {frame_idx}")
            if len(marker_frame_data) > 1:
                raise ValueError(f"Multiple entries for marker '{marker}' at frame {frame_idx}")

            row = marker_frame_data.iloc[0]
            positions[frame_idx, 0] = float(row["x"])
            positions[frame_idx, 1] = float(row["y"])
            positions[frame_idx, 2] = float(row["z"])

        body_trajectories[str(marker)] = positions

    return body_trajectories


# =============================================================================
# SKELETON VISUALIZATION
# =============================================================================
def send_skeleton_data(
    trajectory_data: dict[str, NDArray[np.float64]],
    trajectory_timestamps: NDArray[np.float64],
) -> None:
    """Send skeleton marker and edge data to Rerun at original trajectory timestamps.

    Args:
        trajectory_data: Dictionary mapping marker name to position array of shape (n_frames, 3)
        trajectory_timestamps: Timestamps array of shape (n_frames,)
    """
    n_frames = len(trajectory_timestamps)
    if n_frames == 0:
        return

    t0 = trajectory_timestamps[0]

    marker_times: list[float] = []
    all_positions: list[NDArray[np.float64]] = []
    marker_partition_lengths: list[int] = []
    all_colors: list[NDArray[np.uint8]] = []

    edge_times: list[float] = []
    all_edge_strips: list[NDArray[np.float64]] = []
    edge_partition_lengths: list[int] = []

    for frame_idx in range(n_frames):
        time_val = float(trajectory_timestamps[frame_idx] - t0)

        frame_positions: list[NDArray[np.float64]] = []
        frame_colors: list[NDArray[np.uint8]] = []

        for name in MARKER_NAMES:
            if name in trajectory_data:
                pos = trajectory_data[name][frame_idx]
                if not np.any(np.isnan(pos)):
                    frame_positions.append(pos)
                    color = MARKER_COLORS.get(name, (255, 255, 255))
                    frame_colors.append(np.array(color, dtype=np.uint8))

        if frame_positions:
            marker_times.append(time_val)
            all_positions.extend(frame_positions)
            all_colors.extend(frame_colors)
            marker_partition_lengths.append(len(frame_positions))

        frame_strips: list[NDArray[np.float64]] = []
        for idx_i, idx_j in DISPLAY_EDGES:
            name_i = MARKER_NAMES[idx_i]
            name_j = MARKER_NAMES[idx_j]
            if name_i in trajectory_data and name_j in trajectory_data:
                pos_i = trajectory_data[name_i][frame_idx]
                pos_j = trajectory_data[name_j][frame_idx]
                if not np.any(np.isnan(pos_i)) and not np.any(np.isnan(pos_j)):
                    strip = np.array([pos_i, pos_j], dtype=np.float64)
                    frame_strips.append(strip)

        if frame_strips:
            edge_times.append(time_val)
            all_edge_strips.extend(frame_strips)
            edge_partition_lengths.append(len(frame_strips))

    # Send markers
    if marker_times and all_positions:
        positions_array = np.array(all_positions)
        colors_array = np.array(all_colors)

        rr.send_columns(
            "skeleton/markers",
            indexes=[rr.TimeColumn("time", duration=marker_times)],
            columns=[
                *rr.Points3D.columns(positions=positions_array).partition(
                    lengths=marker_partition_lengths
                ),
                *rr.Points3D.columns(
                    colors=colors_array,
                    radii=[4.0] * len(all_positions),
                ).partition(lengths=marker_partition_lengths),
            ],
        )

    # Send edges
    if edge_times and all_edge_strips:
        rr.send_columns(
            "skeleton/edges",
            indexes=[rr.TimeColumn("time", duration=edge_times)],
            columns=[
                *rr.LineStrips3D.columns(
                    strips=all_edge_strips,
                    colors=[(0, 200, 200)] * len(all_edge_strips),
                    radii=[1.0] * len(all_edge_strips),
                ).partition(lengths=edge_partition_lengths),
            ],
        )


def send_enclosure() -> None:
    """Send a 1m³ enclosure as wireframe box."""
    half = ENCLOSURE_SIZE_MM / 2.0

    corners = np.array(
        [
            [-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0],
            [-half, -half, ENCLOSURE_SIZE_MM], [half, -half, ENCLOSURE_SIZE_MM],
            [half, half, ENCLOSURE_SIZE_MM], [-half, half, ENCLOSURE_SIZE_MM],
        ],
        dtype=np.float64,
    )

    edge_indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    strips = [np.array([corners[i], corners[j]]) for i, j in edge_indices]

    rr.log(
        "skeleton/enclosure",
        rr.LineStrips3D(
            strips=strips,
            colors=[(200, 200, 200, 100)] * len(strips),
            radii=[2] * len(strips),
        ),
        static=True,
    )


# =============================================================================
# TIME SERIES LOGGING
# =============================================================================
def send_head_kinematics(hk: SkullKinematics) -> None:
    """Send head kinematics time series data to Rerun."""
    t0 = hk.timestamps[0]
    times = hk.timestamps - t0

    # Position
    rr.send_columns(
        "position/x",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.position[:, 0]),
    )
    rr.send_columns(
        "position/y",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.position[:, 1]),
    )
    rr.send_columns(
        "position/z",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.position[:, 2]),
    )

    # Euler angles (degrees)
    rr.send_columns(
        "orientation/roll",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.euler_angles_deg[:, 0]),
    )
    rr.send_columns(
        "orientation/pitch",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.euler_angles_deg[:, 1]),
    )
    rr.send_columns(
        "orientation/yaw",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.euler_angles_deg[:, 2]),
    )

    # Angular velocity - world frame (deg/s)
    rr.send_columns(
        "omega_world/x",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.angular_velocity_world_deg_s[:, 0]),
    )
    rr.send_columns(
        "omega_world/y",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.angular_velocity_world_deg_s[:, 1]),
    )
    rr.send_columns(
        "omega_world/z",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.angular_velocity_world_deg_s[:, 2]),
    )

    # Angular velocity - local/head frame (deg/s)
    rr.send_columns(
        "omega_local/roll",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.angular_velocity_local_deg_s[:, 0]),
    )
    rr.send_columns(
        "omega_local/pitch",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.angular_velocity_local_deg_s[:, 1]),
    )
    rr.send_columns(
        "omega_local/yaw",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=hk.angular_velocity_local_deg_s[:, 2]),
    )


def send_head_origin(
    head_origins: NDArray[np.float64],
    timestamps: NDArray[np.float64],
) -> None:
    """Send head origin point to Rerun.

    Args:
        head_origins: (N, 3) array of head origin positions
        timestamps: Array of timestamps
    """
    t0 = timestamps[0]
    times = timestamps - t0
    rr.send_columns(
        "skeleton/head_origin",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Points3D.columns(
                positions=head_origins,
            ),
        ],
    )


def send_head_basis_vectors(
    hk: SkullKinematics,
    head_origins: NDArray[np.float64],
    scale: float = 100.0,
) -> None:
    """Send head basis vectors as arrows in 3D view.

    Args:
        hk: SkullKinematics data
        head_origins: (N, 3) array of head origin positions (e.g. mean of eyes/ears)
        scale: Length of each basis vector arrow in mm (default 100mm)
    """
    t0 = hk.timestamps[0]
    n_frames = len(hk.timestamps)
    times = hk.timestamps - t0

    # Colors for each axis (RGB)
    colors = {
        "x": (255, 107, 107),  # Red
        "y": (78, 255, 96),   # Green
        "z": (55, 20, 255),  # Blue
    }

    basis_data = {
        "x": hk.basis_x,
        "y": hk.basis_y,
        "z": hk.basis_z,
    }

    for axis_name, basis in basis_data.items():
        vectors = basis * scale
        color = colors[axis_name]
        color_array = np.tile(np.array(color, dtype=np.uint8), (n_frames, 1))

        rr.send_columns(
            f"skeleton/head_basis/{axis_name}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=[
                *rr.Arrows3D.columns(
                    origins=head_origins,
                    vectors=vectors,
                    colors=color_array,
                ).partition(lengths=[1] * n_frames),
            ],
        )


# =============================================================================
# GAZE VISUALIZATION
# =============================================================================
def send_gaze_eyeballs(gk: GazeKinematics) -> None:
    """Send eyeball spheres to Rerun.

    Args:
        gk: GazeKinematics data containing eye center positions
    """
    t0 = gk.timestamps[0]
    times = gk.timestamps - t0
    n_frames = len(gk.timestamps)

    # Half sizes for spheres (equal in all dimensions for sphere)
    half_sizes = np.tile(
        np.array([[EYEBALL_RADIUS_MM, EYEBALL_RADIUS_MM, EYEBALL_RADIUS_MM]]),
        (n_frames, 1),
    )

    # Colors arrays
    left_colors = np.tile(np.array(EYEBALL_COLOR_LEFT, dtype=np.uint8), (n_frames, 1))
    right_colors = np.tile(np.array(EYEBALL_COLOR_RIGHT, dtype=np.uint8), (n_frames, 1))

    # Left eye - use send_columns
    rr.send_columns(
        "skeleton/gaze/left_eyeball",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Ellipsoids3D.columns(
                centers=gk.left_eyeball_center_xyz_mm,
                half_sizes=half_sizes,
                colors=left_colors,
            ).partition(lengths=[1] * n_frames),
        ],
    )

    # Right eye - use send_columns
    rr.send_columns(
        "skeleton/gaze/right_eyeball",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Ellipsoids3D.columns(
                centers=gk.right_eyeball_center_xyz_mm,
                half_sizes=half_sizes,
                colors=right_colors,
            ).partition(lengths=[1] * n_frames),
        ],
    )


def send_gaze_vectors(gk: GazeKinematics) -> None:
    """Send gaze direction arrows to Rerun.

    Args:
        gk: GazeKinematics data containing gaze directions and eye centers
    """
    t0 = gk.timestamps[0]
    times = gk.timestamps - t0
    n_frames = len(gk.timestamps)

    # Left eye gaze vectors - use send_columns for efficient batching
    left_vectors = gk.left_gaze_endpoint_mm - gk.left_eyeball_center_xyz_mm
    left_colors = np.tile(np.array(GAZE_VECTOR_COLOR_LEFT, dtype=np.uint8), (n_frames, 1))

    rr.send_columns(
        "skeleton/gaze/left_gaze_vector",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Arrows3D.columns(
                origins=gk.left_eyeball_center_xyz_mm,
                vectors=left_vectors,
                colors=left_colors,
            ).partition(lengths=[1] * n_frames),
        ],
    )

    # Right eye gaze vectors
    right_vectors = gk.right_gaze_endpoint_mm - gk.right_eyeball_center_xyz_mm
    right_colors = np.tile(np.array(GAZE_VECTOR_COLOR_RIGHT, dtype=np.uint8), (n_frames, 1))

    rr.send_columns(
        "skeleton/gaze/right_gaze_vector",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Arrows3D.columns(
                origins=gk.right_eyeball_center_xyz_mm,
                vectors=right_vectors,
                colors=right_colors,
            ).partition(lengths=[1] * n_frames),
        ],
    )


def send_gaze_tracers(gk: GazeKinematics, trail_duration_s: float = GAZE_TRACER_HISTORY_SECONDS) -> None:
    """Send gaze endpoint tracer points with explicit trailing history.

    Logs trail points as a collection of recent positions, not relying on time_ranges.

    Args:
        gk: GazeKinematics data containing gaze endpoints
        trail_duration_s: Duration of trail history in seconds
    """
    t0 = gk.timestamps[0]
    times = gk.timestamps - t0
    n_frames = len(gk.timestamps)

    # Compute frame duration to determine how many frames in trail
    if n_frames < 2:
        return
    median_dt = float(np.median(np.diff(gk.timestamps)))
    trail_frames = max(1, int(trail_duration_s / median_dt))

    for i in range(n_frames):
        rr.set_time("time", duration=float(times[i]))

        # Get trail indices (from max(0, i-trail_frames) to i inclusive)
        start_idx = max(0, i - trail_frames)
        trail_indices = range(start_idx, i + 1)

        # Left eye trail - all points from start_idx to current
        left_trail_positions = gk.left_gaze_endpoint_mm[start_idx:i + 1]
        # Fade colors from dim to bright based on age
        n_trail = len(left_trail_positions)
        left_alphas = np.linspace(50, 255, n_trail).astype(np.uint8)
        left_colors = np.array([
            [GAZE_VECTOR_COLOR_LEFT[0], GAZE_VECTOR_COLOR_LEFT[1], GAZE_VECTOR_COLOR_LEFT[2], a]
            for a in left_alphas
        ], dtype=np.uint8)
        left_radii = np.linspace(GAZE_TRACER_RADIUS_MM * 0.5, GAZE_TRACER_RADIUS_MM, n_trail)

        rr.log(
            "skeleton/gaze/left_tracer",
            rr.Points3D(
                positions=left_trail_positions,
                colors=left_colors,
                radii=left_radii,
            ),
        )

        # Right eye trail
        right_trail_positions = gk.right_gaze_endpoint_mm[start_idx:i + 1]
        right_colors = np.array([
            [GAZE_VECTOR_COLOR_RIGHT[0], GAZE_VECTOR_COLOR_RIGHT[1], GAZE_VECTOR_COLOR_RIGHT[2], a]
            for a in left_alphas  # Same alpha pattern
        ], dtype=np.uint8)
        right_radii = np.linspace(GAZE_TRACER_RADIUS_MM * 0.5, GAZE_TRACER_RADIUS_MM, n_trail)

        rr.log(
            "skeleton/gaze/right_tracer",
            rr.Points3D(
                positions=right_trail_positions,
                colors=right_colors,
                radii=right_radii,
            ),
        )


def send_eye_local_views(
    gk: GazeKinematics,
    left_eye: EyeKinematics,
    right_eye: EyeKinematics,
) -> None:
    """Send eye-local coordinate data for individual eye views.

    In eye-local coordinates:
    - Eyeball is at origin
    - Gaze direction emanates from origin
    - X: medial(-)/lateral(+), Y: up in eye-local coords, Z: forward

    Args:
        gk: GazeKinematics data
        left_eye: Left EyeKinematics data (resampled to same timestamps as gk)
        right_eye: Right EyeKinematics data (resampled to same timestamps as gk)
    """
    t0 = gk.timestamps[0]
    times = gk.timestamps - t0
    n_frames = len(gk.timestamps)

    # Eye data is already resampled - both eyes now share the same timestamps
    left_x = left_eye.eye_angle_x_rad
    left_y = left_eye.eye_angle_y_rad
    right_x = right_eye.eye_angle_x_rad
    right_y = right_eye.eye_angle_y_rad

    # Eye-local eyeball at origin (static)
    half_size = np.array([EYEBALL_RADIUS_MM, EYEBALL_RADIUS_MM, EYEBALL_RADIUS_MM])

    rr.log(
        "eye_local/left/eyeball",
        rr.Ellipsoids3D(
            centers=[[0, 0, 0]],
            half_sizes=[half_size],
            colors=[EYEBALL_COLOR_LEFT],
            fill_mode="solid",
        ),
        static=True,
    )
    rr.log(
        "eye_local/right/eyeball",
        rr.Ellipsoids3D(
            centers=[[0, 0, 0]],
            half_sizes=[half_size],
            colors=[EYEBALL_COLOR_RIGHT],
            fill_mode="solid",
        ),
        static=True,
    )

    if n_frames < 2:
        return

    # Compute trail parameters
    median_dt = float(np.median(np.diff(gk.timestamps)))
    trail_frames = max(1, int(GAZE_TRACER_HISTORY_SECONDS / median_dt))

    # Precompute all gaze directions
    left_dirs = np.zeros((n_frames, 3), dtype=np.float64)
    right_dirs = np.zeros((n_frames, 3), dtype=np.float64)
    origins = np.zeros((n_frames, 3), dtype=np.float64)  # All at [0,0,0]

    for i in range(n_frames):
        # Left eye direction
        left_dir = np.array([
            -left_x[i],  # medial/lateral
            1.0,         # forward (primary direction)
            left_y[i],   # up/down
        ])
        left_dirs[i] = left_dir / np.linalg.norm(left_dir) * EYE_LOCAL_GAZE_LENGTH_MM

        # Right eye direction
        right_dir = np.array([
            -right_x[i],  # medial/lateral
            -1.0,         # forward (primary direction, -Y for right eye)
            right_y[i],   # up/down
        ])
        right_dirs[i] = right_dir / np.linalg.norm(right_dir) * EYE_LOCAL_GAZE_LENGTH_MM

    # Send gaze vectors using send_columns (no trails on arrows)
    left_colors = np.tile(np.array(GAZE_VECTOR_COLOR_LEFT, dtype=np.uint8), (n_frames, 1))
    right_colors_arr = np.tile(np.array(GAZE_VECTOR_COLOR_RIGHT, dtype=np.uint8), (n_frames, 1))

    rr.send_columns(
        "eye_local/left/gaze_vector",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Arrows3D.columns(
                origins=origins,
                vectors=left_dirs,
                colors=left_colors,
            ).partition(lengths=[1] * n_frames),
        ],
    )

    rr.send_columns(
        "eye_local/right/gaze_vector",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Arrows3D.columns(
                origins=origins,
                vectors=right_dirs,
                colors=right_colors_arr,
            ).partition(lengths=[1] * n_frames),
        ],
    )

    # Send tracers with explicit trails (using set_time/log for trail accumulation)
    for i in range(n_frames):
        rr.set_time("time", duration=float(times[i]))

        # Left eye tracer with trail
        start_idx = max(0, i - trail_frames)
        n_trail = i - start_idx + 1
        left_trail = left_dirs[start_idx:i + 1]
        left_alphas = np.linspace(50, 255, n_trail).astype(np.uint8)
        left_colors = np.array([
            [GAZE_VECTOR_COLOR_LEFT[0], GAZE_VECTOR_COLOR_LEFT[1], GAZE_VECTOR_COLOR_LEFT[2], a]
            for a in left_alphas
        ], dtype=np.uint8)
        left_radii = np.linspace(GAZE_TRACER_RADIUS_MM * 0.5, GAZE_TRACER_RADIUS_MM, n_trail)

        rr.log(
            "eye_local/left/tracer",
            rr.Points3D(
                positions=left_trail,
                colors=left_colors,
                radii=left_radii,
            ),
        )

        # Right eye tracer with trail
        right_trail = right_dirs[start_idx:i + 1]
        right_colors = np.array([
            [GAZE_VECTOR_COLOR_RIGHT[0], GAZE_VECTOR_COLOR_RIGHT[1], GAZE_VECTOR_COLOR_RIGHT[2], a]
            for a in left_alphas
        ], dtype=np.uint8)
        right_radii = np.linspace(GAZE_TRACER_RADIUS_MM * 0.5, GAZE_TRACER_RADIUS_MM, n_trail)

        rr.log(
            "eye_local/right/tracer",
            rr.Points3D(
                positions=right_trail,
                colors=right_colors,
                radii=right_radii,
            ),
        )


def send_gaze_kinematics_timeseries(gk: GazeKinematics) -> None:
    """Send gaze kinematics time series data to Rerun.

    Args:
        gk: GazeKinematics data
    """
    t0 = gk.timestamps[0]
    times = gk.timestamps - t0

    # Left eye gaze angles
    rr.send_columns(
        "gaze/left_azimuth",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=np.rad2deg(gk.left_gaze_azimuth_rad)),
    )
    rr.send_columns(
        "gaze/left_elevation",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=np.rad2deg(gk.left_gaze_elevation_rad)),
    )

    # Right eye gaze angles
    rr.send_columns(
        "gaze/right_azimuth",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=np.rad2deg(gk.right_gaze_azimuth_rad)),
    )
    rr.send_columns(
        "gaze/right_elevation",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=np.rad2deg(gk.right_gaze_elevation_rad)),
    )


def send_eye_in_head_timeseries(
    left_eye: EyeKinematics,
    right_eye: EyeKinematics,
) -> None:
    """Send eye-in-head angle time series data to Rerun.

    Args:
        left_eye: Left EyeKinematics data (resampled)
        right_eye: Right EyeKinematics data (resampled)
    """
    # After resampling, left and right timestamps are the same
    t0 = left_eye.timestamps[0]
    times = left_eye.timestamps - t0

    # Left eye - horizontal (X) and vertical (Y) angles in degrees
    rr.send_columns(
        "eye_in_head/left/horizontal",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=np.rad2deg(left_eye.eye_angle_x_rad)),
    )
    rr.send_columns(
        "eye_in_head/left/vertical",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=np.rad2deg(left_eye.eye_angle_y_rad)),
    )

    # Right eye - horizontal (X) and vertical (Y) angles in degrees
    rr.send_columns(
        "eye_in_head/right/horizontal",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=np.rad2deg(right_eye.eye_angle_x_rad)),
    )
    rr.send_columns(
        "eye_in_head/right/vertical",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Scalars.columns(scalars=np.rad2deg(right_eye.eye_angle_y_rad)),
    )


# =============================================================================
# STYLING
# =============================================================================
def setup_plot_styling(include_gaze: bool = False, include_eye_in_head: bool = False) -> None:
    """Configure plot colors and styling.

    Args:
        include_gaze: Whether to include gaze-related styling
        include_eye_in_head: Whether to include eye-in-head styling
    """
    # Position
    for axis, name in [("x", "x"), ("y", "y"), ("z", "z")]:
        rr.log(f"position/{axis}", rr.SeriesLines(colors=AXIS_COLORS[axis], names=name), static=True)
        rr.log(f"position/{axis}", rr.SeriesPoints(colors=AXIS_COLORS[axis], markers="circle", marker_sizes=1.0), static=True)

    # Orientation
    for axis, name in [("roll", "Roll"), ("pitch", "Pitch"), ("yaw", "Yaw")]:
        rr.log(f"orientation/{axis}", rr.SeriesLines(colors=AXIS_COLORS[axis], names=name), static=True)
        rr.log(f"orientation/{axis}", rr.SeriesPoints(colors=AXIS_COLORS[axis], markers="circle", marker_sizes=1.0), static=True)

    # Angular velocity - world frame
    for axis, name in [("x", "ω_x"), ("y", "ω_y"), ("z", "ω_z")]:
        rr.log(f"omega_world/{axis}", rr.SeriesLines(colors=AXIS_COLORS[axis], names=name), static=True)
        rr.log(f"omega_world/{axis}", rr.SeriesPoints(colors=AXIS_COLORS[axis], markers="circle", marker_sizes=1.0), static=True)

    # Angular velocity - local/head frame
    for axis, name in [("roll", "ω_roll"), ("pitch", "ω_pitch"), ("yaw", "ω_yaw")]:
        rr.log(f"omega_local/{axis}", rr.SeriesLines(colors=AXIS_COLORS[axis], names=name), static=True)
        rr.log(f"omega_local/{axis}", rr.SeriesPoints(colors=AXIS_COLORS[axis], markers="circle", marker_sizes=1.0), static=True)

    if include_gaze:
        # Gaze time series styling
        rr.log("gaze/left_azimuth", rr.SeriesLines(colors=GAZE_VECTOR_COLOR_LEFT, names="L_az"), static=True)
        rr.log("gaze/left_elevation", rr.SeriesLines(colors=GAZE_VECTOR_COLOR_LEFT, names="L_el"), static=True)
        rr.log("gaze/right_azimuth", rr.SeriesLines(colors=GAZE_VECTOR_COLOR_RIGHT, names="R_az"), static=True)
        rr.log("gaze/right_elevation", rr.SeriesLines(colors=GAZE_VECTOR_COLOR_RIGHT, names="R_el"), static=True)

    if include_eye_in_head:
        # Eye-in-head time series styling
        rr.log("eye_in_head/left/horizontal", rr.SeriesLines(colors=GAZE_VECTOR_COLOR_LEFT, names="L_horiz"), static=True)
        rr.log("eye_in_head/left/vertical", rr.SeriesLines(colors=(0, 200, 255), names="L_vert"), static=True)
        rr.log("eye_in_head/right/horizontal", rr.SeriesLines(colors=GAZE_VECTOR_COLOR_RIGHT, names="R_horiz"), static=True)
        rr.log("eye_in_head/right/vertical", rr.SeriesLines(colors=(255, 150, 50), names="R_vert"), static=True)


# =============================================================================
# BLUEPRINT
# =============================================================================
def create_blueprint(include_gaze: bool = False, include_eye_in_head: bool = False) -> rrb.Blueprint:
    """Create Rerun blueprint for head kinematics visualization.

    Args:
        include_gaze: Whether to include gaze-related panels
        include_eye_in_head: Whether to include eye-in-head panels
    """
    linked_axis = rrb.archetypes.TimeAxis(link="LinkToGlobal")

    time_series_panels = [
        rrb.TimeSeriesView(
            name="Position (mm)",
            origin="position",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_x=linked_axis,
        ),
        rrb.TimeSeriesView(
            name="Orientation (deg)",
            origin="orientation",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-200.0, 200.0)),
            axis_x=linked_axis,
        ),
        rrb.TimeSeriesView(
            name="Angular Velocity - World Frame (deg/s)",
            origin="omega_world",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_x=linked_axis,
            axis_y=rrb.ScalarAxis(range=(-800.0, 800.0)),

        ),
        rrb.TimeSeriesView(
            name="Angular Velocity - Head Local (deg/s)",
            origin="omega_local",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_x=linked_axis,
            axis_y=rrb.ScalarAxis(range=(-800.0, 800.0)),
        ),
    ]

    if include_gaze:
        time_series_panels.append(
            rrb.TimeSeriesView(
                name="Gaze Direction (deg)",
                origin="gaze",
                plot_legend=rrb.PlotLegend(visible=True),
                axis_x=linked_axis,
                axis_y=rrb.ScalarAxis(range=(-180.0, 180.0)),
            ),
        )

    # Main 3D skeleton view - no time_ranges since tracers handle their own trails
    spatial_3d_view = rrb.Spatial3DView(
        name="3D Skeleton",
        origin="skeleton",
        contents=[
            "+ skeleton/**",
        ],
        eye_controls=rrb.EyeControls3D(
            position=(0.0, -2000.0, 500.0),
            look_target=(0.0, 0.0, 0.0),
            eye_up=(0.0, 0.0, 1.0),
        ),
        line_grid=rrb.LineGrid3D(
            visible=True,
            spacing=100.0,
            plane=rr.components.Plane3D.XY,
            color=[100, 100, 100, 128],
        ),
        spatial_information=rrb.SpatialInformation(
            show_axes=True,
            show_bounding_box=False,
        ),
    )

    # Eye-local 3D views - no time_ranges since tracers handle their own trails
    eye_local_views = []
    if include_eye_in_head:
        eye_local_views = [
            rrb.Spatial3DView(
                name="Left Eye (Local)",
                origin="eye_local/left",
                eye_controls=rrb.EyeControls3D(
                    position=(0.0, -150.0, 50.0),
                    look_target=(0.0, 0.0, 0.0),
                    eye_up=(0.0, 0.0, 1.0),
                ),
                spatial_information=rrb.SpatialInformation(
                    show_axes=True,
                    show_bounding_box=False,
                ),
            ),
            rrb.Spatial3DView(
                name="Right Eye (Local)",
                origin="eye_local/right",
                eye_controls=rrb.EyeControls3D(
                    position=(0.0, 150.0, 50.0),
                    look_target=(0.0, 0.0, 0.0),
                    eye_up=(0.0, 0.0, 1.0),
                ),
                spatial_information=rrb.SpatialInformation(
                    show_axes=True,
                    show_bounding_box=False,
                ),
            ),
        ]

    # Eye-in-head time series panels
    eye_in_head_panels = []
    if include_eye_in_head:
        eye_in_head_panels = [
            rrb.TimeSeriesView(
                name="Left Eye-in-Head (deg)",
                origin="eye_in_head/left",
                plot_legend=rrb.PlotLegend(visible=True),
                axis_x=linked_axis,
                axis_y=rrb.ScalarAxis(range=(-45.0, 45.0)),
            ),
            rrb.TimeSeriesView(
                name="Right Eye-in-Head (deg)",
                origin="eye_in_head/right",
                plot_legend=rrb.PlotLegend(visible=True),
                axis_x=linked_axis,
                axis_y=rrb.ScalarAxis(range=(-45.0, 45.0)),
            ),
        ]

    # Build layout
    if include_eye_in_head:
        # Layout with eye-local views
        left_column = rrb.Vertical(
            spatial_3d_view,
            rrb.Horizontal(*eye_local_views),
            row_shares=[2, 1],
        )
        right_column = rrb.Vertical(
            *time_series_panels,
            *eye_in_head_panels,
        )
        layout = rrb.Horizontal(
            left_column,
            right_column,
            column_shares=[1, 1],
        )
    else:
        layout = rrb.Horizontal(
            spatial_3d_view,
            rrb.Vertical(*time_series_panels),
            column_shares=[1, 1],
        )

    return rrb.Blueprint(
        layout,
        collapse_panels=True,
    )


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================
def run_visualization(
    skull: SkullKinematics,
    trajectory_data: dict[str, NDArray[np.float64]],
    trajectory_timestamps: NDArray[np.float64],
    gaze: GazeKinematics | None = None,
    left_eye_resampled: EyeKinematics | None = None,
    right_eye_resampled: EyeKinematics | None = None,
    application_id: str = "ferret_head_kinematics",
    spawn: bool = True,
) -> None:
    """Run the Rerun visualization for head and gaze kinematics.

    Args:
        skull: SkullKinematics data (resampled)
        trajectory_data: Trajectory data dict mapping marker name to (n_frames, 3) arrays
        trajectory_timestamps: Timestamps array of shape (n_frames,)
        gaze: Optional GazeKinematics data for gaze visualization
        left_eye_resampled: Optional left EyeKinematics data (resampled)
        right_eye_resampled: Optional right EyeKinematics data (resampled)
        application_id: Rerun application ID
        spawn: Whether to spawn the Rerun viewer
    """
    include_gaze = gaze is not None
    include_eye_in_head = (
        left_eye_resampled is not None
        and right_eye_resampled is not None
        and gaze is not None
    )

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

    print("  Computing head origin from trajectory...")

    print("  Sending head origin...")
    send_head_origin(skull.position, skull.timestamps)

    print("  Sending head basis vectors...")
    send_head_basis_vectors(skull, head_origins=skull.position, scale=100.0)

    print("  Sending skeleton data...")
    send_skeleton_data(
        trajectory_data=trajectory_data,
        trajectory_timestamps=trajectory_timestamps,
    )

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
        send_eye_in_head_timeseries(
            left_eye=left_eye_resampled,
            right_eye=right_eye_resampled,
        )

        if gaze is not None:
            print("  Sending eye-local views...")
            send_eye_local_views(
                gk=gaze,
                left_eye=left_eye_resampled,
                right_eye=right_eye_resampled,
            )

    print("Done!")