"""
Ferret Head Kinematics Visualization (Rerun)

Displays:
- 3D skeleton viewer with animation
- Head origin point (mean of eyes and ears, white dot)
- Head basis vectors (x, y, z axes of head coordinate frame, 100mm arrows)
- Head position (mm)
- Head orientation (roll, pitch, yaw in degrees)
- Angular velocity in world frame (x, y, z in deg/s)
- Angular velocity in head-local frame (roll, pitch, yaw rates in deg/s)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray
from rerun.datatypes import Vec3D

from ferret_head_kinematics.ferret_head_kinematics import HeadKinematics


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


# =============================================================================
# DATA LOADING
# =============================================================================
def load_trajectory_data(trajectory_csv_path: Path) -> dict[int, dict[str, NDArray[np.float64]]]:
    """Load trajectory data from CSV and organize by frame."""
    df = pd.read_csv(trajectory_csv_path)
    frames: dict[int, dict[str, NDArray[np.float64]]] = {}

    for frame_idx in df["frame"].unique():
        frame_data = df[df["frame"] == frame_idx]
        frames[int(frame_idx)] = {}

        for _, row in frame_data.iterrows():
            marker = str(row["marker"])
            data_type = str(row["data_type"])
            if data_type == "optimized":
                frames[int(frame_idx)][marker] = np.array(
                    [row["x"], row["y"], row["z"]], dtype=np.float64
                )
            elif marker not in frames[int(frame_idx)]:
                frames[int(frame_idx)][marker] = np.array(
                    [row["x"], row["y"], row["z"]], dtype=np.float64
                )

    return frames


# =============================================================================
# SKELETON VISUALIZATION
# =============================================================================
def send_skeleton_data(
    trajectory_data: dict[int, dict[str, NDArray[np.float64]]],
    timestamps: NDArray[np.float64],
    t0: float,
) -> None:
    """Send skeleton marker and edge data to Rerun."""
    available_frames = sorted(trajectory_data.keys())

    marker_times: list[float] = []
    all_positions: list[NDArray[np.float64]] = []
    marker_partition_lengths: list[int] = []
    all_colors: list[NDArray[np.uint8]] = []

    edge_times: list[float] = []
    all_edge_strips: list[NDArray[np.float64]] = []
    edge_partition_lengths: list[int] = []

    for i, ts in enumerate(timestamps):
        time_val = float(ts - t0)

        closest_frames = [f for f in available_frames if f <= i]
        closest_frame = max(closest_frames) if closest_frames else available_frames[0]
        frame_data = trajectory_data[closest_frame]

        frame_positions: list[NDArray[np.float64]] = []
        frame_colors: list[NDArray[np.uint8]] = []

        for name in MARKER_NAMES:
            if name in frame_data:
                pos = frame_data[name]
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
            if name_i in frame_data and name_j in frame_data:
                pos_i = frame_data[name_i]
                pos_j = frame_data[name_j]
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
def send_head_kinematics(hk: HeadKinematics) -> None:
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
                positions=Vec3D(head_origins),
                # colors=[(255, 255, 255)] * head_origins.shape[0],
                # radii=[10.0]  * head_origins.shape[0],
            ),
        ],
    )


def send_head_basis_vectors(
    hk: HeadKinematics,
    head_origins: NDArray[np.float64],
    scale: float = 100.0,
) -> None:
    """Send head basis vectors as arrows in 3D view.

    Args:
        hk: HeadKinematics data
        head_origins: (N, 3) array of head origin positions (e.g. mean of eyes/ears)
        scale: Length of each basis vector arrow in mm (default 100mm)
    """
    t0 = hk.timestamps[0]
    n_frames = len(hk.timestamps)

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
        times: list[float] = []
        origins: list[NDArray[np.float64]] = []
        vectors: list[NDArray[np.float64]] = []

        for i in range(n_frames):
            time_val = float(hk.timestamps[i] - t0)
            times.append(time_val)
            origins.append(head_origins[i])
            vectors.append(basis[i] * scale)

        origins_array = np.array(origins)
        vectors_array = np.array(vectors)

        rr.send_columns(
            f"skeleton/head_basis/{axis_name}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=[
                *rr.Arrows3D.columns(
                    origins=origins_array,
                    vectors=vectors_array,
                    # colors=[colors[axis_name]] * n_frames,
                    # radii=[3.0] * n_frames,
                ),
            ],
        )


# =============================================================================
# STYLING
# =============================================================================
def setup_plot_styling() -> None:
    """Configure plot colors and styling."""
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


# =============================================================================
# BLUEPRINT
# =============================================================================
def create_blueprint() -> rrb.Blueprint:
    """Create Rerun blueprint for head kinematics visualization."""
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

    spatial_3d_view = rrb.Spatial3DView(
        name="3D Skeleton",
        origin="skeleton",
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

    return rrb.Blueprint(
        rrb.Horizontal(
            spatial_3d_view,
            rrb.Vertical(*time_series_panels),
            column_shares=[1, 1],
        ),
        collapse_panels=True,
    )


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================
def run_visualization(
    hk: HeadKinematics,
    trajectory_data: dict[int, dict[str, NDArray[np.float64]]],
    application_id: str = "ferret_head_kinematics",
    spawn: bool = True,
) -> None:
    """Run the Rerun visualization for head kinematics.

    Args:
        hk: HeadKinematics data
        trajectory_data: Trajectory data for 3D skeleton (required for head origin)
        application_id: Rerun application ID
        spawn: Whether to spawn the Rerun viewer
    """
    rr.init(application_id)

    blueprint = create_blueprint()

    if spawn:
        rr.spawn()

    rr.send_blueprint(blueprint)
    setup_plot_styling()

    print(f"Logging {len(hk.timestamps)} frames...")

    print("  Sending enclosure...")
    send_enclosure()

    print("  Sending head kinematics...")
    send_head_kinematics(hk)

    print("  Computing head origin from trajectory...")

    print("  Sending head origin...")
    send_head_origin(hk.position, hk.timestamps)

    print("  Sending head basis vectors...")
    send_head_basis_vectors(hk, head_origins=hk.position, scale=100.0)

    print("  Sending skeleton data...")
    send_skeleton_data(
        trajectory_data=trajectory_data,
        timestamps=hk.timestamps,
        t0=hk.timestamps[0],
    )

    print("Done!")



