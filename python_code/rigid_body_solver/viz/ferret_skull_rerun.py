"""
Rigid Body Kinematics Visualization (Rerun)

Displays:
- 3D skeleton viewer with animation
  - Skull keypoints (from RigidBodyKinematics - optimized rigid body)
  - Spine keypoints (from raw trajectory data - not rigid)
- Body origin point (white dot)
- Body basis vectors (x, y, z axes as 100mm arrows)
- Position (mm)
- Orientation (roll, pitch, yaw in degrees)
- Angular velocity in world frame (deg/s)
- Angular velocity in body-local frame (deg/s)
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry


# =============================================================================
# CONSTANTS
# =============================================================================

RAD_TO_DEG: float = 180.0 / np.pi

AXIS_COLORS: dict[str, tuple[int, int, int]] = {
    "roll": (255, 107, 107),
    "pitch": (78, 205, 196),
    "yaw": (255, 230, 109),
    "x": (255, 107, 107),
    "y": (78, 255, 96),
    "z": (100, 149, 255),  # Brighter blue for dark backgrounds
}

ENCLOSURE_SIZE_MM: float = 1000.0


# =============================================================================
# TOPOLOGY MODEL
# =============================================================================


class StickFigureTopology:
    """Simple topology class for loading from JSON."""

    def __init__(
        self,
        marker_names: list[str],
        display_edges: list[tuple[str, str]],
        rigid_edges: list[tuple[str, str]] | None = None,
        name: str = "topology",
    ):
        self.marker_names = marker_names
        self.display_edges = display_edges
        self.rigid_edges = rigid_edges or []
        self.name = name

    @classmethod
    def from_json_file(cls, path: Path) -> "StickFigureTopology":
        """Load topology from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            marker_names=data["marker_names"],
            display_edges=[tuple(e) for e in data["display_edges"]],
            rigid_edges=[tuple(e) for e in data.get("rigid_edges", [])],
            name=data.get("name", "topology"),
        )


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================


def load_kinematics_from_tidy_csv(
    csv_path: Path,
    reference_geometry: ReferenceGeometry,
    name: str,
) -> RigidBodyKinematics:
    """
    Load RigidBodyKinematics from a tidy-format CSV file.

    Args:
        csv_path: Path to the tidy-format kinematics CSV
        reference_geometry: The reference geometry for the rigid body
        name: Name for the kinematics object

    Returns:
        RigidBodyKinematics object
    """
    df = pl.read_csv(csv_path)

    # Extract timestamps from position data
    position_df = df.filter(pl.col("trajectory") == "position")
    timestamps = (
        position_df.filter(pl.col("component") == "x")
        .sort("frame")["timestamp_s"]
        .to_numpy()
    )
    n_frames = len(timestamps)

    # Extract position (N, 3)
    position_xyz = _extract_vector3_from_tidy(
        df=df,
        trajectory_name="position",
        components=["x", "y", "z"],
        n_frames=n_frames,
    )

    # Extract orientation quaternions (N, 4)
    quaternions_wxyz = _extract_vector4_from_tidy(
        df=df,
        trajectory_name="orientation",
        components=["w", "x", "y", "z"],
        n_frames=n_frames,
    )

    return RigidBodyKinematics.from_pose_arrays(
        name=name,
        reference_geometry=reference_geometry,
        timestamps=timestamps,
        position_xyz=position_xyz,
        quaternions_wxyz=quaternions_wxyz,
    )


def load_spine_trajectories_from_csv(
    csv_path: Path,
    spine_marker_names: list[str],
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """
    Load spine marker trajectories from tidy CSV.

    Args:
        csv_path: Path to skull_and_spine_trajectories.csv
        spine_marker_names: List of spine marker names to extract

    Returns:
        Tuple of (timestamps, trajectories_dict) where trajectories_dict
        maps marker name to (N, 3) array
    """
    df = pl.read_csv(csv_path)

    # Get timestamps from first marker
    first_marker = spine_marker_names[0]
    marker_df = df.filter(pl.col("trajectory") == first_marker)
    timestamps = (
        marker_df.filter(pl.col("component") == "x")
        .sort("frame")["timestamp"]
        .to_numpy()
    )
    n_frames = len(timestamps)

    trajectories: dict[str, NDArray[np.float64]] = {}
    for marker_name in spine_marker_names:
        marker_df = df.filter(pl.col("trajectory") == marker_name)
        if len(marker_df) == 0:
            continue

        positions = np.zeros((n_frames, 3), dtype=np.float64)
        for i, comp in enumerate(["x", "y", "z"]):
            comp_df = marker_df.filter(pl.col("component") == comp).sort("frame")
            positions[:, i] = comp_df["value"].to_numpy()

        trajectories[marker_name] = positions

    return timestamps, trajectories


def _extract_vector3_from_tidy(
    df: pl.DataFrame,
    trajectory_name: str,
    components: list[str],
    n_frames: int,
) -> NDArray[np.float64]:
    """Extract a (N, 3) array from tidy dataframe."""
    traj_df = df.filter(pl.col("trajectory") == trajectory_name)
    result = np.zeros((n_frames, 3), dtype=np.float64)

    for i, comp in enumerate(components):
        comp_df = traj_df.filter(pl.col("component") == comp).sort("frame")
        result[:, i] = comp_df["value"].to_numpy()

    return result


def _extract_vector4_from_tidy(
    df: pl.DataFrame,
    trajectory_name: str,
    components: list[str],
    n_frames: int,
) -> NDArray[np.float64]:
    """Extract a (N, 4) array from tidy dataframe."""
    traj_df = df.filter(pl.col("trajectory") == trajectory_name)
    result = np.zeros((n_frames, 4), dtype=np.float64)

    for i, comp in enumerate(components):
        comp_df = traj_df.filter(pl.col("component") == comp).sort("frame")
        result[:, i] = comp_df["value"].to_numpy()

    return result


# =============================================================================
# SKELETON VISUALIZATION
# =============================================================================


def send_skull_skeleton_data(
    kinematics: RigidBodyKinematics,
    marker_colors: dict[str, tuple[int, int, int]],
) -> None:
    """
    Send skull skeleton marker and edge data to Rerun.

    Args:
        kinematics: RigidBodyKinematics object with keypoint trajectories
        marker_colors: Dict mapping marker names to RGB colors
    """
    n_frames = kinematics.n_frames
    if n_frames == 0:
        raise ValueError("Kinematics has 0 frames")

    t0 = kinematics.timestamps[0]
    keypoint_names = kinematics.keypoint_names

    if kinematics.reference_geometry.display_edges is None:
        raise ValueError("reference_geometry.display_edges is None - display_edges are required")
    display_edges = kinematics.reference_geometry.display_edges

    # Build marker data
    marker_times: list[float] = []
    all_positions: list[NDArray[np.float64]] = []
    marker_partition_lengths: list[int] = []
    all_colors: list[NDArray[np.uint8]] = []

    # Build edge data
    edge_times: list[float] = []
    all_edge_strips: list[NDArray[np.float64]] = []
    edge_partition_lengths: list[int] = []

    for frame_idx in range(n_frames):
        time_val = float(kinematics.timestamps[frame_idx] - t0)

        # Collect marker positions for this frame
        frame_positions: list[NDArray[np.float64]] = []
        frame_colors: list[NDArray[np.uint8]] = []

        for name in keypoint_names:
            pos = kinematics.keypoint_trajectories[name][frame_idx]
            if not np.any(np.isnan(pos)):
                frame_positions.append(pos)
                color = marker_colors.get(name, (78, 205, 196))  # Cyan default for skull
                frame_colors.append(np.array(color, dtype=np.uint8))

        if frame_positions:
            marker_times.append(time_val)
            all_positions.extend(frame_positions)
            all_colors.extend(frame_colors)
            marker_partition_lengths.append(len(frame_positions))

        # Collect edge data for this frame
        frame_strips: list[NDArray[np.float64]] = []
        for name_i, name_j in display_edges:
            if name_i in kinematics.keypoint_trajectories and name_j in kinematics.keypoint_trajectories:
                pos_i = kinematics.keypoint_trajectories[name_i][frame_idx]
                pos_j = kinematics.keypoint_trajectories[name_j][frame_idx]
                if not np.any(np.isnan(pos_i)) and not np.any(np.isnan(pos_j)):
                    frame_strips.append(np.array([pos_i, pos_j], dtype=np.float64))

        if frame_strips:
            edge_times.append(time_val)
            all_edge_strips.extend(frame_strips)
            edge_partition_lengths.append(len(frame_strips))

    # Send marker data
    if marker_times and all_positions:
        rr.send_columns(
            "skeleton/skull/markers",
            indexes=[rr.TimeColumn("time", duration=marker_times)],
            columns=[
                *rr.Points3D.columns(positions=np.array(all_positions)).partition(
                    lengths=marker_partition_lengths
                ),
                *rr.Points3D.columns(
                    colors=np.array(all_colors),
                    radii=[5.0] * len(all_positions),
                ).partition(lengths=marker_partition_lengths),
            ],
        )

    # Send edge data
    if edge_times and all_edge_strips:
        rr.send_columns(
            "skeleton/skull/edges",
            indexes=[rr.TimeColumn("time", duration=edge_times)],
            columns=[
                *rr.LineStrips3D.columns(
                    strips=all_edge_strips,
                    colors=[(0, 200, 200)] * len(all_edge_strips),
                    radii=[2.0] * len(all_edge_strips),
                ).partition(lengths=edge_partition_lengths)
            ],
        )


def send_spine_skeleton_data(
    timestamps: NDArray[np.float64],
    spine_trajectories: dict[str, NDArray[np.float64]],
    all_trajectories_for_edges: dict[str, NDArray[np.float64]],
    display_edges: list[tuple[str, str]],
    marker_colors: dict[str, tuple[int, int, int]],
) -> None:
    """
    Send spine skeleton marker and edge data to Rerun.

    Args:
        timestamps: (N,) array of timestamps
        spine_trajectories: Dict mapping spine marker name to (N, 3) array (for markers)
        all_trajectories_for_edges: Dict with ALL marker trajectories (skull + spine) for edge drawing
        display_edges: List of (marker_i, marker_j) edges to display
        marker_colors: Dict mapping marker names to RGB colors
    """
    n_frames = len(timestamps)
    if n_frames == 0:
        raise ValueError("Spine timestamps has 0 frames")

    t0 = timestamps[0]
    spine_marker_names = list(spine_trajectories.keys())

    # Build marker data
    marker_times: list[float] = []
    all_positions: list[NDArray[np.float64]] = []
    marker_partition_lengths: list[int] = []
    all_colors: list[NDArray[np.uint8]] = []

    # Build edge data
    edge_times: list[float] = []
    all_edge_strips: list[NDArray[np.float64]] = []
    edge_partition_lengths: list[int] = []

    for frame_idx in range(n_frames):
        time_val = float(timestamps[frame_idx] - t0)

        # Collect marker positions for this frame (only spine markers)
        frame_positions: list[NDArray[np.float64]] = []
        frame_colors: list[NDArray[np.uint8]] = []

        for name in spine_marker_names:
            pos = spine_trajectories[name][frame_idx]
            if not np.any(np.isnan(pos)):
                frame_positions.append(pos)
                color = marker_colors.get(name, (221, 160, 221))  # Pink/purple default for spine
                frame_colors.append(np.array(color, dtype=np.uint8))

        if frame_positions:
            marker_times.append(time_val)
            all_positions.extend(frame_positions)
            all_colors.extend(frame_colors)
            marker_partition_lengths.append(len(frame_positions))

        # Collect edge data for this frame (can include skull-spine connectors)
        frame_strips: list[NDArray[np.float64]] = []
        for name_i, name_j in display_edges:
            if name_i in all_trajectories_for_edges and name_j in all_trajectories_for_edges:
                pos_i = all_trajectories_for_edges[name_i][frame_idx]
                pos_j = all_trajectories_for_edges[name_j][frame_idx]
                if not np.any(np.isnan(pos_i)) and not np.any(np.isnan(pos_j)):
                    frame_strips.append(np.array([pos_i, pos_j], dtype=np.float64))

        if frame_strips:
            edge_times.append(time_val)
            all_edge_strips.extend(frame_strips)
            edge_partition_lengths.append(len(frame_strips))

    # Send marker data (larger, different color)
    if marker_times and all_positions:
        rr.send_columns(
            "skeleton/spine/markers",
            indexes=[rr.TimeColumn("time", duration=marker_times)],
            columns=[
                *rr.Points3D.columns(positions=np.array(all_positions)).partition(
                    lengths=marker_partition_lengths
                ),
                *rr.Points3D.columns(
                    colors=np.array(all_colors),
                    radii=[6.0] * len(all_positions),
                ).partition(lengths=marker_partition_lengths),
            ],
        )

    # Send edge data (dashed appearance via thinner lines, different color)
    if edge_times and all_edge_strips:
        rr.send_columns(
            "skeleton/spine/edges",
            indexes=[rr.TimeColumn("time", duration=edge_times)],
            columns=[
                *rr.LineStrips3D.columns(
                    strips=all_edge_strips,
                    colors=[(200, 100, 200)] * len(all_edge_strips),  # Purple
                    radii=[1.5] * len(all_edge_strips),
                ).partition(lengths=edge_partition_lengths)
            ],
        )


def send_enclosure() -> None:
    """Send a 1m³ enclosure as wireframe box."""
    half = ENCLOSURE_SIZE_MM / 2.0
    corners = np.array(
        [
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
            [-half, -half, ENCLOSURE_SIZE_MM],
            [half, -half, ENCLOSURE_SIZE_MM],
            [half, half, ENCLOSURE_SIZE_MM],
            [-half, half, ENCLOSURE_SIZE_MM],
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


def send_kinematics_timeseries(kinematics: RigidBodyKinematics) -> None:
    """
    Send kinematics time series data to Rerun.

    Args:
        kinematics: RigidBodyKinematics object
    """
    t0 = kinematics.timestamps[0]
    times = kinematics.timestamps - t0

    # Position (mm)
    for i, name in enumerate(["x", "y", "z"]):
        rr.send_columns(
            f"position/{name}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=kinematics.position_xyz[:, i]),
        )

    # Orientation (roll, pitch, yaw in degrees)
    euler_rad = kinematics.orientations.to_euler_xyz_array()
    euler_deg = euler_rad * RAD_TO_DEG
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        rr.send_columns(
            f"orientation/{name}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=euler_deg[:, i]),
        )

    # Angular velocity - global frame (deg/s)
    omega_global_deg_s = kinematics.angular_velocity_global * RAD_TO_DEG
    for i, name in enumerate(["x", "y", "z"]):
        rr.send_columns(
            f"omega_global/{name}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=omega_global_deg_s[:, i]),
        )

    # Angular velocity - body/local frame (deg/s)
    omega_local_deg_s = kinematics.angular_velocity_local * RAD_TO_DEG
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        rr.send_columns(
            f"omega_body/{name}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=omega_local_deg_s[:, i]),
        )


def send_body_origin(kinematics: RigidBodyKinematics) -> None:
    """Send the body origin point over time."""
    t0 = kinematics.timestamps[0]
    times = kinematics.timestamps - t0
    rr.send_columns(
        "skeleton/body_origin",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[*rr.Points3D.columns(positions=kinematics.position_xyz)],
    )


def send_body_basis_vectors(
    kinematics: RigidBodyKinematics,
    scale: float = 100.0,
) -> None:
    """
    Send body-frame basis vectors as arrows.

    Args:
        kinematics: RigidBodyKinematics object
        scale: Length of arrows in mm
    """
    t0 = kinematics.timestamps[0]
    n_frames = kinematics.n_frames
    times = kinematics.timestamps - t0

    # Get rotation matrices for all frames: (N, 3, 3)
    rotation_matrices = kinematics.orientations.to_rotation_matrices()

    colors = {"x": (255, 107, 107), "y": (78, 255, 96), "z": (55, 20, 255)}
    origins = kinematics.position_xyz

    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        # Extract basis vector for this axis: (N, 3) = rotation_matrices[:, :, axis_idx]
        vectors = rotation_matrices[:, :, axis_idx] * scale
        color_array = np.tile(
            np.array(colors[axis_name], dtype=np.uint8),
            (n_frames, 1),
        )
        rr.send_columns(
            f"skeleton/body_basis/{axis_name}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=[
                *rr.Arrows3D.columns(
                    origins=origins,
                    vectors=vectors,
                    colors=color_array,
                ).partition(lengths=[1] * n_frames)
            ],
        )


# =============================================================================
# PLOT STYLING & BLUEPRINT
# =============================================================================


def setup_plot_styling() -> None:
    """Configure plot series styling - both lines AND points for each series."""
    # Position - lines + dots
    for name in ["x", "y", "z"]:
        rr.log(
            f"position/{name}",
            rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name], widths=[1.5]),
            static=True,
        )
        rr.log(
            f"position/{name}",
            rr.SeriesPoints(colors=[AXIS_COLORS[name]], marker_sizes=[2.0]),
            static=True,
        )

    # Orientation - lines + dots
    for name in ["roll", "pitch", "yaw"]:
        rr.log(
            f"orientation/{name}",
            rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name], widths=[1.5]),
            static=True,
        )
        rr.log(
            f"orientation/{name}",
            rr.SeriesPoints(colors=[AXIS_COLORS[name]], marker_sizes=[2.0]),
            static=True,
        )

    # Angular velocity - global - lines + dots
    for name in ["x", "y", "z"]:
        rr.log(
            f"omega_global/{name}",
            rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name], widths=[1.5]),
            static=True,
        )
        rr.log(
            f"omega_global/{name}",
            rr.SeriesPoints(colors=[AXIS_COLORS[name]], marker_sizes=[2.0]),
            static=True,
        )

    # Angular velocity - body/local - lines + dots
    for name in ["roll", "pitch", "yaw"]:
        rr.log(
            f"omega_body/{name}",
            rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name], widths=[1.5]),
            static=True,
        )
        rr.log(
            f"omega_body/{name}",
            rr.SeriesPoints(colors=[AXIS_COLORS[name]], marker_sizes=[2.0]),
            static=True,
        )

    # Body origin styling
    rr.log(
        "skeleton/body_origin",
        rr.Points3D.from_fields(radii=8.0, colors=(255, 255, 255)),
        static=True,
    )


def create_blueprint(time_window_seconds: float) -> rrb.Blueprint:
    """
    Create the Rerun viewer layout blueprint.

    Args:
        time_window_seconds: How many seconds before/after cursor to show in time series plots
    """
    # Scrolling time range centered on cursor
    scrolling_time_range = rrb.VisibleTimeRange(
        timeline="time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    time_series_panels = [
        rrb.TimeSeriesView(
            name="Position (mm)",
            origin="position",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-500.0, 500.0)),
            time_ranges=scrolling_time_range,
        ),
        rrb.TimeSeriesView(
            name="Orientation (deg)",
            origin="orientation",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-180.0, 180.0)),
            time_ranges=scrolling_time_range,
        ),
        rrb.TimeSeriesView(
            name="ω Global Frame (deg/s)",
            origin="omega_global",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-800.0, 800.0)),
            time_ranges=scrolling_time_range,
        ),
        rrb.TimeSeriesView(
            name="ω Body Frame (deg/s)",
            origin="omega_body",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-800.0, 800.0)),
            time_ranges=scrolling_time_range,
        ),
    ]

    spatial_3d_view = rrb.Spatial3DView(
        name="3D Skeleton",
        origin="skeleton",
        contents=["+ skeleton/**"],
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
    )

    layout = rrb.Horizontal(
        spatial_3d_view,
        rrb.Vertical(*time_series_panels),
        column_shares=[1, 1],
    )
    return rrb.Blueprint(layout, collapse_panels=True)


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================


def run_visualization(
    kinematics: RigidBodyKinematics,
    spine_timestamps: NDArray[np.float64],
    spine_trajectories: dict[str, NDArray[np.float64]],
    spine_display_edges: list[tuple[str, str]],
    skull_marker_colors: dict[str, tuple[int, int, int]],
    spine_marker_colors: dict[str, tuple[int, int, int]],
    application_id: str,
    spawn: bool,
    time_window_seconds: float,
) -> None:
    """
    Run the Rerun visualization for RigidBodyKinematics.

    Args:
        kinematics: RigidBodyKinematics object to visualize
        spine_timestamps: Timestamps for spine data
        spine_trajectories: Dict of spine marker trajectories
        spine_display_edges: List of spine display edges
        skull_marker_colors: Dict mapping skull marker names to RGB colors
        spine_marker_colors: Dict mapping spine marker names to RGB colors
        application_id: Rerun application ID
        spawn: Whether to spawn a new viewer window
        time_window_seconds: How many seconds before/after cursor to show in time series
    """
    rr.init(application_id)
    blueprint = create_blueprint(time_window_seconds=time_window_seconds)
    if spawn:
        rr.spawn()
    rr.send_blueprint(blueprint)
    setup_plot_styling()

    print(f"Logging {kinematics.n_frames} frames...")
    print("  Sending enclosure...")
    send_enclosure()
    print("  Sending kinematics time series...")
    send_kinematics_timeseries(kinematics)
    print("  Sending body origin...")
    send_body_origin(kinematics)
    print("  Sending body basis vectors...")
    send_body_basis_vectors(kinematics=kinematics, scale=100.0)
    print("  Sending skull skeleton data...")
    send_skull_skeleton_data(kinematics=kinematics, marker_colors=skull_marker_colors)

    # Build combined trajectories dict for spine edge drawing (includes skull keypoints)
    all_trajectories_for_edges: dict[str, NDArray[np.float64]] = {}
    for keypoint_name in kinematics.keypoint_names:
        all_trajectories_for_edges[keypoint_name] = kinematics.keypoint_trajectories[keypoint_name]
    for spine_name, spine_traj in spine_trajectories.items():
        all_trajectories_for_edges[spine_name] = spine_traj

    print("  Sending spine skeleton data...")
    send_spine_skeleton_data(
        timestamps=spine_timestamps,
        spine_trajectories=spine_trajectories,
        all_trajectories_for_edges=all_trajectories_for_edges,
        display_edges=spine_display_edges,
        marker_colors=spine_marker_colors,
    )

    print("Done!")


# =============================================================================
# FERRET SKULL + SPINE SPECIFIC CONFIGURATION
# =============================================================================


SKULL_MARKER_COLORS: dict[str, tuple[int, int, int]] = {
    "nose": (255, 107, 107),
    "left_eye": (78, 205, 196),
    "right_eye": (78, 205, 196),
    "left_ear": (149, 225, 211),
    "right_ear": (149, 225, 211),
    "base": (255, 230, 109),
    "left_cam_tip": (168, 230, 207),
    "right_cam_tip": (168, 230, 207),
}

SPINE_MARKER_COLORS: dict[str, tuple[int, int, int]] = {
    "spine_t1": (221, 160, 221),
    "sacrum": (255, 182, 193),
    "tail_tip": (255, 105, 180),
}

SPINE_MARKER_NAMES: list[str] = ["spine_t1", "sacrum", "tail_tip"]


def run_ferret_skull_and_spine_visualization(
    output_dir: Path,
    spawn: bool,
    time_window_seconds: float,
) -> RigidBodyKinematics:
    """
    Visualization for ferret skull + spine.

    Required files in output_dir:
        - skull_reference_geometry.json
        - skull_kinematics.csv
        - skull_and_spine_trajectories.csv
        - skull_and_spine_topology.json

    Args:
        output_dir: Directory containing the kinematics output files
        spawn: Whether to spawn a new viewer window
        time_window_seconds: How many seconds before/after cursor to show in time series

    Returns:
        The loaded RigidBodyKinematics object
    """
    output_dir = Path(output_dir)

    # Define required file paths
    reference_geometry_json = output_dir / "skull_reference_geometry.json"
    kinematics_csv = output_dir / "skull_kinematics.csv"
    spine_trajectories_csv = output_dir / "skull_and_spine_trajectories.csv"
    topology_json = output_dir / "skull_and_spine_topology.json"

    # Check all required files exist - FAIL LOUDLY if missing
    for filepath in [reference_geometry_json, kinematics_csv, spine_trajectories_csv, topology_json]:
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")

    print("Loading data from disk...")

    # Load skull kinematics
    reference_geometry = ReferenceGeometry.from_json_file(reference_geometry_json)
    print(f"  Reference geometry: {len(reference_geometry.markers)} markers")

    kinematics = load_kinematics_from_tidy_csv(
        csv_path=kinematics_csv,
        reference_geometry=reference_geometry,
        name="skull",
    )
    print(f"  Skull kinematics: {kinematics.n_frames} frames")

    # Load spine trajectories
    spine_timestamps, spine_trajectories = load_spine_trajectories_from_csv(
        csv_path=spine_trajectories_csv,
        spine_marker_names=SPINE_MARKER_NAMES,
    )
    print(f"  Spine trajectories: {len(spine_trajectories)} markers, {len(spine_timestamps)} frames")

    # Load topology for display edges
    topology = StickFigureTopology.from_json_file(topology_json)
    spine_display_edges = [
        (a, b) for a, b in topology.display_edges
        if a in SPINE_MARKER_NAMES or b in SPINE_MARKER_NAMES
    ]
    print(f"  Topology: {len(spine_display_edges)} spine display edges")

    print("\nLaunching visualization...")
    run_visualization(
        kinematics=kinematics,
        spine_timestamps=spine_timestamps,
        spine_trajectories=spine_trajectories,
        spine_display_edges=spine_display_edges,
        skull_marker_colors=SKULL_MARKER_COLORS,
        spine_marker_colors=SPINE_MARKER_COLORS,
        application_id="ferret_skull_kinematics",
        spawn=spawn,
        time_window_seconds=time_window_seconds,
    )

    return kinematics


if __name__ == "__main__":
    _output_dir = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output"
    )
    run_ferret_skull_and_spine_visualization(
        output_dir=_output_dir,
        spawn=True,
        time_window_seconds=3,
    )