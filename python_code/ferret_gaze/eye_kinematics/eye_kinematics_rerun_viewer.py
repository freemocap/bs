"""
Ferret Eye Kinematics Rerun Viewer - Binocular Version
=======================================================

Visualizes BOTH left and right eye kinematics simultaneously.

COORDINATE SYSTEM (matches model frame):
    +Z = gaze direction (toward pupil, "north pole")
    +Y = superior (up)
    +X = subject's left
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import (
    extract_frame_data,
    load_eye_trajectories_csv,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

NUM_PUPIL_POINTS: int = 8

# Colors (RGB) - Base colors for 3D visualization
COLOR_GAZE_ARROW = [255, 50, 255]
COLOR_EYE_X_AXIS = [255, 0, 0]
COLOR_EYE_Y_AXIS = [0, 255, 0]
COLOR_EYE_Z_AXIS = [0, 100, 255]
COLOR_WORLD_X = [200, 50, 50]
COLOR_WORLD_Y = [50, 200, 50]
COLOR_WORLD_Z = [50, 50, 200]
COLOR_PUPIL_CENTER = [0, 0, 0]
COLOR_PUPIL_BOUNDARY = [0, 0, 139]
COLOR_PUPIL_POINTS = [70, 130, 180]
COLOR_TEAR_DUCT = [255, 140, 0]
COLOR_OUTER_EYE = [255, 100, 0]
COLOR_SOCKET_LINE = [255, 165, 0]
COLOR_SPHERE_WIRE = [20, 20, 20]
COLOR_EYE_CENTER = [128, 0, 128]
COLOR_PUPIL_FACE_RIGHT = [155, 100, 100]
COLOR_PUPIL_FACE_LEFT = [100, 100,155]

# Colors for timeseries (left vs right eye)
COLOR_LEFT_EYE_PRIMARY = [0, 150, 255]  # Blue
COLOR_LEFT_EYE_SECONDARY = [100, 180, 255]  # Light blue
COLOR_RIGHT_EYE_PRIMARY = [255, 100, 0]  # Orange
COLOR_RIGHT_EYE_SECONDARY = [255, 160, 80]  # Light orange


@dataclass
class EyeViewerData:
    """Data for a single eye in the viewer."""

    kinematics: FerretEyeKinematics
    pixel_data: dict[str, NDArray[np.float64]] | None = None
    video_path: Path | None = None


def set_time_seconds(timeline: str, seconds: float) -> None:
    """Set time on timeline with Rerun API compatibility."""
    if hasattr(rr, "set_time"):
        try:
            rr.set_time(timeline, timestamp=seconds)
            return
        except TypeError:
            pass
    if hasattr(rr, "set_time_seconds"):
        try:
            rr.set_time_seconds(timeline, seconds)
            return
        except TypeError:
            pass
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(timeline, int(seconds * 1e9))
        return
    raise RuntimeError("Could not find compatible Rerun time API.")


def rotate_strips(
    strips: list[NDArray[np.float64]],
    rotation_matrix: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Rotate all line strips by a rotation matrix."""
    return [(rotation_matrix @ strip.T).T for strip in strips]


def quaternion_to_rotation_matrix(q_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-10:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def generate_sphere_line_strips_with_pole_at_z(
    radius: float,
    n_lat: int = 10,
    n_lon: int = 20,
) -> list[NDArray[np.float64]]:
    """Generate wireframe sphere with north pole at +Z."""
    strips = []
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        r_circle = radius * np.sin(theta)
        z_pos = radius * np.cos(theta)
        circle = []
        for j in range(n_lon + 1):
            phi = 2 * np.pi * j / n_lon
            circle.append([r_circle * np.cos(phi), r_circle * np.sin(phi), z_pos])
        strips.append(np.array(circle, dtype=np.float64))
    for j in range(0, n_lon, 2):
        phi = 2 * np.pi * j / n_lon
        circle = []
        for i in range(n_lat + 1):
            theta = np.pi * i / n_lat
            circle.append(
                [
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                ]
            )
        strips.append(np.array(circle, dtype=np.float64))
    return strips


def generate_sphere_mesh(
    radius: float,
    n_lat: int = 12,
    n_lon: int = 24,
) -> tuple[NDArray[np.float64], NDArray[np.uint32]]:
    """Generate sphere mesh with pole at +Z."""
    vertices = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            vertices.append(
                [
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                ]
            )
    vertices = np.array(vertices, dtype=np.float64)

    triangles = []
    for i in range(n_lat):
        for j in range(n_lon):
            curr = i * n_lon + j
            next_j = i * n_lon + (j + 1) % n_lon
            below = (i + 1) * n_lon + j
            below_next = (i + 1) * n_lon + (j + 1) % n_lon
            if i > 0:
                triangles.append([curr, below, next_j])
            if i < n_lat - 1:
                triangles.append([next_j, below, below_next])
    return vertices, np.array(triangles, dtype=np.uint32)


def generate_great_circle(
    radius: float,
    normal: NDArray[np.float64],
    n_points: int = 64,
) -> NDArray[np.float64]:
    """Generate points on a great circle perpendicular to the given normal."""
    normal = normal / np.linalg.norm(normal)
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    else:
        u = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    points = np.zeros((n_points + 1, 3), dtype=np.float64)
    for i, angle in enumerate(angles):
        points[i] = radius * (np.cos(angle) * u + np.sin(angle) * v)

    return points


def create_binocular_viewer_blueprint(
    time_window_seconds: float = 2.0,
    has_left_video: bool = False,
    has_right_video: bool = False,
) -> rrb.Blueprint:
    """Create Rerun blueprint for binocular eye viewer layout.

    All timeseries views share the same cursor-relative time window:
    - The time window shows ±time_window_seconds from the cursor position
    - Current time appears as a vertical line in the center
    - All plots scroll together as you move the time cursor

    Note: True linked zoom/pan (axis_x linking) requires Rerun 0.29+.
    In 0.28.x, all plots share the same cursor-relative time window.
    """
    # Shared time range for ALL timeseries views
    # cursor_relative keeps the current time centered in the view
    scrolling_time_range = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    # Top-down camera settings: looking down from +Z axis at the origin
    # Eye is above the scene along Z, looking down at the eyeball
    top_down_eye_controls = rrb.EyeControls3D(
        position=(0.0, 0.0, 15.0),  # Camera positioned along +Z axis
        look_target=(0.0, 0.0, 0.0),  # Looking at origin (eye center)
        eye_up=(0.0, 1.0, 0.0),  # Y+ is "up" in the view
        kind=rrb.Eye3DKind.Orbital,
    )

    # 3D views for each eye with top-down camera
    left_eye_3d = rrb.Spatial3DView(
        name="Left Eye 3D",
        origin="/",
        contents=["+ /left_eye/**", "+ /world_frame/**"],
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=top_down_eye_controls,
    )

    right_eye_3d = rrb.Spatial3DView(
        name="Right Eye 3D",
        origin="/",
        contents=["+ /right_eye/**", "+ /world_frame/**"],
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=top_down_eye_controls,
    )

    # Video views
    left_video_view = rrb.Spatial2DView(name="Left Eye Video", origin="/video/left_eye")
    right_video_view = rrb.Spatial2DView(name="Right Eye Video", origin="/video/right_eye")

    # Left eye timeseries (order: pixels, angles, velocity, acceleration)
    # All share the same scrolling_time_range so they stay synchronized
    # Y range included in name as fallback when Rerun hides tick labels
    left_pupil_view = rrb.TimeSeriesView(
        name="Left Pupil (px) [±40]",
        origin="/timeseries/pupil_position/left_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-40.0, 40.0)),
    )

    left_angle_view = rrb.TimeSeriesView(
        name="Left Angles (deg) [±25]",
        origin="/timeseries/angles/left_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-25.0, 25.0)),
    )

    left_velocity_view = rrb.TimeSeriesView(
        name="Left Velocity (deg/s) [±350]",
        origin="/timeseries/velocity/left_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-350.0, 350.0)),
    )

    left_acceleration_view = rrb.TimeSeriesView(
        name="Left Accel (deg/s²) [±5000]",
        origin="/timeseries/acceleration/left_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-5000.0, 5000.0)),
    )

    # Right eye timeseries (order: pixels, angles, velocity, acceleration)
    # Y range included in name as fallback when Rerun hides tick labels
    right_pupil_view = rrb.TimeSeriesView(
        name="Right Pupil (px) [±40]",
        origin="/timeseries/pupil_position/right_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-40.0, 40.0)),
    )

    right_angle_view = rrb.TimeSeriesView(
        name="Right Angles (deg) [±25]",
        origin="/timeseries/angles/right_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-25.0, 25.0)),
    )

    right_velocity_view = rrb.TimeSeriesView(
        name="Right Velocity (deg/s) [±350]",
        origin="/timeseries/velocity/right_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-350.0, 350.0)),
    )

    right_acceleration_view = rrb.TimeSeriesView(
        name="Right Accel (deg/s²) [±5000]",
        origin="/timeseries/acceleration/right_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-5000.0, 5000.0)),
    )

    # Timeseries stacks (pixels, angles, velocity, acceleration - top to bottom)
    left_timeseries = rrb.Vertical(
        left_pupil_view, left_angle_view, left_velocity_view, left_acceleration_view
    )
    right_timeseries = rrb.Vertical(
        right_pupil_view, right_angle_view, right_velocity_view, right_acceleration_view
    )

    # Build layout: Left eye column | Right eye column
    # Each column: [3D + Video in hbox], timeseries below
    if has_left_video and has_right_video:
        left_top = rrb.Horizontal(left_eye_3d, left_video_view)
        right_top = rrb.Horizontal(right_eye_3d, right_video_view)
        left_column = rrb.Vertical(left_top, left_timeseries, row_shares=[0.20, 0.80])
        right_column = rrb.Vertical(right_top, right_timeseries, row_shares=[0.20, 0.80])
    elif has_left_video:
        left_top = rrb.Horizontal(left_eye_3d, left_video_view)
        left_column = rrb.Vertical(left_top, left_timeseries, row_shares=[0.20, 0.80])
        right_column = rrb.Vertical(right_eye_3d, right_timeseries, row_shares=[0.20, 0.80])
    elif has_right_video:
        right_top = rrb.Horizontal(right_eye_3d, right_video_view)
        left_column = rrb.Vertical(left_eye_3d, left_timeseries, row_shares=[0.20, 0.80])
        right_column = rrb.Vertical(right_top, right_timeseries, row_shares=[0.20, 0.80])
    else:
        left_column = rrb.Vertical(left_eye_3d, left_timeseries, row_shares=[0.20, 0.80])
        right_column = rrb.Vertical(right_eye_3d, right_timeseries, row_shares=[0.20, 0.80])

    return rrb.Blueprint(
        rrb.Horizontal(left_column, right_column, column_shares=[0.5, 0.5]),
        rrb.TimePanel(state="expanded"),
        collapse_panels=True,
    )


def log_static_world_frame(eye_prefix: str, axis_length: float, eye_radius: float) -> None:
    """Log static world reference frame axes and reference circles for an eye."""
    # Log view coordinates to hint at preferred orientation (Y-up, looking down -Y axis)
    rr.log(
        f"{eye_prefix}",
        rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        static=True,
    )

    # World axes
    rr.log(
        f"{eye_prefix}/world_frame/x_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[axis_length, 0, 0]],
            colors=[COLOR_WORLD_X],
            radii=[0.04],
        ),
        static=True,
    )
    rr.log(
        f"{eye_prefix}/world_frame/y_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[0, axis_length, 0]],
            colors=[COLOR_WORLD_Y],
            radii=[0.04],
        ),
        static=True,
    )
    rr.log(
        f"{eye_prefix}/world_frame/z_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[0, 0, axis_length]],
            colors=[COLOR_WORLD_Z],
            radii=[0.04],
        ),
        static=True,
    )

    # Reference circles at eye rest position
    horizontal_circle = generate_great_circle(
        radius=eye_radius,
        normal=np.array([0.0, 1.0, 0.0]),
        n_points=64,
    )
    rr.log(
        f"{eye_prefix}/reference/horizontal_circle",
        rr.LineStrips3D(
            strips=[horizontal_circle],
            colors=[[155, 0, 0]],
            radii=[0.03],
        ),
        static=True,
    )

    vertical_circle = generate_great_circle(
        radius=eye_radius,
        normal=np.array([1.0, 0.0, 0.0]),
        n_points=64,
    )
    rr.log(
        f"{eye_prefix}/reference/vertical_circle",
        rr.LineStrips3D(
            strips=[vertical_circle],
            colors=[[0, 155, 0]],
            radii=[0.03],
        ),
        static=True,
    )


def log_eye_basis_vectors(
    eye_prefix: str, quaternion: NDArray[np.float64], axis_length: float
) -> None:
    """Log eye frame basis vectors (rotate with eye)."""
    R = quaternion_to_rotation_matrix(quaternion)
    axes = [
        np.array([axis_length, 0, 0]),
        np.array([0, axis_length, 0]),
        np.array([0, 0, axis_length]),
    ]
    colors = [COLOR_EYE_X_AXIS, COLOR_EYE_Y_AXIS, COLOR_EYE_Z_AXIS]
    names = ["x", "y", "z"]
    for axis, color, name in zip(axes, colors, names):
        v = R @ axis
        rr.log(
            f"{eye_prefix}/basis/{name}_axis",
            rr.Arrows3D(origins=[[0, 0, 0]], vectors=[v], colors=[color], radii=[0.06]),
        )


def log_rotating_sphere_and_gaze(
    eye_prefix: str,
    quaternion: NDArray[np.float64],
    eye_radius: float,
    gaze_length: float,
) -> None:
    """Log wireframe sphere and gaze arrow."""
    R = quaternion_to_rotation_matrix(quaternion)

    # Wireframe
    local_strips = generate_sphere_line_strips_with_pole_at_z(eye_radius, 8, 16)
    rotated_strips = [s for s in rotate_strips(local_strips, R)]
    rr.log(
        f"{eye_prefix}/sphere/wireframe",
        rr.LineStrips3D(
            strips=rotated_strips,
            colors=[COLOR_SPHERE_WIRE] * len(rotated_strips),
            radii=[0.015],
        ),
    )

    # Mesh
    verts, tris = generate_sphere_mesh(eye_radius * 0.99, 12, 24)
    verts_rotated = (R @ verts.T).T
    rr.log(
        f"{eye_prefix}/sphere/mesh",
        rr.Mesh3D(
            vertex_positions=verts_rotated,
            triangle_indices=tris,
            albedo_factor=[200, 200, 220, 25],
        ),
    )

    # Gaze arrow (rest gaze = +Z)
    gaze_vec = R @ np.array([0, 0, gaze_length])
    rr.log(
        f"{eye_prefix}/gaze_arrow",
        rr.Arrows3D(
            origins=[[0, 0, 0]], vectors=[gaze_vec], colors=[COLOR_GAZE_ARROW], radii=[0.1]
        ),
    )


def log_pupil_geometry(
    eye_prefix: str,
    pupil_center: NDArray[np.float64],
    pupil_points: NDArray[np.float64],
) -> None:
    """Log actual tracked pupil center, boundary, and filled face."""
    pc = pupil_center
    pp = pupil_points

    rr.log(
        f"{eye_prefix}/pupil/center",
        rr.Points3D(positions=[pc], colors=[COLOR_PUPIL_CENTER], radii=[0.12]),
    )
    rr.log(
        f"{eye_prefix}/pupil/boundary",
        rr.LineStrips3D(
            strips=[np.vstack([pp, pp[0:1]])],
            colors=[COLOR_PUPIL_BOUNDARY],
            radii=[0.06],
        ),
    )
    rr.log(
        f"{eye_prefix}/pupil/points",
        rr.Points3D(positions=pp, colors=[COLOR_PUPIL_POINTS] * len(pp), radii=[0.08]),
    )

    n_points = len(pp)
    vertices = np.vstack([pc, pp])
    triangles = []
    for i in range(n_points):
        next_i = (i + 1) % n_points
        triangles.append([0, i + 1, next_i + 1])
    triangles = np.array(triangles, dtype=np.uint32)

    rr.log(
        f"{eye_prefix}/pupil/face",
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=triangles,
            vertex_colors=[COLOR_PUPIL_FACE_RIGHT if eye_prefix == "right_eye" else COLOR_PUPIL_FACE_LEFT
                           ] * len(vertices),
        ),
    )


def log_socket_landmarks(
    eye_prefix: str, tear_duct: NDArray[np.float64], outer_eye: NDArray[np.float64]
) -> None:
    """Log socket landmarks connected to eye center with labels."""
    td = tear_duct
    oe = outer_eye
    eye_center = np.array([0.0, 0.0, 0.0])

    rr.log(
        f"{eye_prefix}/socket/tear_duct",
        rr.Points3D(
            positions=[td],
            colors=[COLOR_TEAR_DUCT],
            radii=[0.15],
            labels=["tear_duct"],
        ),
    )
    rr.log(
        f"{eye_prefix}/socket/outer_eye",
        rr.Points3D(
            positions=[oe],
            colors=[COLOR_OUTER_EYE],
            radii=[0.15],
            labels=["outer_eye"],
        ),
    )
    rr.log(
        f"{eye_prefix}/socket/tear_duct_line",
        rr.LineStrips3D(
            strips=[np.array([eye_center, td])],
            colors=[COLOR_TEAR_DUCT],
            radii=[0.04],
        ),
    )
    rr.log(
        f"{eye_prefix}/socket/outer_eye_line",
        rr.LineStrips3D(
            strips=[np.array([eye_center, oe])],
            colors=[COLOR_OUTER_EYE],
            radii=[0.04],
        ),
    )


def log_timeseries_angles(
    eye_name: str, adduction_deg: float, elevation_deg: float, entity_path: str = "/"
) -> None:
    """Log gaze angles for an eye."""
    rr.log(f"{entity_path}timeseries/angles/{eye_name}/adduction", rr.Scalars(adduction_deg))
    rr.log(f"{entity_path}timeseries/angles/{eye_name}/elevation", rr.Scalars(elevation_deg))


def log_timeseries_velocities(
    eye_name: str, adduction_vel: float, elevation_vel: float, entity_path: str = "/"
) -> None:
    """Log angular velocities for an eye."""
    rr.log(f"{entity_path}timeseries/velocity/{eye_name}/adduction", rr.Scalars(adduction_vel))
    rr.log(f"{entity_path}timeseries/velocity/{eye_name}/elevation", rr.Scalars(elevation_vel))


def log_timeseries_accelerations(
    eye_name: str, adduction_acc: float, elevation_acc: float, entity_path: str = "/"
) -> None:
    """Log angular accelerations for an eye."""
    rr.log(f"{entity_path}timeseries/acceleration/{eye_name}/adduction", rr.Scalars(adduction_acc))
    rr.log(f"{entity_path}timeseries/acceleration/{eye_name}/elevation", rr.Scalars(elevation_acc))


def log_pixel_data(
    eye_name: str, pixel_data: dict[str, NDArray[np.float64]], frame_idx: int
) -> None:
    """Log pupil position for an eye."""
    if "pupil_center_x" in pixel_data:
        rr.log(
            f"timeseries/pupil_position/{eye_name}/horizontal",
            rr.Scalars(-pixel_data["pupil_center_x"][frame_idx]),
        )
    if "pupil_center_y" in pixel_data:
        rr.log(
            f"timeseries/pupil_position/{eye_name}/vertical",
            rr.Scalars(-pixel_data["pupil_center_y"][frame_idx]),
        )


def setup_binocular_timeseries_styling() -> None:
    """Setup styling for timeseries plots with both eyes."""
    # Define eye colors
    eye_colors = {
        "left_eye": COLOR_LEFT_EYE_PRIMARY,
        "right_eye": COLOR_RIGHT_EYE_PRIMARY,
    }
    eye_secondary_colors = {
        "left_eye": COLOR_LEFT_EYE_SECONDARY,
        "right_eye": COLOR_RIGHT_EYE_SECONDARY,
    }

    for eye_name, primary_color in eye_colors.items():
        secondary_color = eye_secondary_colors[eye_name]

        # Pupil position
        rr.log(
            f"timeseries/pupil_position/{eye_name}/horizontal",
            rr.SeriesLines(widths=1.5, colors=[primary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/pupil_position/{eye_name}/horizontal",
            rr.SeriesPoints(keypoint_sizes=2.0, colors=[primary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/pupil_position/{eye_name}/vertical",
            rr.SeriesLines(widths=1.5, colors=[secondary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/pupil_position/{eye_name}/vertical",
            rr.SeriesPoints(keypoint_sizes=2.0, colors=[secondary_color]),
            static=True,
        )

        # Angles
        rr.log(
            f"timeseries/angles/{eye_name}/adduction",
            rr.SeriesLines(widths=1.5, colors=[primary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/angles/{eye_name}/adduction",
            rr.SeriesPoints(keypoint_sizes=2.0, colors=[primary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/angles/{eye_name}/elevation",
            rr.SeriesLines(widths=1.5, colors=[secondary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/angles/{eye_name}/elevation",
            rr.SeriesPoints(keypoint_sizes=2.0, colors=[secondary_color]),
            static=True,
        )

        # Velocities
        rr.log(
            f"timeseries/velocity/{eye_name}/adduction",
            rr.SeriesLines(widths=1.5, colors=[primary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/velocity/{eye_name}/adduction",
            rr.SeriesPoints(keypoint_sizes=2.0, colors=[primary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/velocity/{eye_name}/elevation",
            rr.SeriesLines(widths=1.5, colors=[secondary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/velocity/{eye_name}/elevation",
            rr.SeriesPoints(keypoint_sizes=2.0, colors=[secondary_color]),
            static=True,
        )

        # Accelerations
        rr.log(
            f"timeseries/acceleration/{eye_name}/adduction",
            rr.SeriesLines(widths=1.5, colors=[primary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/acceleration/{eye_name}/adduction",
            rr.SeriesPoints(keypoint_sizes=2.0, colors=[primary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/acceleration/{eye_name}/elevation",
            rr.SeriesLines(widths=1.5, colors=[secondary_color]),
            static=True,
        )
        rr.log(
            f"timeseries/acceleration/{eye_name}/elevation",
            rr.SeriesPoints(keypoint_sizes=2.0, colors=[secondary_color]),
            static=True,
        )


class VideoFrameReader:
    """Stream video frames from file."""

    def __init__(self, video_path: Path, expected_n_frames: int):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frames = min(self.total_frames, expected_n_frames)
        self.current_frame = 0

    def read_frame(self) -> NDArray[np.uint8] | None:
        if self.current_frame >= self.n_frames:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.current_frame += 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def get_eye_radius_from_kinematics(kinematics: FerretEyeKinematics) -> float:
    """Extract eye radius from reference geometry."""
    pc = kinematics.eyeball.reference_geometry.keypoints["pupil_center"]
    r = np.sqrt(pc.x**2 + pc.y**2 + pc.z**2)
    if r < 0.1:
        raise ValueError(f"Eye radius too small: {r}")
    return r


def run_binocular_eye_kinematics_viewer(
    left_eye_data: EyeViewerData | None = None,
    right_eye_data: EyeViewerData | None = None,
    spawn: bool = True,
    recording_id: str = "binocular_eye_kinematics",
    time_window_seconds: float = 2.0,
    playback_speed: float = 0.25,
) -> None:
    """Launch Rerun viewer for binocular FerretEyeKinematics data."""
    if left_eye_data is None and right_eye_data is None:
        raise ValueError("At least one eye's data must be provided")

    rr.init(recording_id, spawn=spawn)

    # Determine frame count and timestamps from available data
    if left_eye_data is not None and right_eye_data is not None:
        # Use minimum frame count to handle off-by-one differences
        n_frames = min(
            left_eye_data.kinematics.n_frames,
            right_eye_data.kinematics.n_frames,
        )
        timestamps = left_eye_data.kinematics.timestamps[:n_frames]
        if left_eye_data.kinematics.n_frames != right_eye_data.kinematics.n_frames:
            print(
                f"Note: Frame count mismatch (left={left_eye_data.kinematics.n_frames}, "
                f"right={right_eye_data.kinematics.n_frames}). Using {n_frames} frames."
            )
    elif left_eye_data is not None:
        n_frames = left_eye_data.kinematics.n_frames
        timestamps = left_eye_data.kinematics.timestamps
    else:
        assert right_eye_data is not None
        n_frames = right_eye_data.kinematics.n_frames
        timestamps = right_eye_data.kinematics.timestamps

    # Prepare data for each eye
    eye_data_dict: dict[str, EyeViewerData] = {}
    if left_eye_data is not None:
        eye_data_dict["left_eye"] = left_eye_data
    if right_eye_data is not None:
        eye_data_dict["right_eye"] = right_eye_data

    # Open video readers
    video_readers: dict[str, VideoFrameReader] = {}
    for eye_name, data in eye_data_dict.items():
        if data.video_path is not None:
            try:
                video_readers[eye_name] = VideoFrameReader(data.video_path, n_frames)
            except Exception as e:
                print(f"Warning: Could not open video for {eye_name}: {e}")

    # Send blueprint
    has_left_video = "left_eye" in video_readers
    has_right_video = "right_eye" in video_readers
    rr.send_blueprint(
        create_binocular_viewer_blueprint(
            time_window_seconds=time_window_seconds,
            has_left_video=has_left_video,
            has_right_video=has_right_video,
        )
    )

    # Setup styling
    setup_binocular_timeseries_styling()

    # Log static elements for each eye
    for eye_name, data in eye_data_dict.items():
        eye_radius = get_eye_radius_from_kinematics(data.kinematics)
        log_static_world_frame(eye_name, eye_radius * 1.5, eye_radius)

    print(f"Logging {n_frames} frames for binocular eye kinematics...")

    try:
        for i in range(n_frames):
            set_time_seconds("time", timestamps[i])

            # Log video frames
            for eye_name, reader in video_readers.items():
                frame = reader.read_frame()
                if frame is not None:
                    rr.log(f"video/{eye_name}/frame", rr.Image(frame))

            # Log data for each eye
            for eye_name, data in eye_data_dict.items():
                kinematics = data.kinematics
                eye_radius = get_eye_radius_from_kinematics(kinematics)

                # 3D visualization
                log_rotating_sphere_and_gaze(
                    eye_name,
                    kinematics.quaternions_wxyz[i],
                    eye_radius,
                    eye_radius * 2.0,
                )
                log_eye_basis_vectors(
                    eye_name, kinematics.quaternions_wxyz[i], eye_radius * 1.2
                )
                log_pupil_geometry(
                    eye_name,
                    kinematics.tracked_pupil_center[i],
                    kinematics.tracked_pupil_points[i],
                )
                log_socket_landmarks(
                    eye_name, kinematics.tear_duct_mm[i], kinematics.outer_eye_mm[i]
                )

                # Timeseries
                adduction_deg = np.degrees(kinematics.adduction_angle.values[i])
                elevation_deg = np.degrees(kinematics.elevation_angle.values[i])
                adduction_vel = np.degrees(kinematics.adduction_velocity.values[i])
                elevation_vel = np.degrees(kinematics.elevation_velocity.values[i])
                adduction_acc = np.degrees(kinematics.adduction_acceleration.values[i])
                elevation_acc = np.degrees(kinematics.elevation_acceleration.values[i])

                log_timeseries_angles(eye_name, adduction_deg, elevation_deg)
                log_timeseries_velocities(eye_name, adduction_vel, elevation_vel)
                log_timeseries_accelerations(eye_name, adduction_acc, elevation_acc)

                if data.pixel_data is not None:
                    log_pixel_data(eye_name, data.pixel_data, i)

    finally:
        for reader in video_readers.values():
            reader.close()

    print("Done!")


def run_binocular_eye_rerun_viewer(
    recording_folder: RecordingFolder,
    spawn: bool = True,
    time_window_seconds: float = 5.0,
    playback_speed: float = 0.25,
) -> None:
    """Load both eyes' data from directory and launch binocular viewer."""
    eye_kinematics_directory_path = recording_folder.eye_output_data
    print(f"Loading eye kinematics from {eye_kinematics_directory_path}...")


    left_eye_trajectories_csv_path = recording_folder.left_eye_data_csv
    right_eye_trajectories_csv_path = recording_folder.right_eye_data_csv
    left_eye_video_path = recording_folder.left_eye_annotated_video
    right_eye_video_path = recording_folder.right_eye_annotated_video

    left_eye_data: EyeViewerData | None = None
    right_eye_data: EyeViewerData | None = None

    if "757" in str(left_eye_video_path):
        left_eye_video_name = "eye0"
        right_eye_video_name = "eye1"
    else:
        left_eye_video_name = "eye1"
        right_eye_video_name = "eye0"

    # Try to load left eye
    try:
        left_kinematics = FerretEyeKinematics.load_from_directory(
            eye_name="left_eye",
            input_directory=eye_kinematics_directory_path,
        )
        left_pixel_data = None
        if left_eye_trajectories_csv_path is not None:
            try:
                df = load_eye_trajectories_csv(
                    csv_path=left_eye_trajectories_csv_path, eye_side="left", video_name=left_eye_video_name
                )
                timestamps, pupil_centers_px, *_ = extract_frame_data(df)
                left_pixel_data = {
                    "pupil_center_x": pupil_centers_px[:, 0],
                    "pupil_center_y": pupil_centers_px[:, 1],
                }
                print(f"Loaded left eye pixel data: {len(timestamps)} frames")
            except Exception as e:
                print(f"Warning: Could not load left eye pixel data: {e}")

        left_eye_data = EyeViewerData(
            kinematics=left_kinematics,
            pixel_data=left_pixel_data,
            video_path=left_eye_video_path,
        )
        print(f"Loaded left eye kinematics: {left_kinematics.n_frames} frames")
    except FileNotFoundError:
        print("Left eye kinematics not found, skipping...")

    # Try to load right eye
    try:
        right_kinematics = FerretEyeKinematics.load_from_directory(
            eye_name="right_eye",
            input_directory=eye_kinematics_directory_path,
        )
        right_pixel_data = None
        if right_eye_trajectories_csv_path is not None:
            try:
                df = load_eye_trajectories_csv(
                    csv_path=right_eye_trajectories_csv_path, eye_side="right", video_name=right_eye_video_name
                )
                timestamps, pupil_centers_px, *_ = extract_frame_data(df)
                right_pixel_data = {
                    "pupil_center_x": pupil_centers_px[:, 0],
                    "pupil_center_y": pupil_centers_px[:, 1],
                }
                print(f"Loaded right eye pixel data: {len(timestamps)} frames")
            except Exception as e:
                print(f"Warning: Could not load right eye pixel data: {e}")

        right_eye_data = EyeViewerData(
            kinematics=right_kinematics,
            pixel_data=right_pixel_data,
            video_path=right_eye_video_path,
        )
        print(f"Loaded right eye kinematics: {right_kinematics.n_frames} frames")
    except FileNotFoundError:
        print("Right eye kinematics not found, skipping...")

    if left_eye_data is None and right_eye_data is None:
        raise FileNotFoundError(
            f"No eye kinematics found in {eye_kinematics_directory_path}"
        )

    run_binocular_eye_kinematics_viewer(
        left_eye_data=left_eye_data,
        right_eye_data=right_eye_data,
        spawn=spawn,
        time_window_seconds=time_window_seconds,
        playback_speed=playback_speed,
    )


# =============================================================================
# Legacy single-eye functions for backwards compatibility
# =============================================================================


def run_eye_kinematics_viewer(
    kinematics: FerretEyeKinematics,
    pixel_data: dict[str, NDArray[np.float64]] | None = None,
    video_path: Path | str | None = None,
    spawn: bool = True,
    recording_id: str | None = None,
    time_window_seconds: float = 2.0,
) -> None:
    """Launch Rerun viewer for single eye (legacy interface)."""
    eye_data = EyeViewerData(
        kinematics=kinematics,
        pixel_data=pixel_data,
        video_path=Path(video_path) if video_path else None,
    )

    if kinematics.eye_side == "left":
        run_binocular_eye_kinematics_viewer(
            left_eye_data=eye_data,
            right_eye_data=None,
            spawn=spawn,
            recording_id=recording_id or f"eye_kinematics_{kinematics.name}",
            time_window_seconds=time_window_seconds,
        )
    else:
        run_binocular_eye_kinematics_viewer(
            left_eye_data=None,
            right_eye_data=eye_data,
            spawn=spawn,
            recording_id=recording_id or f"eye_kinematics_{kinematics.name}",
            time_window_seconds=time_window_seconds,
        )

if __name__ == "__main__":
    # Example usage with both eyes
    recording_folder = RecordingFolder.from_folder_path(
        ""
    )

    run_binocular_eye_rerun_viewer(
        recording_folder=recording_folder,
        time_window_seconds=3.0,
    )