"""
Enhanced Eye Kinematics Rerun Viewer
====================================

A comprehensive Rerun-based visualization for ferret eye kinematics data.

Features:
- Large 3D viewer showing:
  - Wireframe eyeball sphere (rotates with gaze)
  - Pupil center and boundary (p1-p8)
  - Socket landmarks (tear_duct, outer_eye)
  - Gaze direction arrow pointing out the "north pole"
  - Eye-frame RGB basis vectors (rotate with eye)
  - World-frame RGB basis vectors (fixed reference)

- Synchronized video playback from MP4 file

- Stacked timeseries with linked x-axes that scroll with playback (±2s window):
  - Gaze angles (adduction, elevation, torsion) in degrees
  - Angular velocities in deg/s
  - Original X/Y pixel traces for validation

- Animation playback slider synced across all views

Usage:
    # With FerretEyeKinematics object:
    run_eye_kinematics_viewer(kinematics, video_path="eye_video.mp4")

    # From CSV file:
    run_eye_viewer_from_csv(csv_path, eye_side="right", camera_distance_mm=21, video_path="eye.mp4")

    # Standalone demo with synthetic data:
    run_demo_eye_viewer()
"""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import rerun as rr
import rerun.blueprint as rrb

# Optional video support
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# RERUN API COMPATIBILITY
# =============================================================================

def set_time_seconds(timeline: str, seconds: float) -> None:
    """
    Set time on a timeline with compatibility across Rerun versions.

    Different Rerun versions use different APIs:
    - 0.15+: rr.set_time(timeline, seconds=seconds)
    - 0.14-: rr.set_time_seconds(timeline, seconds)
    - Some: rr.set_time_sequence or other variants
    """
    # Try the newer API first (0.15+)
    if hasattr(rr, 'set_time'):
        try:
            rr.set_time(timeline, timestamp=seconds)
            return
        except TypeError:
            pass

    # Try older API (0.14 and earlier)
    if hasattr(rr, 'set_time_seconds'):
        try:
            rr.set_time_seconds(timeline, seconds)
            return
        except TypeError:
            pass

    # Try with timeline as TimeSequenceLike
    if hasattr(rr, 'set_time_sequence'):
        # Convert seconds to nanoseconds for sequence
        rr.set_time_sequence(timeline, int(seconds * 1e9))
        return

    raise RuntimeError(
        f"Could not find compatible Rerun time API. "
        f"Rerun version: {getattr(rr, '__version__', 'unknown')}. "
        f"Available: {[a for a in dir(rr) if 'time' in a.lower()]}"
    )


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_PUPIL_POINTS: int = 8

# Colors (RGB)
COLOR_GAZE_ARROW = [255, 50, 50]  # Bright red
COLOR_EYE_X_AXIS = [255, 0, 0]  # Red
COLOR_EYE_Y_AXIS = [0, 255, 0]  # Green
COLOR_EYE_Z_AXIS = [0, 100, 255]  # Blue
COLOR_WORLD_X = [200, 50, 50]  # Dark red
COLOR_WORLD_Y = [50, 200, 50]  # Dark green
COLOR_WORLD_Z = [50, 50, 200]  # Dark blue
COLOR_PUPIL_CENTER = [0, 0, 0]  # Black
COLOR_PUPIL_BOUNDARY = [0, 0, 139]  # Dark blue
COLOR_PUPIL_POINTS = [70, 130, 180]  # Steel blue
COLOR_TEAR_DUCT = [255, 140, 0]  # Orange
COLOR_OUTER_EYE = [255, 100, 0]  # Dark orange
COLOR_SOCKET_LINE = [255, 165, 0]  # Orange
COLOR_SPHERE_WIRE = [20, 20, 20]  # Light gray
COLOR_EYE_CENTER = [128, 0, 128]  # Purple

COLOR_ADDUCTION = [30, 144, 255]  # Dodger blue
COLOR_ELEVATION = [50, 205, 50]  # Lime green
COLOR_TORSION = [220, 20, 60]  # Crimson


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

# Visualization transform: model has gaze along +X, we want to display with gaze along +Z
# This is a -90 degree rotation around Y: (x, y, z) -> (-z, y, x)
# So model +X becomes viz +Z (gaze out the north pole)

def model_to_viz(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform from model coordinates (gaze = +X) to visualization coordinates (gaze = +Z).

    Model frame: +X = gaze, +Y = left, +Z = up
    Viz frame: +Z = gaze (north pole), +X = forward in horizontal plane, +Y = left
    """
    p = np.asarray(p)
    if p.ndim == 1:
        return np.array([-p[2], p[1], p[0]], dtype=np.float64)
    else:
        # Handle (N, 3) arrays
        result = np.empty_like(p)
        result[..., 0] = -p[..., 2]
        result[..., 1] = p[..., 1]
        result[..., 2] = p[..., 0]
        return result


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

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def generate_pupil_ellipse_points(
    eye_radius: float,
    pupil_radius: float,
    pupil_eccentricity: float,
    n_points: int = NUM_PUPIL_POINTS,
) -> NDArray[np.float64]:
    """Generate pupil boundary points on the eye surface at rest position."""
    a = pupil_radius  # Semi-major (along Y)
    b = pupil_radius * pupil_eccentricity  # Semi-minor (along Z)
    R = eye_radius

    points = []
    for i in range(n_points):
        phi = 2 * np.pi * i / n_points
        y_tangent = a * np.cos(phi)
        z_tangent = b * np.sin(phi)
        tangent_point = np.array([R, y_tangent, z_tangent])
        direction = tangent_point / np.linalg.norm(tangent_point)
        sphere_point = R * direction
        points.append(sphere_point)

    return np.array(points, dtype=np.float64)


# =============================================================================
# BLUEPRINT CREATION
# =============================================================================


def create_eye_viewer_blueprint(
    has_pixel_data: bool,
    has_video: bool = False,
    time_window_seconds: float = 2.0,
) -> rrb.Blueprint:
    """
    Create Rerun blueprint for eye kinematics viewer layout.

    Layout (with video):
    - Left column (40%): 3D viewer on top, video below
    - Right column (60%): Stacked timeseries panels

    Layout (without video):
    - Left (55%): 3D viewer
    - Right (45%): Stacked timeseries panels

    All timeseries scroll with playback, showing ±time_window_seconds around
    the current cursor. Panels are linked (zoom one = zoom all).

    Args:
        has_pixel_data: Whether to include pixel coordinate panels
        has_video: Whether to include video panel
        time_window_seconds: Seconds before/after cursor to show (default ±2s)
    """
    # Scrolling time range centered on cursor (same pattern as skull viewer)
    scrolling_time_range = rrb.VisibleTimeRange(
        timeline="time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    # All views share the same scrolling_time_range = linked zoom/pan + scrolling
    # Fixed Y-axis ranges prevent squiggling during scroll
    angle_view = rrb.TimeSeriesView(
        name="Gaze Angles (°)",
        origin="/timeseries/angles",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-25.0, 25.0)),
    )

    velocity_view = rrb.TimeSeriesView(
        name="Angular Velocity (°/s)",
        origin="/timeseries/velocity",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-350.0, 350.0)),
    )

    timeseries_views = [angle_view, velocity_view]

    # Add pixel data views if available
    if has_pixel_data:
        pixel_x_view = rrb.TimeSeriesView(
            name="Pupil Center X (px)",
            origin="/timeseries/pixels/x",
            plot_legend=rrb.PlotLegend(visible=True),
            time_ranges=scrolling_time_range,
            axis_y=rrb.ScalarAxis(range=(-40.0, 40.0)),
        )
        pixel_y_view = rrb.TimeSeriesView(
            name="Pupil Center Y (px)",
            origin="/timeseries/pixels/y",
            plot_legend=rrb.PlotLegend(visible=True),
            time_ranges=scrolling_time_range,
            axis_y=rrb.ScalarAxis(range=(-30.0, 30.0)),
        )
        timeseries_views.extend([pixel_x_view, pixel_y_view])

    # 3D viewer (no ground plane grid)
    viewer_3d = rrb.Spatial3DView(
        name="Eye 3D View",
        origin="/",
        contents=[
            "+ /eye/**",
            "+ /world_frame/**",
        ],
        line_grid=rrb.LineGrid3D(visible=False),
    )

    if has_video:
        # Video view
        video_view = rrb.Spatial2DView(
            name="Eye Video",
            origin="/video",
        )

        # Layout with video: 3D and video stacked on left, timeseries on right
        left_column = rrb.Vertical(
            viewer_3d,
            video_view,
            row_shares=[0.5, 0.5],
        )

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                left_column,
                rrb.Vertical(*timeseries_views),
                column_shares=[0.40, 0.60],
            ),
            collapse_panels=True,
        )
    else:
        # Layout without video
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                viewer_3d,
                rrb.Vertical(*timeseries_views),
                column_shares=[0.55, 0.45],
            ),
            collapse_panels=True,
        )

    return blueprint


# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================


def log_static_world_frame(axis_length: float) -> None:
    """Log world reference frame axes (static, solid lines with RGB colors)."""
    # World X axis (red) - solid line
    rr.log(
        "world_frame/x_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[axis_length, 0, 0]],
            colors=[COLOR_WORLD_X],
            radii=[0.04],
        ),
        static=True,
    )

    # World Y axis (green) - solid line
    rr.log(
        "world_frame/y_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[0, axis_length, 0]],
            colors=[COLOR_WORLD_Y],
            radii=[0.04],
        ),
        static=True,
    )

    # World Z axis (blue) - solid line (gaze direction at rest)
    rr.log(
        "world_frame/z_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[0, 0, axis_length]],
            colors=[COLOR_WORLD_Z],
            radii=[0.04],
        ),
        static=True,
    )

    # Add axis labels using text at the end of each axis
    label_offset = axis_length * 1.1
    rr.log(
        "world_frame/labels/x",
        rr.Points3D(
            positions=[[label_offset, 0, 0]],
            colors=[COLOR_WORLD_X],
            radii=[0.01],
        ),
        static=True,
    )
    rr.log(
        "world_frame/labels/y",
        rr.Points3D(
            positions=[[0, label_offset, 0]],
            colors=[COLOR_WORLD_Y],
            radii=[0.01],
        ),
        static=True,
    )
    rr.log(
        "world_frame/labels/z",
        rr.Points3D(
            positions=[[0, 0, label_offset]],
            colors=[COLOR_WORLD_Z],
            radii=[0.01],
        ),
        static=True,
    )

    # Eye center marker (static at origin)
    rr.log(
        "eye/center",
        rr.Points3D(
            positions=[[0, 0, 0]],
            colors=[COLOR_EYE_CENTER],
            radii=[0.12],
        ),
        static=True,
    )


def log_eye_basis_vectors(
    quaternion: NDArray[np.float64],
    axis_length: float,
) -> None:
    """
    Log RGB basis vectors that rotate with the eye.

    These show the eye's local coordinate frame:
    - Red (X): points along the eye's primary gaze direction
    - Green (Y): points to the eye's "left"
    - Blue (Z): points "up" relative to the eye

    All vectors are transformed from model coords to viz coords.
    """
    R_eye = quaternion_to_rotation_matrix(quaternion)

    # Eye basis vectors in model coords (at rest: X=gaze, Y=left, Z=up)
    eye_x_model = R_eye @ np.array([axis_length, 0, 0])
    eye_y_model = R_eye @ np.array([0, axis_length, 0])
    eye_z_model = R_eye @ np.array([0, 0, axis_length])

    # Transform to viz coords
    eye_x_viz = model_to_viz(eye_x_model)
    eye_y_viz = model_to_viz(eye_y_model)
    eye_z_viz = model_to_viz(eye_z_model)

    # Log as arrows from origin
    rr.log(
        "eye/basis/x_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[eye_x_viz],
            colors=[COLOR_EYE_X_AXIS],
            radii=[0.06],
        ),
    )
    rr.log(
        "eye/basis/y_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[eye_y_viz],
            colors=[COLOR_EYE_Y_AXIS],
            radii=[0.06],
        ),
    )
    rr.log(
        "eye/basis/z_axis",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[eye_z_viz],
            colors=[COLOR_EYE_Z_AXIS],
            radii=[0.06],
        ),
    )

    # Add labels at tips of basis vectors
    label_scale = 1.15
    rr.log(
        "eye/basis/labels/x",
        rr.Points3D(
            positions=[eye_x_viz * label_scale],
            colors=[COLOR_EYE_X_AXIS],
            radii=[0.01],
        ),
    )
    rr.log(
        "eye/basis/labels/y",
        rr.Points3D(
            positions=[eye_y_viz * label_scale],
            colors=[COLOR_EYE_Y_AXIS],
            radii=[0.01],
        ),
    )
    rr.log(
        "eye/basis/labels/z",
        rr.Points3D(
            positions=[eye_z_viz * label_scale],
            colors=[COLOR_EYE_Z_AXIS],
            radii=[0.01],
        ),
    )


def log_rotating_sphere_and_gaze(
    quaternion: NDArray[np.float64],
    eye_radius: float,
    gaze_length: float,
) -> None:
    """
    Log the wireframe sphere and gaze arrow rotated by the eye orientation.

    Everything is computed in model coords first, then transformed to viz coords.
    """
    # Get eye rotation matrix
    R_eye = quaternion_to_rotation_matrix(quaternion)

    # Generate sphere at rest with "north pole" at +X (model gaze direction)
    local_strips = generate_sphere_line_strips_with_pole_at_x(
        radius=eye_radius,
        n_lat=8,
        n_lon=16,
    )

    # Rotate sphere by eye quaternion (in model coords)
    rotated_strips_model = rotate_strips(local_strips, R_eye)

    # Transform to viz coords
    rotated_strips_viz = [model_to_viz(strip) for strip in rotated_strips_model]

    rr.log(
        "eye/sphere/wireframe",
        rr.LineStrips3D(
            strips=rotated_strips_viz,
            colors=[COLOR_SPHERE_WIRE] * len(rotated_strips_viz),
            radii=[0.015],
        ),
    )

    # Transparent sphere mesh
    vertices_model, triangles = generate_sphere_mesh(
        radius=eye_radius*.99,
        n_lat=12,
        n_lon=24,
    )
    # Rotate vertices
    vertices_rotated = (R_eye @ vertices_model.T).T
    # Transform to viz coords
    vertices_viz = model_to_viz(vertices_rotated)

    # RGBA with alpha for transparency (200, 200, 220, 40)
    rr.log(
        "eye/sphere/mesh",
        rr.Mesh3D(
            vertex_positions=vertices_viz,
            triangle_indices=triangles,
            vertex_colors=[[200, 200, 220, 40]] * len(vertices_viz),
        ),
    )

    # Gaze direction - rotate from rest [gaze_length, 0, 0] by R_eye, then to viz
    gaze_model = R_eye @ np.array([gaze_length, 0, 0])
    gaze_viz = model_to_viz(gaze_model)

    rr.log(
        "eye/gaze_arrow",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[gaze_viz],
            colors=[COLOR_GAZE_ARROW],
            radii=[0.1],
        ),
    )


def generate_sphere_mesh(
    radius: float,
    n_lat: int = 12,
    n_lon: int = 24,
) -> tuple[NDArray[np.float64], NDArray[np.uint32]]:
    """
    Generate sphere mesh vertices and triangle indices with pole at +X.

    Returns:
        vertices: (N, 3) array of vertex positions
        triangles: (M, 3) array of triangle vertex indices
    """
    vertices = []
    triangles = []

    # Generate vertices
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat  # 0 to pi (pole to pole)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon  # 0 to 2pi

            # Sphere with pole at +X
            x = radius * cos_theta
            y = radius * sin_theta * np.cos(phi)
            z = radius * sin_theta * np.sin(phi)

            vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float64)

    # Generate triangles
    for i in range(n_lat):
        for j in range(n_lon):
            # Current vertex index
            curr = i * n_lon + j
            next_j = i * n_lon + (j + 1) % n_lon
            below = (i + 1) * n_lon + j
            below_next = (i + 1) * n_lon + (j + 1) % n_lon

            # Two triangles per quad (skip degenerate triangles at poles)
            if i > 0:
                triangles.append([curr, below, next_j])
            if i < n_lat - 1:
                triangles.append([next_j, below, below_next])

    triangles = np.array(triangles, dtype=np.uint32)

    return vertices, triangles


def generate_sphere_line_strips_with_pole_at_x(
    radius: float,
    n_lat: int = 10,
    n_lon: int = 20,
) -> list[NDArray[np.float64]]:
    """
    Generate wireframe sphere with "north pole" at +X (model gaze direction).

    This way the sphere lines up with the pupil which is at +X at rest.
    """
    strips = []

    # Latitude circles (around X axis)
    for i in range(1, n_lat):
        # Angle from -X to +X
        angle_from_pole = np.pi * i / n_lat  # 0 to pi
        r_circle = radius * np.sin(angle_from_pole)
        x_pos = radius * np.cos(angle_from_pole)  # From +radius to -radius

        circle = []
        for j in range(n_lon + 1):
            theta = 2 * np.pi * j / n_lon
            y = r_circle * np.cos(theta)
            z = r_circle * np.sin(theta)
            circle.append([x_pos, y, z])
        strips.append(np.array(circle, dtype=np.float64))

    # Longitude circles (through X axis) - fewer for cleaner look
    for j in range(0, n_lon, 2):
        theta = 2 * np.pi * j / n_lon
        circle = []
        for i in range(n_lat + 1):
            phi = np.pi * i / n_lat  # 0 to pi
            x = radius * np.cos(phi)
            r = radius * np.sin(phi)
            y = r * np.cos(theta)
            z = r * np.sin(theta)
            circle.append([x, y, z])
        strips.append(np.array(circle, dtype=np.float64))

    return strips


def log_pupil_geometry(
    pupil_center: NDArray[np.float64],
    pupil_points: NDArray[np.float64],
    quaternion: NDArray[np.float64],
) -> None:
    """Log pupil center and boundary points - already rotated, just apply viz transform."""
    # Pupil data from kinematics is already rotated by R_eye in model coords
    # Just apply model_to_viz transform (same as sphere and gaze)
    pupil_center_viz = model_to_viz(pupil_center)
    pupil_points_viz = model_to_viz(pupil_points)

    rr.log(
        "eye/pupil/center",
        rr.Points3D(
            positions=[pupil_center_viz],
            colors=[COLOR_PUPIL_CENTER],
            radii=[0.12],
        ),
    )

    # Pupil boundary as closed loop
    boundary_closed = np.vstack([pupil_points_viz, pupil_points_viz[0:1]])

    rr.log(
        "eye/pupil/boundary",
        rr.LineStrips3D(
            strips=[boundary_closed],
            colors=[COLOR_PUPIL_BOUNDARY],
            radii=[0.06],
        ),
    )

    # Pupil points as small markers
    rr.log(
        "eye/pupil/points",
        rr.Points3D(
            positions=pupil_points_viz,
            colors=[COLOR_PUPIL_POINTS] * len(pupil_points_viz),
            radii=[0.08],
        ),
    )


def log_socket_landmarks(
    tear_duct: NDArray[np.float64],
    outer_eye: NDArray[np.float64],
) -> None:
    """Log socket landmarks with viz transform applied."""
    tear_duct_viz = model_to_viz(tear_duct)
    outer_eye_viz = model_to_viz(outer_eye)

    rr.log(
        "eye/socket/tear_duct",
        rr.Points3D(
            positions=[tear_duct_viz],
            colors=[COLOR_TEAR_DUCT],
            radii=[0.15],
        ),
    )
    rr.log(
        "eye/socket/outer_eye",
        rr.Points3D(
            positions=[outer_eye_viz],
            colors=[COLOR_OUTER_EYE],
            radii=[0.15],
        ),
    )
    rr.log(
        "eye/socket/opening_line",
        rr.LineStrips3D(
            strips=[np.array([tear_duct_viz, outer_eye_viz])],
            colors=[COLOR_SOCKET_LINE],
            radii=[0.04],
        ),
    )


def log_timeseries_angles(
    adduction_deg: float,
    elevation_deg: float,
    torsion_deg: float,
) -> None:
    """Log gaze angle timeseries values with line-and-markers styling."""
    rr.log(
        "timeseries/angles/adduction",
        rr.Scalars(adduction_deg),
    )
    rr.log(
        "timeseries/angles/elevation",
        rr.Scalars(elevation_deg),
    )
    rr.log(
        "timeseries/angles/torsion",
        rr.Scalars(torsion_deg),
    )


def log_timeseries_velocities(
    adduction_vel: float,
    elevation_vel: float,
    torsion_vel: float,
) -> None:
    """Log angular velocity timeseries values with line-and-markers styling."""
    rr.log(
        "timeseries/velocity/adduction",
        rr.Scalars(adduction_vel),
    )
    rr.log(
        "timeseries/velocity/elevation",
        rr.Scalars(elevation_vel),
    )
    rr.log(
        "timeseries/velocity/torsion",
        rr.Scalars(torsion_vel),
    )


def log_pixel_data(
    pixel_data: dict[str, NDArray[np.float64]],
    frame_idx: int,
) -> None:
    """Log pixel coordinate data for validation - only pupil center X and Y."""
    # Only log pupil center - no p1-p8 or landmarks
    if "pupil_center_x" in pixel_data:
        rr.log("timeseries/pixels/x/pupil_center", rr.Scalars(pixel_data["pupil_center_x"][frame_idx]))

    if "pupil_center_y" in pixel_data:
        rr.log("timeseries/pixels/y/pupil_center", rr.Scalars(pixel_data["pupil_center_y"][frame_idx]))


def setup_timeseries_styling() -> None:
    """Set up line-and-markers styling for all timeseries with small markers (size 2)."""
    # Style for angles
    for name in ["adduction", "elevation", "torsion"]:
        rr.log(f"timeseries/angles/{name}", rr.SeriesLines(widths=1.5), static=True)
        rr.log(f"timeseries/angles/{name}", rr.SeriesPoints(marker_sizes=2.0), static=True)

    # Style for velocities
    for name in ["adduction", "elevation", "torsion"]:
        rr.log(f"timeseries/velocity/{name}", rr.SeriesLines(widths=1.5), static=True)
        rr.log(f"timeseries/velocity/{name}", rr.SeriesPoints(marker_sizes=2.0), static=True)

    # Style for pixel data
    rr.log("timeseries/pixels/x/pupil_center", rr.SeriesLines(widths=1.5), static=True)
    rr.log("timeseries/pixels/x/pupil_center", rr.SeriesPoints(marker_sizes=2.0), static=True)
    rr.log("timeseries/pixels/y/pupil_center", rr.SeriesLines(widths=1.5), static=True)
    rr.log("timeseries/pixels/y/pupil_center", rr.SeriesPoints(marker_sizes=2.0), static=True)


# =============================================================================
# VIDEO LOADING
# =============================================================================


class VideoFrameReader:
    """Stream video frames one at a time instead of loading all into memory."""

    def __init__(self, video_path: Path, expected_n_frames: int):
        if not HAS_CV2:
            raise RuntimeError("OpenCV (cv2) not installed. Install with: pip install opencv-python")

        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frames = min(self.total_frames, expected_n_frames)
        self.current_frame = 0

        print(f"Video: {self.video_path.name} ({self.total_frames} frames)")
        if self.total_frames != expected_n_frames:
            print(f"  Warning: Video has {self.total_frames} frames but kinematics has {expected_n_frames}")
            print(f"  Using {self.n_frames} frames")

    def read_frame(self) -> NDArray[np.uint8] | None:
        """Read the next frame, returning None if no more frames."""
        if self.current_frame >= self.n_frames:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.current_frame += 1
        # Convert BGR to RGB for Rerun
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        """Release the video capture."""
        self.cap.release()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def log_video_frame(frame: NDArray[np.uint8]) -> None:
    """Log a single video frame."""
    rr.log("video/frame", rr.Image(frame))


# =============================================================================
# MAIN VIEWER FUNCTION
# =============================================================================


def run_eye_kinematics_viewer(
    kinematics: "FerretEyeKinematics",
    pixel_data: dict[str, NDArray[np.float64]] | None = None,
    video_path: Path | str | None = None,
    spawn: bool = True,
    recording_id: str | None = None,
    time_window_seconds: float = 2.0,
) -> None:
    """
    Launch Rerun viewer for FerretEyeKinematics data.

    Args:
        kinematics: FerretEyeKinematics object with eye movement data
        pixel_data: Optional dict with pixel coordinates for validation
        video_path: Optional path to MP4 video synced with kinematics
        spawn: Whether to spawn the Rerun viewer
        recording_id: Optional recording ID
        time_window_seconds: Seconds before/after cursor to show in timeseries (default ±2s)
    """
    if recording_id is None:
        recording_id = f"eye_kinematics_{kinematics.name}"

    rr.init(recording_id, spawn=spawn)

    # Extract data
    timestamps = kinematics.timestamps
    n_frames = kinematics.n_frames
    eye_radius = kinematics.eyeball.reference_geometry.markers["pupil_center"].x

    pupil_center = kinematics.pupil_center_trajectory
    pupil_points = kinematics.pupil_points_trajectories
    quaternions = kinematics.quaternions_wxyz
    tear_duct = kinematics.tear_duct_mm
    outer_eye = kinematics.outer_eye_mm

    adduction_deg = np.degrees(kinematics.adduction_angle.values)
    elevation_deg = np.degrees(kinematics.elevation_angle.values)
    torsion_deg = np.degrees(kinematics.torsion_angle.values)

    adduction_vel = np.degrees(kinematics.adduction_velocity.values)
    elevation_vel = np.degrees(kinematics.elevation_velocity.values)
    torsion_vel = np.degrees(kinematics.torsion_velocity.values)

    # Set up video reader (streams frames, doesn't load all into memory)
    video_reader: VideoFrameReader | None = None
    has_video = False
    if video_path is not None and HAS_CV2:
        try:
            video_reader = VideoFrameReader(
                video_path=Path(video_path),
                expected_n_frames=n_frames,
            )
            has_video = True
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not open video: {e}")

    # Create and send blueprint
    blueprint = create_eye_viewer_blueprint(
        has_pixel_data=pixel_data is not None,
        has_video=has_video,
        time_window_seconds=time_window_seconds,
    )
    rr.send_blueprint(blueprint)

    # Set up line-and-markers styling for timeseries
    setup_timeseries_styling()

    # Log static world frame (axes at origin)
    log_static_world_frame(axis_length=eye_radius * 1.5)

    # Log time-varying data
    gaze_length = eye_radius * 2.0
    basis_length = eye_radius * 1.2

    print(f"Logging {n_frames} frames for '{kinematics.name}' ({kinematics.eye_side} eye)...")

    try:
        for frame_idx in range(n_frames):
            t = timestamps[frame_idx]
            set_time_seconds("time", t)

            # Video frame (streamed one at a time)
            if video_reader is not None:
                frame = video_reader.read_frame()
                if frame is not None:
                    log_video_frame(frame)

            # 3D geometry - sphere rotates with gaze, gaze points out north pole (+Z)
            log_rotating_sphere_and_gaze(
                quaternion=quaternions[frame_idx],
                eye_radius=eye_radius,
                gaze_length=gaze_length,
            )

            # Eye basis vectors (RGB axes that rotate with eye)
            log_eye_basis_vectors(
                quaternion=quaternions[frame_idx],
                axis_length=basis_length,
            )

            log_pupil_geometry(
                pupil_center=pupil_center[frame_idx],
                pupil_points=pupil_points[frame_idx],
                quaternion=quaternions[frame_idx],
            )
            log_socket_landmarks(
                tear_duct=tear_duct[frame_idx],
                outer_eye=outer_eye[frame_idx],
            )

            # Timeseries
            log_timeseries_angles(
                adduction_deg=adduction_deg[frame_idx],
                elevation_deg=elevation_deg[frame_idx],
                torsion_deg=torsion_deg[frame_idx],
            )
            log_timeseries_velocities(
                adduction_vel=adduction_vel[frame_idx],
                elevation_vel=elevation_vel[frame_idx],
                torsion_vel=torsion_vel[frame_idx],
            )

            # Pixel data
            if pixel_data is not None:
                log_pixel_data(pixel_data=pixel_data, frame_idx=frame_idx)
    finally:
        if video_reader is not None:
            video_reader.close()

    print(f"Done! Viewer ready.")


# =============================================================================
# DEMO MODE (Synthetic Data)
# =============================================================================


def generate_synthetic_eye_data(
    n_frames: int = 500,
    duration_seconds: float = 5.0,
    eye_radius: float = 3.5,
    pupil_radius: float = 0.5,
    pupil_eccentricity: float = 0.8,
) -> tuple[
    NDArray[np.float64],  # timestamps
    NDArray[np.float64],  # quaternions_wxyz (N, 4)
    NDArray[np.float64],  # pupil_center (N, 3)
    NDArray[np.float64],  # pupil_points (N, 8, 3)
    NDArray[np.float64],  # tear_duct (N, 3)
    NDArray[np.float64],  # outer_eye (N, 3)
]:
    """Generate synthetic eye movement data for demo."""
    timestamps = np.linspace(0, duration_seconds, n_frames)

    # Generate oscillating gaze angles
    freq_h = 0.8  # Hz
    freq_v = 0.5
    freq_t = 0.3

    azimuth = np.radians(15.0) * np.sin(2 * np.pi * freq_h * timestamps)
    elevation = np.radians(10.0) * np.sin(2 * np.pi * freq_v * timestamps + 0.5)
    torsion = np.radians(5.0) * np.sin(2 * np.pi * freq_t * timestamps + 1.0)

    # Convert to quaternions
    quaternions = np.zeros((n_frames, 4), dtype=np.float64)
    for i in range(n_frames):
        # Build rotation: torsion (X) -> elevation (Y) -> azimuth (Z)
        # Using ZYX Euler to quaternion
        cy, sy = np.cos(azimuth[i] / 2), np.sin(azimuth[i] / 2)
        cp, sp = np.cos(elevation[i] / 2), np.sin(elevation[i] / 2)
        cr, sr = np.cos(torsion[i] / 2), np.sin(torsion[i] / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        quaternions[i] = [w, x, y, z]

    # Generate pupil geometry in rest frame
    rest_pupil_center = np.array([eye_radius, 0, 0])
    rest_pupil_points = generate_pupil_ellipse_points(
        eye_radius=eye_radius,
        pupil_radius=pupil_radius,
        pupil_eccentricity=pupil_eccentricity,
    )

    # Rotate for each frame
    pupil_center = np.zeros((n_frames, 3), dtype=np.float64)
    pupil_points = np.zeros((n_frames, NUM_PUPIL_POINTS, 3), dtype=np.float64)

    for i in range(n_frames):
        R = quaternion_to_rotation_matrix(quaternions[i])
        pupil_center[i] = R @ rest_pupil_center
        for j in range(NUM_PUPIL_POINTS):
            pupil_points[i, j] = R @ rest_pupil_points[j]

    # Static socket landmarks with small noise
    mean_tear_duct = np.array([0.5, eye_radius * 0.8, 0])
    mean_outer_eye = np.array([0.5, -eye_radius * 0.8, 0])

    noise_scale = 0.05
    tear_duct = mean_tear_duct + noise_scale * np.random.randn(n_frames, 3)
    outer_eye = mean_outer_eye + noise_scale * np.random.randn(n_frames, 3)

    return timestamps, quaternions, pupil_center, pupil_points, tear_duct, outer_eye


def run_demo_eye_viewer(
    spawn: bool = True,
    time_window_seconds: float = 2.0,
) -> None:
    """
    Run a demo of the eye kinematics viewer with synthetic data.

    This doesn't require any external data files.

    Args:
        spawn: Whether to spawn the Rerun viewer
        time_window_seconds: Seconds before/after cursor to show in timeseries
    """
    print("Generating synthetic eye movement data...")

    (
        timestamps,
        quaternions,
        pupil_center,
        pupil_points,
        tear_duct,
        outer_eye,
    ) = generate_synthetic_eye_data(
        n_frames=500,
        duration_seconds=5.0,
    )

    n_frames = len(timestamps)
    eye_radius = 3.5

    # Compute angles from quaternions
    adduction_deg = np.zeros(n_frames)
    elevation_deg = np.zeros(n_frames)
    torsion_deg = np.zeros(n_frames)

    for i in range(n_frames):
        R = quaternion_to_rotation_matrix(quaternions[i])
        # Gaze direction is rotated +X
        gaze = R @ np.array([1, 0, 0])

        # Azimuth (yaw) = atan2(y, x)
        azimuth = np.arctan2(gaze[1], gaze[0])
        # Elevation = atan2(z, sqrt(x^2 + y^2))
        horizontal = np.sqrt(gaze[0] ** 2 + gaze[1] ** 2)
        elev = np.arctan2(gaze[2], horizontal)

        adduction_deg[i] = np.degrees(azimuth)
        elevation_deg[i] = np.degrees(elev)

        # Approximate torsion from roll Euler angle
        # Using arctan2 for roll extraction
        sinr = 2 * (quaternions[i, 0] * quaternions[i, 1] + quaternions[i, 2] * quaternions[i, 3])
        cosr = 1 - 2 * (quaternions[i, 1] ** 2 + quaternions[i, 2] ** 2)
        torsion_deg[i] = np.degrees(np.arctan2(sinr, cosr))

    # Compute velocities via finite differences
    dt = np.diff(timestamps)
    adduction_vel = np.zeros(n_frames)
    elevation_vel = np.zeros(n_frames)
    torsion_vel = np.zeros(n_frames)

    adduction_vel[1:] = np.diff(adduction_deg) / dt
    elevation_vel[1:] = np.diff(elevation_deg) / dt
    torsion_vel[1:] = np.diff(torsion_deg) / dt
    adduction_vel[0] = adduction_vel[1]
    elevation_vel[0] = elevation_vel[1]
    torsion_vel[0] = torsion_vel[1]

    # Generate fake pixel data - only pupil center for validation
    pixel_scale = 50  # pixels per mm
    pixel_offset_x = 320
    pixel_offset_y = 240

    pixel_data = {
        "pupil_center_x": pupil_center[:, 1] * pixel_scale + pixel_offset_x,
        "pupil_center_y": -pupil_center[:, 2] * pixel_scale + pixel_offset_y,
    }

    # Initialize Rerun
    rr.init("eye_kinematics_demo", spawn=spawn)

    # Create and send blueprint
    blueprint = create_eye_viewer_blueprint(
        has_pixel_data=True,
        time_window_seconds=time_window_seconds,
    )
    rr.send_blueprint(blueprint)

    # Set up line-and-markers styling for timeseries
    setup_timeseries_styling()

    # Log static world frame
    log_static_world_frame(axis_length=eye_radius * 1.5)

    # Log time-varying data
    gaze_length = eye_radius * 2.0
    basis_length = eye_radius * 1.2

    print(f"Logging {n_frames} frames...")

    for frame_idx in range(n_frames):
        t = timestamps[frame_idx]
        set_time_seconds("time", t)

        # Sphere rotates with gaze, gaze points out north pole (+Z)
        log_rotating_sphere_and_gaze(
            quaternion=quaternions[frame_idx],
            eye_radius=eye_radius,
            gaze_length=gaze_length,
        )

        # Eye basis vectors (RGB axes that rotate with eye)
        log_eye_basis_vectors(
            quaternion=quaternions[frame_idx],
            axis_length=basis_length,
        )

        log_pupil_geometry(
            pupil_center=pupil_center[frame_idx],
            pupil_points=pupil_points[frame_idx],
            quaternion=quaternions[frame_idx],
        )
        log_socket_landmarks(
            tear_duct=tear_duct[frame_idx],
            outer_eye=outer_eye[frame_idx],
        )
        log_timeseries_angles(
            adduction_deg=adduction_deg[frame_idx],
            elevation_deg=elevation_deg[frame_idx],
            torsion_deg=torsion_deg[frame_idx],
        )
        log_timeseries_velocities(
            adduction_vel=adduction_vel[frame_idx],
            elevation_vel=elevation_vel[frame_idx],
            torsion_vel=torsion_vel[frame_idx],
        )
        log_pixel_data(pixel_data=pixel_data, frame_idx=frame_idx)

    print("Demo viewer ready! Use the timeline scrubber to animate.")


# =============================================================================
# CSV LOADER
# =============================================================================


def run_eye_viewer_from_csv(
    eye_trajectories_csv_path: Path,
    eye_side: Literal["left", "right"],
    camera_distance_mm: float,
    video_path: Path | str | None = None,
    spawn: bool = True,
    time_window_seconds: float = 2.0,
) -> None:
    """
    Load eye data from CSV and launch viewer.

    Args:
        eye_trajectories_csv_path: Path to eye_trajectories.csv
        eye_side: "left" or "right"
        camera_distance_mm: Distance from camera to eye center in mm
        video_path: Optional path to synced MP4 video
        spawn: Whether to spawn the Rerun viewer
        time_window_seconds: Seconds before/after cursor to show in timeseries
    """
    from python_code.ferret_gaze.eye_kinematics.load_eye_data import (
        load_ferret_eye_kinematics,
        load_eye_csv,
        extract_frame_data,
    )

    print(f"Loading eye data from {eye_trajectories_csv_path}...")

    kinematics = load_ferret_eye_kinematics(
        eye_trajectories_csv_path=eye_trajectories_csv_path,
        eye_side=eye_side,
        camera_distance_mm=camera_distance_mm,
    )

    # Also load raw pixel data for validation (only pupil center)
    pixel_data = None
    try:
        df = load_eye_csv(csv_path=eye_trajectories_csv_path, eye_side=eye_side)
        timestamps, pupil_centers_px, pupil_points_px, tear_duct_px, outer_eye_px = extract_frame_data(df)

        # Only keep pupil center for validation plots
        pixel_data = {
            "pupil_center_x": pupil_centers_px[:, 0],
            "pupil_center_y": pupil_centers_px[:, 1],
        }

        print(f"Loaded {len(timestamps)} frames with pixel validation data")
    except Exception as e:
        print(f"Warning: Could not load pixel data: {e}")

    run_eye_kinematics_viewer(
        kinematics=kinematics,
        pixel_data=pixel_data,
        video_path=video_path,
        spawn=spawn,
        time_window_seconds=time_window_seconds,
    )


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eye Kinematics Rerun Viewer")
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Path to eye_trajectories.csv file (omit for demo mode)",
    )
    parser.add_argument(
        "--eye-side",
        type=str,
        choices=["left", "right"],
        default="right",
        help="Which eye to visualize",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=21.0,
        help="Camera to eye distance in mm",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Path to MP4 video synced with kinematics (same frame count)",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Don't spawn viewer (connect to existing)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic data",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        default=5.0,
        help="Seconds before/after cursor to show in timeseries (default: 2.0)",
    )

    args = parser.parse_args()
    if args.demo and args.csv_path is None:
        print("Running demo mode with synthetic data...")
        run_demo_eye_viewer(
            spawn=not args.no_spawn,
            time_window_seconds=args.time_window,
        )
    else:
        if args.csv_path is None:
            args.csv_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_trajectories.csv")
        run_eye_viewer_from_csv(
            eye_trajectories_csv_path=args.csv_path,
            video_path=r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\left_eye_stabilized.mp4",
            eye_side=args.eye_side,
            camera_distance_mm=args.camera_distance,
            spawn=not args.no_spawn,
            time_window_seconds=args.time_window,
        )