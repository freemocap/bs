"""
Ferret Eye Kinematics Rerun Viewer
==================================

COORDINATE SYSTEM (matches model frame):
    +Z = gaze direction (toward pupil, "north pole")
    +Y = superior (up)
    +X = subject's left
"""
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import rerun as rr
import rerun.blueprint as rrb

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import (
    load_eye_trajectories_csv,
    extract_frame_data,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics

import cv2

NUM_PUPIL_POINTS: int = 8

# Colors (RGB)
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


def set_time_seconds(timeline: str, seconds: float) -> None:
    """Set time on timeline with Rerun API compatibility."""
    if hasattr(rr, 'set_time'):
        try:
            rr.set_time(timeline, timestamp=seconds)
            return
        except TypeError:
            pass
    if hasattr(rr, 'set_time_seconds'):
        try:
            rr.set_time_seconds(timeline, seconds)
            return
        except TypeError:
            pass
    if hasattr(rr, 'set_time_sequence'):
        rr.set_time_sequence(timeline, int(seconds * 1e9))
        return
    raise RuntimeError("Could not find compatible Rerun time API.")


def model_to_viz(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform from model coordinates to visualization coordinates.

    With Z+ = gaze convention, model and viz frames are identical.
    """
    return np.asarray(p, dtype=np.float64)


def rotate_strips(
    strips: list[NDArray[np.float64]],
    rotation_matrix: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Rotate all line strips by a rotation matrix."""
    return [(rotation_matrix @ strip.T).T for strip in strips]


def quaternion_to_rotation_matrix(q_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-10:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def generate_sphere_line_strips_with_pole_at_z(
    radius: float,
    n_lat: int = 10,
    n_lon: int = 20,
) -> list[NDArray[np.float64]]:
    """Generate wireframe sphere with north pole at +Z."""
    strips = []
    # Latitude circles
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        r_circle = radius * np.sin(theta)
        z_pos = radius * np.cos(theta)
        circle = []
        for j in range(n_lon + 1):
            phi = 2 * np.pi * j / n_lon
            circle.append([r_circle * np.cos(phi), r_circle * np.sin(phi), z_pos])
        strips.append(np.array(circle, dtype=np.float64))
    # Longitude circles
    for j in range(0, n_lon, 2):
        phi = 2 * np.pi * j / n_lon
        circle = []
        for i in range(n_lat + 1):
            theta = np.pi * i / n_lat
            circle.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
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
            vertices.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
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


def create_eye_viewer_blueprint(
    time_window_seconds: float = 2.0,
) -> rrb.Blueprint:
    """Create Rerun blueprint for eye viewer layout."""
    scrolling_time_range = rrb.VisibleTimeRange(
        timeline="time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    angle_view = rrb.TimeSeriesView(
        name="Gaze Angles (deg)",
        origin="/timeseries/angles",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-25.0, 25.0)),
    )
    velocity_view = rrb.TimeSeriesView(
        name="Angular Velocity (deg/s)",
        origin="/timeseries/velocity",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-350.0, 350.0)),
    )
    timeseries_views = [angle_view, velocity_view]

    timeseries_views.extend([
        rrb.TimeSeriesView(
            name="Pupil Center X (px)",
            origin="/timeseries/pixels/x",
            plot_legend=rrb.PlotLegend(visible=True),
            time_ranges=scrolling_time_range,
            axis_y=rrb.ScalarAxis(range=(-40.0, 40.0)),
        ),
        rrb.TimeSeriesView(
            name="Pupil Center Y (px)",
            origin="/timeseries/pixels/y",
            plot_legend=rrb.PlotLegend(visible=True),
            time_ranges=scrolling_time_range,
            axis_y=rrb.ScalarAxis(range=(-30.0, 30.0)),
        ),
    ])

    viewer_3d = rrb.Spatial3DView(
        name="Eye 3D View",
        origin="/",
        contents=["+ /eye/**", "+ /world_frame/**"],
        line_grid=rrb.LineGrid3D(visible=False),
    )

    video_view = rrb.Spatial2DView(name="Eye Video", origin="/video")
    left_column = rrb.Vertical(viewer_3d, video_view, row_shares=[0.5, 0.5])
    return rrb.Blueprint(
        rrb.Horizontal(left_column, rrb.Vertical(*timeseries_views), column_shares=[0.40, 0.60]),
        collapse_panels=True,
    )
    return rrb.Blueprint(
        rrb.Horizontal(viewer_3d, rrb.Vertical(*timeseries_views), column_shares=[0.55, 0.45]),
        collapse_panels=True,
    )


def log_static_world_frame(axis_length: float) -> None:
    """Log static world reference frame axes."""
    rr.log("world_frame/x_axis", rr.Arrows3D(origins=[[0,0,0]], vectors=[[axis_length,0,0]], colors=[COLOR_WORLD_X], radii=[0.04]), static=True)
    rr.log("world_frame/y_axis", rr.Arrows3D(origins=[[0,0,0]], vectors=[[0,axis_length,0]], colors=[COLOR_WORLD_Y], radii=[0.04]), static=True)
    rr.log("world_frame/z_axis", rr.Arrows3D(origins=[[0,0,0]], vectors=[[0,0,axis_length]], colors=[COLOR_WORLD_Z], radii=[0.04]), static=True)
    rr.log("eye/center", rr.Points3D(positions=[[0,0,0]], colors=[COLOR_EYE_CENTER], radii=[0.12]), static=True)


def log_eye_basis_vectors(quaternion: NDArray[np.float64], axis_length: float) -> None:
    """Log eye frame basis vectors (rotate with eye)."""
    R = quaternion_to_rotation_matrix(quaternion)
    axes = [np.array([axis_length, 0, 0]), np.array([0, axis_length, 0]), np.array([0, 0, axis_length])]
    colors = [COLOR_EYE_X_AXIS, COLOR_EYE_Y_AXIS, COLOR_EYE_Z_AXIS]
    names = ['x', 'y', 'z']
    for axis, color, name in zip(axes, colors, names):
        v = model_to_viz(R @ axis)
        rr.log(f"eye/basis/{name}_axis", rr.Arrows3D(origins=[[0,0,0]], vectors=[v], colors=[color], radii=[0.06]))


def log_rotating_sphere_and_gaze(
    quaternion: NDArray[np.float64],
    eye_radius: float,
    gaze_length: float,
) -> None:
    """Log wireframe sphere and gaze arrow."""
    R = quaternion_to_rotation_matrix(quaternion)

    # Wireframe
    local_strips = generate_sphere_line_strips_with_pole_at_z(eye_radius, 8, 16)
    rotated_strips = [model_to_viz(s) for s in rotate_strips(local_strips, R)]
    rr.log("eye/sphere/wireframe", rr.LineStrips3D(strips=rotated_strips, colors=[COLOR_SPHERE_WIRE] * len(rotated_strips), radii=[0.015]))

    # Mesh
    verts, tris = generate_sphere_mesh(eye_radius * 0.99, 12, 24)
    verts_rotated = model_to_viz((R @ verts.T).T)
    rr.log("eye/sphere/mesh", rr.Mesh3D(vertex_positions=verts_rotated, triangle_indices=tris, vertex_colors=[[200, 200, 220, 40]] * len(verts)))

    # Gaze arrow (rest gaze = +Z)
    gaze_vec = model_to_viz(R @ np.array([0, 0, gaze_length]))
    rr.log("eye/gaze_arrow", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[gaze_vec], colors=[COLOR_GAZE_ARROW], radii=[0.1]))


COLOR_PUPIL_FACE = [100, 100, 100]  # Grey color for pupil face


def log_pupil_geometry(
    pupil_center: NDArray[np.float64],
    pupil_points: NDArray[np.float64],
    quaternion: NDArray[np.float64],
) -> None:
    """Log pupil center, boundary, and filled face."""
    pc = model_to_viz(pupil_center)
    pp = model_to_viz(pupil_points)

    # Log center point
    rr.log("eye/pupil/center", rr.Points3D(positions=[pc], colors=[COLOR_PUPIL_CENTER], radii=[0.12]))

    # Log boundary line
    rr.log("eye/pupil/boundary", rr.LineStrips3D(strips=[np.vstack([pp, pp[0:1]])], colors=[COLOR_PUPIL_BOUNDARY], radii=[0.06]))

    # Log boundary points
    rr.log("eye/pupil/points", rr.Points3D(positions=pp, colors=[COLOR_PUPIL_POINTS] * len(pp), radii=[0.08]))

    # Create filled pupil face as a triangle fan mesh (center + boundary points)
    n_points = len(pp)
    vertices = np.vstack([pc, pp])  # vertex 0 is center, 1..n are boundary
    triangles = []
    for i in range(n_points):
        next_i = (i + 1) % n_points
        # Triangle: center, point i+1, point next_i+1 (offset by 1 because center is vertex 0)
        triangles.append([0, i + 1, next_i + 1])
    triangles = np.array(triangles, dtype=np.uint32)

    # Log grey pupil face mesh
    rr.log("eye/pupil/face", rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangles,
        vertex_colors=[COLOR_PUPIL_FACE] * len(vertices),
    ))


def log_socket_landmarks(tear_duct: NDArray[np.float64], outer_eye: NDArray[np.float64]) -> None:
    """Log socket landmarks."""
    td, oe = model_to_viz(tear_duct), model_to_viz(outer_eye)
    rr.log("eye/socket/tear_duct", rr.Points3D(positions=[td], colors=[COLOR_TEAR_DUCT], radii=[0.15]))
    rr.log("eye/socket/outer_eye", rr.Points3D(positions=[oe], colors=[COLOR_OUTER_EYE], radii=[0.15]))
    rr.log("eye/socket/opening_line", rr.LineStrips3D(strips=[np.array([td, oe])], colors=[COLOR_SOCKET_LINE], radii=[0.04]))


def log_timeseries_angles(adduction_deg: float, elevation_deg: float, torsion_deg: float) -> None:
    rr.log("timeseries/angles/adduction", rr.Scalars(adduction_deg))
    rr.log("timeseries/angles/elevation", rr.Scalars(elevation_deg))
    rr.log("timeseries/angles/torsion", rr.Scalars(torsion_deg))


def log_timeseries_velocities(adduction_vel: float, elevation_vel: float, torsion_vel: float) -> None:
    rr.log("timeseries/velocity/adduction", rr.Scalars(adduction_vel))
    rr.log("timeseries/velocity/elevation", rr.Scalars(elevation_vel))
    rr.log("timeseries/velocity/torsion", rr.Scalars(torsion_vel))


def log_pixel_data(pixel_data: dict[str, NDArray[np.float64]], frame_idx: int) -> None:
    if "pupil_center_x" in pixel_data:
        rr.log("timeseries/pixels/x/pupil_center", rr.Scalars(pixel_data["pupil_center_x"][frame_idx]))
    if "pupil_center_y" in pixel_data:
        rr.log("timeseries/pixels/y/pupil_center", rr.Scalars(pixel_data["pupil_center_y"][frame_idx]))


def setup_timeseries_styling() -> None:
    for name in ["adduction", "elevation", "torsion"]:
        rr.log(f"timeseries/angles/{name}", rr.SeriesLines(widths=1.5), static=True)
        rr.log(f"timeseries/angles/{name}", rr.SeriesPoints(marker_sizes=2.0), static=True)
        rr.log(f"timeseries/velocity/{name}", rr.SeriesLines(widths=1.5), static=True)
        rr.log(f"timeseries/velocity/{name}", rr.SeriesPoints(marker_sizes=2.0), static=True)
    rr.log("timeseries/pixels/x/pupil_center", rr.SeriesLines(widths=1.5), static=True)
    rr.log("timeseries/pixels/x/pupil_center", rr.SeriesPoints(marker_sizes=2.0), static=True)
    rr.log("timeseries/pixels/y/pupil_center", rr.SeriesLines(widths=1.5), static=True)
    rr.log("timeseries/pixels/y/pupil_center", rr.SeriesPoints(marker_sizes=2.0), static=True)


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
    """
    Extract eye radius from reference geometry.

    With Z+ = gaze, pupil_center is at [0, 0, R].
    """
    pc = kinematics.eyeball.reference_geometry.keypoints["pupil_center"]
    # Pupil center should be at [0, 0, R], but use magnitude for robustness
    r = np.sqrt(pc.x**2 + pc.y**2 + pc.z**2)
    if r < 0.1:
        raise ValueError(f"Eye radius too small: {r}")
    return r


def run_eye_kinematics_viewer(
    kinematics: FerretEyeKinematics,
    pixel_data: dict[str, NDArray[np.float64]] | None = None,
    video_path: Path | str | None = None,
    spawn: bool = True,
    recording_id: str | None = None,
    time_window_seconds: float = 2.0,
) -> None:
    """Launch Rerun viewer for FerretEyeKinematics data."""
    rr.init(recording_id or f"eye_kinematics_{kinematics.name}", spawn=spawn)

    timestamps = kinematics.timestamps
    n_frames = kinematics.n_frames
    eye_radius = get_eye_radius_from_kinematics(kinematics)

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

    video_reader: VideoFrameReader | None = None
    if video_path:
        try:
            video_reader = VideoFrameReader(Path(video_path), n_frames)
        except Exception as e:
            raise RuntimeError(f"Could not open video at {video_path}: {e}")

    rr.send_blueprint(create_eye_viewer_blueprint( time_window_seconds))
    setup_timeseries_styling()
    log_static_world_frame(eye_radius * 1.5)

    print(f"Logging {n_frames} frames for '{kinematics.name}' ({kinematics.eye_side} eye)...")

    try:
        for i in range(n_frames):
            set_time_seconds("time", timestamps[i])

            if video_reader:
                frame = video_reader.read_frame()
                if frame is not None:
                    rr.log("video/frame", rr.Image(frame))

            log_rotating_sphere_and_gaze(quaternions[i], eye_radius, eye_radius * 2.0)
            log_eye_basis_vectors(quaternions[i], eye_radius * 1.2)
            log_pupil_geometry(pupil_center[i], pupil_points[i], quaternions[i])
            log_socket_landmarks(tear_duct[i], outer_eye[i])
            log_timeseries_angles(adduction_deg[i], elevation_deg[i], torsion_deg[i])
            log_timeseries_velocities(adduction_vel[i], elevation_vel[i], torsion_vel[i])
            if pixel_data:
                log_pixel_data(pixel_data, i)
    finally:
        if video_reader:
            video_reader.close()

    print("Done!")


def run_eye_rerun_viewer(
    eye_trajectories_csv_path: Path,
    eye_kinematics_directory_path: Path,
    eye_name: Literal["left_eye", "right_eye"],
    eye_video_path: Path | str | None = None,
    spawn: bool = True,
    time_window_seconds: float = 5.0,
) -> None:
    """Load eye data from directory and launch viewer."""
    print(f"Loading eye kinematics from {eye_kinematics_directory_path}...")

    kinematics = FerretEyeKinematics.load_from_directory(
        eye_name=eye_name,
        input_directory=eye_kinematics_directory_path,
    )

    pixel_data = None
    try:
        eye_side: Literal["left", "right"] = "left" if eye_name == "left_eye" else "right"
        df = load_eye_trajectories_csv(csv_path=eye_trajectories_csv_path, eye_side=eye_side)
        timestamps, pupil_centers_px, *_ = extract_frame_data(df)
        pixel_data = {"pupil_center_x": pupil_centers_px[:, 0], "pupil_center_y": pupil_centers_px[:, 1]}
        print(f"Loaded {len(timestamps)} frames with pixel data")
    except Exception as e:
        print(f"Warning: Could not load pixel data: {e}")

    run_eye_kinematics_viewer(
        kinematics=kinematics,
        pixel_data=pixel_data,
        video_path=eye_video_path,
        spawn=spawn,
        time_window_seconds=time_window_seconds,
    )


if __name__ == "__main__":
    _csv = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_trajectories.csv")
    _kin = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye_kinematics")
    _vid = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\left_eye_stabilized.mp4")
    run_eye_rerun_viewer(_csv, _kin, "left_eye", _vid, time_window_seconds=5)