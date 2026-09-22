  
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import (
    COLOR_EYE_X_AXIS,
    COLOR_EYE_Y_AXIS,
    COLOR_EYE_Z_AXIS,
    COLOR_GAZE_ARROW,
    COLOR_OUTER_EYE,
    COLOR_PUPIL_BOUNDARY,
    COLOR_PUPIL_CENTER,
    COLOR_PUPIL_FACE_LEFT,
    COLOR_PUPIL_POINTS,
    COLOR_SPHERE_WIRE,
    COLOR_TEAR_DUCT,
    generate_sphere_line_strips_with_pole_at_z,
    generate_sphere_mesh,
    get_eye_radius_from_kinematics,
    log_static_world_frame,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def get_3d_eye_view(eye_name: str, entity_path: str = "/"):
    # Top-down camera settings: looking down from +Z axis at the origin
    # Eye is above the scene along Z, looking down at the eyeball
    top_down_eye_controls = rrb.EyeControls3D(
        position=(0.0, 0.0, 15.0),  # Camera positioned along +Z axis
        look_target=(0.0, 0.0, 0.0),  # Looking at origin (eye center)
        eye_up=(0.0, 1.0, 0.0),  # Y+ is "up" in the view
        kind=rrb.Eye3DKind.Orbital,
    )

    # 3D views for each eye with top-down camera
    eye_3d = rrb.Spatial3DView(
        name=f"{eye_name.capitalize()} Eye 3D",
        origin=entity_path,
        contents=[f"+ /{eye_name}_eye/**", "+ /world_frame/**"],
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=top_down_eye_controls,
    )

    return eye_3d

def _quaternions_wxyz_to_rerun_xyzw(quaternions_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize (N,4) wxyz quaternions and reorder to Rerun's xyzw (zero-norm -> identity)."""
    norms = np.linalg.norm(quaternions_wxyz, axis=1, keepdims=True)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    unit_wxyz = np.where(norms < 1e-10, identity, quaternions_wxyz / np.maximum(norms, 1e-10))
    return unit_wxyz[:, [1, 2, 3, 0]]


def send_3d_eye(
    eye_prefix: str,
    timestamps: NDArray[np.float64],
    quaternions_wxyz: NDArray[np.float64],
    tracked_pupil_center: NDArray[np.float64],
    tracked_pupil_points: NDArray[np.float64],
    tear_duct_mm: NDArray[np.float64],
    outer_eye_mm: NDArray[np.float64],
    eye_radius: float,
) -> None:
    """Send the animated 3D eye to Rerun as whole columns.

    Same entities and appearance as logging log_rotating_sphere_and_gaze / log_eye_basis_vectors /
    log_pupil_geometry / log_socket_landmarks once per frame, but the eye-fixed geometry (sphere,
    gaze arrow, basis arrows) is logged once, unrotated, and the per-frame eye rotation is sent as a
    Transform3D column. That is ~50x less data than a rotated copy of the sphere mesh per frame, and
    a handful of send_columns calls instead of ~14 rr.log calls per frame.
    """
    n_frames = len(timestamps)
    n_pupil_points = tracked_pupil_points.shape[1]
    time_column = rr.TimeColumn("time", duration=timestamps)
    eye_rotation = rr.Transform3D.columns(quaternion=_quaternions_wxyz_to_rerun_xyzw(quaternions_wxyz))

    # --- Eye-fixed geometry: static, rotated by the per-frame Transform3D on the same entity ---
    sphere_strips = generate_sphere_line_strips_with_pole_at_z(eye_radius, 8, 16)
    rr.log(
        f"{eye_prefix}/sphere/wireframe",
        rr.LineStrips3D(strips=sphere_strips, colors=[COLOR_SPHERE_WIRE] * len(sphere_strips), radii=[0.015]),
        static=True,
    )
    mesh_vertices, mesh_triangles = generate_sphere_mesh(eye_radius * 0.99, 12, 24)
    rr.log(
        f"{eye_prefix}/sphere/mesh",
        rr.Mesh3D(
            vertex_positions=mesh_vertices,
            triangle_indices=mesh_triangles,
            albedo_factor=[200, 200, 220, 25],
        ),
        static=True,
    )
    rr.send_columns(f"{eye_prefix}/sphere", indexes=[time_column], columns=eye_rotation)

    rr.log(
        f"{eye_prefix}/gaze_arrow",
        rr.Arrows3D(
            origins=[[0, 0, 0]], vectors=[[0, 0, eye_radius * 2.0]], colors=[COLOR_GAZE_ARROW], radii=[0.1]
        ),
        static=True,
    )
    rr.send_columns(f"{eye_prefix}/gaze_arrow", indexes=[time_column], columns=eye_rotation)

    axis_length = eye_radius * 1.2
    for axis_index, (name, color) in enumerate(
        zip(("x", "y", "z"), (COLOR_EYE_X_AXIS, COLOR_EYE_Y_AXIS, COLOR_EYE_Z_AXIS))
    ):
        vector = np.zeros(3)
        vector[axis_index] = axis_length
        rr.log(
            f"{eye_prefix}/basis/{name}_axis",
            rr.Arrows3D(origins=[[0, 0, 0]], vectors=[vector], colors=[color], radii=[0.06]),
            static=True,
        )
        rr.send_columns(f"{eye_prefix}/basis/{name}_axis", indexes=[time_column], columns=eye_rotation)

    # --- Tracked pupil (head-frame positions, not rotated by the eye) ---
    rr.log(
        f"{eye_prefix}/pupil/center",
        rr.Points3D.from_fields(colors=[COLOR_PUPIL_CENTER], radii=[0.12]),
        static=True,
    )
    rr.send_columns(
        f"{eye_prefix}/pupil/center",
        indexes=[time_column],
        columns=rr.Points3D.columns(positions=tracked_pupil_center),
    )

    # Closed boundary loop: first point repeated at the end
    boundary_strips = np.concatenate([tracked_pupil_points, tracked_pupil_points[:, :1]], axis=1)
    rr.log(
        f"{eye_prefix}/pupil/boundary",
        rr.LineStrips3D.from_fields(colors=[COLOR_PUPIL_BOUNDARY], radii=[0.06]),
        static=True,
    )
    rr.send_columns(
        f"{eye_prefix}/pupil/boundary",
        indexes=[time_column],
        columns=rr.LineStrips3D.columns(strips=list(boundary_strips)),
    )

    rr.log(
        f"{eye_prefix}/pupil/points",
        rr.Points3D.from_fields(colors=[COLOR_PUPIL_POINTS], radii=[0.08]),
        static=True,
    )
    rr.send_columns(
        f"{eye_prefix}/pupil/points",
        indexes=[time_column],
        columns=rr.Points3D.columns(positions=tracked_pupil_points.reshape(-1, 3)).partition(
            lengths=[n_pupil_points] * n_frames
        ),
    )

    # Filled pupil face: fan of triangles from the center (vertex 0) around the boundary points
    face_triangles = np.array(
        [[0, i + 1, (i + 1) % n_pupil_points + 1] for i in range(n_pupil_points)], dtype=np.uint32
    )
    n_face_vertices = n_pupil_points + 1
    face_vertices = np.concatenate([tracked_pupil_center[:, None, :], tracked_pupil_points], axis=1)
    rr.log(
        f"{eye_prefix}/pupil/face",
        rr.Mesh3D.from_fields(
            triangle_indices=face_triangles,
            # NOTE: the per-frame log_pupil_geometry picks the right-eye color only when
            # eye_prefix == "right_eye", which never matches "//right_eye", so it was always the left color.
            vertex_colors=[COLOR_PUPIL_FACE_LEFT] * n_face_vertices,
        ),
        static=True,
    )
    rr.send_columns(
        f"{eye_prefix}/pupil/face",
        indexes=[time_column],
        columns=rr.Mesh3D.columns(vertex_positions=face_vertices.reshape(-1, 3)).partition(
            lengths=[n_face_vertices] * n_frames
        ),
    )

    # --- Socket landmarks, connected to the eye center ---
    origin = np.zeros((n_frames, 3))
    for name, positions, color in (
        ("tear_duct", tear_duct_mm, COLOR_TEAR_DUCT),
        ("outer_eye", outer_eye_mm, COLOR_OUTER_EYE),
    ):
        rr.log(
            f"{eye_prefix}/socket/{name}",
            rr.Points3D.from_fields(colors=[color], radii=[0.15], labels=[name]),
            static=True,
        )
        rr.send_columns(
            f"{eye_prefix}/socket/{name}",
            indexes=[time_column],
            columns=rr.Points3D.columns(positions=positions),
        )
        rr.log(
            f"{eye_prefix}/socket/{name}_line",
            rr.LineStrips3D.from_fields(colors=[color], radii=[0.04]),
            static=True,
        )
        rr.send_columns(
            f"{eye_prefix}/socket/{name}_line",
            indexes=[time_column],
            columns=rr.LineStrips3D.columns(strips=list(np.stack([origin, positions], axis=1))),
        )


def plot_3d_eye(
    eye_name: str,
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    """Plot 3D eye kinematics."""
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")


    kinematics = FerretEyeKinematics.load_from_directory(
        eye_name=f"{eye_name}_eye", 
        input_directory=recording_folder.left_eye_kinematics if eye_name == "left" else recording_folder.right_eye_kinematics
    )

    timestamps = kinematics.eyeball.timestamps
    timestamps = timestamps - timestamps[0]
    
    print(f"Loaded left eye kinematics: {kinematics.n_frames} frames")

    eye_radius = get_eye_radius_from_kinematics(kinematics)
    log_static_world_frame(f"{entity_path}/{eye_name}_eye", eye_radius * 1.5, eye_radius)

    send_3d_eye(
        eye_prefix=f"{entity_path}/{eye_name}_eye",
        timestamps=timestamps,
        quaternions_wxyz=kinematics.quaternions_wxyz,
        tracked_pupil_center=kinematics.tracked_pupil_center,
        tracked_pupil_points=kinematics.tracked_pupil_points,
        tear_duct_mm=kinematics.tear_duct_mm,
        outer_eye_mm=kinematics.outer_eye_mm,
        eye_radius=eye_radius,
    )

if __name__ == "__main__":
    from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    eye_name = "left"

    recording_folder = RecordingFolder.from_folder_path(folder_path)
    recording_folder.check_eye_postprocessing()

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    rr.init(recording_string, spawn=True)

    view = get_3d_eye_view(eye_name, entity_path="/")

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)

    plot_3d_eye(eye_name=eye_name, recording_folder=recording_folder)
