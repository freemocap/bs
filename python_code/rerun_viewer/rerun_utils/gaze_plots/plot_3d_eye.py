
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import get_eye_radius_from_kinematics, log_eye_basis_vectors, log_pupil_geometry, log_rotating_sphere_and_gaze, log_socket_landmarks, log_static_world_frame, set_time_seconds
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

def plot_3d_eye(
    eye_name: str,
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    """Plot 3D eye kinematics."""
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")


    kinematics = FerretEyeKinematics.load_from_directory(eye_name=f"{eye_name}_eye", input_directory=recording_folder.eye_output_data / "eye_kinematics")

    timestamps = kinematics.eyeball.timestamps
    timestamps = timestamps - timestamps[0]
    
    print(f"Loaded left eye kinematics: {kinematics.n_frames} frames")

    eye_radius = get_eye_radius_from_kinematics(kinematics)
    log_static_world_frame(f"{entity_path}/{eye_name}_eye", eye_radius * 1.5, eye_radius)

    for i in range(kinematics.n_frames):
        set_time_seconds("time", timestamps[i])
        log_rotating_sphere_and_gaze(
            f"{entity_path}/{eye_name}_eye",
            kinematics.quaternions_wxyz[i],
            eye_radius,
            eye_radius * 2.0,
        )
        log_eye_basis_vectors(
            f"{entity_path}/{eye_name}_eye", kinematics.quaternions_wxyz[i], eye_radius * 1.2
        )
        log_pupil_geometry(
            f"{entity_path}/{eye_name}_eye",
            kinematics.tracked_pupil_center[i],
            kinematics.tracked_pupil_points[i],
        )
        log_socket_landmarks(
            f"{entity_path}/{eye_name}_eye", kinematics.tear_duct_mm[i], kinematics.outer_eye_mm[i]
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
