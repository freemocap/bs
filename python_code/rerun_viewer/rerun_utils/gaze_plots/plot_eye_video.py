
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import EyeViewerData, VideoFrameReader, get_eye_radius_from_kinematics, log_eye_basis_vectors, log_pupil_geometry, log_rotating_sphere_and_gaze, log_socket_landmarks, log_static_world_frame, set_time_seconds
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import extract_frame_data, load_eye_trajectories_csv
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def get_eye_video_view(eye_name: str, entity_path: str = "/"):
    if not entity_path.endswith("/"):
        entity_path += "/"
    return rrb.Spatial2DView(name=f"{eye_name.capitalize()} Eye Video", origin=f"{entity_path}video/{eye_name}_eye")

def plot_eye_video(
    eye_name: str,
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    """Plot 3D eye kinematics."""
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")

    eye_kinematics_directory_path = recording_folder.eye_output_data
    print(f"Loading eye kinematics from {eye_kinematics_directory_path}...")


    eye_trajectories_csv_path = recording_folder.left_eye_data_csv if eye_name == "left" else recording_folder.right_eye_data_csv
    eye_video_path = recording_folder.left_eye_annotated_video if eye_name == "left" else recording_folder.right_eye_annotated_video

    if "757" in str(eye_video_path):
        left_eye_video_name = "eye0"
        right_eye_video_name = "eye1"
    else:
        left_eye_video_name = "eye1"
        right_eye_video_name = "eye0"

    eye_video_name = left_eye_video_name if eye_name == "left" else right_eye_video_name

    kinematics = FerretEyeKinematics.load_from_directory(
        eye_name="left_eye",
        input_directory=eye_kinematics_directory_path,
    )
    df = load_eye_trajectories_csv(
        csv_path=eye_trajectories_csv_path, eye_side=eye_name, video_name=eye_video_name
    )
    timestamps, pupil_centers_px, *_ = extract_frame_data(df)
    left_pixel_data = {
        "pupil_center_x": pupil_centers_px[:, 0],
        "pupil_center_y": pupil_centers_px[:, 1],
    }
    print(f"Loaded {eye_name} eye pixel data: {len(timestamps)} frames")

    eye_data = EyeViewerData(
        kinematics=kinematics,
        pixel_data=left_pixel_data,
        video_path=eye_video_path,
    )
    print(f"Loaded {eye_name} eye kinematics: {kinematics.n_frames} frames")

    timestamps = kinematics.timestamps

    try:
        video_reader = VideoFrameReader(eye_data.video_path, kinematics.n_frames)
    except Exception as e:
        print(f"Warning: Could not open video for {eye_name}: {e}")

    try:
        for i in range(kinematics.n_frames):
            set_time_seconds("time", timestamps[i])

            # Log video frames
            frame = video_reader.read_frame()
            if frame is not None:
                rr.log(f"{entity_path}video/{eye_name}/frame", rr.Image(frame))
    finally:
        video_reader.close()

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

    view = get_eye_video_view(eye_name, entity_path="/")

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)

    plot_eye_video(eye_name=eye_name, recording_folder=recording_folder)
