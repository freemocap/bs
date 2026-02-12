
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import EyeViewerData, VideoFrameReader, set_time_seconds
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import extract_frame_data, load_eye_trajectories_csv
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def get_eye_video_view(eye_name: str, entity_path: str = "/"):
    if not entity_path.endswith("/"):
        entity_path += "/"
    return rrb.Spatial2DView(
        name=f"{eye_name.capitalize()} Eye Video", 
        origin=f"{entity_path}video/{eye_name}_eye"
    )

def plot_eye_video(
    eye_name: str,
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    """Plot 3D eye kinematics."""
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")

    eye_kinematics_directory_path = recording_folder.eye_output_data / "eye_kinematics"
    print(f"Loading eye kinematics from {eye_kinematics_directory_path}...")

    eye_video_path = recording_folder.left_eye_display_video if eye_name == "left" else recording_folder.right_eye_display_video

    kinematics = FerretEyeKinematics.load_from_directory(
        eye_name="left_eye",
        input_directory=eye_kinematics_directory_path,
    )
    eye_data = EyeViewerData(
        kinematics=kinematics,
        pixel_data=None,
        video_path=eye_video_path,
    )
    print(f"Loaded {eye_name} eye kinematics: {kinematics.n_frames} frames")

    timestamps = kinematics.eyeball.timestamps - kinematics.eyeball.timestamps[0]

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
                if eye_name == "right":
                    frame = video_reader.flip_frame(frame)
                rr.log(f"{entity_path}video/{eye_name}_eye", rr.Image(frame))
    finally:
        video_reader.close()

if __name__ == "__main__":
    from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    eye_name = "right"

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
