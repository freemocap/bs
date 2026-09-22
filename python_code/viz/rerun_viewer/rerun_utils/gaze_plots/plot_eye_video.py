
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.viz.rerun_viewer.rerun_utils.process_videos import log_video_file

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
    resize_factor: float = 1.0,
    jpeg_quality: int = 80,
):
    """Plot the eye display video (JPEG-encoded, optionally downscaled) on the "time" timeline."""
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")

    # NOTE: must use the resampled kinematics directory here, not the raw
    # eye_output_data/eye_kinematics one - the video being displayed
    # (recording_folder.left/right_eye_display_video) is the resampled/clipped
    # display video, so its frame count and timestamps must come from the same
    # resampled domain, or the video ends up indexed against the full-session
    # (much longer) raw kinematics and desyncs from the rest of the "time" timeline.
    eye_kinematics_directory_path = (
        recording_folder.left_eye_kinematics if eye_name == "left" else recording_folder.right_eye_kinematics
    )
    print(f"Loading eye kinematics from {eye_kinematics_directory_path}...")

    eye_video_path = recording_folder.left_eye_display_video if eye_name == "left" else recording_folder.right_eye_display_video

    kinematics = FerretEyeKinematics.load_from_directory(
        eye_name=f"{eye_name}_eye",
        input_directory=eye_kinematics_directory_path,
    )
    print(f"Loaded {eye_name} eye kinematics: {kinematics.n_frames} frames")

    timestamps = kinematics.eyeball.timestamps - kinematics.eyeball.timestamps[0]

    log_video_file(
        video_path=eye_video_path,
        entity_path=f"{entity_path}video/{eye_name}_eye",
        timestamps=timestamps,
        resize_factor=resize_factor,
        jpeg_quality=jpeg_quality,
    )

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
