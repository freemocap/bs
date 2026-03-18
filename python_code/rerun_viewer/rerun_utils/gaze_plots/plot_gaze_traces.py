import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.calculate_gaze.calculate_ferret_gaze import batch_rotate_vector_by_quaternion
from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import COLOR_LEFT_EYE_PRIMARY, COLOR_LEFT_EYE_SECONDARY, COLOR_RIGHT_EYE_PRIMARY, COLOR_RIGHT_EYE_SECONDARY, set_time_seconds
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.rigid_body_solver.viz.ferret_skull_rerun import RAD_TO_DEG
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def get_gaze_trace_views(
    eye_name: str,
    entity_path: str = "/",
    time_window_seconds: float = 5.0
):
    if not entity_path.endswith("/"):
        entity_path += "/"

    scrolling_time_range = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    angle_view = rrb.TimeSeriesView(
        name=f"{eye_name.capitalize()} Gaze (deg) [±180]",
        origin=f"{entity_path}timeseries/angles/{eye_name}_gaze",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-180.0, 180.0)),
    )
    return [angle_view]

def log_gaze_trace_style(
    eye_name: str,
    entity_path: str = "/",
):
    primary_color = COLOR_LEFT_EYE_PRIMARY if eye_name == "left" else COLOR_RIGHT_EYE_PRIMARY
    secondary_color = COLOR_LEFT_EYE_SECONDARY if eye_name == "left" else COLOR_RIGHT_EYE_SECONDARY

    # Angles
    rr.log(
        f"{entity_path}timeseries/angles/{eye_name}_gaze/horizontal",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"{entity_path}timeseries/angles/{eye_name}_gaze/horizontal",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"{entity_path}timeseries/angles/{eye_name}_gaze/vertical",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"{entity_path}timeseries/angles/{eye_name}_gaze/vertical",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
        static=True,
    )



def log_timeseries_gaze(
    eye_name: str, horizontal_deg: float, vertical_deg, entity_path: str = "/"
) -> None:
    """Log gaze angles for an eye."""
    rr.log(f"{entity_path}timeseries/angles/{eye_name}/horizontal", rr.Scalars(horizontal_deg))
    rr.log(f"{entity_path}timeseries/angles/{eye_name}/vertical", rr.Scalars(vertical_deg))



def compute_gaze_angles(gaze_kinematics: RigidBodyKinematics, eye_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract horizontal and elevation gaze angles from output kinematics.

    Uses gaze direction = batch_rotate_vector_by_quaternion(q, [0,0,1]).

    Horizontal: atan2(x, -y) for right eye (rest gaze = -Y), atan2(x, y) for left eye (rest gaze = +Y).
    Both conventions yield 0° at rest and positive values when looking rightward (toward nose).
    Elevation:  atan2(z, sqrt(x²+y²))
    """
    gaze_quats = gaze_kinematics.quaternions_wxyz
    gaze_dir = batch_rotate_vector_by_quaternion(gaze_quats, np.array([0.0, 0.0, 1.0]))
    gaze_x, gaze_y, gaze_z = gaze_dir[:, 0], gaze_dir[:, 1], gaze_dir[:, 2]
    # TODO: check this decision about left/right, and document whatever the decision is 
    if eye_name == "left":
        horizontal = np.degrees(np.arctan2(gaze_x, gaze_y))
    else:
        horizontal = np.degrees(np.arctan2(gaze_x, -gaze_y))
    elevation = np.degrees(np.arctan2(gaze_z, np.sqrt(gaze_x**2 + gaze_y**2)))
    return horizontal, elevation



def plot_gaze_traces(
    eye_name: str,
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")

    kinematics = FerretEyeKinematics.load_from_directory(
        eye_name=f"{eye_name}_gaze", 
        input_directory=recording_folder.gaze_kinematics
    )
    horizontal, vertical = compute_gaze_angles(kinematics.eyeball, eye_name)
    timestamps = kinematics.eyeball.timestamps
    timestamps = timestamps - timestamps[0]
    print(f"Loaded left eye kinematics: {kinematics.n_frames} frames")

    for i in range(kinematics.n_frames):
        set_time_seconds("time", timestamps[i])
        log_timeseries_gaze(f"{eye_name}_gaze", horizontal_deg=horizontal[i], vertical_deg=vertical[i])

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

    views = get_gaze_trace_views(eye_name, entity_path="/")

    blueprint = rrb.Horizontal(*views)

    rr.send_blueprint(blueprint)
    log_gaze_trace_style(eye_name, entity_path="/")

    plot_gaze_traces(eye_name=eye_name, recording_folder=recording_folder)