import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import set_time_seconds
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.rigid_body_solver.viz.ferret_skull_rerun import RAD_TO_DEG
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

AXIS_COLORS: dict[str, tuple[int, int, int]] = {
    "roll": (255, 107, 107),
    "pitch": (78, 205, 196),
    "yaw": (255, 230, 109),
    "x": (255, 107, 107),
    "y": (78, 255, 96),
    "z": (100, 149, 255),  # Brighter blue for dark backgrounds
}

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
    # Angles
    rr.log(
        f"timeseries/angles/{eye_name}_gaze/roll",
        rr.SeriesLines(widths=1.5, colors=[AXIS_COLORS["roll"]]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}_gaze/roll",
        rr.SeriesPoints(marker_sizes=2.0, colors=[AXIS_COLORS["roll"]]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}_gaze/pitch",
        rr.SeriesLines(widths=1.5, colors=[AXIS_COLORS["pitch"]]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}_gaze/pitch",
        rr.SeriesPoints(marker_sizes=2.0, colors=[AXIS_COLORS["pitch"]]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}_gaze/yaw",
        rr.SeriesLines(widths=1.5, colors=[AXIS_COLORS["yaw"]]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}_gaze/yaw",
        rr.SeriesPoints(marker_sizes=2.0, colors=[AXIS_COLORS["yaw"]]),
        static=True,
    )



def log_timeseries_gaze(
    eye_name: str, roll_deg: float, pitch_deg: float, yaw_deg: float, entity_path: str = "/"
) -> None:
    """Log gaze angles for an eye."""
    rr.log(f"{entity_path}timeseries/angles/{eye_name}/roll", rr.Scalars(roll_deg))
    rr.log(f"{entity_path}timeseries/angles/{eye_name}/pitch", rr.Scalars(pitch_deg))
    rr.log(f"{entity_path}timeseries/angles/{eye_name}/yaw", rr.Scalars(yaw_deg))



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
    euler_rad = kinematics.eyeball.orientations.to_euler_xyz_array()
    euler_deg = euler_rad * RAD_TO_DEG
    timestamps = kinematics.eyeball.timestamps
    timestamps = timestamps - timestamps[0]
    print(f"Loaded left eye kinematics: {kinematics.n_frames} frames")

    for i in range(kinematics.n_frames):
        set_time_seconds("time", timestamps[i])
        roll_deg = euler_deg[i, 0]
        pitch_deg = euler_deg[i, 1]
        yaw_deg = euler_deg[i, 2]

        log_timeseries_gaze(f"{eye_name}_gaze", roll_deg, pitch_deg, yaw_deg)

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