import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import set_time_seconds
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.rigid_body_solver.viz.ferret_skull_rerun import RAD_TO_DEG, load_kinematics_from_tidy_csv
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

AXIS_COLORS: dict[str, tuple[int, int, int]] = {
    "roll":  (255, 107, 107),
    "pitch": (78, 205, 196),
    "yaw":   (255, 230, 109),
}


def get_naive_gaze_trace_views(
    eye_name: str,
    entity_path: str = "/",
    time_window_seconds: float = 5.0,
):
    if not entity_path.endswith("/"):
        entity_path += "/"

    scrolling_time_range = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    angle_view = rrb.TimeSeriesView(
        name=f"{eye_name.capitalize()} Naive Gaze (deg) [±180]",
        origin=f"{entity_path}timeseries/angles/{eye_name}_naive_gaze",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-180.0, 180.0)),
    )
    return [angle_view]


def log_naive_gaze_trace_style(
    eye_name: str,
    entity_path: str = "/",
):
    for component in ["roll", "pitch", "yaw"]:
        rr.log(
            f"timeseries/angles/{eye_name}_naive_gaze/{component}",
            rr.SeriesLines(widths=1.5, colors=[AXIS_COLORS[component]]),
            static=True,
        )
        rr.log(
            f"timeseries/angles/{eye_name}_naive_gaze/{component}",
            rr.SeriesPoints(marker_sizes=2.0, colors=[AXIS_COLORS[component]]),
            static=True,
        )


def plot_naive_gaze_traces(
    eye_name: str,
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")

    eye_kinematics = FerretEyeKinematics.load_from_directory(
        eye_name=f"{eye_name}_eye",
        input_directory=(
            recording_folder.left_eye_kinematics
            if eye_name == "left"
            else recording_folder.right_eye_kinematics
        ),
    )
    timestamps = eye_kinematics.eyeball.timestamps
    timestamps = timestamps - timestamps[0]
    print(f"Loaded {eye_name} eye kinematics: {eye_kinematics.n_frames} frames")

    reference_geometry = ReferenceGeometry.from_json_file(recording_folder.skull_reference_geometry)
    skull_kinematics = load_kinematics_from_tidy_csv(
        csv_path=recording_folder.skull_kinematics_csv,
        reference_geometry=reference_geometry,
        name="skull",
    )

    eye_euler_deg = eye_kinematics.eyeball.orientations.to_euler_xyz_array() * RAD_TO_DEG
    skull_euler_deg = skull_kinematics.orientations.to_euler_xyz_array() * RAD_TO_DEG

    naive_roll_deg  = np.degrees(eye_kinematics.elevation_angle.values) + skull_euler_deg[:, 0]
    naive_pitch_deg = skull_euler_deg[:, 1]
    naive_yaw_deg   = np.degrees(eye_kinematics.adduction_angle.values) + skull_euler_deg[:, 2]

    for i in range(eye_kinematics.n_frames):
        set_time_seconds("time", timestamps[i])
        rr.log(f"timeseries/angles/{eye_name}_naive_gaze/roll",  rr.Scalars(naive_roll_deg[i]))
        rr.log(f"timeseries/angles/{eye_name}_naive_gaze/pitch", rr.Scalars(naive_pitch_deg[i]))
        rr.log(f"timeseries/angles/{eye_name}_naive_gaze/yaw",   rr.Scalars(naive_yaw_deg[i]))


if __name__ == "__main__":
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    eye_name = "right"

    recording_folder = RecordingFolder.from_folder_path(folder_path)

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    rr.init(recording_string, spawn=True)
    rr.send_blueprint(rrb.Horizontal(*get_naive_gaze_trace_views(eye_name, entity_path="/")))
    log_naive_gaze_trace_style(eye_name, entity_path="/")
    plot_naive_gaze_traces(eye_name=eye_name, recording_folder=recording_folder)
