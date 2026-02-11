import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import COLOR_LEFT_EYE_PRIMARY, COLOR_LEFT_EYE_SECONDARY, COLOR_RIGHT_EYE_PRIMARY, COLOR_RIGHT_EYE_SECONDARY, get_eye_radius_from_kinematics, log_static_world_frame, log_timeseries_accelerations, log_timeseries_angles, log_timeseries_velocities, set_time_seconds
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def get_eye_trace_views(
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
    # Left eye timeseries (order: pixels, angles, velocity, acceleration)
    # All share the same scrolling_time_range so they stay synchronized
    # Y range included in name as fallback when Rerun hides tick labels
    # pupil_view = rrb.TimeSeriesView(
    #     name=f"{eye_name.capitalize()} Pupil (px) [±40]",
    #     origin=f"{entity_path}timeseries/pupil_position/{eye_name}_eye",
    #     plot_legend=rrb.PlotLegend(visible=True),
    #     time_ranges=scrolling_time_range,
    #     axis_y=rrb.ScalarAxis(range=(-40.0, 40.0)),
    # )

    # angle_view = rrb.TimeSeriesView(
    #     name=f"{eye_name.capitalize()} Angles (deg) [±25]",
    #     origin=f"{entity_path}timeseries/angles/{eye_name}_eye",
    #     plot_legend=rrb.PlotLegend(visible=True),
    #     time_ranges=scrolling_time_range,
    #     axis_y=rrb.ScalarAxis(range=(-25.0, 25.0)),
    # )

    velocity_view = rrb.TimeSeriesView(
        name=f"{eye_name.capitalize()} Velocity (deg/s) [±350]",
        origin=f"{entity_path}timeseries/velocity/{eye_name}_eye",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-350.0, 350.0)),
    )

    # acceleration_view = rrb.TimeSeriesView(
    #     name=f"{eye_name.capitalize()} Accel (deg/s²) [±5000]",
    #     origin=f"{entity_path}timeseries/acceleration/{eye_name}_eye",
    #     plot_legend=rrb.PlotLegend(visible=True),
    #     time_ranges=scrolling_time_range,
    #     axis_y=rrb.ScalarAxis(range=(-5000.0, 5000.0)),
    # )

    # timeseries_views = [angle_view, velocity_view, acceleration_view]

    # return timeseries_views
    return [velocity_view]

def log_eye_trace_style(
    eye_name: str,
    entity_path: str = "/",
):
    if not eye_name.endswith("_eye"):
        eye_name+="_eye"
    primary_color = COLOR_LEFT_EYE_PRIMARY if eye_name == "left_eye" else COLOR_RIGHT_EYE_PRIMARY
    secondary_color = COLOR_LEFT_EYE_SECONDARY if eye_name == "left_eye" else COLOR_RIGHT_EYE_SECONDARY

    # Pupil position
    rr.log(
        f"timeseries/pupil_position/{eye_name}/horizontal",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/pupil_position/{eye_name}/horizontal",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/pupil_position/{eye_name}/vertical",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/pupil_position/{eye_name}/vertical",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
        static=True,
    )

    # Angles
    rr.log(
        f"timeseries/angles/{eye_name}/adduction",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}/adduction",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}/elevation",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}/elevation",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
        static=True,
    )

    # Velocities
    rr.log(
        f"timeseries/velocity/{eye_name}/adduction",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/velocity/{eye_name}/adduction",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/velocity/{eye_name}/elevation",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/velocity/{eye_name}/elevation",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
        static=True,
    )

    # Accelerations
    rr.log(
        f"timeseries/acceleration/{eye_name}/adduction",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/acceleration/{eye_name}/adduction",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/acceleration/{eye_name}/elevation",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/acceleration/{eye_name}/elevation",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
        static=True,
    )



def plot_eye_traces(
    eye_name: str,
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")

    kinematics = FerretEyeKinematics.load_from_directory(eye_name=f"{eye_name}_eye", input_directory=recording_folder.eye_output_data / "eye_kinematics")
    timestamps = kinematics.timestamps
    print(f"Loaded left eye kinematics: {kinematics.n_frames} frames")

    for i in range(kinematics.n_frames):
        set_time_seconds("time", timestamps[i])
        adduction_deg = np.degrees(kinematics.adduction_angle.values[i])
        elevation_deg = np.degrees(kinematics.elevation_angle.values[i])
        adduction_vel = np.degrees(kinematics.adduction_velocity.values[i])
        elevation_vel = np.degrees(kinematics.elevation_velocity.values[i])
        adduction_acc = np.degrees(kinematics.adduction_acceleration.values[i])
        elevation_acc = np.degrees(kinematics.elevation_acceleration.values[i])

        # log_timeseries_angles(f"{eye_name}_eye", adduction_deg, elevation_deg)
        log_timeseries_velocities(f"{eye_name}_eye", adduction_vel, elevation_vel)
        # log_timeseries_accelerations(f"{eye_name}_eye", adduction_acc, elevation_acc)

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

    view = get_eye_trace_views(eye_name, entity_path="/")

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)
    log_eye_trace_style(eye_name, entity_path="/")

    plot_eye_traces(eye_name=eye_name, recording_folder=recording_folder)