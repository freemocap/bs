import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import COLOR_LEFT_EYE_PRIMARY, COLOR_LEFT_EYE_SECONDARY, COLOR_RIGHT_EYE_PRIMARY, COLOR_RIGHT_EYE_SECONDARY, get_eye_radius_from_kinematics, log_static_world_frame, log_timeseries_accelerations, log_timeseries_angles, log_timeseries_velocities, set_time_seconds
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.rigid_body_solver.viz.ferret_skull_rerun import load_kinematics_from_tidy_csv
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def get_naive_gaze_trace_views(
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
    #     origin=f"{entity_path}timeseries/pupil_position/{eye_name}_naive_gaze",
    #     plot_legend=rrb.PlotLegend(visible=True),
    #     time_ranges=scrolling_time_range,
    #     axis_y=rrb.ScalarAxis(range=(-40.0, 40.0)),
    # )

    angle_view = rrb.TimeSeriesView(
        name=f"{eye_name.capitalize()} Angles (deg) [±180]",
        origin=f"{entity_path}timeseries/angles/{eye_name}_naive_gaze",
        plot_legend=rrb.PlotLegend(visible=True),
        time_ranges=scrolling_time_range,
        axis_y=rrb.ScalarAxis(range=(-180.0, 180.0)),
    )

    # velocity_view = rrb.TimeSeriesView(
    #     name=f"{eye_name.capitalize()} Velocity (deg/s) [±350]",
    #     origin=f"{entity_path}timeseries/velocity/{eye_name}_naive_gaze",
    #     plot_legend=rrb.PlotLegend(visible=True),
    #     time_ranges=scrolling_time_range,
    #     axis_y=rrb.ScalarAxis(range=(-350.0, 350.0)),
    # )

    # acceleration_view = rrb.TimeSeriesView(
    #     name=f"{eye_name.capitalize()} Accel (deg/s²) [±5000]",
    #     origin=f"{entity_path}timeseries/acceleration/{eye_name}_naive_gaze",
    #     plot_legend=rrb.PlotLegend(visible=True),
    #     time_ranges=scrolling_time_range,
    #     axis_y=rrb.ScalarAxis(range=(-5000.0, 5000.0)),
    # )

    # timeseries_views = [angle_view, velocity_view, acceleration_view]

    # return timeseries_views
    return [angle_view]

def log_naive_gaze_trace_style(
    eye_name: str,
    entity_path: str = "/",
):
    if not eye_name.endswith("_eye"):
        eye_name+="_eye"
    primary_color = COLOR_LEFT_EYE_PRIMARY if eye_name == "left_eye" else COLOR_RIGHT_EYE_PRIMARY
    secondary_color = COLOR_LEFT_EYE_SECONDARY if eye_name == "left_eye" else COLOR_RIGHT_EYE_SECONDARY

    # Pupil position
    rr.log(
        f"timeseries/pupil_position/{eye_name}_naive_gaze/horizontal",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/pupil_position/{eye_name}_naive_gaze/horizontal",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/pupil_position/{eye_name}_naive_gaze/vertical",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/pupil_position/{eye_name}_naive_gaze/vertical",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
        static=True,
    )

    # Angles
    rr.log(
        f"timeseries/angles/{eye_name}_naive_gaze/adduction",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}_naive_gaze/adduction",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}_naive_gaze/elevation",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/angles/{eye_name}_naive_gaze/elevation",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
        static=True,
    )

    # Velocities
    rr.log(
        f"timeseries/velocity/{eye_name}_naive_gaze/adduction",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/velocity/{eye_name}_naive_gaze/adduction",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/velocity/{eye_name}_naive_gaze/elevation",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/velocity/{eye_name}_naive_gaze/elevation",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
        static=True,
    )

    # Accelerations
    rr.log(
        f"timeseries/acceleration/{eye_name}_naive_gaze/adduction",
        rr.SeriesLines(widths=1.5, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/acceleration/{eye_name}_naive_gaze/adduction",
        rr.SeriesPoints(marker_sizes=2.0, colors=[primary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/acceleration/{eye_name}_naive_gaze/elevation",
        rr.SeriesLines(widths=1.5, colors=[secondary_color]),
        static=True,
    )
    rr.log(
        f"timeseries/acceleration/{eye_name}_naive_gaze/elevation",
        rr.SeriesPoints(marker_sizes=2.0, colors=[secondary_color]),
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
        input_directory=recording_folder.left_eye_kinematics if eye_name=="left" else recording_folder.right_eye_kinematics
    )
    timestamps = eye_kinematics.eyeball.timestamps
    timestamps = timestamps - timestamps[0]
    print(f"Loaded left eye kinematics: {eye_kinematics.n_frames} frames")

    reference_geometry = ReferenceGeometry.from_json_file(recording_folder.skull_reference_geometry)
    print(f"  Reference geometry: {len(reference_geometry.keypoints)} keypoints")

    skull_kinematics = load_kinematics_from_tidy_csv(
        csv_path=recording_folder.skull_kinematics_csv,
        reference_geometry=reference_geometry,
        name="skull",
    )

    gaze_adduction_deg = np.degrees(eye_kinematics.adduction_angle.values) + np.degrees(skull_kinematics.yaw.values)
    gaze_elevation_deg = np.degrees(eye_kinematics.elevation_angle.values) + np.degrees(skull_kinematics.pitch.values)

    print(f"eye adduction shape: {eye_kinematics.adduction_angle.values.shape}")
    print(f"head yaw shape: {skull_kinematics.yaw.values.shape}")

    for i in range(eye_kinematics.n_frames):
        set_time_seconds("time", timestamps[i])
        log_timeseries_angles(f"{eye_name}_naive_gaze", gaze_adduction_deg[i], gaze_elevation_deg[i])
        # log_timeseries_velocities(f"{eye_name}_naive_gaze", adduction_vel, elevation_vel)
        # log_timeseries_accelerations(f"{eye_name}_naive_gaze", adduction_acc, elevation_acc)

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

    views = get_naive_gaze_trace_views(eye_name, entity_path="/")

    blueprint = rrb.Horizontal(*views)

    rr.send_blueprint(blueprint)
    log_naive_gaze_trace_style(eye_name, entity_path="/")

    plot_naive_gaze_traces(eye_name=eye_name, recording_folder=recording_folder)