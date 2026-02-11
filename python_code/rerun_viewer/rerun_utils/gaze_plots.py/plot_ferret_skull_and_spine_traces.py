import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.rigid_body_solver.viz.ferret_skull_rerun import load_kinematics_from_tidy_csv, send_kinematics_timeseries
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def get_ferret_skull_and_spine_traces_view(
    entity_path: str = "/",
    time_window_seconds: float = 5.0
):
    scrolling_time_range = rrb.VisibleTimeRange(
        timeline="time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    time_series_panels = [
        rrb.TimeSeriesView(
            name="Position (mm)",
            origin="position",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-500.0, 500.0)),
            time_ranges=scrolling_time_range,
        ),
        rrb.TimeSeriesView(
            name="Orientation (deg)",
            origin="orientation",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-180.0, 180.0)),
            time_ranges=scrolling_time_range,
        ),
        rrb.TimeSeriesView(
            name="ω Global Frame (deg/s)",
            origin="omega_global",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-800.0, 800.0)),
            time_ranges=scrolling_time_range,
        ),
        rrb.TimeSeriesView(
            name="ω Body Frame (deg/s)",
            origin="omega_body",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-800.0, 800.0)),
            time_ranges=scrolling_time_range,
        ),
    ]

    return rrb.Vertical(*time_series_panels)


def plot_ferret_skull_and_spine_traces(
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    recording_folder.check_gaze_postprocessing(enforce_toy=True, enforce_annotated=False)
    print("Loading data from disk...")

    # Load skull kinematics
    reference_geometry = ReferenceGeometry.from_json_file(recording_folder.skull_reference_geometry)
    print(f"  Reference geometry: {len(reference_geometry.keypoints)} keypoints")

    kinematics = load_kinematics_from_tidy_csv(
        csv_path=recording_folder.skull_kinematics_csv,
        reference_geometry=reference_geometry,
        name="skull",
    )
    print(f"  Skull kinematics: {kinematics.n_frames} frames")
    print("  Sending kinematics time series...")
    send_kinematics_timeseries(entity_path=entity_path, kinematics=kinematics)

if __name__ == "__main__":
    from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    rr.init(recording_string, spawn=True)

    view = get_ferret_skull_and_spine_traces_view(entity_path="/")

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)

    plot_ferret_skull_and_spine_traces(recording_folder=recording_folder, entity_path="/")