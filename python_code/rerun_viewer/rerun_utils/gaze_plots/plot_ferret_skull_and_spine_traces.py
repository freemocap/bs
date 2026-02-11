import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.rigid_body_solver.viz.ferret_skull_rerun import load_kinematics_from_tidy_csv, send_kinematics_timeseries
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

AXIS_COLORS: dict[str, tuple[int, int, int]] = {
    "roll": (255, 107, 107),
    "pitch": (78, 205, 196),
    "yaw": (255, 230, 109),
    "x": (255, 107, 107),
    "y": (78, 255, 96),
    "z": (100, 149, 255),  # Brighter blue for dark backgrounds
}

def get_ferret_skull_and_spine_traces_views(
    entity_path: str = "/",
    time_window_seconds: float = 5.0
):
    if not entity_path.endswith("/"):
        entity_path += "/"
    scrolling_time_range = rrb.VisibleTimeRange(
        timeline="time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-time_window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=time_window_seconds),
    )

    time_series_panels = [
        # rrb.TimeSeriesView(
        #     name="Position (mm)",
        #     origin=f"{entity_path}position",
        #     plot_legend=rrb.PlotLegend(visible=True),
        #     axis_y=rrb.ScalarAxis(range=(-500.0, 500.0)),
        #     time_ranges=scrolling_time_range,
        # ),
        rrb.TimeSeriesView(
            name="Orientation (deg)",
            origin=f"{entity_path}orientation",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-180.0, 180.0)),
            time_ranges=scrolling_time_range,
        ),
        # rrb.TimeSeriesView(
        #     name="ω Global Frame (deg/s)",
        #     origin=f"{entity_path}omega_global",
        #     plot_legend=rrb.PlotLegend(visible=True),
        #     axis_y=rrb.ScalarAxis(range=(-800.0, 800.0)),
        #     time_ranges=scrolling_time_range,
        # ),
        rrb.TimeSeriesView(
            name="ω Body Frame (deg/s)",
            origin=f"{entity_path}omega_body",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-800.0, 800.0)),
            time_ranges=scrolling_time_range,
        ),
    ]

    return time_series_panels

def log_ferret_skull_and_spine_traces_style(
    entity_path: str = "/",
):
    if not entity_path.endswith("/"):
        entity_path += "/"
    # Position - lines + dots
    for name in ["x", "y", "z"]:
        rr.log(
            f"{entity_path}position/{name}",
            rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name], widths=[1.5]),
            static=True,
        )
        rr.log(
            f"{entity_path}position/{name}",
            rr.SeriesPoints(colors=[AXIS_COLORS[name]], marker_sizes=[2.0]),
            static=True,
        )

    # Orientation - lines + dots
    for name in ["roll", "pitch", "yaw"]:
        rr.log(
            f"{entity_path}orientation/{name}",
            rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name], widths=[1.5]),
            static=True,
        )
        rr.log(
            f"{entity_path}orientation/{name}",
            rr.SeriesPoints(colors=[AXIS_COLORS[name]], marker_sizes=[2.0]),
            static=True,
        )

    # Angular velocity - global - lines + dots
    for name in ["x", "y", "z"]:
        rr.log(
            f"{entity_path}omega_global/{name}",
            rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name], widths=[1.5]),
            static=True,
        )
        rr.log(
            f"{entity_path}omega_global/{name}",
            rr.SeriesPoints(colors=[AXIS_COLORS[name]], marker_sizes=[2.0]),
            static=True,
        )

    # Angular velocity - body/local - lines + dots
    for name in ["roll", "pitch", "yaw"]:
        rr.log(
            f"{entity_path}omega_body/{name}",
            rr.SeriesLines(colors=[AXIS_COLORS[name]], names=[name], widths=[1.5]),
            static=True,
        )
        rr.log(
            f"{entity_path}omega_body/{name}",
            rr.SeriesPoints(colors=[AXIS_COLORS[name]], marker_sizes=[2.0]),
            static=True,
        )


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

    view = get_ferret_skull_and_spine_traces_views(entity_path="/")

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)
    log_ferret_skull_and_spine_traces_style(entity_path="/")

    plot_ferret_skull_and_spine_traces(recording_folder=recording_folder, entity_path="/")