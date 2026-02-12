"""Video encode images using av and stream them to Rerun with optimized performance."""

from pathlib import Path
import numpy as np
from datetime import datetime
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D
import toml

from python_code.rerun_viewer.rerun_utils.gaze_plots.plot_3d_eye import (
    plot_3d_eye,
    get_3d_eye_view,
)
from python_code.rerun_viewer.rerun_utils.gaze_plots.plot_eye_traces import (
    plot_eye_traces,
    get_eye_trace_views,
    log_eye_trace_style,
)
from python_code.rerun_viewer.rerun_utils.gaze_plots.plot_ferret_skull_and_spine_3d import (
    log_ferret_skull_and_spine_3d_style,
    plot_ferret_skull_and_spine_3d,
    get_ferret_skull_and_spine_3d_view,
)
from python_code.rerun_viewer.rerun_utils.gaze_plots.plot_ferret_skull_and_spine_traces import (
    log_ferret_skull_and_spine_traces_style,
    plot_ferret_skull_and_spine_traces,
    get_ferret_skull_and_spine_traces_views,
)
from python_code.rerun_viewer.rerun_utils.plot_eye_video import add_eye_video_context, get_eye_video_views, plot_eye_video, eye_connections, eye_landmarks
from python_code.rerun_viewer.rerun_utils.video_data import AlignedEyeVideoData
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

# Configuration
GOOD_PUPIL_POINT = "p2"
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)


def create_rerun_recording(
    recording_folder: RecordingFolder,
    eye_name: str = "left",
) -> None:
    """Process both eye videos and visualize them with Rerun."""
    # Initialize Rerun
    recording_string = f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    left_eye = AlignedEyeVideoData.create(
        annotated_video_path=recording_folder.left_eye_stabilized_canvas,
        raw_video_path=recording_folder.left_eye_stabilized_canvas,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy,
        data_csv_path=recording_folder.left_eye_plot_points_csv,
        data_name="Left Eye"
    )
    right_eye = AlignedEyeVideoData.create(
        annotated_video_path=recording_folder.right_eye_stabilized_canvas,
        raw_video_path=recording_folder.right_eye_stabilized_canvas,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy,
        data_csv_path=recording_folder.right_eye_plot_points_csv,
        data_name="Right Eye"
    )

    eye_video_data = left_eye if eye_name=='left' else right_eye
    flip_horizontal = False if eye_name=='left' else True

    rr.init(recording_string, spawn=True)

    eye_3d_view = get_3d_eye_view(eye_name, entity_path="/")
    eye_trace_views = get_eye_trace_views(eye_name, entity_path="/")
    ferret_skull_3d_view = get_ferret_skull_and_spine_3d_view(entity_path="/")
    ferret_skull_traces_views = get_ferret_skull_and_spine_traces_views(entity_path="/")

    eye_videos_entity_path = "/eye_videos"

    eye_video_views = get_eye_video_views(left_eye, right_eye, eye_videos_entity_path)
    eye_video_view = eye_video_views[1] if eye_name == "left" else eye_video_views[0]


    eye_horizontal = rrb.Horizontal(
        *[
            eye_video_view,
            eye_3d_view
        ]
    )
    left_side = rrb.Vertical(
        *[
            eye_horizontal,
            ferret_skull_3d_view
        ]
    )
    time_series = [*ferret_skull_traces_views, *eye_trace_views]
    right_side = rrb.Vertical(*time_series)

    blueprint = rrb.Horizontal(
        *[
            left_side,
            right_side
        ]
    )

    rr.send_blueprint(blueprint)
    add_eye_video_context(eye_landmarks, eye_connections, eye_videos_entity_path)
    log_eye_trace_style(eye_name=eye_name)
    log_ferret_skull_and_spine_3d_style()
    log_ferret_skull_and_spine_traces_style()

    plot_eye_video(eye_video=eye_video_data, entity_path=f"{eye_videos_entity_path}/{eye_name}_eye", landmarks=eye_landmarks, flip_horizontal=flip_horizontal)
    plot_3d_eye(eye_name=eye_name, recording_folder=recording_folder)
    plot_eye_traces(eye_name=eye_name, recording_folder=recording_folder)
    plot_ferret_skull_and_spine_3d(recording_folder=recording_folder) 
    plot_ferret_skull_and_spine_traces(recording_folder=recording_folder)

    print(
        f"Processing complete! Rerun recording '{recording_folder.recording_name}' is ready."
    )


if __name__ == "__main__":
    recording_folder = RecordingFolder.from_folder_path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    eye_to_plot = "right"
    create_rerun_recording(
        recording_folder=recording_folder,
        eye_name=eye_to_plot
    )