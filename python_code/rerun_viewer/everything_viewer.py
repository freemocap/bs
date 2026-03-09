from pathlib import Path
import numpy as np
import toml
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.groundplane_and_origin import log_groundplane_and_origin
from python_code.rerun_viewer.rerun_utils.load_tidy_dataset import load_solver_outputs, load_tidy_trajectory_dataset
from python_code.rerun_viewer.rerun_utils.plot_3d_data import (
    add_3d_data_context,
    get_3d_data_view,
    plot_3d_data,
)
from python_code.rerun_viewer.rerun_utils.plot_eye_traces import (
    get_eye_trace_views,
    plot_eye_traces,
)
from python_code.rerun_viewer.rerun_utils.plot_eye_video import (
    add_eye_video_context,
    get_eye_video_views,
    plot_eye_video,
    eye_connections,
    eye_landmarks,
)
from python_code.rerun_viewer.rerun_utils.plot_mocap_video import get_mocap_video_view
from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
from python_code.rerun_viewer.rerun_utils.video_data import (
    AlignedEyeVideoData,
    EyeVideoData,
    MocapVideoData,
)
from python_code.utilities.connections_and_landmarks import (
    ferret_head_spine_connections,
    ferret_head_spine_landmarks,
    toy_connections,
    toy_landmarks,
)


def main_rerun_viewer_maker(
    recording_folder: RecordingFolder,
    body_data_3d: np.ndarray,
    include_side_videos: bool = False,
    calibration_path: str | None = None,
    toy_data_3d: np.ndarray | None = None,
):
    """Main function to run the eye tracking visualization."""
    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=recording_folder.topdown_annotated_video_path,
        raw_video_path=recording_folder.topdown_video_path,
        timestamps_npy_path=recording_folder.topdown_timestamps_npy_path,
        data_name="TopDown Mocap",
    )

    if include_side_videos:
        side_0_video = MocapVideoData.create(
            annotated_video_path=recording_folder.side_0_annotated_video_path,
            raw_video_path=recording_folder.side_0_video_path,
            timestamps_npy_path=recording_folder.side_0_timestamps_npy_path,
            data_name="Side 0 Mocap",
            resize_factor=0.5,
        )

        side_1_video = MocapVideoData.create(
            annotated_video_path=recording_folder.side_1_annotated_video_path,
            raw_video_path=recording_folder.side_1_video_path,
            timestamps_npy_path=recording_folder.side_1_timestamps_npy_path,
            data_name="Side 1 Mocap",
            resize_factor=0.5,
        )

        side_2_video = MocapVideoData.create(
            annotated_video_path=recording_folder.side_2_annotated_video_path,
            raw_video_path=recording_folder.side_2_video_path,
            timestamps_npy_path=recording_folder.side_2_timestamps_npy_path,
            data_name="Side 2 Mocap",
            resize_factor=0.5,
        )

        side_3_video = MocapVideoData.create(
            annotated_video_path=recording_folder.side_3_annotated_video_path,
            raw_video_path=recording_folder.side_3_video_path,
            timestamps_npy_path=recording_folder.side_3_timestamps_npy_path,
            data_name="Side 3 Mocap",
            resize_factor=0.5,
        )

        side_videos = [side_0_video, side_1_video, side_2_video, side_3_video]
        recording_start_time = np.min(
            [
                float(topdown_mocap_video.timestamps[0]),
                float(side_0_video.timestamps[0]),
                float(side_1_video.timestamps[0]),
                float(side_2_video.timestamps[0]),
                float(side_3_video.timestamps[0]),
            ]
        )
    else:
        side_videos = []
        recording_start_time = float(topdown_mocap_video.timestamps[0])
    topdown_mocap_video.timestamps -= recording_start_time
    for side_video in side_videos:
        side_video.timestamps -= recording_start_time

    aligned_left_eye = AlignedEyeVideoData.create(
        annotated_video_path=recording_folder.left_eye_aligned_canvas_path,
        raw_video_path=recording_folder.left_eye_aligned_canvas_path,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy_path,
        data_csv_path=recording_folder.left_eye_plot_points_csv_path,
        data_name="Left Eye",
    )

    aligned_right_eye = AlignedEyeVideoData.create(
        annotated_video_path=recording_folder.right_eye_aligned_canvas_path,
        raw_video_path=recording_folder.right_eye_aligned_canvas_path,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy_path,
        data_csv_path=recording_folder.right_eye_plot_points_csv_path,
        data_name="Right Eye",
    )

    left_eye = EyeVideoData.create(
        annotated_video_path=recording_folder.left_eye_annotated_video_path,
        raw_video_path=recording_folder.left_eye_video_path,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy_path,
        data_csv_path=recording_folder.eye_data_csv_path,
        data_name="Left Eye",
    )

    right_eye = EyeVideoData.create(
        annotated_video_path=recording_folder.right_eye_annotated_video_path,
        raw_video_path=recording_folder.right_eye_video_path,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy_path,
        data_csv_path=recording_folder.eye_data_csv_path,
        data_name="Right Eye",
    )

    for eye_video in [left_eye, right_eye, aligned_left_eye, aligned_right_eye]:
        eye_video.timestamps -= recording_start_time

    if calibration_path is not None:
        calibration = toml.load(calibration_path)
    else:
        calibration = None

    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    eye_videos_entity_path = "/eye_videos"
    add_eye_video_context(eye_landmarks, eye_connections, eye_videos_entity_path)
    eye_plots_entity_path = "/eye_plots"
    mocap_video_entity_path = "/topdown_mocap"
    data_3d_entity_path = "/data_3d"
    mocap_entity_path = f"{data_3d_entity_path}/tracked_object"
    toy_entity_path = f"{data_3d_entity_path}/toy_object"

    add_3d_data_context(
        entity_path=mocap_entity_path,
        landmarks=ferret_head_spine_landmarks,
        connections=ferret_head_spine_connections,
    )

    add_3d_data_context(
        entity_path=toy_entity_path,
        landmarks=ferret_head_spine_landmarks,
        connections=ferret_head_spine_connections,
    )

    log_groundplane_and_origin(entity_path=data_3d_entity_path)

    eye_trace_views = get_eye_trace_views(entity_path=eye_plots_entity_path)

    eye_video_views = get_eye_video_views(left_eye, right_eye, eye_videos_entity_path)

    mocap_video_view = get_mocap_video_view(
        mocap_video=topdown_mocap_video, entity_path=mocap_video_entity_path
    )

    data_3d_view = get_3d_data_view(entity_path=data_3d_entity_path)

    eye_video_horizontal = rrb.Horizontal(
        *eye_video_views
    )

    eye_vertical = rrb.Vertical(
        eye_video_horizontal,
        eye_trace_views
    )

    mocap_view = rrb.Vertical(
        mocap_video_view,
        data_3d_view,
    )

    blueprint = rrb.Horizontal(
        eye_vertical,
        mocap_view
    )

    rr.send_blueprint(blueprint)

    plot_eye_video(
        eye_video=aligned_left_eye,
        entity_path=f"{eye_videos_entity_path}/left_eye",
        landmarks=eye_landmarks,
    )
    plot_eye_video(
        eye_video=aligned_right_eye,
        entity_path=f"{eye_videos_entity_path}/right_eye",
        landmarks=eye_landmarks,
        flip_horizontal=True,
    )
    plot_eye_traces(
        right_eye_video_data=right_eye,
        left_eye_video_data=left_eye,
        entity_path=eye_plots_entity_path,
    )
    process_video(
        video_data=topdown_mocap_video,
        entity_path=mocap_video_entity_path,
        include_annotated=True,
    )

    plot_3d_data(
        mocap_video=topdown_mocap_video,
        landmarks=ferret_head_spine_landmarks,
        data_3d=body_data_3d,
        entity_path=mocap_entity_path,
    )

    if toy_data_3d is not None:
        plot_3d_data(
            mocap_video=topdown_mocap_video,
            landmarks=ferret_head_spine_landmarks,
            data_3d=toy_data_3d,
            entity_path=toy_entity_path,
        )


if __name__ == "__main__":
    from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
    from datetime import datetime

    recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(
        recording_name,
        clip_name,
        base_recordings_folder=Path("/home/scholl-lab/ferret_recordings"),
    )
    # recording_folder = RecordingFolder.create_full_recording(recording_name, base_recordings_folder="/home/scholl-lab/ferret_recordings")

    body_data_3d_path = (
        recording_folder.mocap_data_folder
        / "output_data"
        / "dlc"
        / "head_body_rigid_3d_xyz.npy"
    )

    toy_data_3d_path = (
        recording_folder.mocap_data_folder
        / "output_data"
        / "dlc"
        / "toy_body_rigid_3d_xyz.npy"
    )

    body_data_3d = np.load(body_data_3d_path)
    solver_output_path = (
        recording_folder.mocap_data_folder
        / "output_data"
        / "solver_output"
        / "skull_and_spine_trajectories.csv"
    )
    body_data_3d = load_solver_outputs(
        csv_path=solver_output_path,
        landmarks=ferret_head_spine_landmarks,
    )
    toy_data_3d = np.load(toy_data_3d_path)
    # toy_data_3d = None

    main_rerun_viewer_maker(
        recording_folder=recording_folder,
        body_data_3d=body_data_3d,
        include_side_videos = False,
        calibration_path = None,
        toy_data_3d=toy_data_3d,
    )
