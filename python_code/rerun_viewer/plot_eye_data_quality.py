"""
Visualize eye data quality: both eye videos with DLC keypoint annotations
alongside confidence and boolean quality flag timeseries from
eye_model_v3_mean_confidence.csv.

Layout:
    Top:    Left Eye Video  |  Right Eye Video
    Bottom: Left Quality    |  Right Quality
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from python_code.rerun_viewer.rerun_utils.plot_eye_data_quality import (
    get_eye_quality_view,
    log_eye_quality_style,
    plot_eye_quality,
)
from python_code.rerun_viewer.rerun_utils.plot_eye_video import (
    add_eye_video_context,
    get_eye_video_views,
    plot_eye_video,
    eye_landmarks,
    eye_connections,
)
from python_code.rerun_viewer.rerun_utils.video_data import AlignedEyeVideoData
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


def plot_eye_data_quality(recording_folder: RecordingFolder) -> None:
    left_eye = AlignedEyeVideoData.create(
        annotated_video_path=recording_folder.left_eye_stabilized_canvas,
        raw_video_path=recording_folder.left_eye_stabilized_canvas,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy,
        data_csv_path=recording_folder.left_eye_plot_points_csv,
        data_name="Left Eye",
    )

    right_eye = AlignedEyeVideoData.create(
        annotated_video_path=recording_folder.right_eye_stabilized_canvas,
        raw_video_path=recording_folder.right_eye_stabilized_canvas,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy,
        data_csv_path=recording_folder.right_eye_plot_points_csv,
        data_name="Right Eye",
    )

    confidence_df = pd.read_csv(recording_folder.eye_mean_confidence)

    # Timestamps relative to recording start, indexed by frame number
    left_timestamps = left_eye.timestamps - left_eye.timestamps[0]
    right_timestamps = right_eye.timestamps - right_eye.timestamps[0]

    recording_string = f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    rr.init(recording_string, spawn=True)

    eye_videos_entity_path = "/eye_videos"
    quality_entity_path = "/quality_plots"

    add_eye_video_context(eye_landmarks, eye_connections, eye_videos_entity_path)
    log_eye_quality_style("left", quality_entity_path)
    log_eye_quality_style("right", quality_entity_path)

    video_views = get_eye_video_views(left_eye, right_eye, eye_videos_entity_path)
    left_quality_view = get_eye_quality_view("left", quality_entity_path)
    right_quality_view = get_eye_quality_view("right", quality_entity_path)

    blueprint = rrb.Vertical(
        rrb.Horizontal(*video_views),
        rrb.Horizontal(left_quality_view, right_quality_view),
    )
    rr.send_blueprint(blueprint)

    plot_eye_video(eye_video=left_eye, entity_path=f"{eye_videos_entity_path}/left_eye", landmarks=eye_landmarks)
    plot_eye_video(eye_video=right_eye, entity_path=f"{eye_videos_entity_path}/right_eye", landmarks=eye_landmarks, flip_horizontal=True)

    plot_eye_quality(
        eye_name="left",
        camera_name=recording_folder.left_eye_name,
        confidence_df=confidence_df,
        all_timestamps=left_timestamps,
        entity_path=quality_entity_path,
    )
    plot_eye_quality(
        eye_name="right",
        camera_name=recording_folder.right_eye_name,
        confidence_df=confidence_df,
        all_timestamps=right_timestamps,
        entity_path=quality_entity_path,
    )


if __name__ == "__main__":
    recording_folder = RecordingFolder.from_folder_path(
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s")
    )

    plot_eye_data_quality(recording_folder=recording_folder)
