from pathlib import Path
from typing import Literal
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D
from pydantic import BaseModel

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.viz.rerun_viewer.utils.process_videos import process_video
from python_code.viz.rerun_viewer.utils.video_data import AlignedEyeVideoData, EyeVideoData

eye_landmarks = {
    "p1": 0,
    "p2": 1,
    "p3": 2,
    "p4": 3,
    "p5": 4,
    "p6": 5,
    "p7": 6,
    "p8": 7,
    "tear_duct": 8,
    "outer_eye": 9
}

eye_connections = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (0,7)
)


class EyeVideoLandmarksContext(BaseModel):
    eye_name: Literal["left", "right"]
    entity_path: str = "/eye_videos"
    flip_horizontal: bool = False
    flip_vertical: bool = False


def required_data_available(recording_folder: RecordingFolder, context: "EyeVideoLandmarksContext") -> bool:
    stabilized_canvas = (
        recording_folder.left_eye_stabilized_canvas
        if context.eye_name == "left"
        else recording_folder.right_eye_stabilized_canvas
    )
    plot_points_csv = (
        recording_folder.left_eye_plot_points_csv
        if context.eye_name == "left"
        else recording_folder.right_eye_plot_points_csv
    )
    timestamps_npy = (
        recording_folder.left_eye_timestamps_npy
        if context.eye_name == "left"
        else recording_folder.right_eye_timestamps_npy
    )
    return stabilized_canvas is not None and plot_points_csv is not None and timestamps_npy is not None


def get_eye_video_landmarks_view(eye_name: str, entity_path: str = "/eye_videos"):
    return rrb.Spatial2DView(
        name=f"{eye_name.capitalize()} Eye Video",
        origin=f"{entity_path}/{eye_name}_eye",
    )


def add_eye_video_context(eye_landmarks: dict[str, int], eye_connections: tuple, entity_path: str = ""):
    rr.log(
        entity_path,
        rr.AnnotationContext(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=1, label="eye_points"),
                keypoint_annotations=[
                    rr.AnnotationInfo(id=value, label=key)
                    for key, value in eye_landmarks.items()
                ],
                keypoint_connections=eye_connections,
            ),
        ),
        static=True,
    )


def log_eye_video_landmarks_style(eye_name: str, entity_path: str = "/eye_videos"):
    add_eye_video_context(eye_landmarks, eye_connections, f"{entity_path}/{eye_name}_eye")


def plot_eye_video_landmarks_from_data(eye_video: AlignedEyeVideoData, landmarks: dict[str, int], entity_path: str = "", flip_horizontal: bool = False, flip_vertical: bool = False):
    eye_data_array = eye_video.data_array()
    timestamps = eye_video.timestamps - eye_video.timestamps[0]
    time_column = rr.TimeColumn("time", duration=timestamps)
    class_ids = np.ones(shape=eye_video.frame_count)
    keypoints = np.array(list(landmarks.values()))
    keypoint_ids = np.repeat(keypoints[np.newaxis, :], eye_video.frame_count, axis=0)
    show_labels = np.full(shape=eye_data_array.shape, fill_value=False, dtype=bool)
    radii = np.full(shape=keypoint_ids.shape, fill_value=6.0)
    if flip_horizontal:
        eye_data_array = eye_video.flip_data_horizontal(array=eye_data_array, image_width=eye_video.width)
    if flip_vertical:
        eye_data_array = eye_video.flip_data_vertical(array=eye_data_array, image_height=eye_video.height)
    rr.send_columns(
        entity_path=f"{entity_path}/points",
        indexes=[time_column],
        columns=[
            *rr.Points2D.columns(positions=eye_data_array),
            *rr.Points2D.columns(
                radii=radii,
                class_ids=class_ids,
                keypoint_ids=keypoint_ids,
                show_labels=show_labels
            ),
        ],
    )

    process_video(video_data=eye_video, entity_path=entity_path, include_annotated=False, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)


def plot_eye_video_landmarks(recording_folder: RecordingFolder, eye_name: str, entity_path: str = "/eye_videos", flip_horizontal: bool = False, flip_vertical: bool = False):
    context = EyeVideoLandmarksContext(eye_name=eye_name, entity_path=entity_path, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)
    if not required_data_available(recording_folder, context):
        print(f"Skipping eye video landmarks for {context.eye_name}: required data not found")
        return

    stabilized_canvas = (
        recording_folder.left_eye_stabilized_canvas
        if context.eye_name == "left"
        else recording_folder.right_eye_stabilized_canvas
    )
    timestamps_npy = (
        recording_folder.left_eye_timestamps_npy
        if context.eye_name == "left"
        else recording_folder.right_eye_timestamps_npy
    )
    plot_points_csv = (
        recording_folder.left_eye_plot_points_csv
        if context.eye_name == "left"
        else recording_folder.right_eye_plot_points_csv
    )

    eye_video = AlignedEyeVideoData.create(
        annotated_video_path=stabilized_canvas,
        raw_video_path=stabilized_canvas,
        timestamps_npy_path=timestamps_npy,
        data_csv_path=plot_points_csv,
        data_name=f"{context.eye_name.capitalize()} Eye",
    )

    plot_eye_video_landmarks_from_data(
        eye_video=eye_video,
        entity_path=f"{context.entity_path}/{context.eye_name}_eye",
        landmarks=eye_landmarks,
        flip_horizontal=context.flip_horizontal,
        flip_vertical=context.flip_vertical,
    )


if __name__ == "__main__":
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-12_ferret_420_E03/full_recording"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    entity_path = "/eye_videos"

    views = [
        get_eye_video_landmarks_view("right", entity_path),
        get_eye_video_landmarks_view("left", entity_path),
    ]
    rr.send_blueprint(rrb.Horizontal(*views))

    log_eye_video_landmarks_style(eye_name="left", entity_path=entity_path)
    log_eye_video_landmarks_style(eye_name="right", entity_path=entity_path)

    plot_eye_video_landmarks(recording_folder=recording_folder, eye_name="left", entity_path=entity_path)
    plot_eye_video_landmarks(recording_folder=recording_folder, eye_name="right", entity_path=entity_path, flip_horizontal=True)
