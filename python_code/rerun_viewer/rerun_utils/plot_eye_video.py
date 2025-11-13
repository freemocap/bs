from pathlib import Path
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.video_data import AlignedEyeVideoData, EyeVideoData

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

def plot_eye_video(eye_video: AlignedEyeVideoData, landmarks: dict[str, int], entity_path: str = "", flip_horizontal: bool = False, flip_vertical: bool = False):
    time_column = rr.TimeColumn("time", duration=eye_video.timestamps)
    class_ids = np.ones(shape=eye_video.frame_count)
    keypoints = np.array(list(landmarks.values()))
    keypoint_ids = np.repeat(keypoints[np.newaxis, :], eye_video.frame_count, axis=0)
    radii = np.full(shape=keypoint_ids.shape, fill_value=6.0)
    print(f" shape of radii: {radii.shape}")
    eye_data_array = eye_video.data_array()
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
            ),
        ],
    )

    process_video(video_data=eye_video, entity_path=entity_path, include_annotated=False, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)


def get_eye_video_views(left_eye: EyeVideoData | AlignedEyeVideoData, right_eye: EyeVideoData | AlignedEyeVideoData, eye_videos_entity_path: str):
    left_eye_view = rrb.Spatial2DView(
            name="Left Eye Video",
            origin=f"{eye_videos_entity_path}/left_eye",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, left_eye.resized_width),
                    y_range=(0, left_eye.resized_height),
                )
            ),
        )

    right_eye_view = rrb.Spatial2DView(
        name="Right Eye Video",
        origin=f"{eye_videos_entity_path}/right_eye",
        visual_bounds=VisualBounds2D.from_fields(
            range=Range2D(
                x_range=(0, right_eye.resized_width),
                y_range=(0, right_eye.resized_height),
            )
        ),
    )

    views = [right_eye_view, left_eye_view]
    return views

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

if __name__ == "__main__":
    from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
    from datetime import datetime

    recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name, base_recordings_folder=Path("/home/scholl-lab/ferret_recordings"))
    # recording_folder = RecordingFolder.create_full_recording(recording_name, base_recordings_folder="/home/scholl-lab/ferret_recordings")

    left_eye = AlignedEyeVideoData.create(
        annotated_video_path=recording_folder.left_eye_aligned_canvas_path,
        raw_video_path=recording_folder.left_eye_aligned_canvas_path,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy_path,
        data_csv_path=recording_folder.left_eye_plot_points_csv_path,
        data_name="Left Eye"
    )

    # print(left_eye.get_point_names())
    # left_eye.get_dataframe()

    right_eye = AlignedEyeVideoData.create(
        annotated_video_path=recording_folder.right_eye_aligned_canvas_path,
        raw_video_path=recording_folder.right_eye_aligned_canvas_path,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy_path,
        data_csv_path=recording_folder.right_eye_plot_points_csv_path,
        data_name="Right Eye"
    )

    # print(right_eye.get_point_names())
    # right_eye.get_dataframe()



    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    eye_videos_entity_path = "/eye_videos"

    views = get_eye_views(left_eye, right_eye, eye_videos_entity_path)

    blueprint = rrb.Horizontal(*views)

    rr.send_blueprint(blueprint)

    add_eye_video_context(eye_landmarks, eye_connections, eye_videos_entity_path)

    plot_eye_video(eye_video=left_eye, entity_path=f"{eye_videos_entity_path}/left_eye", landmarks=eye_landmarks)
    plot_eye_video(eye_video=right_eye, entity_path=f"{eye_videos_entity_path}/right_eye", landmarks=eye_landmarks, flip_horizontal=True)
