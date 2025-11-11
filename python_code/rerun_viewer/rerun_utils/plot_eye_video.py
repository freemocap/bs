import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.video_data import EyeVideoData

def plot_eye_video(eye_video: EyeVideoData, landmarks: dict[str, int], connections: list[tuple[int, int]], entity_path: str = ""):
    time_column = rr.TimeColumn("time", duration=eye_video.timestamps)
    class_ids = np.ones(shape=eye_video.frame_count)
    keypoints = np.array(list(landmarks.values()))
    keypoint_ids = np.repeat(keypoints[np.newaxis, :], eye_video.frame_count, axis=0)
    radii = np.full(shape=keypoint_ids.shape, fill_value=6.0)
    print(f" shape of radii: {radii.shape}")
    eye_data_array = eye_video.data_array()
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

    process_video(video_data=eye_video, entity_path=entity_path, include_annotated=False)


if __name__ == "__main__":
    from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
    from datetime import datetime

    recording_name = "session_2025-07-01_ferret_757_EyeCameras_P33EO5"
    clip_name = "1m_20s-2m_20s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name)

    left_eye = EyeVideoData.create(
        annotated_video_path=recording_folder.left_eye_annotated_video_path,
        raw_video_path=recording_folder.left_eye_video_path,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy_path,
        data_csv_path=recording_folder.eye_data_csv_path,
        data_name="Left Eye"
    )

    # print(left_eye.get_point_names())
    # left_eye.get_dataframe()

    right_eye = EyeVideoData.create(
        annotated_video_path=recording_folder.right_eye_annotated_video_path,
        raw_video_path=recording_folder.right_eye_video_path,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy_path,
        data_csv_path=recording_folder.eye_data_csv_path,
        data_name="Right Eye"
    )

    # print(right_eye.get_point_names())
    # right_eye.get_dataframe()

    landmarks = {
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

    connections = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0,7)
    )

    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    eye_videos_entity_path = "/eye_videos"

    left_eye_view = rrb.Vertical(
        rrb.Spatial2DView(
            name="Left Eye Video (Annotated)",
            origin=f"{eye_videos_entity_path}/left_eye",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, left_eye.resized_width),
                    y_range=(0, left_eye.resized_height),
                )
            ),
        ),
        rrb.Spatial2DView(
            name="Left Eye Video (Raw)",
            origin=f"{eye_videos_entity_path}/left_eye",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, right_eye.resized_width),
                    y_range=(0, right_eye.resized_height),
                )
            ),
            visible=False,
        ),
    )

    right_eye_view = rrb.Vertical(
        rrb.Spatial2DView(
            name="Right Eye Video (Annotated)",
            origin=f"{eye_videos_entity_path}/right_eye",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, right_eye.resized_width),
                    y_range=(0, right_eye.resized_height),
                )
            ),
        ),
        rrb.Spatial2DView(
            name="Right Eye Video (Raw)",
            origin=f"{eye_videos_entity_path}/right_eye",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, right_eye.resized_width),
                    y_range=(0, right_eye.resized_height),
                )
            ),
            visible=False,
        ),
    )

    views = [right_eye_view, left_eye_view]

    blueprint = rrb.Horizontal(*views)

    rr.send_blueprint(blueprint)

    rr.log(
        eye_videos_entity_path,
        rr.AnnotationContext(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=1, label="eye_points"),
                keypoint_annotations=[
                    rr.AnnotationInfo(id=value, label=key)
                    for key, value in landmarks.items()
                ],
                keypoint_connections=connections,
            ),
        ),
        static=True,
    )

    plot_eye_video(eye_video=left_eye, entity_path=f"{eye_videos_entity_path}/left_eye", landmarks=landmarks, connections=connections)
    plot_eye_video(eye_video=right_eye, entity_path=f"{eye_videos_entity_path}/right_eye", landmarks=landmarks, connections=connections)
