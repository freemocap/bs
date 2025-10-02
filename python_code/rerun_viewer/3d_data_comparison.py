"""Video encode images using av and stream them to Rerun with optimized performance."""

from pathlib import Path
import numpy as np
from datetime import datetime
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.freemocap_recording_folder import (
    FreemocapRecordingFolder,
)
from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
from python_code.rerun_viewer.rerun_utils.video_data import MocapVideoData

# Configuration
GOOD_PUPIL_POINT = "p2"
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)


def create_rerun_recording(
    recording_name: str,
    data_1_name: str,
    data_3d_1: np.ndarray,
    data_2_name: str,
    data_3d_2: np.ndarray,
    topdown_mocap_video: MocapVideoData,
    side_videos: list[MocapVideoData],
    landmarks: dict[str, int],
    connections: tuple[tuple[int, int], ...],
    include_side_videos: bool = False,
) -> None:
    """Process both eye videos and visualize them with Rerun."""
    # Initialize Rerun
    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    rr.log(
        "/",
        rr.AnnotationContext(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=1, label="Tracked_object"),
                keypoint_annotations=[
                    rr.AnnotationInfo(id=value, label=key)
                    for key, value in landmarks.items()
                ],
                keypoint_connections=connections,
            ),
        ),
        static=True,
    )

    topdown_view = rrb.Vertical(
        rrb.Spatial2DView(
            name="TopDown Mocap Video(Annotated)",
            origin=f"/mocap_video/top_down/annotated",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, topdown_mocap_video.resized_width),
                    y_range=(0, topdown_mocap_video.resized_height),
                )
            ),
        ),
        rrb.Spatial2DView(
            name="TopDown Mocap Video(Raw)",
            origin=f"/mocap_video/top_down/raw",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, topdown_mocap_video.resized_width),
                    y_range=(0, topdown_mocap_video.resized_height),
                )
            ),
            visible=False,
        ),
    )
    if include_side_videos:
        side_view_vertical_0 = rrb.Vertical(
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="TopDown Mocap Video(Annotated)",
                    origin=f"/mocap_video/side_0/annotated",
                    visual_bounds=VisualBounds2D.from_fields(
                        range=Range2D(
                            x_range=(0, side_videos[0].resized_width),
                            y_range=(0, side_videos[0].resized_height),
                        )
                    ),
                ),
                rrb.Spatial2DView(
                    name="TopDown Mocap Video(Raw)",
                    origin=f"/mocap_video/side_0/raw",
                    visual_bounds=VisualBounds2D.from_fields(
                        range=Range2D(
                            x_range=(0, side_videos[0].resized_width),
                            y_range=(0, side_videos[0].resized_height),
                        )
                    ),
                    visible=False,
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="TopDown Mocap Video(Annotated)",
                    origin=f"/mocap_video/side_1/annotated",
                    visual_bounds=VisualBounds2D.from_fields(
                        range=Range2D(
                            x_range=(0, side_videos[1].resized_width),
                            y_range=(0, side_videos[1].resized_height),
                        )
                    ),
                ),
                rrb.Spatial2DView(
                    name="TopDown Mocap Video(Raw)",
                    origin=f"/mocap_video/side_1/raw",
                    visual_bounds=VisualBounds2D.from_fields(
                        range=Range2D(
                            x_range=(0, side_videos[1].resized_width),
                            y_range=(0, side_videos[1].resized_height),
                        )
                    ),
                    visible=False,
                ),
            ),
        )

        side_view_vertical_1 = rrb.Vertical(
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="TopDown Mocap Video(Annotated)",
                    origin=f"/mocap_video/side_2/annotated",
                    visual_bounds=VisualBounds2D.from_fields(
                        range=Range2D(
                            x_range=(0, side_videos[2].resized_width),
                            y_range=(0, side_videos[2].resized_height),
                        )
                    ),
                ),
                rrb.Spatial2DView(
                    name="TopDown Mocap Video(Raw)",
                    origin=f"/mocap_video/side_2/raw",
                    visual_bounds=VisualBounds2D.from_fields(
                        range=Range2D(
                            x_range=(0, side_videos[2].resized_width),
                            y_range=(0, side_videos[2].resized_height),
                        )
                    ),
                    visible=False,
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="TopDown Mocap Video(Annotated)",
                    origin=f"/mocap_video/side_3/annotated",
                    visual_bounds=VisualBounds2D.from_fields(
                        range=Range2D(
                            x_range=(0, side_videos[3].resized_width),
                            y_range=(0, side_videos[3].resized_height),
                        )
                    ),
                ),
                rrb.Spatial2DView(
                    name="TopDown Mocap Video(Raw)",
                    origin=f"/mocap_video/side_3/raw",
                    visual_bounds=VisualBounds2D.from_fields(
                        range=Range2D(
                            x_range=(0, side_videos[3].resized_width),
                            y_range=(0, side_videos[3].resized_height),
                        )
                    ),
                    visible=False,
                ),
            ),
        )

    spatial_3d_view_1 = rrb.Spatial3DView(
        name=data_1_name,
        origin=f"/tracked_object_1/pose/points",
    )
    spatial_3d_view_2 = rrb.Spatial3DView(
        name=data_2_name,
        origin=f"/tracked_object_2/pose/points",
    )

    if include_side_videos:
        views = [
            topdown_view,
            side_view_vertical_0,
            side_view_vertical_1,
            spatial_3d_view_1,
            spatial_3d_view_2,
        ]
    else:
        views = [topdown_view, spatial_3d_view_1, spatial_3d_view_2]

    blueprint = rrb.Horizontal(*views)

    rr.send_blueprint(blueprint)

    time_column = rr.TimeColumn("time", duration=topdown_mocap_video.timestamps_array)
    class_ids = np.ones(shape=data_3d_1.shape[0])
    keypoints = np.array(list(landmarks.values()))
    keypoint_ids = np.repeat(keypoints[np.newaxis, :], data_3d_1.shape[0], axis=0)
    rr.send_columns(
        entity_path="tracked_object_1/pose/points",
        indexes=[time_column],
        columns=[
            *rr.Points3D.columns(positions=data_3d_1),
            *rr.Points3D.columns(
                class_ids=class_ids,
                keypoint_ids=keypoint_ids,
            ),
        ],
    )
    class_ids = np.ones(shape=data_3d_2.shape[0])
    keypoints = np.array(list(landmarks.values()))
    keypoint_ids = np.repeat(keypoints[np.newaxis, :], data_3d_2.shape[0], axis=0)
    rr.send_columns(
        entity_path="tracked_object_2/pose/points",
        indexes=[time_column],
        columns=[
            *rr.Points3D.columns(positions=data_3d_2),
            *rr.Points3D.columns(
                class_ids=class_ids,
                keypoint_ids=keypoint_ids,
            ),
        ],
    )

    # Process mocap video
    process_video(video_data=topdown_mocap_video, entity_path="mocap_video/top_down")
    if include_side_videos:
        for i, side_video in enumerate(side_videos):
            process_video(video_data=side_video,
                        entity_path=f"mocap_video/side_{i}")

    print(f"Processing complete! Rerun recording '{recording_name}' is ready.")


def main_rerun_viewer_maker(
    recording_folder: RecordingFolder,
    data_1_name: str,
    data_3d_1: np.ndarray,
    data_2_name: str,
    data_3d_2: np.ndarray,
    landmarks: dict[str, int],
    connections: tuple[tuple[int, int], ...],
    include_side_videos: bool = False,
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
                float(topdown_mocap_video.timestamps_array[0]),
                float(side_0_video.timestamps_array[0]),
                float(side_1_video.timestamps_array[0]),
                float(side_2_video.timestamps_array[0]),
                float(side_3_video.timestamps_array[0]),
            ]
        )
    else:
        side_videos = []
        recording_start_time = float(topdown_mocap_video.timestamps_array[0])
    topdown_mocap_video.timestamps_array -= recording_start_time
    for side_video in side_videos:
        side_video.timestamps_array -= recording_start_time
    # Process and visualize the eye videos
    create_rerun_recording(
        data_1_name=data_1_name,
        data_3d_1=data_3d_1,
        data_2_name=data_2_name,
        data_3d_2=data_3d_2,
        topdown_mocap_video=topdown_mocap_video,
        side_videos=side_videos,
        recording_name=recording_folder.recording_name,
        landmarks=landmarks,
        connections=connections,
        include_side_videos=include_side_videos,
    )


if __name__ == "__main__":
    recording_name = "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name)

    data_1_name = "resnet_50"
    data_3d_1_path = (
        recording_folder.mocap_data_folder
        / "output_data_head_body_eyecam_v1_model_outputs_iteration_17_old_toml"
        / "dlc"
        / "dlc_body_rigid_3d_xyz.npy"
    )
    data_2_name = "rtmpose_x"
    data_3d_2_path = (
        recording_folder.mocap_data_folder
        / "output_data_head_body_eyecam_retrain_test_v2_model_outputs_iteration_1_using_old_toml"
        / "dlc"
        / "dlc_body_rigid_3d_xyz.npy"
    )

    landmarks = {
        "nose": 0,
        "left_cam_tip": 1,
        "right_cam_tip": 2,
        "base": 3,
        "left_eye": 4,
        "right_eye": 5,
        "left_ear": 6,
        "right_ear": 7,
        "spine_t1": 8,
        "tail_base": 9,
        "tail_tip": 10,
    }

    connections = (
        (0, 5),
        (0, 4),
        (5, 7),
        (4, 6),
        (3, 1),
        (3, 2),
        (3, 8),
        (8, 9),
        (9, 10),
    )

    # for freemocap:
    # recording_name = "session_2025-05-28_12_46_54/recording_12_50_03_gmt-6"

    # recording_folder = FreemocapRecordingFolder.create_from_clip(recording_name)
    # import mediapipe as mp
    # import mediapipe.python.solutions.pose as mp_pose
    # landmarks = {lm.name: lm.value for lm in mp_pose.PoseLandmark}
    # connections = mp_pose.POSE_CONNECTIONS
    # data_3d_path = recording_folder.mocap_output_data_folder / "mediapipe_body_3d_xyz.npy"

    data_3d_1 = np.load(data_3d_1_path)
    data_3d_2 = np.load(data_3d_2_path)
    main_rerun_viewer_maker(
        recording_folder=recording_folder,
        data_1_name=data_1_name,
        data_3d_1=data_3d_1,
        data_2_name=data_2_name,
        data_3d_2=data_3d_2,
        landmarks=landmarks,
        connections=connections,
        include_side_videos=False,
    )
