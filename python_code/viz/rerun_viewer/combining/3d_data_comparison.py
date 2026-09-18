"""Video encode images using av and stream them to Rerun with optimized performance."""

import numpy as np
from datetime import datetime
from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.utilities.folder_utilities.recording_folder import BaslerCamera, RecordingFolder
from python_code.viz.rerun_viewer.utils.groundplane_and_origin import log_groundplane_and_origin
from python_code.viz.rerun_viewer.utils.process_videos import process_video
from python_code.viz.rerun_viewer.utils.video_data import MocapVideoData

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
        origin=f"/tracked_object_1/",
    )
    log_groundplane_and_origin(entity_path="/tracked_object_1")
    spatial_3d_view_2 = rrb.Spatial3DView(
        name=data_2_name,
        origin=f"/tracked_object_2/",
    )
    log_groundplane_and_origin(entity_path="/tracked_object_2")

    if include_side_videos:
        views = [
            topdown_view,
            side_view_vertical_0,
            side_view_vertical_1,
            spatial_3d_view_1,
            spatial_3d_view_2,
        ]
    else:
        # views = [topdown_view, spatial_3d_view_1, spatial_3d_view_2]
        views = [spatial_3d_view_1, spatial_3d_view_2]

    blueprint = rrb.Horizontal(*views)

    rr.send_blueprint(blueprint)

    time_column = rr.TimeColumn("time", duration=topdown_mocap_video.timestamps)
    class_ids = np.ones(shape=data_3d_1.shape[0])
    show_labels = np.full(shape=data_3d_1.shape, fill_value=False, dtype=bool)
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
                show_labels=show_labels
            ),
        ],
    )
    class_ids = np.ones(shape=data_3d_2.shape[0])
    show_labels = np.full(shape=data_3d_2.shape, fill_value=False, dtype=bool)
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
                show_labels=show_labels
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
        annotated_video_path=recording_folder.get_annotated_video_by_name(BaslerCamera.TOPDOWN.value),
        raw_video_path=recording_folder.get_synchronized_video_by_name(BaslerCamera.TOPDOWN.value),
        timestamps_npy_path=recording_folder.get_timestamp_by_name(BaslerCamera.TOPDOWN.value),
        data_name="TopDown Mocap",
    )

    if include_side_videos:
        side_cameras = [BaslerCamera.SIDE_0, BaslerCamera.SIDE_1, BaslerCamera.SIDE_2, BaslerCamera.SIDE_3]
        side_videos = [
            MocapVideoData.create(
                annotated_video_path=recording_folder.get_annotated_video_by_name(camera.value),
                raw_video_path=recording_folder.get_synchronized_video_by_name(camera.value),
                timestamps_npy_path=recording_folder.get_timestamp_by_name(camera.value),
                data_name=f"Side {i} Mocap",
                resize_factor=0.5,
            )
            for i, camera in enumerate(side_cameras)
        ]
    else:
        side_videos = []

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
    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)
    print(recording_folder.mocap_data)

    data_1_name = "resnet_50_no_thresholding"
    data_3d_1_path = (
        recording_folder.mocap_data
        / "output_data_archive"
        / "output_data_resnet_50_no_confidence_thresholding"
        / "dlc"
        / "dlc_body_rigid_3d_xyz.npy"
    )
    data_2_name = "rtmpose_no_thresholding"
    data_3d_2_path = (
        recording_folder.mocap_data
        / "output_data_archive"
        / "output_data_head_body_eyecam_retrain_test_v2_confidence_0"
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
