"""Video encode images using av and stream them to Rerun with optimized performance."""

import numpy as np
from datetime import datetime
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D
import toml

from python_code.rerun_viewer.rerun_utils.load_tidy_dataset import load_solver_outputs
from python_code.rerun_viewer.rerun_utils.groundplane_and_origin import log_groundplane_and_origin
from python_code.rerun_viewer.rerun_utils.log_cameras import log_cameras
from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
from python_code.rerun_viewer.rerun_utils.video_data import MocapVideoData
from python_code.rerun_viewer.rerun_utils.groundplane_and_origin import log_groundplane_and_origin

# Configuration
GOOD_PUPIL_POINT = "p2"
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)


def create_rerun_recording(
    recording_name: str,
    data_3d: np.ndarray,
    topdown_mocap_video: MocapVideoData,
    side_videos: list[MocapVideoData],
    landmarks: dict[str, int],
    connections: tuple[tuple[int, int], ...],
    include_side_videos: bool = False,
    calibration: dict | None = None,
    toy_data_3d: np.ndarray | None = None,
    toy_landmarks: dict[str, int] | None = None,
    toy_connections: tuple[tuple[int, int], ...] | None = None
) -> None:
    """Process both eye videos and visualize them with Rerun."""
    # Initialize Rerun
    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    rr.log(
        "/tracked_object",
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

    log_groundplane_and_origin()

    if calibration:
        log_cameras(calibration)

    if toy_data_3d is not None and toy_landmarks is not None and toy_connections is not None:
        rr.log(
            "/toy_object",
            rr.AnnotationContext(
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=2, label="Toy"),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=value, label=key)
                        for key, value in toy_landmarks.items()
                    ],
                    keypoint_connections=toy_connections,
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

    spatial_3d_view = rrb.Spatial3DView(
        name="3D Data",
        origin=f"/",
    )

    if include_side_videos:
        views = [
            topdown_view,
            side_view_vertical_0,
            side_view_vertical_1,
            spatial_3d_view,
        ]
    else:
        views = [topdown_view, spatial_3d_view]

    blueprint = rrb.Horizontal(*views)

    rr.send_blueprint(blueprint)

    time_column = rr.TimeColumn("time", duration=topdown_mocap_video.timestamps)
    class_ids = np.ones(shape=data_3d.shape[0])
    show_labels = np.full(shape=data_3d.shape, fill_value=False, dtype=bool)
    keypoints = np.array(list(landmarks.values()))
    keypoint_ids = np.repeat(keypoints[np.newaxis, :], data_3d.shape[0], axis=0)
    rr.send_columns(
        entity_path="tracked_object/pose/points",
        indexes=[time_column],
        columns=[
            *rr.Points3D.columns(positions=data_3d),
            *rr.Points3D.columns(
                class_ids=class_ids,
                keypoint_ids=keypoint_ids,
                show_labels=show_labels,
            ),
        ],
    )

    if toy_data_3d is not None and toy_landmarks is not None and toy_connections is not None:
        class_ids = np.ones(shape=toy_data_3d.shape[0])
        show_labels = np.full(shape=toy_data_3d.shape, fill_value=False, dtype=bool)
        keypoints = np.array(list(landmarks.values()))
        keypoint_ids = np.repeat(keypoints[np.newaxis, :], toy_data_3d.shape[0], axis=0)
        rr.send_columns(
            entity_path="toy_object/pose/points",
            indexes=[time_column],
            columns=[
                *rr.Points3D.columns(positions=toy_data_3d),
                *rr.Points3D.columns(
                    class_ids=class_ids,
                    keypoint_ids=keypoint_ids,
                    show_labels=show_labels,
                ),
            ],
        )

    # Process mocap video
    # process_video(video_data=topdown_mocap_video, entity_path="mocap_video/top_down")
    # if include_side_videos:
    #     for i, side_video in enumerate(side_videos):
    #         process_video(video_data=side_video,
    #                                 entity_path=f"mocap_video/side_{i}")

    print(f"Processing complete! Rerun recording '{recording_name}' is ready.")


def main_rerun_viewer_maker(
    recording_folder: RecordingFolder,
    body_data_3d: np.ndarray,
    landmarks: dict[str, int],
    connections: tuple[tuple[int, int], ...],
    include_side_videos: bool = False,
    calibration_path: str | None = None,
    toy_data_3d: np.ndarray | None = None, 
    toy_landmarks: dict[str, int] | None = None,
    toy_connections: tuple[tuple[int, int], ...] | None = None
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
    if calibration_path is not None:
        calibration = toml.load(calibration_path)
    else:
        calibration = None
    # Process and visualize the eye videos
    create_rerun_recording(
        data_3d=body_data_3d,
        topdown_mocap_video=topdown_mocap_video,
        side_videos=side_videos,
        recording_name=recording_folder.recording_name,
        landmarks=landmarks,
        connections=connections,
        include_side_videos=include_side_videos,
        calibration=calibration,
        toy_data_3d=toy_data_3d, 
        toy_connections=toy_connections,
        toy_landmarks=toy_landmarks
    )


if __name__ == "__main__":
    # recording_name = "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    # clip_name = "0m_37s-1m_37s"
    # recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name, base_recordings_folder="/home/scholl-lab/ferret_recordings")
    recording_folder = RecordingFolder.create_full_recording(recording_name="session_2025-07-09_ferret_753_EyeCameras_P41_E13", base_recordings_folder="/home/scholl-lab/ferret_recordings")
    calibration_path = "/home/scholl-lab/ferret_recordings/session_2025-10-20_ferret_420_E010/calibration/session_2025-10-20_calibration_camera_calibration.toml"

    body_data_3d_path = (
        recording_folder.mocap_data_folder
        / "output_data"
        / "dlc"
        / "head_body_rigid_3d_xyz.npy"
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
        "sacrum": 9,
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

    toy_data_3d_path = (
        recording_folder.mocap_data_folder
        / "output_data"
        / "dlc"
        / "toy_body_rigid_3d_xyz.npy"
    )

    toy_landmarks = {
        "toy_face": 0,
        "toy_top": 1,
        "toy_tail": 2,
    }

    toy_connections = (
        (0, 1),
        (1, 2),
    )

    # for freemocap:
    # recording_name = "session_2025-05-28_12_46_54/recording_12_50_03_gmt-6"

    # recording_folder = FreemocapRecordingFolder.create_from_clip(recording_name)
    # import mediapipe as mp
    # import mediapipe.python.solutions.pose as mp_pose
    # landmarks = {lm.name: lm.value for lm in mp_pose.PoseLandmark}
    # connections = mp_pose.POSE_CONNECTIONS
    # data_3d_path = recording_folder.mocap_output_data_folder / "mediapipe_body_3d_xyz.npy"

    # body_data_3d = np.load(body_data_3d_path)
    solver_output_path = (
        recording_folder.mocap_data_folder
        / "output_data"
        / "solver_output"
        / "skull_and_spine_trajectories.csv"
    )
    body_data_3d = load_solver_outputs(
        csv_path=solver_output_path,
        landmarks=landmarks,
    )
    toy_data_3d = np.load(toy_data_3d_path)
    main_rerun_viewer_maker(
        recording_folder=recording_folder,
        body_data_3d=body_data_3d,
        landmarks=landmarks,
        connections=connections,
        include_side_videos=False,
        toy_data_3d=toy_data_3d,
        toy_landmarks=toy_landmarks,
        toy_connections=toy_connections,
        # calibration_path=calibration_path,
    )
