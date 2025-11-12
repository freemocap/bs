from pathlib import Path
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.video_data import MocapVideoData

ferret_head_spine_landmarks = {
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
    "center": 11,
}

ferret_head_spine_connections = (
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

toy_landmarks = {
    "front": 0,
    "top": 1,
    "back": 2,
}

toy_connections = (
    (0, 1),
    (1, 2),
)

def add_3d_data_context(entity_path: str, landmarks: dict, connections: tuple):
    rr.log(
        entity_path,
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

def plot_3d_data(mocap_video: MocapVideoData, landmarks: dict[str, int], data_3d: np.ndarray, entity_path: str = ""):
    time_column = rr.TimeColumn("time", duration=mocap_video.timestamps)
    class_ids = np.ones(shape=data_3d.shape[0])
    show_labels = np.full(shape=data_3d.shape, fill_value=False, dtype=bool)
    keypoints = np.array(list(landmarks.values()))
    keypoint_ids = np.repeat(keypoints[np.newaxis, :], data_3d.shape[0], axis=0)
    rr.send_columns(
        entity_path=f"{entity_path}/pose/points",
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

def get_3d_data_view(entity_path: str = "/"):
    spatial_3d_view = rrb.Spatial3DView(
        name="3D Data",
        origin=entity_path,
    )
    return spatial_3d_view

if __name__ == "__main__":
    from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
    from datetime import datetime

    recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name, base_recordings_folder=Path("/home/scholl-lab/ferret_recordings"))
    # recording_folder = RecordingFolder.create_full_recording(recording_name, base_recordings_folder="/home/scholl-lab/ferret_recordings")

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=recording_folder.topdown_annotated_video_path,
        raw_video_path=recording_folder.topdown_video_path,
        timestamps_npy_path=recording_folder.topdown_timestamps_npy_path,
        data_name="TopDown Mocap",
    )

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
    # solver_output_path = (
    #     recording_folder.mocap_data_folder
    #     / "output_data"
    #     / "solver_output"
    #     / "tidy_trajectory_data.csv"
    # )
    # body_data_3d = load_tidy_dataset(
    #     csv_path=solver_output_path,
    #     landmarks=landmarks,
    #     data_type="optimized"
    # )
    # toy_data_3d = np.load(toy_data_3d_path)
    toy_data_3d = None

    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    mocap_entity_path = "/tracked_object"
    toy_entity_path = "/toy_object"

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

    view = get_3d_data_view(entity_path="/")

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)

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

