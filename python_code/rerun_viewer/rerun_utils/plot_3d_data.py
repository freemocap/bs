from pathlib import Path
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.load_tidy_dataset import load_tidy_dataset
from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.video_data import MocapVideoData
from python_code.utilities.connections_and_landmarks import (
    ferret_head_spine_connections,
    ferret_head_spine_landmarks,
    toy_connections,
    toy_landmarks,
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
    from python_code.utilities.folder_utilities.recording_folder import RecordingFolder, BaslerCamera
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-09_ferret_753_EyeCameras_P41_E13/full_recording"
    )

    include_toy = True
    use_solver_output = True

    recording_folder = RecordingFolder.from_folder_path(folder_path)
    if use_solver_output:
        recording_folder.check_skull_postprocessing(enforce_toy=include_toy, enforce_annotated=True)
    else:
        recording_folder.check_triangulation(enforce_toy=include_toy, enforce_annotated=True)

    topdown_synchronized_video = recording_folder.get_synchronized_video_by_name(BaslerCamera.TOPDOWN.value)
    topdown_annotated_video = recording_folder.get_annotated_video_by_name(BaslerCamera.TOPDOWN.value)
    topdown_timestamps_npy = recording_folder.get_timestamp_by_name(BaslerCamera.TOPDOWN.value)

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=topdown_annotated_video,
        raw_video_path=topdown_synchronized_video,
        timestamps_npy_path=topdown_timestamps_npy,
        data_name="TopDown Mocap",
    )

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    if use_solver_output:
        solver_output_path = (
            recording_folder.mocap_solver_output
            / "tidy_trajectory_data.csv"
        )
        body_data_3d = load_tidy_dataset(
            csv_path=solver_output_path,
            landmarks=ferret_head_spine_landmarks,
            data_type="optimized"
        )

    else:
        body_data_3d_path = (
            recording_folder.mocap_3d_data
            / "head_body_rigid_3d_xyz.npy"
        )
        body_data_3d = np.load(body_data_3d_path)

    if include_toy:
        toy_data_3d_path = (
            recording_folder.toy_3d_data
            / "toy_body_rigid_3d_xyz.npy"
        )
        toy_data_3d = np.load(toy_data_3d_path)
    else:
        toy_data_3d = None

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

