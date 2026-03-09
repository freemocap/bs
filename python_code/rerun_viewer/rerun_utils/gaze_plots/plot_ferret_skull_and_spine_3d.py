import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import rerun as rr
import rerun.blueprint as rrb
import polars as pl

from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.kinematics_core.stick_figure_topology_model import StickFigureTopology
from python_code.rigid_body_solver.viz.ferret_skull_rerun import SPINE_MARKER_NAMES, load_kinematics_from_tidy_csv, load_spine_trajectories_from_csv, send_body_basis_vectors, send_body_origin, send_enclosure, send_skull_skeleton_data, send_spine_skeleton_data
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

toy_landmarks = {
    "toy_face": 0,
    "toy_top": 1,
    "toy_tail": 2,
}

SKULL_MARKER_COLORS: dict[str, tuple[int, int, int]] = {
    "nose": (255, 107, 107),
    "left_eye": (78, 205, 196),
    "right_eye": (78, 205, 196),
    "left_ear": (149, 225, 211),
    "right_ear": (149, 225, 211),
    "base": (255, 230, 109),
    "left_cam_tip": (168, 230, 207),
    "right_cam_tip": (168, 230, 207),
}

SPINE_MARKER_COLORS: dict[str, tuple[int, int, int]] = {
    "spine_t1": (221, 160, 221),
    "sacrum": (255, 182, 193),
    "tail_tip": (255, 105, 180),
}

def get_ferret_skull_and_spine_3d_view(
    entity_path: str = "/",
):
    spatial_3d_view = rrb.Spatial3DView(
        name="3D Skeleton",
        origin=f"{entity_path}/skeleton",
        contents=["+ skeleton/**"],
        eye_controls=rrb.EyeControls3D(
            position=(0.0, -2000.0, 500.0),
            look_target=(0.0, 0.0, 0.0),
            eye_up=(0.0, 0.0, 1.0),
        ),
        line_grid=rrb.LineGrid3D(
            visible=True,
            spacing=100.0,
            plane=rr.components.Plane3D.XY,
            color=[100, 100, 100, 128],
        ),
    )

    return spatial_3d_view

def log_ferret_skull_and_spine_3d_style(
    entity_path: str = "/",
):
    rr.log(
        f"{entity_path}skeleton/body_origin",
        rr.Points3D.from_fields(radii=8.0, colors=(255, 255, 255)),
        static=True,
    )

    rr.log(
        f"{entity_path}skeleton/toy",
        rr.Points3D.from_fields(radii=8.0, colors=(0, 0, 255)),
        static=True,
    )

def load_toy_data_from_tidy_csv(
    csv_path: Path,
    landmarks: dict[str, int] = toy_landmarks,
):
    df = pl.read_csv(csv_path)

    num_frames = df.select(pl.col("frame").n_unique()).item()

    num_keypoints = len(landmarks)
    data = np.zeros((num_frames, num_keypoints, 3))
    for i, keypoint in enumerate(landmarks.keys()):
        masked_df = df.filter(pl.col("trajectory") == keypoint)

        data[:, i, :] = (
            masked_df
            .pivot(index="frame", on="component", values="value", aggregate_function="first")
            .select(["x", "y", "z"])          # enforce order
            .to_numpy()
        )

    return data

def send_toy_data(
    data_3d: np.ndarray,
    landmarks: dict[str, int],
    timestamps: np.ndarray,
    entity_path: str = "/",
):
    time_column = rr.TimeColumn("time", duration=timestamps)
    class_ids = np.ones(shape=data_3d.shape[0])
    show_labels = np.full(shape=data_3d.shape, fill_value=False, dtype=bool)
    keypoints = np.array(list(landmarks.values()))
    keypoint_ids = np.repeat(keypoints[np.newaxis, :], data_3d.shape[0], axis=0)
    rr.send_columns(
        entity_path=f"{entity_path}/skeleton/toy",
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

def plot_ferret_skull_and_spine_3d(
    recording_folder: RecordingFolder,
    entity_path: str = "/",
):
    recording_folder.check_gaze_postprocessing(enforce_toy=True, enforce_annotated=False)
    print("Loading data from disk...")

    # Load skull kinematics
    reference_geometry = ReferenceGeometry.from_json_file(recording_folder.skull_reference_geometry)
    print(f"  Reference geometry: {len(reference_geometry.keypoints)} keypoints")

    kinematics = load_kinematics_from_tidy_csv(
        csv_path=recording_folder.skull_kinematics_csv,
        reference_geometry=reference_geometry,
        name="skull",
    )
    print(f"  Skull kinematics: {kinematics.n_frames} frames")

    # Load spine trajectories
    spine_timestamps, spine_trajectories = load_spine_trajectories_from_csv(
        csv_path=recording_folder.skull_and_spine_resampled_trajectories,
        spine_keypoint_names=SPINE_MARKER_NAMES,
    )
    print(f"  Spine trajectories: {len(spine_trajectories)} keypoints, {len(spine_timestamps)} frames")

    # Load topology for display edges
    with open(recording_folder.skull_kinematics / "skull_and_spine_topology.json", "r") as f:
        topology_json_data = json.load(f)

    topology = StickFigureTopology(**topology_json_data)
    spine_display_edges = [
        (a, b) for a, b in topology.display_edges
        if a in SPINE_MARKER_NAMES or b in SPINE_MARKER_NAMES
    ]
    print(f"  Topology: {len(spine_display_edges)} spine display edges")

    print(f"Logging {kinematics.n_frames} frames...")
    print("  Sending enclosure...")
    send_enclosure(entity_path=entity_path)
    print("  Sending body origin...")
    send_body_origin(kinematics, entity_path=entity_path)
    print("  Sending body basis vectors...")
    send_body_basis_vectors(kinematics=kinematics, scale=100.0, entity_path=entity_path)
    print("  Sending skull skeleton data...")
    send_skull_skeleton_data(kinematics=kinematics, keypoint_colors=SKULL_MARKER_COLORS, entity_path=entity_path)

    all_trajectories_for_edges: dict[str, NDArray[np.float64]] = {}
    for keypoint_name in kinematics.keypoint_names:
        all_trajectories_for_edges[keypoint_name] = kinematics.keypoint_trajectories[keypoint_name]
    for spine_name, spine_traj in spine_trajectories.items():
        all_trajectories_for_edges[spine_name] = spine_traj

    print("  Sending spine skeleton data...")
    send_spine_skeleton_data(
        timestamps=spine_timestamps,
        spine_trajectories=spine_trajectories,
        all_trajectories_for_edges=all_trajectories_for_edges,
        display_edges=spine_display_edges,
        keypoint_colors=SPINE_MARKER_COLORS,
        entity_path=entity_path
    )

    print("  Sending toy data...")
    toy_data = load_toy_data_from_tidy_csv(csv_path=recording_folder.toy_resampled_trajectories)
    send_toy_data(
        data_3d=toy_data,
        landmarks=toy_landmarks,
        timestamps=kinematics.timestamps,
        entity_path=entity_path
    )

if __name__ == "__main__":
    from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    entity_path = "/"

    rr.init(recording_string, spawn=True)

    view = get_ferret_skull_and_spine_3d_view(entity_path=entity_path)

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)
    log_ferret_skull_and_spine_3d_style(entity_path=entity_path)

    plot_ferret_skull_and_spine_3d(recording_folder=recording_folder, entity_path=entity_path) 