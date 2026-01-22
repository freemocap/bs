from pathlib import Path
from typing import Literal

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import (
    run_binocular_eye_rerun_viewer,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import (
    eye_camera_distance_from_skull_geometry,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.ferret_eyeball_reference_geometry import NUM_PUPIL_POINTS
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.kinematics_core.stick_figure_topology_model import StickFigureTopology


def create_eye_topology(eye_name: str) -> StickFigureTopology:
    """
    Create a StickFigureTopology for an eye.

    Marker names: tear_duct, outer_eye, pupil_center, p1-p8
    Rigid edges: tear_duct <-> outer_eye (fixed socket landmarks)
    Display edges: tear_duct-outer_eye line, pupil boundary ring (p1->p2->...->p8->p1)

    Args:
        eye_name: Name of the eye (e.g., "left_eye" or "right_eye")

    Returns:
        StickFigureTopology for the eye
    """
    marker_names = [
        "tear_duct",
        "outer_eye",
        "pupil_center",
    ] + [f"p{i}" for i in range(1, NUM_PUPIL_POINTS + 1)]

    # Rigid edges: only the socket landmarks maintain fixed distance
    rigid_edges: list[tuple[str, str]] = [
        ("tear_duct", "outer_eye"),
    ]

    # Display edges: socket line + pupil boundary ring
    display_edges: list[tuple[str, str]] = [
        ("tear_duct", "outer_eye"),
    ]
    # Add pupil boundary connections: p1->p2, p2->p3, ..., p8->p1
    for i in range(1, NUM_PUPIL_POINTS + 1):
        next_i = (i % NUM_PUPIL_POINTS) + 1
        display_edges.append((f"p{i}", f"p{next_i}"))

    return StickFigureTopology(
        name=eye_name,
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        display_edges=display_edges,
    )


def load_eye_kinematics_from_csv(
    left_eye_trajectories_csv_path: Path,
    right_eye_trajectories_csv_path: Path,
    skull_reference_geometry_json_path: Path,
) -> dict[str, FerretEyeKinematics]:
    skull_reference_geometry = ReferenceGeometry.from_json_file(skull_reference_geometry_json_path)
    kinematics_by_eye: dict[str, FerretEyeKinematics] = {}
    for eye_name, csv_path in zip(
        ["left_eye", "right_eye"],
        [left_eye_trajectories_csv_path, right_eye_trajectories_csv_path],
        strict=True,
    ):
        eye_side: Literal["left", "right"] = "left" if eye_name == "left_eye" else "right"
        eye_camera_distance_mm = eye_camera_distance_from_skull_geometry(
            skull_reference_geometry=skull_reference_geometry,
            eye_side=eye_side,
        )
        kinematics_by_eye[eye_name] = FerretEyeKinematics.calculate_from_trajectories(
            eye_trajectories_csv_path=csv_path,
            eye_camera_distance_mm=eye_camera_distance_mm,
            eye_name=eye_name,
        )
    return kinematics_by_eye


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION - Edit these values
    # =========================================================================
    _left_eye_trajectories_csv_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye0_data.csv"
    )
    _right_eye_trajectories_csv_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye1_data.csv"
    )
    _skull_reference_geometry_json_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output\skull_reference_geometry.json"
    )
    _eye_kinematics_output_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye_kinematics"
    )

    # Video paths (optional)
    _right_eye_vid = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\left_eye_stabilized.mp4"
    )
    _left_eye_vid = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\right_eye_stabilized.mp4"
    )
    # =========================================================================

    _eye_kinematics_output_path.mkdir(exist_ok=True, parents=True)
    print(f"Loading eye data from {_left_eye_trajectories_csv_path}...")

    eye_kinematics_by_eye = load_eye_kinematics_from_csv(
        left_eye_trajectories_csv_path=_left_eye_trajectories_csv_path,
        right_eye_trajectories_csv_path=_right_eye_trajectories_csv_path,
        skull_reference_geometry_json_path=_skull_reference_geometry_json_path,
    )

    for _eye_name, eye_kinematics in eye_kinematics_by_eye.items():
        print(f"Saving {_eye_name} kinematics to {_eye_kinematics_output_path}")
        eye_kinematics.save_to_disk(output_directory=_eye_kinematics_output_path)

        # Save eye topology
        eye_topology = create_eye_topology(_eye_name)
        topology_path = _eye_kinematics_output_path / f"{_eye_name}_topology.json"
        eye_topology.save_json(topology_path)
        print(f"Saved {_eye_name} topology to {topology_path}")

    # Launch binocular viewer with both eyes
    run_binocular_eye_rerun_viewer(
        eye_kinematics_directory_path=_eye_kinematics_output_path,
        left_eye_trajectories_csv_path=_left_eye_trajectories_csv_path,
        right_eye_trajectories_csv_path=_right_eye_trajectories_csv_path,
        left_eye_video_path=_left_eye_vid,
        right_eye_video_path=_right_eye_vid,
        time_window_seconds=3.0,
    )

    print("Done.")