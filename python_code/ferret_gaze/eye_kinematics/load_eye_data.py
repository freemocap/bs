from pathlib import Path
from typing import Literal

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import \
    eye_camera_distance_from_skull_geometry
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry


def load_eye_kinematics_from_csv(eye_trajectories_csv_path: Path,
                                 skull_reference_geometry_json_path: Path) -> dict[str, FerretEyeKinematics]:
    skull_reference_geometry = ReferenceGeometry.from_json_file(skull_reference_geometry_json_path)
    kinematics_by_eye = {}
    for eye_name in ['left_eye', 'right_eye']:
        eye_side: Literal['left', 'right'] = 'left' if eye_name == 'left_eye' else 'right'
        eye_camera_distance_mm = eye_camera_distance_from_skull_geometry(
            skull_reference_geometry=skull_reference_geometry,
            eye_side=eye_side
        )
        kinematics_by_eye[eye_name] = FerretEyeKinematics.calculate_from_trajectories(
            eye_trajectories_csv_path=eye_trajectories_csv_path,
            eye_camera_distance_mm=eye_camera_distance_mm,
            eye_name=eye_name)
    return kinematics_by_eye


if __name__ == "__main__":

    # =========================================================================
    # CONFIGURATION - Edit these values
    # =========================================================================
    _eye_trajectories_csv_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_trajectories.csv")
    _skull_reference_geometry_json_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output\skull_reference_geometry.json")
    # =========================================================================

    print(f"Loading eye data from {_eye_trajectories_csv_path}...")
    eye_kinematics_by_eye = load_eye_kinematics_from_csv(
        eye_trajectories_csv_path=_eye_trajectories_csv_path,
        skull_reference_geometry_json_path=_skull_reference_geometry_json_path
    )
    print("Done.")