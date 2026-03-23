import matplotlib.pyplot as plt
import numpy as np

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import (
    COLOR_LEFT_EYE_PRIMARY,
    COLOR_LEFT_EYE_SECONDARY,
    COLOR_RIGHT_EYE_PRIMARY,
    COLOR_RIGHT_EYE_SECONDARY,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import (
    FerretEyeKinematics,
)
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.rerun_viewer.rerun_utils.gaze_plots.plot_ferret_skull_and_spine_3d import load_toy_data_from_tidy_csv
from python_code.rigid_body_solver.viz.ferret_skull_rerun import (
    RAD_TO_DEG,
    load_kinematics_from_tidy_csv,
)
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


def get_eye_kinematics_from_recording_folder(
    recording_folder: RecordingFolder, eye_name: str = "left"
) -> tuple[FerretEyeKinematics, np.ndarray]:
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")
    kinematics = FerretEyeKinematics.load_from_directory(
        eye_name=f"{eye_name}_eye",
        input_directory=recording_folder.eye_output_data / "eye_kinematics",
    )
    timestamps = kinematics.eyeball.timestamps
    timestamps = timestamps - timestamps[0]
    print(f"Loaded {eye_name} eye kinematics: {kinematics.n_frames} frames")

    return kinematics, timestamps


def get_gaze_kinematics_from_recording_folder(
    recording_folder: RecordingFolder, eye_name: str = "left"
) -> tuple[FerretEyeKinematics, np.ndarray]:
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")
    kinematics = FerretEyeKinematics.load_from_directory(
        eye_name=f"{eye_name}_gaze",
        input_directory=recording_folder.gaze_kinematics,
    )
    timestamps = kinematics.eyeball.timestamps
    timestamps = timestamps - timestamps[0]
    print(f"Loaded {eye_name} eye kinematics: {kinematics.n_frames} frames")

    return kinematics, timestamps


def get_head_kinematics_from_recording_folder(
    recording_folder: RecordingFolder,
) -> tuple[RigidBodyKinematics, np.ndarray]:
    # Load skull kinematics
    reference_geometry = ReferenceGeometry.from_json_file(
        recording_folder.skull_reference_geometry
    )
    print(f"  Reference geometry: {len(reference_geometry.keypoints)} keypoints")

    kinematics = load_kinematics_from_tidy_csv(
        csv_path=recording_folder.skull_kinematics_csv,
        reference_geometry=reference_geometry,
        name="skull",
    )

    t0 = kinematics.timestamps[0]
    times = kinematics.timestamps - t0

    return kinematics, times

def plot_eye_kinematics_distributions(
    left_eye_kinematics: FerretEyeKinematics,
    right_eye_kinematics: FerretEyeKinematics,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot left eye distributions
    axes[0].hist(np.degrees(left_eye_kinematics.adduction_angle.values), bins=30, color=COLOR_LEFT_EYE_PRIMARY, alpha=0.7)
    axes[0].set_title("Left Eye Azimuth (degrees)")
    axes[1].hist(np.degrees(left_eye_kinematics.elevation_angle.values), bins=30, color=COLOR_LEFT_EYE_SECONDARY, alpha=0.7)
    axes[1].set_title("Left Eye Elevation (degrees)")

    # Plot right eye distributions
    axes[3].hist(np.degrees(right_eye_kinematics.adduction_angle.values), bins=30, color=COLOR_RIGHT_EYE_PRIMARY, alpha=0.7)
    axes[3].set_title("Right Eye Azimuth (degrees)")
    axes[4].hist(np.degrees(right_eye_kinematics.elevation_angle.values), bins=30, color=COLOR_RIGHT_EYE_SECONDARY, alpha=0.7)
    axes[4].set_title("Right Eye Elevation (degrees)")

    plt.tight_layout()
    plt.show()

def plot_gaze_kinematics_distributions(
    left_gaze_kinematics: FerretEyeKinematics,
    right_gaze_kinematics: FerretEyeKinematics,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot left gaze distributions
    axes[0].hist(np.degrees(left_gaze_kinematics.adduction_angle.values), bins=30, color=COLOR_LEFT_EYE_PRIMARY, alpha=0.7)
    axes[0].set_title("Left Gaze Azimuth (degrees)")
    axes[1].hist(np.degrees(left_gaze_kinematics.elevation_angle.values), bins=30, color=COLOR_LEFT_EYE_SECONDARY, alpha=0.7)
    axes[1].set_title("Left Gaze Elevation (degrees)")

    # Plot right gaze distributions
    axes[3].hist(np.degrees(right_gaze_kinematics.adduction_angle.values), bins=30, color=COLOR_RIGHT_EYE_PRIMARY, alpha=0.7)
    axes[3].set_title("Right Gaze Azimuth (degrees)")
    axes[4].hist(np.degrees(right_gaze_kinematics.elevation_angle.values), bins=30, color=COLOR_RIGHT_EYE_SECONDARY, alpha=0.7)
    axes[4].set_title("Right Gaze Elevation (degrees)")

if __name__ == "__main__":
    recording_folder = RecordingFolder.from_folder_path(
        "/Users/philipqueen/session_2025-07-01_ferret_757_EyeCameras_P33_EO5/clips/1m_20s-2m_20s/"
    )