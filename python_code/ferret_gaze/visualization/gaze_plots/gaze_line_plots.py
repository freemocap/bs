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


def plot_eye_position(
    recording_folder: RecordingFolder,
    eye_name: str = "left",
    frame_range: tuple[int, int] = (0, 1200),
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    kinematics, timestamps = get_eye_kinematics_from_recording_folder(
        recording_folder, eye_name
    )
    adduction_deg = np.degrees(kinematics.adduction_angle.values)
    elevation_deg = np.degrees(kinematics.elevation_angle.values)

    primary_color = "#0096FF" if eye_name == "left" else "#FF6400"
    secondary_color = "#64B4FF" if eye_name == "left" else "#FFA050"

    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        adduction_deg[frame_range[0] : frame_range[1]],
        label="Adduction",
        color=primary_color,
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        elevation_deg[frame_range[0] : frame_range[1]],
        label="Elevation",
        color=secondary_color,
    )
    ax.set_title(f"{eye_name.capitalize()} Eye Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Degrees")
    ax.legend()

def plot_gaze_position(
    recording_folder: RecordingFolder,
    eye_name: str = "left",
    frame_range: tuple[int, int] = (0, 1200),
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    kinematics, timestamps = get_gaze_kinematics_from_recording_folder(
        recording_folder, eye_name
    )
    adduction_deg = np.degrees(kinematics.adduction_angle.values)
    elevation_deg = np.degrees(kinematics.elevation_angle.values)

    adduction_deg, elevation_deg = elevation_deg, adduction_deg

    primary_color = "#0096FF" if eye_name == "left" else "#FF6400"
    secondary_color = "#64B4FF" if eye_name == "left" else "#FFA050"

    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        adduction_deg[frame_range[0] : frame_range[1]],
        label="Adduction",
        color=primary_color,
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        elevation_deg[frame_range[0] : frame_range[1]],
        label="Elevation",
        color=secondary_color,
    )
    ax.set_title(f"{eye_name.capitalize()} Gaze Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Degrees")
    ax.legend()

def plot_gaze_velocity(
    recording_folder: RecordingFolder,
    eye_name: str = "left",
    frame_range: tuple[int, int] = (0, 1200),
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    kinematics, timestamps = get_gaze_kinematics_from_recording_folder(
        recording_folder, eye_name
    )
    adduction_deg_s = np.rad2deg(kinematics.adduction_velocity.values)
    elevation_deg_s = np.rad2deg(kinematics.elevation_velocity.values)

    adduction_deg_s, elevation_deg_s = elevation_deg_s, adduction_deg_s

    primary_color = "#0096FF" if eye_name == "left" else "#FF6400"
    secondary_color = "#64B4FF" if eye_name == "left" else "#FFA050"

    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        adduction_deg_s[frame_range[0] : frame_range[1]],
        label="Adduction",
        color=primary_color,
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        elevation_deg_s[frame_range[0] : frame_range[1]],
        label="Elevation",
        color=secondary_color,
    )
    ax.set_title(f"{eye_name.capitalize()} Gaze Velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (deg/s)")
    ax.legend()


def plot_head_position(
    recording_folder: RecordingFolder, frame_range: tuple[int, int] = (0, 1200), ax=None
):
    if ax is None:
        fig, ax = plt.subplots()

    kinematics, timestamps = get_head_kinematics_from_recording_folder(recording_folder)
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        kinematics.position_xyz[frame_range[0] : frame_range[1], 0],
        label="X",
        color=(255 / 255, 107 / 255, 107 / 255),
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        kinematics.position_xyz[frame_range[0] : frame_range[1], 1],
        label="Y",
        color=(78 / 255, 255 / 255, 96 / 255),
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        kinematics.position_xyz[frame_range[0] : frame_range[1], 2],
        label="Z",
        color=(100 / 255, 149 / 255, 255 / 255),
    )
    ax.set_title("Head Position")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Position (mm)")
    ax.legend()

def plot_head_rotation(
    recording_folder: RecordingFolder, frame_range: tuple[int, int] = (0, 1200), ax=None
):
    if ax is None:
        fig, ax = plt.subplots()

    kinematics, timestamps = get_head_kinematics_from_recording_folder(recording_folder)
    euler_rad = kinematics.orientations.to_euler_xyz_array()
    euler_deg = euler_rad * RAD_TO_DEG
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        euler_deg[frame_range[0] : frame_range[1], 0],
        label="Roll",
        color=(255 / 255, 107 / 255, 107 / 255),
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        euler_deg[frame_range[0] : frame_range[1], 1],
        label="Pitch",
        color=(78 / 255, 205 / 255, 196 / 255),
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        euler_deg[frame_range[0] : frame_range[1], 2],
        label="Yaw",
        color=(255 / 255, 230 / 255, 109 / 255),
    )
    ax.set_title("Head Rotation")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Rotation (degrees)")
    ax.legend()

def plot_head_velocity(
    recording_folder: RecordingFolder, frame_range: tuple[int, int] = (0, 1200), ax=None
):
    if ax is None:
        fig, ax = plt.subplots()

    kinematics, timestamps = get_head_kinematics_from_recording_folder(recording_folder)
    omega_local_deg_s = kinematics.angular_velocity_local * RAD_TO_DEG
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        omega_local_deg_s[frame_range[0] : frame_range[1], 0],
        label="Roll",
        color=(255 / 255, 107 / 255, 107 / 255),
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        omega_local_deg_s[frame_range[0] : frame_range[1], 1],
        label="Pitch",
        color=(78 / 255, 205 / 255, 196 / 255),
    )
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        omega_local_deg_s[frame_range[0] : frame_range[1], 2],
        label="Yaw",
        color=(255 / 255, 230 / 255, 109 / 255),
    )
    ax.set_title("Local Head Velocity")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Velocity (deg/s)")
    ax.legend()

def plot_prey_distance(
    recording_folder: RecordingFolder, frame_range: tuple[int, int] = (0, 1200), ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    toy_data = load_toy_data_from_tidy_csv(csv_path=recording_folder.toy_resampled_trajectories)
    toy_mean_xyz = np.mean(toy_data, axis=1)
    kinematics, timestamps = get_head_kinematics_from_recording_folder(recording_folder)

    distances = np.linalg.norm(toy_mean_xyz - kinematics.keypoint_trajectories["nose"], axis=1)
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]],
        distances[frame_range[0] : frame_range[1]],
        color=(100 / 255, 255 / 255, 149 / 255),
        label="Distance",
    )
    ax.set_title("Distance of Nose to Prey")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Distance (mm)")
    ax.legend()

def plot_head_speed(
    recording_folder: RecordingFolder, frame_range: tuple[int, int] = (0, 1200), ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    kinematics, timestamps = get_head_kinematics_from_recording_folder(recording_folder)
    head_speed = np.linalg.norm(np.diff(kinematics.position_xyz, axis=0), axis=1) / np.diff(timestamps)
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]-1],
        head_speed[frame_range[0] : frame_range[1]-1],
        color=(100 / 255, 255 / 255, 149 / 255),
        label="Distance",
    )
    ax.set_title("Head Speed")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (mm/s)")
    ax.legend()

def plot_prey_speed(
    recording_folder: RecordingFolder, frame_range: tuple[int, int] = (0, 1200), ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    toy_data = load_toy_data_from_tidy_csv(csv_path=recording_folder.toy_resampled_trajectories)
    toy_mean_xyz = np.mean(toy_data, axis=1)

    _, timestamps = get_head_kinematics_from_recording_folder(recording_folder)
    
    toy_xy_speed = np.linalg.norm(np.diff(toy_mean_xyz[:, :2], axis=0), axis=1) / np.diff(timestamps)
    ax.plot(
        timestamps[frame_range[0] : frame_range[1]-1],
        toy_xy_speed[frame_range[0] : frame_range[1]-1],
        color=(100 / 255, 255 / 255, 149 / 255),
        label="Distance",
    )
    ax.set_title("Prey Speed")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (mm/s)")
    ax.legend()


def make_superplot(recording_folder: RecordingFolder):
    fig, axes = plt.subplots(12, 1, figsize=(10, 24))
    frame_range = (1200, 2400)

    plot_head_position(recording_folder=recording_folder, frame_range=frame_range, ax=axes[0])
    plot_head_rotation(recording_folder=recording_folder, frame_range=frame_range, ax=axes[1])
    plot_eye_position(recording_folder=recording_folder, eye_name="left", frame_range=frame_range, ax=axes[2])
    plot_eye_position(recording_folder=recording_folder, eye_name="right", frame_range=frame_range, ax=axes[3])
    plot_gaze_position(recording_folder=recording_folder, eye_name="left", frame_range=frame_range, ax=axes[4])
    plot_gaze_position(recording_folder=recording_folder, eye_name="right", frame_range=frame_range, ax=axes[5])
    plot_gaze_velocity(recording_folder=recording_folder, eye_name="left", frame_range=frame_range, ax=axes[6])
    plot_gaze_velocity(recording_folder=recording_folder, eye_name="right", frame_range=frame_range, ax=axes[7])
    plot_head_velocity(recording_folder=recording_folder, frame_range=frame_range, ax=axes[8])
    plot_prey_distance(recording_folder=recording_folder, frame_range=frame_range, ax=axes[9])
    plot_head_speed(recording_folder=recording_folder, frame_range=frame_range, ax=axes[10])
    plot_prey_speed(recording_folder=recording_folder, frame_range=frame_range, ax=axes[11])

    plt.tight_layout()
    return fig



if __name__ == "__main__":
    recording_folder = RecordingFolder.from_folder_path(
        "/Users/philipqueen/session_2025-07-01_ferret_757_EyeCameras_P33_EO5/clips/1m_20s-2m_20s/"
    )

    fig = make_superplot(recording_folder)
    fig.savefig(f"{recording_folder.recording_name}_superplot.svg", format="svg")
    # plt.show()

    # plot_head_position(recording_folder)
    # plot_head_rotation(recording_folder)
    # plot_eye_position(recording_folder, eye_name="left")
    # plot_eye_position(recording_folder, eye_name="right")
    # plot_gaze_position(recording_folder, eye_name="left")
    # plot_gaze_position(recording_folder, eye_name="right")
    # plot_head_velocity(recording_folder)
    # plot_prey_distance(recording_folder)
    # plot_prey_speed(recording_folder)
    # plt.show()
