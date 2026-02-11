import numpy as np

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import EyeViewerData, get_eye_radius_from_kinematics, log_static_world_frame, log_timeseries_accelerations, log_timeseries_angles, log_timeseries_velocities, set_time_seconds
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import extract_frame_data, load_eye_trajectories_csv
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


def plot_eye_traces(
    eye_name: str,
    recording_folder: RecordingFolder,
):
    if eye_name not in ["left", "right"]:
        raise ValueError(f"Invalid eye name: {eye_name} - expected 'left' or 'right'")
    
    if eye_name == "left":
        trajectory_csv_path = recording_folder.left_eye_data_csv
        video_path = recording_folder.left_eye_video
        if "757" in str(video_path):
            video_name = "eye0"
        else:
            video_name = "eye1"
    else:
        trajectory_csv_path = recording_folder.right_eye_data_csv
        video_path = recording_folder.right_eye_video
        if "757" in str(video_path):
            video_name = "eye1"
        else:
            video_name = "eye0"


    try:
        kinematics = FerretEyeKinematics.load_from_directory(eye_name=f"{eye_name}_eye", input_directory=recording_folder.eye_output_data)
        left_pixel_data = None
        if trajectory_csv_path is not None:
            try:
                df = load_eye_trajectories_csv(
                    csv_path=trajectory_csv_path, eye_side=eye_name, video_name=video_name
                )
                timestamps, pupil_centers_px, *_ = extract_frame_data(df)
                left_pixel_data = {
                    "pupil_center_x": pupil_centers_px[:, 0],
                    "pupil_center_y": pupil_centers_px[:, 1],
                }
                print(f"Loaded left eye pixel data: {len(timestamps)} frames")
            except Exception as e:
                print(f"Warning: Could not load left eye pixel data: {e}")
        timestamps = kinematics.timestamps
        print(f"Loaded left eye kinematics: {kinematics.n_frames} frames")
    except FileNotFoundError:
        print("Left eye kinematics not found, skipping...")

    eye_radius = get_eye_radius_from_kinematics(kinematics)
    log_static_world_frame(eye_name, eye_radius * 1.5, eye_radius)
    for i in range(kinematics.n_frames):
        set_time_seconds("time", timestamps[i])
        adduction_deg = np.degrees(kinematics.adduction_angle.values[i])
        elevation_deg = np.degrees(kinematics.elevation_angle.values[i])
        adduction_vel = np.degrees(kinematics.adduction_velocity.values[i])
        elevation_vel = np.degrees(kinematics.elevation_velocity.values[i])
        adduction_acc = np.degrees(kinematics.adduction_acceleration.values[i])
        elevation_acc = np.degrees(kinematics.elevation_acceleration.values[i])

        log_timeseries_angles(eye_name, adduction_deg, elevation_deg)
        log_timeseries_velocities(eye_name, adduction_vel, elevation_vel)
        log_timeseries_accelerations(eye_name, adduction_acc, elevation_acc)