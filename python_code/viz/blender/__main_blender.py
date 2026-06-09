from pathlib import Path

import pandas as pd
from freemocap.core.pipeline.posthoc.video_group_helper import VideoGroupHelper
from freemocap.core.tasks.calibration.shared.calibration_result import CalibrationResult

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import \
    load_ferret_eye_kinematics_from_directory
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

# TODO - Shouldn't need to target the INNER `full_recording` folder
RECORDING_PATH = Path("/media/jon-alien-pop/DATA/bs/session_2025-10-22_ferret_420_EO13/session_2025-10-22_ferret_420_EO13/full_recording")

# Load recording data
recording = RecordingFolder.from_folder_path(RECORDING_PATH)


## Camera Calibration data (via FMC CameraModels)
calibration_toml_path = recording.calibration_toml_path
if calibration_toml_path is None:
    raise ValueError("Calibration file not found")
calibration = CalibrationResult.load_anipose_toml(calibration_toml_path)

cameras = calibration.cameras

timestamps = recording.common_timestamps

## Synchronized Videos  (via FMC VideoGroup)
mocap_videos_folder_path = recording.mocap_synchronized_videos
display_videos_path = recording.display_videos
if mocap_videos_folder_path is None:
    raise ValueError("Mocap synchronized video not found")
if display_videos_path is None:
    raise ValueError("Display videos path not found")
#TODO - Why is everything optional???
mocap_videos = VideoGroupHelper.from_video_folder_path(mocap_videos_folder_path)
display_videos = VideoGroupHelper.from_video_folder_path(display_videos_path)

## Trajectories
### Skull&Spine
skull_and_spine_trajectories = pd.read_csv( recording.skull_and_spine_resampled_trajectories)

### Toy
toy_trajectories = pd.read_csv( recording.toy_resampled_trajectories)

## RigidBodies
### Skull
skull_dir = recording.skull_kinematics
if not skull_dir:
    raise ValueError("Skull kinematics not found")
#TODO - Again with the optionals :'(

#TODO - This should be a factory method on the
skull = RigidBodyKinematics.load_from_disk(
    reference_geometry_json_path=skull_dir / "skull_reference_geometry.json",
    kinematics_csv_path=skull_dir / "skull_kinematics.csv",
)

#% ## Eyes

right_eye_kinematics_folder = recording.right_eye_kinematics
left_eye_kinematics_folder = recording.left_eye_kinematics

if left_eye_kinematics_folder is None:
    raise ValueError("Left eye kinematics not found")
if right_eye_kinematics_folder is None:
    raise ValueError("Right eye kinematics not found")

# TODO - This should be an easier-to-use factory method on the FerretEyeKinematics object
left_eye = load_ferret_eye_kinematics_from_directory(
    input_directory=left_eye_kinematics_folder,
    eye_name="left_eye",
)

right_eye = load_ferret_eye_kinematics_from_directory(
    input_directory=right_eye_kinematics_folder,
    eye_name="right_eye",
)

print('Done!')