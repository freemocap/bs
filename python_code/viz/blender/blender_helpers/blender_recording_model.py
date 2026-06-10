from pathlib import Path

import numpy as np
import polars as pl
from freemocap.core.pipeline.posthoc.video_group_helper import VideoGroupHelper
from freemocap.core.tasks.calibration.shared.calibration_result import CalibrationResult
from polars import DataFrame
from pydantic import BaseModel, model_validator

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import \
    load_ferret_eye_kinematics_from_directory
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


class BlenderVideoGroups(BaseModel):
    mocap_videos: VideoGroupHelper
    display_videos: VideoGroupHelper

    @model_validator(mode="after")
    def model_validate_frame_count(self):
        self._validate_frame_count()
        return self

    def validate_frame_count(self)->int:
        frame_counts = {self.mocap_videos.frame_count, self.display_videos.frame_count}
        if len(set(frame_counts)) != 1:
            raise ValueError("All videos in VideoGroup must have the same frame count.")
        return list(frame_counts)[0]

    @property
    def frame_count(self) -> int:
        return self._validate_frame_count()

class BlenderData(BaseModel):
    timestamps: np.ndarray
    calibration: CalibrationResult
    skull_and_spine_trajectories_df: DataFrame  # Replace with friendlier class, TrajectoryModel or similar (in kinematics_core)
    toy_trajectories_df: DataFrame  # Ditto
    skull_kinematics: RigidBodyKinematics
    right_eye_kinematics: FerretEyeKinematics
    left_eye_kinematics: FerretEyeKinematics

    @model_validator(mode="after")
    def model_validate_frame_count(self):
        self.validate_frame_count()

    def validate_frame_count(self)->int:
        frame_counts = {self.timestamps.shape[0],
                        self.skull_and_spine_trajectories_df.shape[0],
                        self.toy_trajectories_df.shape[0],
                        self.skull_kinematics.n_frames,
                        self.right_eye_kinematics.n_frames,
                        self.left_eye_kinematics.n_frames,}

        if len(set(frame_counts)) != 1:
            raise ValueError("All videos in VideoGroup must have the same frame count.")
        return list(frame_counts)[0]


    @property
    def frame_count(self) -> int:
        return self.validate_frame_count()


class BlenderRecording(BaseModel):
    recording_path: Path
    videos: BlenderVideoGroups
    data: BlenderData

    @property
    def name(self)->str:
        return self.recording_path.stem

    @property
    def frame_count(self) -> int:
        return self.validate_frame_count()

    @model_validator(mode="after")
    def model_validate_frame_count(self):
        self.validate_frame_count()

    @model_validator(mode="after")
    def model_validate_recording_path(self):
        if not Path(self.recording_path).exists():
            raise ValueError(f"Recording path does not exist at: {self.recording_path}")

    def validate_frame_count(self) -> int:
        frame_counts = {self.videos.frame_count, self.data.frame_count }
        if len(set(frame_counts)) != 1:
            raise ValueError("All videos in VideoGroup must have the same frame count.")
        return list(frame_counts)[0]



    @classmethod
    def from_recording_path(cls, recording_path: str | Path) -> 'BlenderRecording':
        # Load recording data
        recording_path = Path(recording_path)
        recording = RecordingFolder.from_folder_path(recording_path)

        ## Camera Calibration data (via FMC CameraModels)
        calibration_toml_path = recording.calibration_toml_path
        if calibration_toml_path is None:
            raise ValueError("Calibration file not found")
        calibration = CalibrationResult.load_anipose_toml(calibration_toml_path)


        timestamps = np.load(str(recording.common_timestamps))

        ## Synchronized Videos  (via FMC VideoGroup)
        mocap_videos_folder_path = recording.mocap_synchronized_videos
        display_videos_path = recording.display_videos
        if mocap_videos_folder_path is None:
            raise ValueError("Mocap synchronized video not found")
        if display_videos_path is None:
            raise ValueError("Display videos path not found")
        # TODO - Why is everything optional???
        mocap_videos = VideoGroupHelper.from_video_folder_path(mocap_videos_folder_path)
        display_videos = VideoGroupHelper.from_video_folder_path(display_videos_path)

        ## Trajectories
        ### Skull&Spine

        if recording.skull_and_spine_resampled_trajectories is None:
            raise ValueError("Skull kinematics not found")
        skull_and_spine_trajectories_csv:Path = recording.skull_and_spine_resampled_trajectories

        skull_and_spine_trajectories = pl.read_csv(skull_and_spine_trajectories_csv)

        ### Toy
        if recording.toy_resampled_trajectories is None:
            raise ValueError("Skull kinematics not found")
        toy_trajectories_csv:Path = recording.toy_resampled_trajectories
        toy_trajectories = pl.read_csv(toy_trajectories_csv)

        ## RigidBodies
        ### Skull
        skull_dir = recording.skull_kinematics
        if not skull_dir:
            raise ValueError("Skull kinematics not found")
        # TODO - Again with the gd optional everything :'(

        skull = RigidBodyKinematics.load_from_disk(
            reference_geometry_json_path=skull_dir / "skull_reference_geometry.json",
            kinematics_csv_path=skull_dir / "skull_kinematics.csv",
        )

        # % ## Eyes

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
        return cls(
            videos=BlenderVideoGroups(
                mocap_videos=mocap_videos,
                display_videos=display_videos,
            ),
            data=BlenderData(
                timestamps=timestamps,
                calibration=calibration,
                skull_and_spine_trajectories_df=skull_and_spine_trajectories,
                toy_trajectories_df=toy_trajectories,
                skull_kinematics=skull,
                right_eye_kinematics=right_eye,
                left_eye_kinematics=left_eye,
            ),
        )
