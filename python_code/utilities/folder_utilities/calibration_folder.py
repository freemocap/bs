from pathlib import Path
from typing import Tuple
from enum import Enum

from pydantic import BaseModel


class CalibrationPipelineStep(Enum):
    RAW = "raw"
    SYNCHRONIZED = "synchronized"
    CALIBRATED = "calibrated"


class RecordingFolder(BaseModel):
    folder: Path
    base_recordings_folder: Path
    recording_name: str
    processing_step: CalibrationPipelineStep = CalibrationPipelineStep.RAW

    @classmethod
    def from_folder_path(cls, folder: Path | str, expected_processing_step: CalibrationPipelineStep = CalibrationPipelineStep.RAW) -> "RecordingFolder":
        folder = Path(folder)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")
        if not folder.is_dir():
            raise ValueError(f"Folder is not a directory: {folder}")

        if folder.name != "calibration":
            raise ValueError("Must create calibration folder from '../calibration' path")

        base_recordings_folder = folder.parent
        recording_name = base_recordings_folder.name
        recording_folder = cls(
            folder=folder,
            base_recordings_folder=base_recordings_folder,
            recording_name=recording_name,
        )

        match expected_processing_step:
            case CalibrationPipelineStep.CALIBRATED:
                try:
                    recording_folder.check_calibration()
                    recording_folder.processing_step = CalibrationPipelineStep.CALIBRATED
                    print(f"Folder is calibrated: {folder}")
                except ValueError as e:
                    print(f"Folder is not calibrated: {e}")
                    raise ValueError(
                        f"Folder is not calibrated: {folder}"
                    )
            case CalibrationPipelineStep.SYNCHRONIZED:
                try:
                    recording_folder.check_synchronization()
                    recording_folder.processing_step = CalibrationPipelineStep.SYNCHRONIZED
                    print(f"Folder is synchronized: {folder}")
                except ValueError as e:
                    print(f"Folder is not synchronized: {e}")
                    raise ValueError(
                        f"Folder is not synchronized: {folder}"
                    )
            case CalibrationPipelineStep.RAW:
                pass
            case _:
                raise ValueError(f"Unknown processing step: {expected_processing_step}")

        return recording_folder

    @property
    def raw_videos(self) -> Path:
        return self.folder / "raw_videos"

    @property
    def synchronized_videos(self) -> Path:
        pass

    def check_synchronization(self):
        for name, path in {
            "eye_videos_folder": self.eye_videos,
            "left_eye_video": self.left_eye_video,
            "left_eye_timestamps_npy": self.left_eye_timestamps_npy,
            "right_eye_video": self.right_eye_video,
            "right_eye_timestamps_npy": self.right_eye_timestamps_npy,
            "mocap_synchronized_videos": self.mocap_synchronized_videos,
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, synchronization failed")
            
        for video in [
            BaslerCamera.TOPDOWN.value,
            BaslerCamera.SIDE_0.value,
            BaslerCamera.SIDE_1.value,
            BaslerCamera.SIDE_2.value,
            BaslerCamera.SIDE_3.value,
        ]:
            try:
                self.get_synchronized_video_by_name(video)
            except ValueError:
                raise ValueError(f"Could not find synchronized video for {video} in {self.mocap_synchronized_videos}")
            try:
                self.get_timestamp_by_name(video)
            except ValueError:
                raise ValueError(f"Could not find timestamp for {video} in {self.mocap_synchronized_videos}")
            try:
                self.get_annotated_video_by_name(video)
            except ValueError:
                raise ValueError(f"Could not find annotated video for {video} in {self.head_body_annotated_videos}")
            

    def check_dlc_output(self, enforce_toy: bool = True, enforce_annotated: bool = True):
        try:
            self.check_synchronization()
        except ValueError:
            raise ValueError("Synchronization failed, dlc output cannot be checked")

        for name, path in {
            "eye_dlc_output_folder": self.eye_dlc_output,
            "eye_dlc_output_flipped_folder": self.eye_dlc_output_flipped,
            "eye_skellyclicker_labels": self.eye_data_skellyclicker_labels,
            "eye_skellyclicker_labels_flipped": self.eye_data_skellyclicker_labels_flipped,
            "mocap_dlc_output_folder": self.head_body_dlc_output,
            "mocap_skellyclicker_labels": self.head_body_skellyclicker_labels
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, dlc output failed")
            
        if enforce_toy:
            for name, path in {
                "toy_dlc_output_folder": self.toy_dlc_output,
                "toy_skellyclicker_labels": self.toy_skellyclicker_labels
            }.items():
                if path is None:
                    raise ValueError(f"{name} does not exist, dlc output failed")
        
        if enforce_annotated:
            for name, path in {
                "left_eye_annotated_video": self.left_eye_annotated_video,
                "left_eye_annotated_flipped_video": self.left_eye_annotated_flipped_video,
                "right_eye_annotated_video": self.right_eye_annotated_video,
                "right_eye_annotated_flipped_video": self.right_eye_annotated_flipped_video,
            }.items():
                if path is None:
                    raise ValueError(f"{name} does not exist, dlc output failed")
            for video in [
                BaslerCamera.TOPDOWN.value,
                BaslerCamera.SIDE_0.value,
                BaslerCamera.SIDE_1.value,
                BaslerCamera.SIDE_2.value,
                BaslerCamera.SIDE_3.value,
            ]:
                try:
                    self.get_annotated_video_by_name(video)
                except ValueError:
                    raise ValueError(f"Could not find annotated video for {video} in {self.head_body_annotated_videos}")

    def check_synchronization(self, enforce_toy: bool = True, enforce_annotated: bool = True):
        try:
            self.check_dlc_output(enforce_toy=enforce_toy)
        except ValueError as e:
            print(f"DLC output failed with error: {e}")
            raise ValueError("DLC output failed, triangulation cannot be checked")

        for name, path in {
            "output_data_folder": self.mocap_output_data,
            "mocap_3d_data": self.mocap_3d_data,
            "head_body_3d_xyz.csv": self.mocap_3d_data / "head_body_3d_xyz.csv" if self.mocap_3d_data else None,
            "head_body_3d_xyz.npy": self.mocap_3d_data / "head_body_3d_xyz.npy" if self.mocap_3d_data else None,
            "head_body_rigid_3d_xyz.csv": self.mocap_3d_data / "head_body_rigid_3d_xyz.csv" if self.mocap_3d_data else None,
            "head_body_rigid_3d_xyz.npy": self.mocap_3d_data / "head_body_rigid_3d_xyz.npy" if self.mocap_3d_data else None,
            "head_freemocap_data_by_frame.csv": self.mocap_3d_data / "head_freemocap_data_by_frame.csv" if self.mocap_3d_data else None,
            "head_freemocap_data_by_frame.parquet": self.mocap_3d_data / "head_freemocap_data_by_frame.parquet" if self.mocap_3d_data else None,
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, triangulation failed")

        if enforce_toy:
            for name, path in {
                "toy_body_3d_xyz.csv": self.mocap_3d_data / "head_body_3d_xyz.csv" if self.mocap_3d_data else None,
                "toy_body_3d_xyz.npy": self.mocap_3d_data / "head_body_3d_xyz.npy" if self.mocap_3d_data else None,
                "toy_body_rigid_3d_xyz.csv": self.mocap_3d_data / "head_body_rigid_3d_xyz.csv" if self.mocap_3d_data else None,
                "toy_body_rigid_3d_xyz.npy": self.mocap_3d_data / "head_body_rigid_3d_xyz.npy" if self.mocap_3d_data else None,
                "toy_freemocap_data_by_frame.csv": self.mocap_3d_data / "head_freemocap_data_by_frame.csv" if self.mocap_3d_data else None,
                "toy_freemocap_data_by_frame.parquet": self.mocap_3d_data / "head_freemocap_data_by_frame.parquet" if self.mocap_3d_data else None,
            }.items():
                if path is None:
                    raise ValueError(f"{name} does not exist, triangulation failed")
        

    def check_calibration(self, enforce_toy: bool = True, enforce_annotated: bool = True):
        try:
            self.check_synchronization(enforce_toy=enforce_toy, enforce_annotated=enforce_annotated)
        except ValueError as e:
            print(f"Triangulation failed with error: {e}")
            raise ValueError("Triangulation failed, postprocessing cannot be checked")

        for name, path in {
            "solver_output_folder": self.mocap_solver_output,
            "reference_geometry": self.mocap_solver_output / "reference_geometry.json" if self.mocap_solver_output else None,
            "reference_geometry_skull": self.mocap_solver_output / "reference_geometry_skull.json" if self.mocap_solver_output else None,
            "rotation_translation_data.csv": self.mocap_solver_output / "rotation_translation_data.csv" if self.mocap_solver_output else None,
            "tidy_trajectory_data.csv": self.mocap_solver_output / "tidy_trajectory_data.csv" if self.mocap_solver_output else None,
            "trajectory_data.csv": self.mocap_solver_output / "trajectory_data.csv" if self.mocap_solver_output else None,
            "topology.json": self.mocap_solver_output / "topology.json" if self.mocap_solver_output else None,
            "metrics.json": self.mocap_solver_output / "metrics.json" if self.mocap_solver_output else None,
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, head solver failed")
            
        for name, path in {
            "eye_data.csv": self.eye_data_csv,
            "eye_model_v3_mean_confidence.csv": self.eye_mean_confidence,
            "left_eye_plot_points.csv": self.left_eye_plot_points_csv,
            "right_eye_plot_points.csv": self.right_eye_plot_points_csv,
            "left_eye_stabilized_canvas.mp4": self.left_eye_stabilized_canvas,
            "right_eye_stabilized_canvas.mp4": self.right_eye_stabilized_canvas,
            "eye_output_data": self.eye_output_data,
            "eye0_alignment_summary.json": self.eye_output_data / "eye0_alignment_summary.json" if self.eye_output_data else None,
            "eye1_alignment_summary.json": self.eye_output_data / "eye1_alignment_summary.json" if self.eye_output_data else None,
            "eye0_data.csv": self.eye_output_data / "eye0_data.csv" if self.eye_output_data else None,
            "eye1_data.csv": self.eye_output_data / "eye1_data.csv" if self.eye_output_data else None,
            "eye0_correction_comparison.png": self.eye_output_data / "eye0_correction_comparison.png" if self.eye_output_data else None,
            "eye1_correction_comparison.png": self.eye_output_data / "eye1_correction_comparison.png" if self.eye_output_data else None,
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, eye postprocessing failed")


    def csv_report(self):
        pass # TODO: implement a csv report that can be passed into a dataframe easily

if __name__ == "__main__":
    RecordingFolder.from_folder_path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-18_ferret_420_E09/full_recording",
        expected_processing_step=PipelineStep.POST_PROCESSED
    )
