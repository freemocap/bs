from pathlib import Path
from typing import Tuple
from enum import Enum

from pydantic import BaseModel


class CalibrationPipelineStep(Enum):
    RAW = "raw"
    SYNCHRONIZED = "synchronized"
    CALIBRATED = "calibrated"


class CalibrationFolder(BaseModel):
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
    def synchronized_videos(self) -> Path | None:
        synchronized_videos = self.folder / "synchronized_videos"
        if not synchronized_videos.exists():
            synchronized_videos = self.folder / "synchronized_corrected_videos"
        
        return synchronized_videos if synchronized_videos.exists() else None

    @property
    def annotated_videos(self) -> Path | None:
        annotated_videos = self.folder / "charuco_annotated_videos"
        return annotated_videos if annotated_videos.exists() else None
    
    @property
    def output_data(self) -> Path | None:
        output_data = self.folder / "output_data"
        return output_data if output_data.exists() else None
    
    @property
    def calibration_toml(self) -> Path | None:
        toml = self.folder.glob("*camera_calibration.toml")
        return next(toml, None) if toml else None

    def check_synchronization(self):
        if self.synchronized_videos is None:
            raise ValueError("Synchronization failed, no synchronized videos found")

    def check_calibration(self, enforce_annotated: bool = True):
        try:
            self.check_synchronization()
        except ValueError as e:
            raise ValueError(
                f"Folder is not synchronized: {e}"
            )

        if self.calibration_toml is None:
            raise ValueError("Calibration failed, no calibration toml found")

        if self.output_data is None:
            raise ValueError("Calibration failed, no output data found")

        if not (self.output_data / "charuco_3d_xyz.npy").exists():
            raise ValueError("Calibration failed, no charuco_3d_xyz.npy found in output data folder")

        if enforce_annotated:
            if self.annotated_videos is None:
                raise ValueError("Calibration failed, no annotated videos found")

    def csv_report(self):
        pass # TODO: implement a csv report that can be passed into a dataframe easily

if __name__ == "__main__":
    CalibrationFolder.from_folder_path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-18_ferret_420_E09/calibration",
        expected_processing_step=CalibrationPipelineStep.CALIBRATED
    )
