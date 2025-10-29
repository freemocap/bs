from pathlib import Path
from pydantic import BaseModel


class TopLevelFolder(BaseModel):
    top_level_folder: Path
    calibration_folder: Path
    base_data_folder: Path
    full_recording_folder: Path
    clips_folder: Path

    @classmethod
    def create_from_top_level_folder(cls, top_level_folder: Path) -> "TopLevelFolder":
        return cls(
            top_level_folder=top_level_folder,
            calibration_folder=top_level_folder / "calibration",
            base_data_folder=top_level_folder / "base_data",
            full_recording_folder=top_level_folder / "full_recordings",
            clips_folder=top_level_folder / "clips",
        )
    
    @property
    def clips(self) -> list[str]:
        return [path.stem for path in self.clips_folder.iterdir() if path.is_dir()]
