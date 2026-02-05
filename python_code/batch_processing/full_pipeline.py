"""
process the entire pipeline in one go
use boolean parameters to turn steps on and off
"""
from pathlib import Path

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def full_pipeline(
    recording_folder: Path,
    overwrite_synchronization: bool = False,
    overwrite_dlc: bool = False,
    overwrite_triangulation: bool = False,
    overwrite_eye_postprocessing: bool = False,
    overwrite_skull_postprocessing: bool = False,
    overwrite_gaze: bool = False
):
    pass

if __name__=="__main__":
    recording_folder_path = Path(
        ""
    )
    recording_folder = RecordingFolder.from_folder_path(folder=recording_folder_path)

    try:
        recording_folder.check_synchronization()
        synchronized = True
    except ValueError:
        synchronized = False