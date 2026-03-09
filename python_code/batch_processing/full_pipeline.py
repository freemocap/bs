"""
process the entire pipeline in one go
use boolean parameters to turn steps on and off
"""
from pathlib import Path
import subprocess
import os

from python_code.batch_processing.postprocess_recording import process_recording
from python_code.cameras.postprocess import postprocess
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

def run_skellyclicker_subprocess(
        recording_folder_path: Path,
        venv_path: str = "/home/scholl-lab/anaconda3/envs/skellyclicker/bin/python",
        script_path: str = "/home/scholl-lab/skellyclicker/skellyclicker/scripts/process_recording.py",
        include_eye: bool = True,):
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    clean_env.pop("VIRTUAL_ENV", None)

    command_list = [venv_path, script_path, recording_folder_path]
    if not include_eye:
        command_list.append("--skip-eye")

    result = subprocess.run(
        command_list,
        env=clean_env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Script failed!")
        print(result.stderr)
    else:
        print(result.stdout)


def run_triangulation_subprocess(
        recording_folder_path: Path,
        calibration_toml_path: Path,
        venv_path: str = "/home/scholl-lab/Documents/git_repos/dlc_to_3d/.venv/bin/python",
        script_path: str = "/home/scholl-lab/Documents/git_repos/dlc_to_3d/dlc_reconstruction/dlc_to_3d.py",,
        skip_toy: bool = False
    ):

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    clean_env.pop("VIRTUAL_ENV", None)

    command_list = [venv_path, script_path, recording_folder_path, calibration_toml_path]
    if skip_toy:
        command_list.append("--skip-toy")

    result = subprocess.run(
        command_list,
        env=clean_env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Script failed!")
        print(result.stderr)
    else:
        print(result.stdout)



def full_pipeline(
    recording_folder_path: Path,
    calibration_toml_path: Path | None = None,
    include_eye: bool = True,
    overwrite_synchronization: bool = False,
    overwrite_dlc: bool = False,
    overwrite_triangulation: bool = False,
    overwrite_eye_postprocessing: bool = False,
    overwrite_skull_postprocessing: bool = False,
    overwrite_gaze: bool = False
):
    recording_folder = RecordingFolder.from_folder_path(folder=recording_folder_path)

    # Synchronization
    try:
        recording_folder.check_synchronization()
        synchronized = True
    except:
        synchronized = False
    if overwrite_synchronization or synchronized is False:
        postprocess(session_folder_path=recording_folder.base_recordings_folder, include_eyes=include_eye)

    # DLC
    try:
        recording_folder.check_dlc_output()
        dlc_output = True
    except:
        dlc_output = False
    if overwrite_dlc or dlc_output is False:
        pass

    # Triangulation
    try:
        recording_folder.check_triangulation()
        triangulation = True
    except:
        triangulation = False
    if overwrite_triangulation or triangulation is False:
        if calibration_toml_path is None:
            calibration_toml_path = recording_folder.calibration_toml_path
        if calibration_toml_path is None:
            raise ValueError("No calibration toml file found, could not run triangulation")
        run_triangulation_subprocess(recording_folder_path=recording_folder_path, calibration_toml_path=calibration_toml_path)

    # Eye postprocessing
    try:
        recording_folder.check_eye_postprocessing()
        eye_postprocessing = True
    except:
        eye_postprocessing = False

    # Skull postprocessing
    try:
        recording_folder.check_skull_postprocessing()
        skull_postprocessing = True
    except:
        skull_postprocessing = False

    # Gaze
    try:
        recording_folder.check_gaze_postprocessing()
        gaze_postprocessing = True
    except:
        gaze_postprocessing = False

    run_eye_postprocessing = include_eye and (overwrite_eye_postprocessing or eye_postprocessing is False)
    run_skull_postprocessing = overwrite_skull_postprocessing or skull_postprocessing is False
    run_gaze_postprocessing = include_eye and (overwrite_gaze or gaze_postprocessing is False)
    process_recording(
        recording_folder=recording_folder,
        skip_eye=not run_eye_postprocessing,
        skip_skull=not run_skull_postprocessing,
        skip_gaze=not run_gaze_postprocessing
    )


if __name__=="__main__":
    recording_folder_path = Path(
        ""
    )

    full_pipeline(
        recording_folder_path=recording_folder_path,
        overwrite_synchronization=False,
        overwrite_dlc=True,
        overwrite_triangulation=True,
        overwrite_eye_postprocessing=True,
        overwrite_skull_postprocessing=True,
        overwrite_gaze=True
    )
