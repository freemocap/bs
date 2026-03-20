"""
process the entire pipeline in one go
use boolean parameters to turn steps on and off

requires the following repos/bracnhes installed:
    skellyclicker: https://github.com/freemocap/skellyclicker
    dlc_to_3d: https://github.com/philipqueen/freemocap_playground@philip/bs
    freemocap: https://github.com/freemocap/freemocap
"""
from pathlib import Path
import subprocess
import os
import sys

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

    command_list = [venv_path, "-u",script_path, recording_folder_path]
    if not include_eye:
        command_list.append("--skip-eye")

    process = subprocess.Popen(
        command_list,
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    while True:
        char = process.stdout.read(1)
        if not char:
            break
        sys.stdout.write(char)
        sys.stdout.flush()

    process.wait()


def run_triangulation_subprocess(
        recording_folder_path: Path,
        calibration_toml_path: Path,
        venv_path: str = "/home/scholl-lab/Documents/git_repos/dlc_to_3d/.venv/bin/python",
        script_path: str = "/home/scholl-lab/Documents/git_repos/dlc_to_3d/dlc_reconstruction/dlc_to_3d.py",
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


def run_calibration_subprocess(
        calibration_videos_path: Path,
        venv_path: str = "/home/scholl-lab/anaconda3/envs/fmc/bin/python",
        script_path: str = "/home/scholl-lab/Documents/git_repos/freemocap/experimental/batch_process/headless_calibration.py",
    ):

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    clean_env.pop("VIRTUAL_ENV", None)

    command_list = [
        venv_path, 
        script_path, 
        calibration_videos_path,
        "--square-size",
        "57",
        "--5x3",
        "--use-groundplane"
    ]

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
    overwrite_calibration: bool = False,
    overwrite_dlc: bool = False,
    overwrite_triangulation: bool = False,
    overwrite_eye_postprocessing: bool = False,
    overwrite_skull_postprocessing: bool = False,
    overwrite_gaze: bool = False
):
    recording_folder = RecordingFolder.from_folder_path(folder=recording_folder_path)

    # Propagate overwrite flags through dependent steps
    if overwrite_synchronization:
        overwrite_calibration = True

    if overwrite_dlc:
        overwrite_eye_postprocessing = True
        if overwrite_calibration:
            overwrite_triangulation = True

    if overwrite_triangulation:
        overwrite_skull_postprocessing = True

    if overwrite_eye_postprocessing or overwrite_skull_postprocessing:
        overwrite_gaze = True

    # Synchronization
    try:
        recording_folder.check_synchronization()
        synchronized = True
    except ValueError as e:
        print(f"Session not synchronized: {e}")
        synchronized = False
    if overwrite_synchronization or synchronized is False:
        print(f"Synchronizing videos at {recording_folder.base_recordings_folder}")
        postprocess(session_folder_path=recording_folder.base_recordings_folder, include_eyes=include_eye)
    recording_folder.check_synchronization()
    print("Synchronizing videos completed")

    # Calibration
    try:
        recording_folder.check_calibration()
        calibrated = True
    except ValueError as e:
        print(f"Session not calibrated: {e}")
        calibrated = False
    if overwrite_calibration or calibrated is False:
        print("Calibrating session...")
        run_calibration_subprocess(calibration_videos_path=recording_folder.calibration_videos)
    recording_folder.check_calibration()
    print("Calibration complete")

    # DLC
    try:
        recording_folder.check_dlc_output()
        dlc_output = True
    except ValueError as e:
        print("DLC not processed")
        dlc_output = False
    if overwrite_dlc or dlc_output is False:
        print("Running pose estimation...")
        run_skellyclicker_subprocess(recording_folder_path=recording_folder_path)
    recording_folder.check_dlc_output()
    print("Pose estimation complete")

    # Triangulation
    try:
        recording_folder.check_triangulation()
        triangulation = True
    except ValueError as e:
        triangulation = False
    if overwrite_triangulation or triangulation is False:
        if calibration_toml_path is None:
            calibration_toml_path = recording_folder.calibration_toml_path
        if calibration_toml_path is None:
            raise ValueError("No calibration toml file found, could not run triangulation")
        print("Running triangulation...")
        run_triangulation_subprocess(recording_folder_path=recording_folder_path, calibration_toml_path=calibration_toml_path)
    recording_folder.check_triangulation()
    print("Triangulation complete")

    # Eye postprocessing
    try:
        recording_folder.check_eye_postprocessing()
        eye_postprocessing = True
    except ValueError as e:
        eye_postprocessing = False

    # Skull postprocessing
    try:
        recording_folder.check_skull_postprocessing()
        skull_postprocessing = True
    except ValueError as e:
        skull_postprocessing = False


    # Gaze
    try:
        recording_folder.check_gaze_postprocessing()
        gaze_postprocessing = True
    except ValueError as e:
        gaze_postprocessing = False
    

    run_eye_postprocessing = include_eye and (overwrite_eye_postprocessing or not eye_postprocessing)
    run_skull_postprocessing = overwrite_skull_postprocessing or not skull_postprocessing
    run_gaze_postprocessing = include_eye and (overwrite_gaze or not gaze_postprocessing)
    if run_eye_postprocessing or run_skull_postprocessing or run_gaze_postprocessing: 
        print("Running gaze processing...")
        process_recording(
            recording_folder=recording_folder,
            skip_eye=not run_eye_postprocessing,
            skip_skull=not run_skull_postprocessing,
            skip_gaze=not run_gaze_postprocessing
        )
    recording_folder.check_eye_postprocessing()
    recording_folder.check_skull_postprocessing()
    recording_folder.check_gaze_postprocessing()
    print("Gaze calculations complete")
    print(f"Session processed: {recording_folder_path}")


if __name__=="__main__":
    recording_folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )

    if "clips" not in str(recording_folder_path) and "full_recording" not in str(recording_folder_path):
        recording_folder_path = recording_folder_path / "full_recording"


    recording_folder_path.mkdir(exist_ok=True, parents=False)
    (recording_folder_path / "mocap_data").mkdir(exist_ok=True, parents=False)
    (recording_folder_path / "eye_data").mkdir(exist_ok=True, parents=False)
    print(f"Processing {recording_folder_path}")

    full_pipeline(
        recording_folder_path=recording_folder_path,
        overwrite_synchronization=False,
        overwrite_calibration=False,
        overwrite_dlc=False,
        overwrite_triangulation=False,
        overwrite_eye_postprocessing=True,
        overwrite_skull_postprocessing=True,
        overwrite_gaze=True
    )
