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


def _run_subprocess_streaming(command_list: list, clean_env: dict, use_pty: bool = False) -> None:
    """
    Run a subprocess and stream its output in real time, raising on non-zero exit.

    Args:
        command_list: Command and arguments to run.
        clean_env: Environment dict for the subprocess.
        use_pty: If True, allocate a pseudo-terminal so the child process believes
                 it is writing to a real terminal.  This keeps tqdm and other
                 progress-bar libraries in single-line overwrite mode (\\r) instead
                 of printing every update as a new line.
    """
    if use_pty and sys.platform != "win32":
        import pty
        import fcntl
        import termios
        import struct
        master_fd, slave_fd = pty.openpty()
        # Match the PTY window size to the parent terminal so tqdm sizes its
        # bar correctly.  Falls back silently if stdout is not a TTY.
        try:
            term_size = os.get_terminal_size(sys.stdout.fileno())
            winsize = struct.pack("HHHH", term_size.lines, term_size.columns, 0, 0)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass
        process = subprocess.Popen(
            command_list,
            env=clean_env,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=subprocess.DEVNULL,
        )
        os.close(slave_fd)
        try:
            while True:
                try:
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()
                except OSError:
                    # Raised when the slave end is closed (process exited)
                    break
        finally:
            os.close(master_fd)
    else:
        process = subprocess.Popen(
            command_list,
            env=clean_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command_list[0])


def run_skellyclicker_subprocess(
        recording_folder_path: Path,
        venv_path: str = "/home/scholl-lab/anaconda3/envs/skellyclicker/bin/python",
        script_path: str = "/home/scholl-lab/skellyclicker/skellyclicker/scripts/process_recording.py",
        include_eye: bool = True,):
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    clean_env.pop("VIRTUAL_ENV", None)

    command_list = [venv_path, "-u", script_path, recording_folder_path]
    if not include_eye:
        command_list.append("--skip-eye")

    _run_subprocess_streaming(command_list, clean_env, use_pty=True)


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

    _run_subprocess_streaming(command_list, clean_env)


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

    _run_subprocess_streaming(command_list, clean_env)



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
    if overwrite_synchronization or not recording_folder.is_synchronized():
        print(f"Synchronizing videos at {recording_folder.base_recordings_folder}")
        postprocess(session_folder_path=recording_folder.base_recordings_folder, include_eyes=include_eye)
    recording_folder.check_synchronization()
    print("Synchronizing videos completed")

    # Calibration
    if overwrite_calibration or not recording_folder.is_calibrated():
        print("Calibrating session...")
        run_calibration_subprocess(calibration_videos_path=recording_folder.calibration_videos)
    recording_folder.check_calibration()
    print("Calibration complete")

    # DLC
    if overwrite_dlc or not recording_folder.is_dlc_processed():
        print("Running pose estimation...")
        run_skellyclicker_subprocess(recording_folder_path=recording_folder_path)
    recording_folder.check_dlc_output()
    print("Pose estimation complete")

    # Triangulation
    if overwrite_triangulation or not recording_folder.is_triangulated():
        if calibration_toml_path is None:
            calibration_toml_path = recording_folder.calibration_toml_path
        if calibration_toml_path is None:
            raise ValueError("No calibration toml file found, could not run triangulation")
        print("Running triangulation...")
        run_triangulation_subprocess(recording_folder_path=recording_folder_path, calibration_toml_path=calibration_toml_path)
    recording_folder.check_triangulation()
    print("Triangulation complete")

    eye_postprocessing = recording_folder.is_eye_postprocessed()
    skull_postprocessing = recording_folder.is_skull_postprocessed()
    gaze_postprocessing = recording_folder.is_gaze_postprocessed()
    

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
        "/home/scholl-lab/ferret_recordings/session_2025-10-22_ferret_420_EO13/full_recording"
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
        overwrite_skull_postprocessing=False,
        overwrite_gaze=True
    )
