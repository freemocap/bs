# processes everything after model inference
from pathlib import Path
import sys

from python_code.eye_analysis.process_eye_session import process_eye_session_from_recording_folder
from python_code.ferret_gaze.data_resampling.data_resampling_helpers import ResamplingStrategy
from python_code.ferret_gaze.run_gaze_pipeline import run_gaze_pipeline
from python_code.rigid_body_solver.ferret_skull_solver import run_ferret_skull_solver_from_recording_folder
from python_code.utilities.find_bad_eye_data import bad_eye_data
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


def process_recording(
    recording_folder: RecordingFolder, 
    skip_eye: bool = False, 
    skip_skull: bool = False, 
    skip_gaze: bool = False):
    if not skip_eye:
        # process eye data
        process_eye_session_from_recording_folder(recording_folder=recording_folder.folder_path)

        # # run eye confidence analysis
        bad_eye_data(recording_folder=recording_folder.folder_path)

    if not skip_skull:
        # process ceres solver
        run_ferret_skull_solver_from_recording_folder(recording_folder=recording_folder.folder_path)

    if not skip_gaze:
        run_gaze_pipeline(
            recording_path=recording_folder.folder_path,
            resampling_strategy=ResamplingStrategy.FASTEST,
            reprocess_all=True,
        )




def pre_recording_validation(recording_folder: RecordingFolder):
    recording_folder.check_triangulation(enforce_toy=False, enforce_annotated=False)

def post_recording_validation(recording_folder: RecordingFolder):
    recording_folder.check_skull_postprocessing(enforce_toy=False, enforce_annotated=False)

if __name__ == "__main__":
    recording_folder_path = Path("/home/scholl-lab/synched_ferret_recordings/session_2025-07-05_ferret_753_EyeCameras_P37_EO9/full_recording")
    skip_eye = True
    skip_skull = False
    skip_gaze = False

    if len(sys.argv) >= 2:
        recording_folder = Path(sys.argv[1])
    else:
        recording_folder = recording_folder_path
        print(f"Using default directory: {recording_folder}")

    if not recording_folder.exists():
        print(f"Error: Directory does not exist: {recording_folder}")
        print("\nUsage: python plot_vor_correlation_grid.py [recording_folder] [output_dir]")
        sys.exit(1)

    flags = [a for a in sys.argv[1:] if a.startswith("-")]

    # Process boolean flags
    for flag in flags:
        if flag in ("--skip-eye", "-e"):
            skip_eye = True
        elif flag in ("--skip-skull", "-s"):
            skip_skull = True
        elif flag in ("--skip-gaze", "-g"):
            skip_gaze = True
        else:
            print(f"Warning: unknown flag {flag}")

    recording_folder = RecordingFolder.from_folder_path(recording_folder_path)
    pre_recording_validation(recording_folder=recording_folder)
    process_recording(
        recording_folder=recording_folder,
        skip_eye=skip_eye,
        skip_skull=skip_skull,
        skip_gaze=skip_gaze
    )
    post_recording_validation(recording_folder=recording_folder)
