# processes everything after model inference
from pathlib import Path

from python_code.eye_analysis.process_eye_session import process_eye_session_from_recording_folder
from python_code.rigid_body_solver.ferret_skull_solver import run_ferret_skull_solver_from_recording_folder
from python_code.utilities.find_bad_eye_data import bad_eye_data
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


def process_recording(recording_folder: RecordingFolder):
    # process eye data
    process_eye_session_from_recording_folder(recording_folder=recording_folder.folder_path)

    # run eye confidence analysis
    bad_eye_data(recording_folder=recording_folder.folder_path)

    # process ceres solver
    run_ferret_skull_solver_from_recording_folder(recording_folder=recording_folder.folder_path)


def pre_recording_validation(recording_folder: RecordingFolder):
    recording_folder.check_triangulation(enforce_toy=False, enforce_annotated=False)

def post_recording_validation(recording_folder: RecordingFolder):
    recording_folder.check_postprocessing(enforce_toy=False, enforce_annotated=False)

if __name__ == "__main__":
    recording_folder_path = Path("/home/scholl-lab/ferret_recordings/session_2025-10-18_ferret_420_E09/full_recording")
    recording_folder = RecordingFolder.from_folder_path(recording_folder_path)
    pre_recording_validation(recording_folder=recording_folder)
    process_recording(recording_folder=recording_folder)
    post_recording_validation(recording_folder=recording_folder)
