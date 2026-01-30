# processes everything after model inference
from pathlib import Path

from python_code.eye_analysis.process_eye_session import process_eye_session_from_recording_folder
from python_code.pyceres_solvers.rigid_body_solver.examples.ferret_head_solver import run_ferret_skull_solver_from_recording_folder
from python_code.utilities.find_bad_eye_data import bad_eye_data


def process_recording(recording_folder: Path):
    # process eye data
    process_eye_session_from_recording_folder(recording_folder=recording_folder)

    # run eye confidence analysis
    bad_eye_data(recording_folder=recording_folder)

    # process ceres solver
    run_ferret_skull_solver_from_recording_folder(recording_folder=recording_folder)


def pre_recording_validation(recording_folder: Path):
    eye_data_folder = recording_folder / "eye_data" 
    eye_video_folder = eye_data_folder / "eye_videos"
    eye_annotated_video_folder = eye_data_folder / "annotated_videos"
    eye_dlc_output_folder = eye_data_folder / "dlc_output"
    eye_model_folder = eye_dlc_output_folder / "eye_model_v3"
    eye_model_flipped_folder = eye_dlc_output_folder / "eye_model_v3_flipped"
    
    mocap_data_folder = recording_folder / "mocap_data"
    mocap_video_folder = mocap_data_folder / "synchronized_videos"
    if not mocap_video_folder.exists(): 
        mocap_video_folder = mocap_data_folder / "synchronized_corrected_videos" 
    mocap_annotated_video_folder = mocap_data_folder / "annotated_videos"
    mocap_dlc_output_folder = mocap_data_folder / "dlc_output"
    head_model_folder = mocap_dlc_output_folder / "head_body_eyecam_retrain_test_v2"
    toy_model_folder = mocap_dlc_output_folder / "toy_model_v2"

    for path in [
        eye_data_folder,
        eye_video_folder,
        eye_annotated_video_folder,
        eye_dlc_output_folder,
        eye_model_folder,
        eye_model_flipped_folder,
        mocap_data_folder,
        mocap_video_folder,
        mocap_annotated_video_folder,
        mocap_dlc_output_folder,
        head_model_folder,
        toy_model_folder,
    ]:
        if not path.exists():
            raise ValueError(f"Path required for processing does not exist: {path}")
        
    mocap_output_data_folder = mocap_data_folder / "output_data"
    data_3d_folder = mocap_output_data_folder / "dlc"
    data_3d_csv = data_3d_folder / "head_freemocap_data_by_frame.csv"

    for path in [
        mocap_output_data_folder,
        data_3d_folder,
        data_3d_csv,
    ]:
        if not path.exists():
            raise ValueError(f"Triangulated 3d data required for processing does not exist: {path}")


def post_recording_validation(recording_folder: Path):
    pass

if __name__ == "__main__":
    recording_folder = Path("/home/scholl-lab/ferret_recordings/session_2025-10-18_ferret_420_E09/full_recording")
    pre_recording_validation(recording_folder=recording_folder)
    process_recording(recording_folder=recording_folder)
    post_recording_validation(recording_folder=recording_folder)
