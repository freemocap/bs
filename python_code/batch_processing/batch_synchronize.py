from pathlib import Path

from python_code.cameras.postprocess import postprocess

def synchronize_recordings(recording_folders: list[Path]):
    for folder in recording_folders:
        postprocess(session_folder_path=folder, include_eyes=True)

if __name__=="__main__":
    sessions_to_process = [
        Path("/home/scholl-lab/ferret_recordings/session_2025-06-28_ferret_757_EyeCameras_P30_EO2"),
        Path("/home/scholl-lab/ferret_recordings/session_2025-06-29_ferret_757_EyeCameras_P31_EO3"),
        Path("/home/scholl-lab/ferret_recordings/session_2025-06-29_ferret_757_EyeCameras_P31_EO3__1")
    ]

    synchronize_recordings(recording_folders=sessions_to_process)