from pathlib import Path

from python_code.cameras.postprocess import postprocess

def synchronize_recordings(recording_folders: list[Path]):
    for folder in recording_folders:
        postprocess(session_folder_path=folder, include_eyes=True)

if __name__=="__main__":
    sessions_to_process = [
        Path("/home/scholl-lab/ferret_recordings/session_2025-06-29_ferret_753_EyeCameras_P31_EO3"),
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_753_EyeCameras_P33_EO5"),
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5__2"),
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-03_ferret_757_EyeCameras_P35_EO7"),
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-05_ferret_753_EyeCameras_P37_EO9"),
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-05_ferret_757_EyeCameras_P37_EO9"),
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-09_ferret_753_EyeCameras_P41_E13")
    ]

    synchronize_recordings(recording_folders=sessions_to_process)