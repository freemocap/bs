from pathlib import Path

from python_code.cameras.postprocess import postprocess

def synchronize_recordings(recording_folders: list[Path]):
    for folder in recording_folders:
        postprocess(session_folder_path=folder, include_eyes=False)

if __name__=="__main__":
    sessions_to_process = [
        Path("/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp1_0.5hz_pitch_05-04-26"),
        Path("//home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp1_0.5hz_roll_05-04-26"),
        Path("/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp1_0.5hz_yaw_05-04-26"),
        Path("/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp2_0.25hz_pitch_05-04-26"),
        Path("/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp2_0.25hz_roll_05-04-26"),
        Path("/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp2_0.25hz_yaw_05-04-26"),
        Path("/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp3_0.1hz_pitch_05-04-26"),
        Path("/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp3_0.1hz_roll_05-04-26"),
        Path("/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp3_0.1hz_yaw_05-04-26")
    ]

    synchronize_recordings(recording_folders=sessions_to_process)