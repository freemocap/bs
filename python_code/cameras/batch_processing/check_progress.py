import pandas as pd

from pathlib import Path



def check_progress(df: pd.DataFrame):
    for index, row in df.iterrows():
        recording_path = Path(row["recording_path"])
        if not recording_path.exists():
            raise ValueError(f"Recording path {recording_path} does not exist")
        df.at[index, "calibration_recorded"] = check_calibration_exists(recording_path)
        df.at[index, "calibration_synchronized_corrected"] = check_calibration_synchronized(recording_path)
        df.at[index, "calibration_toml"] = check_calibration_toml(recording_path)
        df.at[index, "pupil_recording"] = check_pupil_recording(recording_path)
        df.at[index, "synchronized_corrected_videos"] = check_synchronized_corrected_videos(recording_path)
        df.at[index, "basler_pupil_synchronized_videos"] = check_basler_pupil_synchronized_videos(recording_path)
        df.at[index, "combined_videos"] = check_combined_videos(recording_path)


def check_calibration_exists(path: Path) -> bool:
    calibration_path = path.parent / "calibration"
    return calibration_path.exists() and calibration_path.is_dir()

def check_calibration_synchronized(path: Path) -> bool:
    synched_calibration_path = path.parent / "calibration" / "synchronized_corrected_videos"
    return synched_calibration_path.exists() and synched_calibration_path.is_dir()

def check_calibration_toml(path: Path) -> bool:
    calibration_path = path.parent / "calibration"
    calibration_toml_paths = list(calibration_path.glob("*camera_calibration.toml"))
    if len(calibration_toml_paths) == 0:
        return False
    return calibration_toml_paths[0].exists()

def check_pupil_recording(path: Path) -> bool:
    pupil_path = path / "pupil_output"
    if not pupil_path.exists() or not pupil_path.is_dir():
        return False
    return True

    # pupil_eye0_video_path = pupil_path / "eye0.mp4"
    # pupil_eye1_video_path = pupil_path / "eye1.mp4"
    # return pupil_eye0_video_path.exists() and pupil_eye1_video_path.exists()

def check_synchronized_corrected_videos(path: Path) -> bool:
    synchronized_video_path = path / "synchronized_corrected_videos"
    return synchronized_video_path.exists() and synchronized_video_path.is_dir()

def check_basler_pupil_synchronized_videos(path: Path) -> bool:
    synchronized_video_path = path / "basler_pupil_synchronized"
    if not synchronized_video_path.exists() and synchronized_video_path.is_dir():
        return False

    pupil_eye0_video_path = synchronized_video_path / "eye0.mp4"
    pupil_eye1_video_path = synchronized_video_path / "eye1.mp4"
    return pupil_eye0_video_path.exists() and pupil_eye1_video_path.exists()

def check_combined_videos(path: Path) -> bool:
    synchronized_video_path = path / "basler_pupil_synchronized"
    if not synchronized_video_path.exists() and synchronized_video_path.is_dir():
        return False

    combined_video_path = synchronized_video_path / "combined.mp4"
    return combined_video_path.exists()
    

if __name__ == "__main__":
    from python_code.cameras.batch_processing.setup_csv import load_recording_progress
    pd.set_option('max_colwidth', 100)
    
    recording_progress = load_recording_progress()

    print(recording_progress.columns)

    check_progress(recording_progress)

    for column in ["calibration_recorded", "pupil_recording"]:
        print(recording_progress[["recording_path", column]])