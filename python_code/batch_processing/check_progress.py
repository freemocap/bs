import pandas as pd

from pathlib import Path



def check_progress(df: pd.DataFrame, pass_missing_folder: bool = False):
    for index, row in df.iterrows():
        recording_path = Path(row["recording_path"])
        if not recording_path.exists():
            raise ValueError(f"Recording path {recording_path} does not exist")
        data_path = Path(row["data_path"])
        if not data_path.exists():
            if pass_missing_folder:
                print(f"Data path {data_path} does not exist")
                continue
            else:
                raise ValueError(f"Data path {data_path} does not exist")
        base_data_folder = data_path / "base_data"
        if not base_data_folder.exists():
            raise ValueError(f"Base data folder {base_data_folder} does not exist")
        df.at[index, "calibration_recorded"] = check_calibration_exists(data_path)
        df.at[index, "calibration_synchronized_corrected"] = check_calibration_synchronized(data_path)
        df.at[index, "calibration_toml"] = check_calibration_toml(data_path)
        df.at[index, "pupil_recording"] = check_pupil_recording(base_data_folder)
        if not check_raw_videos(base_data_folder):
            raise ValueError(f"Raw videos not found in {base_data_folder}")
        df.at[index, "synchronized_corrected_videos"] = check_synchronized_corrected_videos(base_data_folder)
        df.at[index, "full_recording"] = check_full_recording(data_path)
        df.at[index, "clips"] = check_clips(data_path)
        # df.at[index, "basler_pupil_synchronized_videos"] = check_basler_pupil_synchronized_videos(recording_path)
        # df.at[index, "combined_videos"] = check_combined_videos(recording_path)


def check_calibration_exists(path: Path) -> bool:
    calibration_path = path / "calibration"
    return calibration_path.exists() and calibration_path.is_dir()

def check_calibration_synchronized(path: Path) -> bool:
    synched_calibration_path = path / "calibration" / "synchronized_corrected_videos"
    return synched_calibration_path.exists() and synched_calibration_path.is_dir()

def check_calibration_toml(path: Path) -> bool:
    calibration_path = path / "calibration"
    if not calibration_path.exists() or not calibration_path.is_dir():
        return False
    calibration_toml_paths = list(calibration_path.glob("*camera_calibration.toml"))
    if len(calibration_toml_paths) == 0:
        return False
    return calibration_toml_paths[0].exists()

def check_pupil_recording(path: Path) -> bool:
    pupil_path = path / "pupil_output"
    if not pupil_path.exists() or not pupil_path.is_dir():
        return False

    pupil_eye0_video_path = pupil_path / "eye0.mp4"
    pupil_eye1_video_path = pupil_path / "eye1.mp4"
    return pupil_eye0_video_path.exists() and pupil_eye1_video_path.exists()

def check_raw_videos(path: Path) -> bool:
    raw_video_path = path / "raw_videos"
    if not raw_video_path.exists() and raw_video_path.is_dir():
        return False
    
    if len(list(raw_video_path.glob("*.mp4"))) < 4:
        return False
    
    return True

def check_synchronized_corrected_videos(path: Path) -> bool:
    synchronized_video_path = path / "synchronized_corrected_videos"
    if not synchronized_video_path.exists() and synchronized_video_path.is_dir():
        return False
    
    if len(list(synchronized_video_path.glob("*.mp4"))) < 4:
        print(f"Not enough synchronized videos found in {synchronized_video_path}")
        return False
    
    if len(list(synchronized_video_path.glob("*timestamps_utc.npy"))) < 4:
        print(f"Not enough synchronized utc timestamps found in {synchronized_video_path}")
        return False
    
    return True

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

def check_full_recording(path: Path) -> bool:
    full_recording_path = path / "full_recording"
    if not full_recording_path.exists() or not full_recording_path.is_dir():
        return False
    eye_data_path = full_recording_path / "eye_data"
    if not eye_data_path.exists() or not eye_data_path.is_dir():
        return False
    mocap_data_path = full_recording_path / "mocap_data"
    if not mocap_data_path.exists() or not mocap_data_path.is_dir():
        return False
    return True

def check_clips(path: Path) -> bool:
    clips_path = path / "clips"
    if not clips_path.exists() or not clips_path.is_dir():
        return False
    if not len(list(clips_path.iterdir())) > 0:
        print(f"No clips found in {clips_path}")
        return False
    return True
    

if __name__ == "__main__":
    from python_code.batch_processing.setup_csv import load_recording_progress, save_recording_progress
    pd.set_option('max_colwidth', 100)
    
    recording_progress = load_recording_progress()

    print(recording_progress.columns)

    check_progress(recording_progress)

    save_recording_progress(recording_progress, pass_missing_folder=True)

    for column in ["calibration_recorded", "pupil_recording"]:
        print(recording_progress[["recording_path", column]])