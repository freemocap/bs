import pandas as pd

from pathlib import Path

RECORDING_SCHEDULE_PATH = Path(__file__).parent / "Ferret Recording Schedule - recording_info_v1.csv"
RECORDING_PROGRESS_PATH = Path(__file__).parent / "recording_progress.csv"

def setup_csv(input_path: Path = RECORDING_SCHEDULE_PATH) -> pd.DataFrame:
    """
    Function to set up initial csv file for recording schedule
    
    file exported as csv from: https://docs.google.com/spreadsheets/d/16isURxaoivt7_ctsXTxqbmJNtOCfIB5x2R04bQBAeb0/edit?gid=1748073986#gid=1748073986
    """
    df = pd.read_csv(input_path)

    new_columns = {
        "calibration_recorded": [False] * len(df),
        "calibration_synchronized_corrected": [False] * len(df),
        "calibration_toml": [False] * len(df),
        "pupil_recording": [False] * len(df),
        "synchronized_corrected_videos": [False] * len(df),
        "basler_pupil_synchronized_videos": [False] * len(df),
        "combined_videos": [False] * len(df),
    }

    df = df.assign(**new_columns)

    print(df.head())
    print(df.columns)

    return df

def load_and_save_new_csv(input_path: Path = RECORDING_SCHEDULE_PATH):
    df = setup_csv(input_path)
    df.to_csv(RECORDING_PROGRESS_PATH, index=False)

def load_recording_progress() -> pd.DataFrame:
    df = pd.read_csv(RECORDING_PROGRESS_PATH)
    return df

if __name__ == "__main__":
    load_and_save_new_csv()