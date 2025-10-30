import pandas as pd

from pathlib import Path

RECORDING_SCHEDULE_PATH = Path(__file__).parent / "Ferret Recording Schedule - recording_info_v1.csv"
RECORDING_PROGRESS_PATH = Path(__file__).parent / "recording_progress.csv"

def setup_csv(input_path: Path = RECORDING_SCHEDULE_PATH) -> pd.DataFrame:
    """
    Function to set up initial csv file for recording schedule
    
    file exported as csv from: https://docs.google.com/spreadsheets/d/16isURxaoivt7_ctsXTxqbmJNtOCfIB5x2R04bQBAeb0/edit?gid=1748073986#gid=1748073986
    """
    df = pd.read_csv(input_path, sep=",")

    new_columns = {
        "data_path": [""] * len(df),
        "calibration_recorded": [False] * len(df),
        "calibration_synchronized_corrected": [False] * len(df),
        "calibration_toml": [False] * len(df),
        "pupil_recording": [False] * len(df),
        "synchronized_corrected_videos": [False] * len(df),
        "full_recording": [False] * len(df),
        "clips": [False] * len(df),
    }

    df = df.assign(**new_columns)

    for index, row in df.iterrows():
        split_recording_path = row["recording_path"].split("/")
        df.at[index, "data_path"] = Path("/home/scholl-lab/ferret_recordings") / f"{split_recording_path[-2]}_{split_recording_path[-1]}"

    print(df.head())
    print(df["data_path"])
    print(df.columns)

    return df

def load_and_save_new_csv(input_path: Path = RECORDING_SCHEDULE_PATH):
    df = setup_csv(input_path)
    df.to_csv(RECORDING_PROGRESS_PATH, index=False)

def load_recording_progress() -> pd.DataFrame:
    df = pd.read_csv(RECORDING_PROGRESS_PATH)
    return df

def save_recording_progress(df: pd.DataFrame):
    df.to_csv(RECORDING_PROGRESS_PATH)

if __name__ == "__main__":
    load_and_save_new_csv()