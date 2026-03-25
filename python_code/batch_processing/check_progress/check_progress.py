import pandas as pd

from pathlib import Path

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


_COLUMNS = [
    "recording_path",
    "synchronized",
    "calibrated",
    "dlc_processed",
    "triangulated",
    "eye_postprocessed",
    "skull_postprocessed",
    "gaze_postprocessed",
]

_EMPTY_ROW = {col: False for col in _COLUMNS if col != "recording_path"}


def check_progress(ferret_recordings_path: Path) -> pd.DataFrame:
    rows = []
    for subdir in sorted(ferret_recordings_path.iterdir()):
        if not subdir.is_dir():
            continue

        full_recording_path = subdir / "full_recording"
        row = {"recording_path": subdir}

        if not full_recording_path.exists():
            row.update(_EMPTY_ROW)
            rows.append(row)
            continue

        try:
            recording_folder = RecordingFolder.from_folder_path(full_recording_path)
        except ValueError:
            row.update(_EMPTY_ROW)
            rows.append(row)
            continue

        row.update({
            "synchronized": recording_folder.is_synchronized(),
            "calibrated": recording_folder.is_calibrated(),
            "dlc_processed": recording_folder.is_dlc_processed(),
            "triangulated": recording_folder.is_triangulated(),
            "eye_postprocessed": recording_folder.is_eye_postprocessed(),
            "skull_postprocessed": recording_folder.is_skull_postprocessed(),
            "gaze_postprocessed": recording_folder.is_gaze_postprocessed(),
        })
        rows.append(row)

    return pd.DataFrame(rows, columns=_COLUMNS)
    

if __name__ == "__main__":
    ferret_recordings_path = Path("/Users/philipqueen/Documents/GitHub/bs/ferret_recordings")
    # check if recording_progress.csv exists, if it does, load it and update it, if it doesn't, create it
    df = check_progress(ferret_recordings_path)
    print(df)

    #save df to csv
    df.to_csv(ferret_recordings_path / "recording_progress.csv", index=False)