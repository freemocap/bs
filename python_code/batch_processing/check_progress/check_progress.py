import json
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
    "head_dlc_iteration",
    "eye_dlc_iteration",
    "toy_dlc_iteration",
]

_EMPTY_ROW = {
    **{col: False for col in _COLUMNS if col not in ("recording_path", "head_dlc_iteration", "eye_dlc_iteration", "toy_dlc_iteration")},
    "head_dlc_iteration": -1,
    "eye_dlc_iteration": -1,
    "toy_dlc_iteration": -1,
}


def _read_dlc_iteration(dlc_output_folder: Path | None) -> int:
    """Return the iteration from skellyclicker_metadata.json, or -1 if unavailable."""
    if dlc_output_folder is None:
        return -1
    metadata_path = dlc_output_folder / "skellyclicker_metadata.json"
    if not metadata_path.exists():
        return -1
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata.get("iteration", -1)


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
            "head_dlc_iteration": _read_dlc_iteration(recording_folder.head_body_dlc_output),
            "eye_dlc_iteration": _read_dlc_iteration(recording_folder.eye_dlc_output),
            "toy_dlc_iteration": _read_dlc_iteration(recording_folder.toy_dlc_output),
        })
        rows.append(row)

    return pd.DataFrame(rows, columns=_COLUMNS)
    

if __name__ == "__main__":
    ferret_recordings_path = Path("/home/scholl-lab/ferret_recordings")
    # check if recording_progress.csv exists, if it does, load it and update it, if it doesn't, create it
    df = check_progress(ferret_recordings_path)
    print(df)

    #save df to csv
    df.to_csv(ferret_recordings_path / "recording_progress.csv", index=False)