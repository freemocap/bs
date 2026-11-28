import multiprocessing
from pathlib import Path
import pandas as pd

from python_code.eye_analysis.data_processing.align_data.eye_anatomical_alignment import eye_alignment_main, merge_eye_output_csvs
from python_code.eye_analysis.video_viewers.stabilized_eye_viewer import create_stabilized_eye_videos
from python_code.utilities.temporal_filters import FilterSpec

EYE_DLC_MODEL_FOLDER: str = "eye_model_v3"
# Chosen from the Nov 2026 filter comparison (variant dlc_median_sg_55ms_p3):
# - 5-frame median, identical to DLC's filterpredictions, removes single-frame outliers
# - then Savitzky-Golay 55 ms (7 samples at 120 Hz), polyorder 3
# Keeps ~86-95% of saccade peak velocity (the old 6 Hz Butterworth kept ~45%).
EYE_FILTER_SPEC: FilterSpec = FilterSpec.savgol(window_s=0.055, polyorder=3, median_frames=5)


def find_eye_dlc_csv(recording_path: Path, eye_name: str) -> Path:
    """Unfiltered DLC output (excludes DLC's median-filtered `_filtered.csv`)."""
    folder = recording_path / "eye_data" / "dlc_output" / EYE_DLC_MODEL_FOLDER
    candidates = sorted(p for p in folder.glob(f"{eye_name}*snapshot*.csv") if "_filtered" not in p.name)
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one unfiltered {eye_name} DLC csv in {folder}, found {candidates}")
    return candidates[0]

def merge_head_data(merged_eye_output: pd.DataFrame, head_data: Path) -> pd.DataFrame:
    head_data_csv = pd.read_csv(head_data)
    topdown_video_name = "24676894"
    head_data_topdown = head_data_csv[head_data_csv["video"].str.startswith(topdown_video_name)]
    return pd.concat([merged_eye_output, head_data_topdown], axis=0).sort_values(["timestamp"])


def process_eye_session(
    session_folder: Path,
    clip_folder: Path,
    clip_name: str,
    left_eye_dlc_csv: Path,
    right_eye_dlc_csv: Path,
    left_eye_timestamps_npy: Path,
    right_eye_timestamps_npy: Path,
    left_eye_video_path: Path,
    right_eye_video_path: Path,
    filter_spec: FilterSpec = EYE_FILTER_SPEC,
):
    recording_name = session_folder.stem
    output_data_folder = clip_folder / "eye_data" / "output_data"
    output_data_folder.mkdir(parents=True, exist_ok=True)

    # process both eyes in parallel
    alignment_args = [
        (f"{recording_name}_{clip_name}_left_eye", left_eye_dlc_csv, left_eye_timestamps_npy, output_data_folder, filter_spec),
        (f"{recording_name}_{clip_name}_right_eye", right_eye_dlc_csv, right_eye_timestamps_npy, output_data_folder, filter_spec),
    ]
    with multiprocessing.Pool(processes=2) as pool:
        pool.starmap(eye_alignment_main, alignment_args)

    # merge eye csvs
    merged_eye_output = merge_eye_output_csvs(eye_data_path=clip_folder / "eye_data")
    merged_eye_output.to_csv(clip_folder / "eye_data" / f"eye_data.csv", index=False)

    # run video creation for both eyes in parallel
    video_args = [
        (clip_folder, left_eye_video_path, left_eye_timestamps_npy, left_eye_dlc_csv),
        (clip_folder, right_eye_video_path, right_eye_timestamps_npy, right_eye_dlc_csv),
    ]
    with multiprocessing.Pool(processes=2) as pool:
        pool.starmap(create_stabilized_eye_videos, video_args)


def process_eye_session_from_recording_folder(
    recording_folder: Path,
    filter_spec: FilterSpec = EYE_FILTER_SPEC,
) -> dict:
    """Process both eyes; returns the parameters used, for processing_metadata.json."""
    clip_name = recording_folder.stem
    if "session" in recording_folder.parent.stem:
        session_folder = recording_folder.parent
    else:
        session_folder = recording_folder.parent.parent
    eye_videos_folder = recording_folder / "eye_data" / "eye_videos"

    left_eye_dlc_csv  = find_eye_dlc_csv(recording_folder, "left_eye")
    right_eye_dlc_csv = find_eye_dlc_csv(recording_folder, "right_eye")

    left_eye_timestamps_npy  = next(eye_videos_folder.glob("left_eye*timestamps_utc*.npy"))
    right_eye_timestamps_npy = next(eye_videos_folder.glob("right_eye*timestamps_utc*.npy"))

    left_eye_video_path  = next(eye_videos_folder.glob("left_eye*.mp4"))
    right_eye_video_path = next(eye_videos_folder.glob("right_eye*.mp4"))

    process_eye_session(
        session_folder=session_folder,
        clip_folder=recording_folder,
        clip_name=clip_name,
        left_eye_dlc_csv=left_eye_dlc_csv,
        right_eye_dlc_csv=right_eye_dlc_csv,
        left_eye_timestamps_npy=left_eye_timestamps_npy,
        right_eye_timestamps_npy=right_eye_timestamps_npy,
        left_eye_video_path=left_eye_video_path,
        right_eye_video_path=right_eye_video_path,
        filter_spec=filter_spec,
    )
    return {
        "filter": filter_spec.model_dump(mode="json"),
        "filter_label": filter_spec.label(),
        "dlc_model_folder": EYE_DLC_MODEL_FOLDER,
        "left_eye_dlc_csv": str(left_eye_dlc_csv),
        "right_eye_dlc_csv": str(right_eye_dlc_csv),
    }


if __name__ == "__main__":
    recording_folder = Path("/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5/clips/1m_20s-2m_20s")

    process_eye_session_from_recording_folder(recording_folder=recording_folder)