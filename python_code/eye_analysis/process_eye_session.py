import multiprocessing
from pathlib import Path
import pandas as pd

from python_code.eye_analysis.data_processing.align_data.eye_anatomical_alignment import eye_alignment_main, merge_eye_output_csvs
from python_code.eye_analysis.video_viewers.stabilized_eye_viewer import create_stabilized_eye_videos

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
):
    recording_name = session_folder.stem
    output_data_folder = clip_folder / "eye_data" / "output_data"
    output_data_folder.mkdir(parents=True, exist_ok=True)

    # process left eye
    eye_alignment_main(
        recording_name=f"{recording_name}_{clip_name}_left_eye",
        csv_path=left_eye_dlc_csv,
        timestamps_path=left_eye_timestamps_npy,
        output_path=output_data_folder,
    )

    # process right eye
    eye_alignment_main(
        recording_name=f"{recording_name}_{clip_name}_right_eye",
        csv_path=right_eye_dlc_csv,
        timestamps_path=right_eye_timestamps_npy,
        output_path=output_data_folder,
    )

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


def process_eye_session_from_recording_folder(recording_folder: Path):
    clip_name = recording_folder.stem
    if "session" in recording_folder.parent.stem:
        session_folder = recording_folder.parent
    else:
        session_folder = recording_folder.parent.parent
    dlc_output_folder = recording_folder / "eye_data" / "dlc_output"
    eye_videos_folder = recording_folder / "eye_data" / "eye_videos"

    left_eye_dlc_csv  = next((dlc_output_folder / "eye_model_v3").glob("left_eye*snapshot*.csv"))
    right_eye_dlc_csv = next((dlc_output_folder / "eye_model_v3").glob("right_eye*snapshot*.csv"))

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
    )


if __name__ == "__main__":
    recording_folder = Path("/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5/clips/1m_20s-2m_20s")

    process_eye_session_from_recording_folder(recording_folder=recording_folder)