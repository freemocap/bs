from pathlib import Path
import pandas as pd

from python_code.eye_analysis.data_processing.align_data.eye_anatomical_alignment import eye_alignment_main, merge_eye_output_csvs

def merge_head_data(merged_eye_output: pd.DataFrame, head_data: Path) -> pd.DataFrame:
    head_data_csv = pd.read_csv(head_data)
    topdown_video_name = "24676894"
    head_data_topdown = head_data_csv[head_data_csv["video"].str.startswith(topdown_video_name)]
    return pd.concat([merged_eye_output, head_data_topdown], axis=0).sort_values(["timestamp"])


def process_eye_session(
    session_folder: Path,
    clip_folder: Path,
    clip_name: str, 
    eye_0_dlc_csv: Path,
    eye_1_dlc_csv: Path,
    eye_0_timestamps_npy: Path,
    eye_1_timestamps_npy: Path,
    head_data: Path | None = None
):
    recording_name = session_folder.stem
    output_data_folder = clip_folder / "eye_data" / "output_data"
    output_data_folder.mkdir(parents=True, exist_ok=True)

    # process eye 0
    eye_alignment_main(
        recording_name=f"{recording_name}_{clip_name}_eye0",
        csv_path=eye_0_dlc_csv,
        timestamps_path=eye_0_timestamps_npy,
        output_path=output_data_folder,
    )

    # process eye 1
    eye_alignment_main(
        recording_name=f"{recording_name}_{clip_name}_eye1",
        csv_path=eye_1_dlc_csv,
        timestamps_path=eye_1_timestamps_npy,
        output_path=output_data_folder,
    )

    # merge eye csvs
    merged_eye_output = merge_eye_output_csvs(eye_data_path=clip_folder / "eye_data")
    merged_eye_output.to_csv(clip_folder / "eye_data" / f"{recording_name}_{clip_name}_eye_data.csv", index=False)

    if head_data:
        head_and_eye_data = merge_head_data(merged_eye_output=merged_eye_output, head_data=head_data)
        head_and_eye_data.to_csv(clip_folder / "eye_data" / f"{recording_name}_{clip_name}_head_and_eye_data.csv", index=False)

    # run video creation

if __name__ == "__main__":
    session_folder = Path("/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1")

    clip_name = "0m_37s-1m_37s"
    clip_folder = session_folder / "clips" / clip_name
    

    eye_0_dlc_csv = clip_folder / "eye_data" / "dlc_output" / "eye_model_v2_model_outputs_iteration_0" / f"eye0_clipped_4354_11523DLC_Resnet50_eye_model_v2_shuffle1_snapshot_020.csv"
    eye_1_dlc_csv = clip_folder / "eye_data" / "dlc_output" / "eye_model_v2_model_outputs_iteration_0" / f"eye1_clipped_4371_11541DLC_Resnet50_eye_model_v2_shuffle1_snapshot_020.csv"

    eye_0_timestamps_npy = clip_folder / "eye_data" / "eye_videos" / f"eye0_timestamps_utc_clipped_4354_11523.npy"
    eye_1_timestamps_npy = clip_folder / "eye_data" / "eye_videos" / f"eye1_timestamps_utc_clipped_4371_11541.npy"

    head_data = Path("/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/mocap_data/tidy_head_data.csv")
    process_eye_session(
        session_folder=session_folder,
        clip_folder=clip_folder,
        clip_name=clip_name,
        eye_0_dlc_csv=eye_0_dlc_csv,
        eye_1_dlc_csv=eye_1_dlc_csv,
        eye_0_timestamps_npy=eye_0_timestamps_npy,
        eye_1_timestamps_npy=eye_1_timestamps_npy,
        head_data=head_data
    )