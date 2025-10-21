from pathlib import Path

from python_code.eye_analysis.data_processing.align_data.eye_anatomical_alignment import eye_alignment_main, merge_eye_output_csvs

def process_eye_session(
    session_folder: Path,
    clip_folder: Path,
    clip_name: str, 
    eye_0_dlc_csv: Path,
    eye_1_dlc_csv: Path,
    eye_0_timestamps_npy: Path,
    eye_1_timestamps_npy: Path
):
    recording_name = session_folder.stem

    # process eye 0
    eye_alignment_main(
        recording_name=f"{recording_name}_{clip_name}_eye0",
        csv_path=eye_0_dlc_csv,
        timestamps_path=eye_0_timestamps_npy,
        output_path=clip_folder / "eye_data" / "output_data",
    )

    # process eye 1
    eye_alignment_main(
        recording_name=f"{recording_name}_{clip_name}_eye1",
        csv_path=eye_1_dlc_csv,
        timestamps_path=eye_1_timestamps_npy,
        output_path=clip_folder / "eye_data" / "output_data",
    )

    # merge eye csvs
    merged_eye_output = merge_eye_output_csvs(eye_data_path=clip_folder / "eye_data")
    merged_eye_output.to_csv(clip_folder / "eye_data" / f"{recording_name}_{clip_name}_eye_data.csv")

    # run video creation

if __name__ == "__main__":
    session_folder = Path("/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/")

    clip_name = "0m_37s-1m_37s"
    clip_folder = session_folder / "clips" / clip_name
    

    eye_0_dlc_csv = clip_folder / "eye_data" / "dlc_output" / "model_outputs_iteration_11" / f"eye0_clipped_4340_11510DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    eye_1_dlc_csv = clip_folder / "eye_data" / "dlc_output" / "model_outputs_iteration_11" / f"eye1_clipped_4358_11527DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"

    eye_0_timestamps_npy = clip_folder / "eye_data" / "eye_videos" / f"eye0_timestamps_utc_clipped_4340_11510.npy"
    eye_1_timestamps_npy = clip_folder / "eye_data" / "eye_videos" / f"eye1_timestamps_utc_clipped_4358_11527.npy"

    process_eye_session(
        session_folder=session_folder,
        clip_folder=clip_folder,
        clip_name=clip_name,
        eye_0_dlc_csv=eye_0_dlc_csv,
        eye_1_dlc_csv=eye_1_dlc_csv,
        eye_0_timestamps_npy=eye_0_timestamps_npy,
        eye_1_timestamps_npy=eye_1_timestamps_npy
    )