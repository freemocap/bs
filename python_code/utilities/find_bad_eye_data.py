import pandas as pd
import numpy as np
from pathlib import Path
from time import perf_counter

from python_code.utilities.get_mean_dlc_confidence import get_mean_dlc_confidence

# def check_single_row(row, vertical_threshold: float, analysis_df: pd.DataFrame) -> int:
#     camera = row["camera"]
#     if camera == "eye0":
#         camera = "eye0DLC"

#     frame = row["frames"]

#     mask = (
#         (analysis_df["frame"] == frame) &
#         (analysis_df["video"] == camera)
#     )

#     filtered_rows = analysis_df[mask]

#     tear_duct_row = filtered_rows[filtered_rows["keypoint"] == 'tear_duct']
#     outer_eye_row = filtered_rows[filtered_rows["keypoint"] == 'outer_eye']
#     pupil_points = filtered_rows[filtered_rows["keypoint"].isin(['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])]
    
#     tear_duct_x = tear_duct_row['x'].iloc[0]
#     outer_eye_x = outer_eye_row['x'].iloc[0]
#     pupil_center_x = pupil_points['x'].mean()
#     pupil_center_y = pupil_points['y'].mean()
    
#     if pupil_center_x < tear_duct_x or pupil_center_x > outer_eye_x:
#         return 0
#     if abs(pupil_center_y) > vertical_threshold:
#         return 0
        
#     return 1

def check_single_eye(camera: str, frame: int, vertical_threshold: float, analysis_df: pd.DataFrame) -> int:
    filtered_rows = analysis_df[analysis_df["frame"] == frame]
    
    if len(filtered_rows) == 0:
        print(f"Warning: no rows remaining after filtering for frame {frame} camera {camera}")
        return 0
        
    # Get keypoints efficiently
    tear_duct_row = filtered_rows[filtered_rows["keypoint"] == 'tear_duct']
    outer_eye_row = filtered_rows[filtered_rows["keypoint"] == 'outer_eye']
    pupil_points = filtered_rows[filtered_rows["keypoint"].isin(['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])]
    
    # Extract coordinates directly
    tear_duct_x = tear_duct_row['x'].iloc[0]
    outer_eye_x = outer_eye_row['x'].iloc[0]
    pupil_center_x = pupil_points['x'].mean()
    pupil_center_y = pupil_points['y'].mean()
    
    # Check conditions
    if pupil_center_x < tear_duct_x or pupil_center_x > outer_eye_x:
        return 0
    if abs(pupil_center_y) > vertical_threshold:
        return 0
        
    return 1

def find_bad_eye_data(confidence_df: pd.DataFrame, analysis_df: pd.DataFrame):
    blink_threshold = 0.45
    single_eye_threshold = 0.4
    vertical_threshold = 25
    confidence_df["good_data"] = [1] * len(confidence_df)
    confidence_df["blink_threshold"] = [1] * len(confidence_df)
    confidence_df["confidence_threshold"] = np.where(confidence_df["mean_confidence"] > single_eye_threshold, 1, 0)
    confidence_df["eye_position_threshold"] = [1] * len(confidence_df)
    eye0_mask = confidence_df["camera"]=="eye0"
    eye1_mask = confidence_df["camera"]=="eye1"
    eye0_confidence_df = confidence_df[eye0_mask]
    eye1_confidence_df = confidence_df[eye1_mask]

    cleaned_mask = (analysis_df["processing_level"]=="cleaned")
    # cleaned_analysis_df = analysis_df[cleaned_mask]
    # confidence_df['eye_position_threshold'] = confidence_df.apply(
    #     lambda x: check_single_row(x, vertical_threshold=vertical_threshold, analysis_df=cleaned_analysis_df),
    #     axis=1
    # )
    # eye0_analysis_mask = (analysis_df["video"] == "eye0") & cleaned_mask
    eye0_analysis_mask = (analysis_df["video"] == "eye0") & cleaned_mask
    eye1_analysis_mask = (analysis_df["video"] == "eye1") & cleaned_mask

    eye0_analysis = analysis_df[eye0_analysis_mask]
    eye1_analysis = analysis_df[eye1_analysis_mask]

    for frame in confidence_df["frames"]:
        frame_mask = confidence_df["frames"]==frame

        eye0_row = eye0_confidence_df.loc[frame_mask]
        eye1_row = eye1_confidence_df.loc[frame_mask]
        if len(eye0_row)==0 or len(eye1_row)==0:
            continue
        eye0_confidence = eye0_row.iloc[0]['mean_confidence']
        eye1_confidence = eye1_row.iloc[0]['mean_confidence']

        if eye0_confidence < blink_threshold and eye1_confidence < blink_threshold:
            confidence_df.loc[frame_mask & eye0_mask, "blink_threshold"] = 0
            confidence_df.loc[frame_mask & eye1_mask, "blink_threshold"] = 0
            continue

        confidence_df.loc[frame_mask & eye0_mask, "eye_position_threshold"]=check_single_eye("eye0", frame, vertical_threshold, eye0_analysis)
        confidence_df.loc[frame_mask & eye1_mask, "eye_position_threshold"]=check_single_eye("eye1", frame, vertical_threshold, eye1_analysis)

    confidence_df["good_data"] = ((confidence_df["blink_threshold"] ==1) & (confidence_df["confidence_threshold"]==1) & (confidence_df["eye_position_threshold"]==1)).astype(int)

    return confidence_df

def bad_eye_data(recording_folder: Path):
    eye_data_folder = recording_folder / "eye_data"

    dlc_path = eye_data_folder / "dlc_output" / "eye_model_v3"
    synched_video_path = eye_data_folder / "eye_videos"
    camera_names  = ["eye0", "eye1"]

    get_mean_dlc_confidence(
        path_to_folder_with_dlc_csvs=dlc_path,
        path_to_synchronized_video_folder=synched_video_path,
        camera_names=camera_names
    )

    dlc_confidence_csv = eye_data_folder / "eye_model_v3_mean_confidence.csv"
    eye_analysis_output = eye_data_folder / "eye_data.csv"

    dlc_confidence_df = pd.read_csv(dlc_confidence_csv)
    eye_analysis_df = pd.read_csv(eye_analysis_output)

    start_time = perf_counter()
    updated_df = find_bad_eye_data(confidence_df=dlc_confidence_df, analysis_df=eye_analysis_df)
    end_time = perf_counter()
    updated_df.to_csv(dlc_confidence_csv, index=False)
    print(f"Searching for eye data took {end_time - start_time} s")
    percent_zeros = (updated_df['good_data'] == 0).mean() * 100
    print(f"Percent of bad data found was {percent_zeros:.2f}%")

if __name__=='__main__':
    recording_folder = Path("/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s")

    bad_eye_data(recording_folder=recording_folder)