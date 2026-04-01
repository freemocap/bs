import pandas as pd
import numpy as np
from pathlib import Path
from time import perf_counter

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.utilities.get_mean_dlc_confidence import get_mean_dlc_confidence


def check_single_eye(frame: int, vertical_threshold: float, horizontal_threshold: float, analysis_df: pd.DataFrame) -> int:
    filtered_rows = analysis_df[analysis_df["frame"] == frame]

    if len(filtered_rows) == 0:
        print(f"Warning: no rows remaining after filtering for frame {frame}")
        return 0

    # Get keypoints efficiently
    tear_duct_row = filtered_rows[filtered_rows["keypoint"] == 'tear_duct']
    outer_eye_row = filtered_rows[filtered_rows["keypoint"] == 'outer_eye']
    pupil_points = filtered_rows[filtered_rows["keypoint"].isin(['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])]

    if tear_duct_row.empty or outer_eye_row.empty or pupil_points.empty:
        print(f"Warning: missing keypoints for frame {frame}")
        return 0

    # Extract coordinates directly
    tear_duct_x = tear_duct_row['x'].iloc[0]
    outer_eye_x = outer_eye_row['x'].iloc[0]
    pupil_center_x = pupil_points['x'].mean()
    pupil_center_y = pupil_points['y'].mean()

    # Check conditions
    if (outer_eye_x - tear_duct_x) < horizontal_threshold:
        return 0
    if pupil_center_x < tear_duct_x or pupil_center_x > outer_eye_x:
        return 0
    if abs(pupil_center_y) > vertical_threshold:
        return 0

    return 1

def find_bad_eye_data(
        confidence_df: pd.DataFrame,
        analysis_df: pd.DataFrame,
        blink_threshold: float = 0.8,
        single_eye_threshold: float = 0.7,
        vertical_threshold: float = 25,
        horizontal_threshold: float = 100,
    ) -> pd.DataFrame:
    confidence_df["good_data"] = [1] * len(confidence_df)
    confidence_df["blink_threshold"] = [1] * len(confidence_df)
    confidence_df["confidence_threshold"] = np.where(confidence_df["mean_confidence"] > single_eye_threshold, 1, 0)
    confidence_df["eye_position_threshold"] = [1] * len(confidence_df)
    eye0_mask = confidence_df["camera"]=="eye0"
    eye1_mask = confidence_df["camera"]=="eye1"
    eye0_confidence_df = confidence_df[eye0_mask]
    eye1_confidence_df = confidence_df[eye1_mask]

    cleaned_mask = (analysis_df["processing_level"]=="cleaned")
    eye0_analysis_mask = (analysis_df["video"] == "eye0") & cleaned_mask
    eye1_analysis_mask = (analysis_df["video"] == "eye1") & cleaned_mask

    eye0_analysis = analysis_df[eye0_analysis_mask]
    eye1_analysis = analysis_df[eye1_analysis_mask]

    for frame in confidence_df["frames"].unique():
        frame_mask = confidence_df["frames"]==frame

        eye0_row = eye0_confidence_df[eye0_confidence_df["frames"] == frame]
        eye1_row = eye1_confidence_df[eye1_confidence_df["frames"] == frame]
        if len(eye0_row)==0 or len(eye1_row)==0:
            continue
        eye0_confidence = eye0_row.iloc[0]['mean_confidence']
        eye1_confidence = eye1_row.iloc[0]['mean_confidence']

        if eye0_confidence < blink_threshold and eye1_confidence < blink_threshold:
            confidence_df.loc[frame_mask & eye0_mask, "blink_threshold"] = 0
            confidence_df.loc[frame_mask & eye1_mask, "blink_threshold"] = 0
            continue

        confidence_df.loc[frame_mask & eye0_mask, "eye_position_threshold"]=check_single_eye(frame, vertical_threshold, horizontal_threshold, eye0_analysis)
        confidence_df.loc[frame_mask & eye1_mask, "eye_position_threshold"]=check_single_eye(frame, vertical_threshold, horizontal_threshold, eye1_analysis)

    confidence_df["good_data"] = ((confidence_df["blink_threshold"] ==1) & (confidence_df["confidence_threshold"]==1) & (confidence_df["eye_position_threshold"]==1)).astype(int)

    return confidence_df

def bad_eye_data(recording_folder: RecordingFolder):
    get_mean_dlc_confidence(recording_folder=recording_folder)

    dlc_confidence_csv = recording_folder.eye_data / "eye_model_v3_mean_confidence.csv"

    dlc_confidence_df = pd.read_csv(dlc_confidence_csv)
    eye_analysis_df = pd.read_csv(recording_folder.eye_data_csv)

    start_time = perf_counter()
    updated_df = find_bad_eye_data(confidence_df=dlc_confidence_df, analysis_df=eye_analysis_df)
    end_time = perf_counter()
    updated_df.to_csv(dlc_confidence_csv, index=False)
    print(f"Searching for eye data took {end_time - start_time} s")
    percent_zeros = (updated_df['good_data'] == 0).mean() * 100
    print(f"Percent of bad data found was {percent_zeros:.2f}%")

if __name__=='__main__':
    recording_folder = RecordingFolder.from_folder_path(
        Path("/Users/philipqueen/session_2025-07-01_ferret_757_EyeCameras_P33_EO5/clips/1m_20s-2m_20s/")
    )

    bad_eye_data(recording_folder=recording_folder)