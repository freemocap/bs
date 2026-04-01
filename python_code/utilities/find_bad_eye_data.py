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
        confidence_n_std: float = 2.0,
        blink_baseline_window: int = 500,
        blink_n_std: float = 2.0,
        blink_trailing_frames: int = 10,
        vertical_threshold: float = 25,
        horizontal_threshold: float = 100,
        density_window: int = 150,
        max_bad_fraction: float = 0.4,
    ) -> pd.DataFrame:
    confidence_df["good_data"] = [1] * len(confidence_df)
    confidence_df["blink_threshold"] = [1] * len(confidence_df)
    confidence_df["confidence_threshold"] = [1] * len(confidence_df)
    confidence_df["eye_position_threshold"] = [1] * len(confidence_df)
    eye0_mask = confidence_df["camera"] == "eye0"
    eye1_mask = confidence_df["camera"] == "eye1"

    # Confidence threshold: per-camera, flag frames more than confidence_n_std below session mean
    for camera_mask in [eye0_mask, eye1_mask]:
        conf = confidence_df.loc[camera_mask, "mean_confidence"]
        threshold = conf.mean() - confidence_n_std * conf.std()
        confidence_df.loc[camera_mask, "confidence_threshold"] = (conf > threshold).astype(int).values

    # Blink detection: both eyes must dip simultaneously below their rolling median baseline.
    # Dip is normalized by session std so the threshold is session-independent.
    eye0_sorted = confidence_df[eye0_mask].sort_values("frames")
    eye1_sorted = confidence_df[eye1_mask].sort_values("frames")
    eye0_conf = eye0_sorted["mean_confidence"]
    eye1_conf = eye1_sorted["mean_confidence"]
    eye0_dip = (eye0_conf.rolling(blink_baseline_window, center=True, min_periods=1).median() - eye0_conf) / eye0_conf.std()
    eye1_dip = (eye1_conf.rolling(blink_baseline_window, center=True, min_periods=1).median() - eye1_conf) / eye1_conf.std()
    eye0_dip_by_frame = dict(zip(eye0_sorted["frames"], eye0_dip))
    eye1_dip_by_frame = dict(zip(eye1_sorted["frames"], eye1_dip))
    blink_frame_set = {
        f for f in eye0_dip_by_frame.keys() & eye1_dip_by_frame.keys()
        if eye0_dip_by_frame[f] > blink_n_std and eye1_dip_by_frame[f] > blink_n_std
    }

    cleaned_mask = (analysis_df["processing_level"] == "cleaned")
    eye0_analysis = analysis_df[(analysis_df["video"] == "eye0") & cleaned_mask]
    eye1_analysis = analysis_df[(analysis_df["video"] == "eye1") & cleaned_mask]

    blink_countdown = 0

    for frame in sorted(confidence_df["frames"].unique()):
        frame_mask = confidence_df["frames"] == frame

        if frame in blink_frame_set:
            confidence_df.loc[frame_mask & eye0_mask, "blink_threshold"] = 0
            confidence_df.loc[frame_mask & eye1_mask, "blink_threshold"] = 0
            blink_countdown = blink_trailing_frames
            continue
        elif blink_countdown > 0:
            confidence_df.loc[frame_mask & eye0_mask, "blink_threshold"] = 0
            confidence_df.loc[frame_mask & eye1_mask, "blink_threshold"] = 0
            blink_countdown -= 1
            continue

        confidence_df.loc[frame_mask & eye0_mask, "eye_position_threshold"] = check_single_eye(frame, vertical_threshold, horizontal_threshold, eye0_analysis)
        confidence_df.loc[frame_mask & eye1_mask, "eye_position_threshold"] = check_single_eye(frame, vertical_threshold, horizontal_threshold, eye1_analysis)

    confidence_df["good_data"] = ((confidence_df["blink_threshold"] == 1) & (confidence_df["confidence_threshold"] == 1) & (confidence_df["eye_position_threshold"] == 1)).astype(int)

    # Density check: mark frames where the surrounding window has too many bad frames.
    # Done per camera so each eye is evaluated independently.
    confidence_df["density_threshold"] = 1
    for camera in confidence_df["camera"].unique():
        camera_mask = confidence_df["camera"] == camera
        camera_df = confidence_df[camera_mask].sort_values("frames")
        bad_fraction = (1 - camera_df["good_data"]).rolling(window=density_window, center=True, min_periods=1).mean()
        density_ok = (bad_fraction <= max_bad_fraction).astype(int)
        confidence_df.loc[camera_mask, "density_threshold"] = density_ok.values

    confidence_df["good_data"] = ((confidence_df["blink_threshold"] == 1) & (confidence_df["confidence_threshold"] == 1) & (confidence_df["eye_position_threshold"] == 1) & (confidence_df["density_threshold"] == 1)).astype(int)

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
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s")
    )

    bad_eye_data(recording_folder=recording_folder)