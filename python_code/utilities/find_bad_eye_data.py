import pandas as pd
from pathlib import Path
from time import perf_counter

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.utilities.get_mean_dlc_confidence import get_mean_dlc_confidence


def check_single_eye(frame: int, vertical_threshold: float, horizontal_threshold: float, analysis_df: pd.DataFrame) -> int:
    filtered_rows = analysis_df[analysis_df["frame"] == frame]

    if len(filtered_rows) == 0:
        print(f"Warning: no rows remaining after filtering for frame {frame}")
        return 0

    tear_duct_row = filtered_rows[filtered_rows["keypoint"] == 'tear_duct']
    outer_eye_row = filtered_rows[filtered_rows["keypoint"] == 'outer_eye']
    pupil_points = filtered_rows[filtered_rows["keypoint"].isin(['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])]

    if tear_duct_row.empty or outer_eye_row.empty or pupil_points.empty:
        print(f"Warning: missing keypoints for frame {frame}")
        return 0

    tear_duct_x = tear_duct_row['x'].iloc[0]
    outer_eye_x = outer_eye_row['x'].iloc[0]
    pupil_center_x = pupil_points['x'].mean()
    pupil_center_y = pupil_points['y'].mean()

    if (outer_eye_x - tear_duct_x) < horizontal_threshold:
        return 0
    if pupil_center_x < tear_duct_x or pupil_center_x > outer_eye_x:
        return 0
    if abs(pupil_center_y) > vertical_threshold:
        return 0

    return 1


def _find_blink_frames(eye_sorted: pd.DataFrame, drop_threshold: float, n_recovery: int) -> set:
    conf = eye_sorted["mean_confidence"].tolist()
    frames = eye_sorted["frames"].tolist()
    diff = [0.0] + [conf[i] - conf[i - 1] for i in range(1, len(conf))]
    bad_frames = set()
    for i in range(1, len(diff) - 1):
        if diff[i] < -drop_threshold and diff[i + 1] < -drop_threshold:
            onset_conf = conf[i]
            bad_frames.add(frames[i])
            if i + 1 < len(frames):
                bad_frames.add(frames[i + 1])
            for j in range(i + 2, min(i + 2 + n_recovery, len(frames))):
                if conf[j] < onset_conf:
                    bad_frames.add(frames[j])
    return bad_frames


def find_bad_eye_data(
        confidence_df: pd.DataFrame,
        analysis_df: pd.DataFrame,
        # Low-level params (loose — fewest frames rejected)
        confidence_n_std_low: float = 6.0,
        # Medium-level params (current defaults)
        confidence_n_std: float = 4.0,
        blink_drop_n_std: float = 1.0,
        blink_n_std: float = 3.0,
        # High-level params (strict — most frames rejected)
        confidence_n_std_high: float = 3.0,
        blink_drop_n_std_high: float = 0.7,
        blink_n_std_high: float = 2.0,
        # Shared params
        blink_n_recovery_frames: int = 3,
        blink_baseline_window: int = 500,
        blink_trailing_frames: int = 10,
        vertical_threshold: float = 25,
        horizontal_threshold: float = 100,
        density_window: int = 150,
        max_bad_fraction: float = 0.4,
    ) -> pd.DataFrame:
    confidence_df["confidence_threshold_low"] = 1
    confidence_df["confidence_threshold"] = 1
    confidence_df["confidence_threshold_high"] = 1
    confidence_df["blink_threshold"] = 1
    confidence_df["blink_threshold_high"] = 1
    confidence_df["combined_blink_threshold"] = 1
    confidence_df["combined_blink_threshold_high"] = 1
    confidence_df["eye_position_threshold"] = 1
    eye0_mask = confidence_df["camera"] == "eye0"
    eye1_mask = confidence_df["camera"] == "eye1"

    # Confidence thresholds: all three levels computed per-camera in one pass
    for camera_mask in [eye0_mask, eye1_mask]:
        conf = confidence_df.loc[camera_mask, "mean_confidence"]
        trimmed_median = conf[conf >= conf.quantile(0.10)].median()
        std = conf.std()
        confidence_df.loc[camera_mask, "confidence_threshold_low"]  = (conf > trimmed_median - confidence_n_std_low  * std).astype(int).values
        confidence_df.loc[camera_mask, "confidence_threshold"]      = (conf > trimmed_median - confidence_n_std      * std).astype(int).values
        confidence_df.loc[camera_mask, "confidence_threshold_high"] = (conf > trimmed_median - confidence_n_std_high * std).astype(int).values

    # Per-eye shape-based blink detection at medium and high params
    eye0_sorted = confidence_df[eye0_mask].sort_values("frames").reset_index(drop=True)
    eye1_sorted = confidence_df[eye1_mask].sort_values("frames").reset_index(drop=True)
    eye0_std = eye0_sorted["mean_confidence"].std()
    eye1_std = eye1_sorted["mean_confidence"].std()

    eye0_blink_frames = _find_blink_frames(eye0_sorted, blink_drop_n_std * eye0_std, blink_n_recovery_frames)
    eye1_blink_frames = _find_blink_frames(eye1_sorted, blink_drop_n_std * eye1_std, blink_n_recovery_frames)
    eye0_blink_frames_high = _find_blink_frames(eye0_sorted, blink_drop_n_std_high * eye0_std, blink_n_recovery_frames)
    eye1_blink_frames_high = _find_blink_frames(eye1_sorted, blink_drop_n_std_high * eye1_std, blink_n_recovery_frames)

    # Combined blink detection (both eyes simultaneously) at medium and high params
    eye0_conf = eye0_sorted["mean_confidence"]
    eye1_conf = eye1_sorted["mean_confidence"]
    eye0_dip = (eye0_conf.rolling(blink_baseline_window, center=True, min_periods=1).median() - eye0_conf) / eye0_conf.std()
    eye1_dip = (eye1_conf.rolling(blink_baseline_window, center=True, min_periods=1).median() - eye1_conf) / eye1_conf.std()
    eye0_dip_by_frame = dict(zip(eye0_sorted["frames"], eye0_dip))
    eye1_dip_by_frame = dict(zip(eye1_sorted["frames"], eye1_dip))
    shared_frames = eye0_dip_by_frame.keys() & eye1_dip_by_frame.keys()
    combined_blink_frame_set = {
        f for f in shared_frames
        if eye0_dip_by_frame[f] > blink_n_std and eye1_dip_by_frame[f] > blink_n_std
    }
    combined_blink_frame_set_high = {
        f for f in shared_frames
        if eye0_dip_by_frame[f] > blink_n_std_high and eye1_dip_by_frame[f] > blink_n_std_high
    }

    cleaned_mask = analysis_df["processing_level"] == "cleaned"
    eye0_analysis = analysis_df[(analysis_df["video"] == "eye0") & cleaned_mask]
    eye1_analysis = analysis_df[(analysis_df["video"] == "eye1") & cleaned_mask]

    eye0_blink_countdown = 0
    eye1_blink_countdown = 0
    eye0_blink_countdown_high = 0
    eye1_blink_countdown_high = 0
    combined_blink_countdown = 0
    combined_blink_countdown_high = 0

    for frame in sorted(confidence_df["frames"].unique()):
        frame_mask = confidence_df["frames"] == frame
        eye0_is_blink = frame in eye0_blink_frames
        eye1_is_blink = frame in eye1_blink_frames
        eye0_is_blink_high = frame in eye0_blink_frames_high
        eye1_is_blink_high = frame in eye1_blink_frames_high

        # Medium per-eye blink
        if eye0_is_blink:
            confidence_df.loc[frame_mask & eye0_mask, "blink_threshold"] = 0
            eye0_blink_countdown = blink_trailing_frames
        elif eye0_blink_countdown > 0:
            confidence_df.loc[frame_mask & eye0_mask, "blink_threshold"] = 0
            eye0_blink_countdown -= 1

        if eye1_is_blink:
            confidence_df.loc[frame_mask & eye1_mask, "blink_threshold"] = 0
            eye1_blink_countdown = blink_trailing_frames
        elif eye1_blink_countdown > 0:
            confidence_df.loc[frame_mask & eye1_mask, "blink_threshold"] = 0
            eye1_blink_countdown -= 1

        # High per-eye blink
        if eye0_is_blink_high:
            confidence_df.loc[frame_mask & eye0_mask, "blink_threshold_high"] = 0
            eye0_blink_countdown_high = blink_trailing_frames
        elif eye0_blink_countdown_high > 0:
            confidence_df.loc[frame_mask & eye0_mask, "blink_threshold_high"] = 0
            eye0_blink_countdown_high -= 1

        if eye1_is_blink_high:
            confidence_df.loc[frame_mask & eye1_mask, "blink_threshold_high"] = 0
            eye1_blink_countdown_high = blink_trailing_frames
        elif eye1_blink_countdown_high > 0:
            confidence_df.loc[frame_mask & eye1_mask, "blink_threshold_high"] = 0
            eye1_blink_countdown_high -= 1

        # Combined medium blink
        if frame in combined_blink_frame_set:
            confidence_df.loc[frame_mask & eye0_mask, "combined_blink_threshold"] = 0
            confidence_df.loc[frame_mask & eye1_mask, "combined_blink_threshold"] = 0
            combined_blink_countdown = blink_trailing_frames
        elif combined_blink_countdown > 0:
            confidence_df.loc[frame_mask & eye0_mask, "combined_blink_threshold"] = 0
            confidence_df.loc[frame_mask & eye1_mask, "combined_blink_threshold"] = 0
            combined_blink_countdown -= 1

        # Combined high blink
        if frame in combined_blink_frame_set_high:
            confidence_df.loc[frame_mask & eye0_mask, "combined_blink_threshold_high"] = 0
            confidence_df.loc[frame_mask & eye1_mask, "combined_blink_threshold_high"] = 0
            combined_blink_countdown_high = blink_trailing_frames
        elif combined_blink_countdown_high > 0:
            confidence_df.loc[frame_mask & eye0_mask, "combined_blink_threshold_high"] = 0
            confidence_df.loc[frame_mask & eye1_mask, "combined_blink_threshold_high"] = 0
            combined_blink_countdown_high -= 1

        # Eye position — computed once per frame, skipped during medium blink windows
        if not eye0_is_blink and eye0_blink_countdown == 0:
            confidence_df.loc[frame_mask & eye0_mask, "eye_position_threshold"] = check_single_eye(frame, vertical_threshold, horizontal_threshold, eye0_analysis)
        if not eye1_is_blink and eye1_blink_countdown == 0:
            confidence_df.loc[frame_mask & eye1_mask, "eye_position_threshold"] = check_single_eye(frame, vertical_threshold, horizontal_threshold, eye1_analysis)

    # Compute density from the medium-level pre-density good_data
    good_data_medium_pre_density = (
        (confidence_df["confidence_threshold"] == 1)
        & (confidence_df["blink_threshold"] == 1)
        & (confidence_df["combined_blink_threshold"] == 1)
        & (confidence_df["eye_position_threshold"] == 1)
    )
    confidence_df["density_threshold"] = 1
    for camera in confidence_df["camera"].unique():
        camera_mask = confidence_df["camera"] == camera
        camera_df = good_data_medium_pre_density[camera_mask].reset_index(drop=True)
        bad_fraction = (1 - camera_df).rolling(window=density_window, center=True, min_periods=1).mean()
        confidence_df.loc[camera_mask, "density_threshold"] = (bad_fraction <= max_bad_fraction).astype(int).values

    # Final good_data columns for each level
    confidence_df["good_data_low"] = (
        (confidence_df["confidence_threshold_low"] == 1)
        & (confidence_df["eye_position_threshold"] == 1)
    ).astype(int)

    confidence_df["good_data_medium"] = (
        good_data_medium_pre_density
        & (confidence_df["density_threshold"] == 1)
    ).astype(int)

    confidence_df["good_data_high"] = (
        (confidence_df["confidence_threshold_high"] == 1)
        & (confidence_df["blink_threshold_high"] == 1)
        & (confidence_df["combined_blink_threshold_high"] == 1)
        & (confidence_df["eye_position_threshold"] == 1)
        & (confidence_df["density_threshold"] == 1)
    ).astype(int)

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
    for level in ("low", "medium", "high"):
        pct = (updated_df[f"good_data_{level}"] == 0).mean() * 100
        print(f"  {level}: {pct:.2f}% bad")


if __name__ == '__main__':
    recording_folder = RecordingFolder.from_folder_path(
        Path("/home/scholl-lab/ferret_recordings/session_2025-07-09_ferret_757_EyeCameras_P41_E13/full_recording")
    )

    bad_eye_data(recording_folder=recording_folder)
