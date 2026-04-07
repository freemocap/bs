import pandas as pd
import polars as pl
from pathlib import Path
from time import perf_counter

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.utilities.get_mean_dlc_confidence import get_mean_dlc_confidence


def _compute_eye_position_threshold(
    analysis_df: pd.DataFrame,
    vertical_threshold: float,
    horizontal_threshold: float,
) -> pd.Series:
    """Return a Series indexed by frame with value 1 (good) or 0 (bad).
    Frames with missing keypoints default to 0."""
    tear_duct = analysis_df[analysis_df["keypoint"] == "tear_duct"].groupby("frame")["x"].first()
    outer_eye = analysis_df[analysis_df["keypoint"] == "outer_eye"].groupby("frame")["x"].first()
    pupils = analysis_df[analysis_df["keypoint"].isin(["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"])]
    pupil_x = pupils.groupby("frame")["x"].mean()
    pupil_y = pupils.groupby("frame")["y"].mean()

    valid = tear_duct.index.intersection(outer_eye.index).intersection(pupil_x.index)
    td, oe, px, py = tear_duct[valid], outer_eye[valid], pupil_x[valid], pupil_y[valid]

    ok = (
        ((oe - td) >= horizontal_threshold)
        & (px >= td) & (px <= oe)
        & (py.abs() <= vertical_threshold)
    )
    result = pd.Series(0, index=valid)
    result[valid] = ok.astype(int)
    return result


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


def _expand_blink_trail(blink_frames: set, sorted_frames: list, trailing: int) -> set:
    if not blink_frames or trailing == 0:
        return set(blink_frames)
    frame_to_idx = {f: i for i, f in enumerate(sorted_frames)}
    expanded = set(blink_frames)
    for f in blink_frames:
        idx = frame_to_idx[f]
        for j in range(1, trailing + 1):
            if idx + j < len(sorted_frames):
                expanded.add(sorted_frames[idx + j])
    return expanded


def find_bad_eye_data(
        confidence_df: pd.DataFrame,
        analysis_df: pd.DataFrame,
        # Low/medium-level params (shared confidence threshold)
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
    confidence_df["confidence_threshold"] = 1
    confidence_df["confidence_threshold_high"] = 1
    confidence_df["blink_threshold"] = 1
    confidence_df["blink_threshold_high"] = 1
    confidence_df["combined_blink_threshold"] = 1
    confidence_df["combined_blink_threshold_high"] = 1
    confidence_df["eye_position_threshold"] = 1
    eye0_mask = confidence_df["camera"] == "eye0"
    eye1_mask = confidence_df["camera"] == "eye1"

    # Confidence thresholds: low/medium share the same threshold, high uses tighter params
    for camera_mask in [eye0_mask, eye1_mask]:
        conf = confidence_df.loc[camera_mask, "mean_confidence"]
        trimmed_median = conf[conf >= conf.quantile(0.10)].median()
        std = conf.std()
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

    # Expand all blink sets with trailing frames
    sorted_frames = sorted(confidence_df["frames"].unique())
    eye0_blink_all          = _expand_blink_trail(eye0_blink_frames,             sorted_frames, blink_trailing_frames)
    eye1_blink_all          = _expand_blink_trail(eye1_blink_frames,             sorted_frames, blink_trailing_frames)
    eye0_blink_all_high     = _expand_blink_trail(eye0_blink_frames_high,        sorted_frames, blink_trailing_frames)
    eye1_blink_all_high     = _expand_blink_trail(eye1_blink_frames_high,        sorted_frames, blink_trailing_frames)
    combined_blink_all      = _expand_blink_trail(combined_blink_frame_set,      sorted_frames, blink_trailing_frames)
    combined_blink_all_high = _expand_blink_trail(combined_blink_frame_set_high, sorted_frames, blink_trailing_frames)

    # Blink threshold columns — vectorized assignment
    confidence_df.loc[confidence_df["frames"].isin(eye0_blink_all) & eye0_mask, "blink_threshold"] = 0
    confidence_df.loc[confidence_df["frames"].isin(eye1_blink_all) & eye1_mask, "blink_threshold"] = 0
    confidence_df.loc[confidence_df["frames"].isin(eye0_blink_all_high) & eye0_mask, "blink_threshold_high"] = 0
    confidence_df.loc[confidence_df["frames"].isin(eye1_blink_all_high) & eye1_mask, "blink_threshold_high"] = 0
    confidence_df.loc[confidence_df["frames"].isin(combined_blink_all), "combined_blink_threshold"] = 0
    confidence_df.loc[confidence_df["frames"].isin(combined_blink_all_high), "combined_blink_threshold_high"] = 0

    # Eye position threshold — computed for all frames at once
    cleaned_mask = analysis_df["processing_level"] == "cleaned"
    eye0_analysis = analysis_df[(analysis_df["video"] == "eye0") & cleaned_mask]
    eye1_analysis = analysis_df[(analysis_df["video"] == "eye1") & cleaned_mask]
    eye0_pos = _compute_eye_position_threshold(eye0_analysis, vertical_threshold, horizontal_threshold)
    eye1_pos = _compute_eye_position_threshold(eye1_analysis, vertical_threshold, horizontal_threshold)
    confidence_df.loc[eye0_mask, "eye_position_threshold"] = (
        confidence_df.loc[eye0_mask, "frames"].map(eye0_pos).fillna(0).astype(int).values
    )
    confidence_df.loc[eye1_mask, "eye_position_threshold"] = (
        confidence_df.loc[eye1_mask, "frames"].map(eye1_pos).fillna(0).astype(int).values
    )

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
        (confidence_df["confidence_threshold"] == 1)
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


def save_eye_data_quality_csv(recording_folder: RecordingFolder, confidence_df: pl.DataFrame) -> None:
    analyzable_output = recording_folder.folder_path / "analyzable_output"

    eye_to_trajectory = {
        recording_folder.left_eye_name:  "left_eye_data_quality",
        recording_folder.right_eye_name: "right_eye_data_quality",
    }
    component_map = {
        "good_data_low":    "low_threshold",
        "good_data_medium": "medium_threshold",
        "good_data_high":   "high_threshold",
    }

    chunks = []
    for camera_name, trajectory_name in eye_to_trajectory.items():
        chunk = (
            confidence_df
            .filter(pl.col("camera") == camera_name)
            .select(["frames", "timestamps", "good_data_low", "good_data_medium", "good_data_high"])
            .rename({"frames": "frame", "timestamps": "timestamp_s"})
            .unpivot(
                on=["good_data_low", "good_data_medium", "good_data_high"],
                index=["frame", "timestamp_s"],
                variable_name="component",
                value_name="value",
            )
            .with_columns([
                pl.col("component").replace(component_map).cast(pl.Categorical),
                pl.lit(trajectory_name).alias("trajectory").cast(pl.Categorical),
                pl.lit("boolean").alias("units").cast(pl.Categorical),
            ])
            .select(["frame", "timestamp_s", "trajectory", "component", "value", "units"])
        )
        chunks.append(chunk)

    pl.concat(chunks).write_csv(analyzable_output / "eye_data_quality.csv")


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
