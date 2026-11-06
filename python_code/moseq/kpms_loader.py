"""
Keypoint MoSeq Loader
=====================

Custom data loaders for Keypoint MoSeq (kpms).

load_3d_data_kpms
    Reads head_freemocap_data_by_frame.csv from each recording's
    mocap_3d_data directory.  Accepts a single RecordingFolder or a list.

load_solver_output_kpms
    Reads the mocap solver tidy output:
        frame, timestamp, marker, data_type, x, y, z
    Only rows where data_type == 'optimized' are used.
    Accepts a single RecordingFolder or a list of them.

load_3d_eye_kpms
    Reads left_eye_trajectories_resampled.csv and
    right_eye_trajectories_resampled.csv from a RecordingFolder (or list).
    CSV format: frame, timestamp, trajectory, component, value, units
    Keys returned: "{recording_name}_left_eye", "{recording_name}_right_eye"

load_head_with_pupil_points_kpms
    Reads skull_kinematics.parquet (head keypoints) and
    left/right_gaze_kinematics.parquet (tracked pupil boundary points, world
    space) from a RecordingFolder (or list), merging them into a single
    combined keypoint set per recording.

load_both_eyes_skull_frame_kpms
    Reads both eyes' resampled eye-camera keypoints and expresses them in
    the skull body frame (head motion fully removed), with confidences from
    analyzable_output/eye_data_quality.csv. For eye-only models fit with
    fix_heading=True.

load_eye_in_head_kpms
    Reads the `eye_in_head` trajectory (adduction, elevation -- anatomical
    gaze angles already expressed in the skull's own body frame) from
    left/right_eye_kinematics.csv. Not a keypoint loader -- used alongside
    load_head_with_pupil_points_kpms for syllable-level eye vs. head movement
    analysis (see moseq/visualization/eye_syllable_viz.py).

All return data in the format expected by kpms.load_keypoints():
    coordinates: dict[str, NDArray[(N, K, 3)]]
    confidences: dict[str, NDArray[(N, K)]]
    bodyparts: list[str]
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

from python_code.kinematics_core.tidy_dataframe_io import read_parquet_or_csv
from python_code.moseq.utils.bodyparts import BOTH_EYES_BODYPARTS, HEAD_BODYPARTS
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


_3D_REQUIRED_COLUMNS = {"frame", "keypoint", "x", "y", "z", "trajectory"}
_3D_TRAJECTORY_FILTER = "rigid_3d_xyz"
_3D_FILENAME = "head_freemocap_data_by_frame.csv"


def _pivot_to_kpms_arrays(
    df: pd.DataFrame,
    keypoint_col: str,
    sparse: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """
    Pivot a filtered DataFrame into kpms-compatible arrays.

    Parameters
    ----------
    df:
        Pre-filtered DataFrame with at least columns: frame, x, y, z, and
        the column named by ``keypoint_col``.
    keypoint_col:
        Column whose values become the bodypart labels (e.g. "keypoint" or
        "marker").
    sparse:
        If False (default), frames in the pivot are assumed contiguous and
        confidences are all 1.0.
        If True, the output is allocated from frame 0 to max_frame with NaN
        defaults; confidences are 1.0 where data is present, 0.0 elsewhere.

    Returns
    -------
    coords:   (N_frames, N_keypoints, 3)
    confs:    (N_frames, N_keypoints)
    bodyparts: ordered list of keypoint names
    """
    pivoted = df.pivot(index="frame", columns=keypoint_col, values=["x", "y", "z"])
    pivoted = pivoted.sort_index()

    bodyparts: list[str] = pivoted["x"].columns.tolist()
    n_kp = len(bodyparts)

    if sparse:
        n_frames = int(df["frame"].max()) + 1
        pivoted = pivoted.reindex(
            columns=pd.MultiIndex.from_product([["x", "y", "z"], bodyparts])
        )
        coords = np.full((n_frames, n_kp, 3), np.nan, dtype=np.float64)
        confs = np.zeros((n_frames, n_kp), dtype=np.float64)
        frame_indices = pivoted.index.to_numpy()
        coords[frame_indices] = np.stack(
            [pivoted[c][bodyparts].to_numpy() for c in ("x", "y", "z")], axis=-1
        )
        row_has_data = ~np.isnan(coords[frame_indices]).all(axis=-1)
        confs[frame_indices] = row_has_data.astype(np.float64)
    else:
        n_frames = len(pivoted)
        coords = np.stack(
            [pivoted[c][bodyparts].to_numpy() for c in ("x", "y", "z")], axis=-1
        )  # (N_frames, N_keypoints, 3)
        confs = np.ones((n_frames, n_kp), dtype=np.float64)

    return coords, confs, bodyparts


def load_3d_data_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load 3D triangulated keypoint data for Keypoint MoSeq.

    Reads ``head_freemocap_data_by_frame.csv`` from each recording's
    ``mocap_3d_data`` directory, keeping only rows where
    ``trajectory == 'rigid_3d_xyz'``.

    Parameters
    ----------
    recording_folder:
        A single RecordingFolder or a list of them.

    Returns
    -------
    coordinates:
        Dict mapping ``recording_folder.recording_name`` to array of shape
        (n_frames, n_keypoints, 3).
    confidences:
        Same keys; values are (n_frames, n_keypoints) arrays of ones
        (reprojection_error is not bounded 0-1).
    bodyparts:
        Ordered list of keypoint names. Consistent across all folders.
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}
    bodyparts: list[str] | None = None

    for rf in folders:
        if rf.mocap_3d_data is None:
            raise FileNotFoundError(
                f"mocap_3d_data does not exist for recording '{rf.recording_name}'"
            )

        csv_path = rf.mocap_3d_data / _3D_FILENAME
        if not csv_path.exists():
            raise FileNotFoundError(f"3D data CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        missing = _3D_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing required columns {missing}")

        df = df[df["trajectory"] == _3D_TRAJECTORY_FILTER]
        if df.empty:
            raise ValueError(
                f"{csv_path}: no rows with trajectory == '{_3D_TRAJECTORY_FILTER}'"
            )

        coords, confs, file_bodyparts = _pivot_to_kpms_arrays(df, keypoint_col="keypoint")

        if bodyparts is None:
            bodyparts = file_bodyparts
        elif file_bodyparts != bodyparts:
            raise ValueError(
                f"Bodyparts mismatch in {csv_path}.\n"
                f"Expected: {bodyparts}\n"
                f"Got:      {file_bodyparts}"
            )

        name = rf.recording_name
        if name in coordinates:
            raise ValueError(
                f"Duplicate recording name '{name}'. Each RecordingFolder must "
                "have a unique recording_name."
            )

        coordinates[name] = coords
        confidences[name] = confs

    assert bodyparts is not None
    return coordinates, confidences, bodyparts


_EYE_IN_HEAD_TRAJECTORY = "eye_in_head"
_EYE_IN_HEAD_COMPONENTS = ["adduction", "elevation"]
_EYE_KINEMATICS_REQUIRED_COLUMNS = {"frame", "trajectory", "component", "value"}


def _load_eye_kinematics_components(
    recording_folder: RecordingFolder | list[RecordingFolder],
    trajectory: str,
    components: list[str],
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """
    Read one trajectory's components from each RecordingFolder's
    `{left,right}_eye_kinematics.csv`, as `{recording_name: (n_frames,
    len(components)) array}` per eye, columns in `components` order.
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    left: dict[str, NDArray] = {}
    right: dict[str, NDArray] = {}

    for rf in folders:
        for side, csv_path, out in [
            ("left", rf.left_eye_kinematics_csv, left),
            ("right", rf.right_eye_kinematics_csv, right),
        ]:
            if csv_path is None:
                raise FileNotFoundError(
                    f"{side}_eye_kinematics.csv not found for recording '{rf.recording_name}'"
                )

            df = pd.read_csv(csv_path)
            missing = _EYE_KINEMATICS_REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path}: missing required columns {missing}")

            df = df[df["trajectory"] == trajectory]
            if df.empty:
                raise ValueError(f"{csv_path}: no rows with trajectory == '{trajectory}'")

            wide = df.pivot_table(
                index="frame", columns="component", values="value", aggfunc="first"
            ).sort_index()
            missing_components = set(components) - set(wide.columns)
            if missing_components:
                raise ValueError(
                    f"{csv_path}: missing {trajectory} components {missing_components}"
                )

            name = rf.recording_name
            if name in out:
                raise ValueError(
                    f"Duplicate recording name '{name}'. Each RecordingFolder must "
                    "have a unique recording_name."
                )
            out[name] = wide[components].to_numpy()

    return left, right


def load_eye_in_head_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
    quality_level: str | None = None,
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """
    Load head-relative eye gaze angles (adduction, elevation) for Keypoint
    MoSeq syllable analysis.

    Reads the `eye_in_head` trajectory from each RecordingFolder's
    `{left,right}_eye_kinematics.csv`. These angles are computed by the eye
    kinematics pipeline in the skull's own body frame (see
    `python_code/ferret_gaze/analyzable_output_csvs.md`), so unlike the raw
    pupil-point keypoints used by `load_head_with_pupil_points_kpms` they are
    unaffected by head rotation. They are also resampled onto the same
    common-timestamp grid as `skull_kinematics.parquet` (see
    `ferret_data_resampler.resample_ferret_data`), so they line up
    frame-for-frame with `load_head_with_pupil_points_kpms` output and with a
    fitted model's `results[...]["syllable"]`/`"centroid"`/`"heading"` arrays
    for the same recording -- no further alignment is needed, though callers
    should still assert matching frame counts before combining them.

    Parameters
    ----------
    recording_folder:
        A single RecordingFolder or a list of them.
    quality_level:
        If given (one of `EYE_DATA_QUALITY_LEVELS`), set each eye's angles to
        NaN on frames where that eye fails `load_eye_data_quality` at this
        level (blinks, tracking dropouts), so they don't show up as large
        spurious eye movements. None (default) returns every frame.

    Returns
    -------
    eye_in_head_left, eye_in_head_right:
        Each a dict mapping `recording_folder.recording_name` to an array of
        shape (n_frames, 2) with columns [adduction_deg, elevation_deg].
    """
    left, right = _load_eye_kinematics_components(
        recording_folder, _EYE_IN_HEAD_TRAJECTORY, _EYE_IN_HEAD_COMPONENTS
    )
    left = {name: np.degrees(v) for name, v in left.items()}
    right = {name: np.degrees(v) for name, v in right.items()}

    if quality_level is not None:
        folders = (
            [recording_folder] if isinstance(recording_folder, RecordingFolder) else recording_folder
        )
        for rf in folders:
            name = rf.recording_name
            left_good, right_good = load_eye_data_quality(rf, quality_level)
            for side, angles, good in [("left", left, left_good), ("right", right, right_good)]:
                if len(good) != len(angles[name]):
                    raise ValueError(
                        f"'{name}': {side} eye_in_head has {len(angles[name])} frames but "
                        f"eye_data_quality.csv has {len(good)}."
                    )
                angles[name] = np.where(good[:, None], angles[name], np.nan)

    return left, right


def load_pupil_radius_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """
    Load per-eye pupil size as the mean 3D distance of the 8 tracked pupil
    boundary points (p1..p8) from the tracked pupil center, from each
    RecordingFolder's `{left,right}_eye_trajectories_resampled.csv`. Same
    common-timestamp grid as `load_eye_in_head_kpms`.

    Used instead of `{side}_eye_kinematics.csv`'s `pupil_axis`, which is an
    ellipse fit to the same 8 points and much less stable frame to frame.
    Being derived from the same points a pupil-point model is fit on, this
    isn't an independent measurement -- it's a direct readout of the pupil
    size information present in the keypoints, to check whether a fitted
    model's latent space keeps it (see `visualization/eye_pc_viz.py`).

    Returns
    -------
    pupil_radius_left, pupil_radius_right:
        Each a dict mapping `recording_folder.recording_name` to an array of
        shape (n_frames,), mm.
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )
    boundary_points = [f"p{i}" for i in range(1, 9)]

    left: dict[str, NDArray] = {}
    right: dict[str, NDArray] = {}
    for rf in folders:
        for side, csv_path, out in [
            ("left", rf.left_eye_resampled_trajectories, left),
            ("right", rf.right_eye_resampled_trajectories, right),
        ]:
            if csv_path is None:
                raise FileNotFoundError(
                    f"{side} eye resampled trajectories not found for '{rf.recording_name}'"
                )
            df = pd.read_csv(csv_path)
            df = df[df["trajectory"].isin(boundary_points + ["pupil_center"])]
            wide = _long_to_wide_xyz(df, csv_path)
            pivoted = wide.pivot(index="frame", columns="trajectory", values=["x", "y", "z"]).sort_index()
            xyz = lambda names: np.stack(
                [pivoted[c][names].to_numpy() for c in ("x", "y", "z")], axis=-1
            )
            center = xyz(["pupil_center"])  # (n_frames, 1, 3)
            name = rf.recording_name
            if name in out:
                raise ValueError(
                    f"Duplicate recording name '{name}'. Each RecordingFolder must "
                    "have a unique recording_name."
                )
            out[name] = np.linalg.norm(xyz(boundary_points) - center, axis=-1).mean(axis=1)

    return left, right


_EYE_DATA_QUALITY_FILENAME = "eye_data_quality.csv"
EYE_DATA_QUALITY_LEVELS = ("low", "medium", "high")


def _resample_raw_eye_quality_flags(
    good: NDArray[np.bool_],
    raw_timestamps: NDArray[np.float64],
    common_timestamps_original: NDArray[np.float64],
    csv_path: Path,
    side: str,
) -> NDArray[np.bool_]:
    """
    Some older `eye_data_quality.csv` files were written at the raw eye
    camera rate with absolute (unzeroed) timestamps instead of on the
    common-timestamp grid. Map those onto the grid with the same
    nearest-neighbor rule `find_bad_eye_data.save_eye_data_quality_csv`
    uses now, rather than requiring the source confidence CSV to regenerate
    the file.
    """
    tolerance = 2 * np.median(np.diff(common_timestamps_original))
    if (
        raw_timestamps[0] > common_timestamps_original[0] + tolerance
        or raw_timestamps[-1] < common_timestamps_original[-1] - tolerance
    ):
        raise ValueError(
            f"{csv_path}: {side} eye has {len(good)} rows (common grid has "
            f"{len(common_timestamps_original)}) and its timestamps don't cover "
            "common_timestamps_original.npy, so it can't be mapped onto the grid."
        )
    insert_idx = np.clip(np.searchsorted(raw_timestamps, common_timestamps_original), 0, len(raw_timestamps) - 1)
    left_idx = np.maximum(insert_idx - 1, 0)
    use_left = (common_timestamps_original - raw_timestamps[left_idx]) < (
        raw_timestamps[insert_idx] - common_timestamps_original
    )
    nearest_idx = np.where(use_left, left_idx, insert_idx)
    print(
        f"Warning: {csv_path} {side} eye is at the raw eye-camera rate "
        f"({len(good)} rows); nearest-neighbor resampled onto the "
        f"{len(common_timestamps_original)}-frame common grid."
    )
    return good[nearest_idx]


def load_eye_data_quality(
    recording_folder: RecordingFolder,
    level: str = "medium",
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Load per-frame good/bad eye data flags for one recording.

    Reads `analyzable_output/eye_data_quality.csv`, written by
    `find_bad_eye_data.save_eye_data_quality_csv` during the gaze pipeline:
    per-eye boolean flags from DLC confidence, blink detection (single- and
    both-eye confidence dips), pupil-inside-eye position checks, and a
    rolling bad-data density check, already nearest-neighbor resampled onto
    the common-timestamp grid (so frame-aligned with the resampled eye/skull
    trajectories).

    Parameters
    ----------
    level:
        One of `EYE_DATA_QUALITY_LEVELS`. "low" only checks confidence and
        pupil position; "medium" adds blink and density checks; "high" uses
        stricter thresholds for all of them (see `find_bad_eye_data`).

    Returns
    -------
    left_good, right_good:
        `(n_frames,)` bool arrays, True where that eye's data passes.
    """
    if level not in EYE_DATA_QUALITY_LEVELS:
        raise ValueError(f"level must be one of {EYE_DATA_QUALITY_LEVELS}, got '{level}'")
    if recording_folder.analyzable_output is None:
        raise FileNotFoundError(f"No analyzable_output for '{recording_folder.recording_name}'")
    csv_path = recording_folder.analyzable_output / _EYE_DATA_QUALITY_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- it's written by the gaze pipeline "
            "(run_gaze_pipeline) when eye_model_v3_mean_confidence.csv exists."
        )

    common_timestamps = np.load(recording_folder.analyzable_output / "common_timestamps.npy")
    common_timestamps_original = np.load(
        recording_folder.analyzable_output / "common_timestamps_original.npy"
    )

    df = pd.read_csv(csv_path)
    df = df[df["component"] == f"{level}_threshold"]

    flags = []
    for side in ("left", "right"):
        side_df = df[df["trajectory"] == f"{side}_eye_data_quality"].sort_values("frame")
        frames = side_df["frame"].to_numpy()
        if side_df.empty or not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(
                f"{csv_path}: {side} eye {level}_threshold rows missing or not "
                "contiguous from frame 0"
            )
        good = side_df["value"].to_numpy() == 1
        if len(good) != len(common_timestamps):
            good = _resample_raw_eye_quality_flags(
                good, side_df["timestamp_s"].to_numpy(), common_timestamps_original, csv_path, side
            )
        flags.append(good)

    return flags[0], flags[1]


_SOLVER_REQUIRED_COLUMNS = {"frame", "marker", "data_type", "x", "y", "z"}

_SOLVER_DATA_TYPE_FILTER = "optimized"
_SOLVER_FILENAME = "tidy_trajectory_data.csv"


def load_solver_output_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load mocap solver output for Keypoint MoSeq.

    Reads ``tidy_trajectory_data.csv`` from each recording's
    ``mocap_solver_output`` directory, keeping only rows where
    ``data_type == 'optimized'``.

    Parameters
    ----------
    recording_folder:
        A single RecordingFolder or a list of them.

    Returns
    -------
    coordinates:
        Dict mapping ``recording_folder.recording_name`` to array of shape
        (n_frames, n_keypoints, 3). Missing frames have NaN values.
    confidences:
        Same keys; values are (n_frames, n_keypoints) with 1.0 where data is
        present and 0.0 for missing frames.
    bodyparts:
        Sorted list of marker names. Consistent across all folders.
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}
    bodyparts: list[str] | None = None

    for rf in folders:
        if rf.mocap_solver_output is None:
            raise FileNotFoundError(
                f"mocap_solver_output does not exist for recording '{rf.recording_name}'"
            )

        csv_path = rf.mocap_solver_output / _SOLVER_FILENAME
        if not csv_path.exists():
            raise FileNotFoundError(f"Solver output CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        missing = _SOLVER_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing required columns {missing}")

        df = df[df["data_type"] == _SOLVER_DATA_TYPE_FILTER]
        if df.empty:
            raise ValueError(
                f"{csv_path}: no rows with data_type == '{_SOLVER_DATA_TYPE_FILTER}'"
            )

        coords, confs, file_bodyparts = _pivot_to_kpms_arrays(df, keypoint_col="marker", sparse=True)

        if bodyparts is None:
            bodyparts = file_bodyparts
        elif file_bodyparts != bodyparts:
            raise ValueError(
                f"Marker mismatch in {csv_path}.\n"
                f"Expected: {bodyparts}\n"
                f"Got:      {file_bodyparts}"
            )

        name = rf.recording_name
        if name in coordinates:
            raise ValueError(
                f"Duplicate recording name '{name}'. Each RecordingFolder must "
                "have a unique recording_name."
            )

        coordinates[name] = coords
        confidences[name] = confs

    assert bodyparts is not None
    return coordinates, confidences, bodyparts

_SKULL_GAZE_FILENAME = "skull_and_spine_trajectories_resampled.csv"
_SKULL_GAZE_REQUIRED_COLUMNS = {"frame", "trajectory", "component", "value"}
_GAZE_TRAJECTORY_FILTER = "keypoint__gaze_target"
_GAZE_REQUIRED_COLUMNS = {"frame", "trajectory", "component", "value"}

_EYE_REQUIRED_COLUMNS = {"frame", "trajectory", "component", "value"}
_EYE_XYZ = {"x", "y", "z"}


def _long_to_wide_xyz(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    Reshape a long-format eye trajectory DataFrame to wide format.

    Input columns:  frame, trajectory, component (x/y/z), value, ...
    Output columns: frame, trajectory, x, y, z
    """
    df = df[df["component"].isin(_EYE_XYZ)]
    if df.empty:
        raise ValueError(f"{csv_path}: no rows with component in {_EYE_XYZ}")

    wide = df.pivot_table(
        index=["frame", "trajectory"],
        columns="component",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def load_3d_eye_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load 3D eye trajectory data for Keypoint MoSeq.

    Reads ``left_eye_trajectories_resampled.csv`` and
    ``right_eye_trajectories_resampled.csv`` from each RecordingFolder,
    producing two entries per recording keyed as
    ``"{recording_name}_left_eye"`` and ``"{recording_name}_right_eye"``.

    CSV format: frame, timestamp, trajectory, component, value, units

    Parameters
    ----------
    recording_folder:
        A single RecordingFolder or a list of them.

    Returns
    -------
    coordinates:
        Dict with shape (n_frames, n_keypoints, 3) per eye per recording.
    confidences:
        Same keys; all values are 1.0 (no confidence score in this format).
    bodyparts:
        Ordered list of trajectory names. Consistent across all eyes/folders.
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}
    bodyparts: list[str] | None = None

    for rf in folders:
        for side, csv_path in [
            ("left_eye", rf.left_eye_resampled_trajectories),
            ("right_eye", rf.right_eye_resampled_trajectories),
        ]:
            if csv_path is None:
                raise FileNotFoundError(
                    f"{side} resampled trajectories not found for '{rf.recording_name}'"
                )

            df = pd.read_csv(csv_path)

            missing = _EYE_REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path}: missing required columns {missing}")

            wide = _long_to_wide_xyz(df, csv_path)
            coords, confs, file_bodyparts = _pivot_to_kpms_arrays(wide, keypoint_col="trajectory")

            if bodyparts is None:
                bodyparts = file_bodyparts
            elif file_bodyparts != bodyparts:
                raise ValueError(
                    f"Trajectory mismatch in {csv_path}.\n"
                    f"Expected: {bodyparts}\n"
                    f"Got:      {file_bodyparts}"
                )

            name = f"{rf.recording_name}_{side}"
            if name in coordinates:
                raise ValueError(
                    f"Duplicate key '{name}'. Each RecordingFolder must have a unique recording_name."
                )

            coordinates[name] = coords
            confidences[name] = confs

    assert bodyparts is not None
    return coordinates, confidences, bodyparts


def load_both_eyes_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load both eyes combined into a single entry for Keypoint MoSeq.

    Reads ``left_eye_trajectories_resampled.csv`` and
    ``right_eye_trajectories_resampled.csv`` from each RecordingFolder.
    Trajectory names are prefixed with ``"left_"`` / ``"right_"`` to
    disambiguate, and each eye is shifted ±4 mm in x so the left eye sits at
    positive x and the right eye at negative x (preventing overlap since each
    eye spans roughly ±3 mm).

    The result is a single dict entry per recording (keyed by
    ``recording_name``) with shape (n_frames, 22, 3).

    Parameters
    ----------
    recording_folder:
        A single RecordingFolder or a list of them.

    Returns
    -------
    coordinates:
        Dict mapping ``recording_folder.recording_name`` to array of shape
        (n_frames, 22, 3).
    confidences:
        Same keys; all values are 1.0.
    bodyparts:
        ``BOTH_EYES_BODYPARTS`` (22 names).
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}

    for rf in folders:
        for csv_path in (rf.left_eye_resampled_trajectories, rf.right_eye_resampled_trajectories):
            if csv_path is None:
                raise FileNotFoundError(
                    f"Eye resampled trajectories not found for '{rf.recording_name}'"
                )

        left_df = pd.read_csv(rf.left_eye_resampled_trajectories)
        right_df = pd.read_csv(rf.right_eye_resampled_trajectories)

        # Shift eyes apart in x so they don't overlap (each is ~±3 mm wide)
        left_df.loc[left_df["component"] == "x", "value"] += 4.0
        right_df.loc[right_df["component"] == "x", "value"] -= 4.0

        left_df["trajectory"] = "left_" + left_df["trajectory"]
        right_df["trajectory"] = "right_" + right_df["trajectory"]

        combined = pd.concat(
            [_long_to_wide_xyz(left_df, rf.left_eye_resampled_trajectories),
             _long_to_wide_xyz(right_df, rf.right_eye_resampled_trajectories)],
            ignore_index=True,
        )

        coords, confs, _ = _pivot_to_kpms_arrays(combined, keypoint_col="trajectory")

        name = rf.recording_name
        if name in coordinates:
            raise ValueError(
                f"Duplicate recording name '{name}'. Each RecordingFolder must "
                "have a unique recording_name."
            )

        coordinates[name] = coords
        confidences[name] = confs

    return coordinates, confidences, BOTH_EYES_BODYPARTS


_SKULL_REFERENCE_GEOMETRY_FILENAME = "skull_reference_geometry.json"

# kpms's defaults are tuned for pixel-scale data: `format_data` adds
# uniform +-0.1 noise to every coordinate (`added_noise_level`), and the
# observation model assumes ~1.8 units std of keypoint noise (sigmasq init
# 1, error_estimator noise prior 10**0.5 ~ 3.2 units^2 at conf=1). In mm,
# both are as large as or larger than the eye signal itself (pupil points
# move ~0.15 mm std; measured tracking jitter ~0.001 mm). Scaling mm -> units of 10 um puts the assumed noise
# (~18 um) between the real jitter and the signal, makes the added noise
# negligible, and gives a signal-to-assumed-noise ratio (~10x) comparable to
# pixel data and to the mm-scale head models.
EYE_SKULL_FRAME_COORDINATE_SCALE = 100.0


def _eye_points_in_skull_frame(
    csv_path: Path,
    side: str,
    eye_bodyparts: list[str],
    eye_center_in_skull: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Read one eye's resampled eye-camera-frame keypoints (eyeball-centered,
    mm) and express them in the skull body frame, ordered as `eye_bodyparts`.

    Uses the same fixed socket mounting (`get_eye_to_skull_rotation_matrix`)
    and eyeball center (skull reference geometry `{side}_eye`) as the gaze
    pipeline, so the result is identical to taking the world-space tracked
    pupil points (`project_tracked_pupil_to_world`) and undoing the skull's
    per-frame world pose -- without needing the gaze parquet, and without
    round-tripping through world space.
    """
    from python_code.ferret_gaze.calculate_gaze.calculate_ferret_gaze import (
        get_eye_to_skull_rotation_matrix,
    )

    df = pd.read_csv(csv_path)
    missing = _EYE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")

    wide = _long_to_wide_xyz(df, csv_path)
    pivoted = wide.pivot(index="frame", columns="trajectory", values=["x", "y", "z"]).sort_index()
    frames = pivoted.index.to_numpy()
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError(f"{csv_path}: frames are not contiguous from 0")
    missing_bodyparts = set(eye_bodyparts) - set(pivoted["x"].columns)
    if missing_bodyparts:
        raise ValueError(f"{csv_path}: missing trajectories {sorted(missing_bodyparts)}")

    points_eye = np.stack(
        [pivoted[c][eye_bodyparts].to_numpy() for c in ("x", "y", "z")], axis=-1
    )  # (n_frames, n_kp, 3)
    rotation_eye_to_skull = get_eye_to_skull_rotation_matrix(side)
    return points_eye @ rotation_eye_to_skull.T + eye_center_in_skull


def replace_static_shape(
    coordinates: dict[str, NDArray],
    confidences: dict[str, NDArray],
) -> dict[str, NDArray]:
    """
    Replace each session's static keypoint configuration with the mean
    across sessions, keeping each session's movement around it.

    Per session, each keypoint's median position (over frames with nonzero
    confidence) is subtracted and the across-session mean of those medians
    is added back. This removes per-session offsets such as headmount /
    eye-camera placement and skull reference geometry, which otherwise make
    animal identity the dominant PC (see `animal_identity_decoding`). It also
    removes real per-session static differences, e.g. mean pupil size.

    Only valid for models with no per-frame rotation (`fix_heading=True`):
    kpms's centroid subtraction is linear, so a constant per-keypoint shift
    doesn't change the movement it sees.
    """
    medians = {}
    for name, coords in coordinates.items():
        good = confidences[name] > 0
        medians[name] = np.stack(
            [
                np.nanmedian(coords[good[:, k], k] if good[:, k].any() else coords[:, k], axis=0)
                for k in range(coords.shape[1])
            ]
        )
    template = np.mean(list(medians.values()), axis=0)
    return {name: coords - medians[name] + template for name, coords in coordinates.items()}


def load_both_eyes_skull_frame_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
    quality_level: str | None = "medium",
    coordinate_scale: float = EYE_SKULL_FRAME_COORDINATE_SCALE,
    recenter_static_shape: bool = False,
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load both eyes' keypoints in the skull body frame for an eye-only
    Keypoint MoSeq model.

    Same 22 keypoints as `load_both_eyes_kpms` (per eye: tear_duct, p1..p8,
    outer_eye, pupil_center), but rotated/translated into the skull frame
    (+X nose, +Y left eye, +Z up; see `_eye_points_in_skull_frame`) instead
    of stacking the two eye-camera frames side by side with a fake ±4 mm
    offset. Head motion is therefore fully removed (all 6 DOF, not just the
    translation + yaw kpms's own alignment removes), and both eyes share one
    anatomical frame. Intended to be fit with `fix_heading=True` (see
    `_LOADER_CONFIG[KPMS_Loader.BOTH_EYES_SKULL_FRAME]`) -- there's no
    heading left to infer.

    tear_duct/outer_eye are kept on purpose: kpms subtracts the keypoint
    centroid every Gibbs iteration, so a pupil-only keypoint set would have
    its rigid displacement (i.e. the eye movement itself) absorbed into the
    centroid. The eye corners (fixed in the eye-camera frame) anchor it.

    Parameters
    ----------
    recording_folder:
        A single RecordingFolder or a list of them.
    quality_level:
        `load_eye_data_quality` level used to build confidences: each eye's
        11 keypoints get confidence 1.0 on frames where that eye passes and
        0.0 where it doesn't (blinks, low DLC confidence, pupil outside the
        eye, ...). kpms then interpolates those frames for PCA
        initialization and gives them a much larger noise prior during
        fitting. Pass None for all-ones confidences.
    coordinate_scale:
        Multiplier applied to the mm coordinates (default: units of 10 um;
        see `EYE_SKULL_FRAME_COORDINATE_SCALE` for why kpms needs this).
    recenter_static_shape:
        Replace each session's static keypoint configuration with the
        across-session mean (`replace_static_shape`), so the model sees only
        eye movement, not per-animal headmount/geometry offsets.

    Returns
    -------
    coordinates:
        Dict mapping `recording_name` to (n_frames, 22, 3), skull frame, in
        mm * `coordinate_scale`.
    confidences:
        Same keys, (n_frames, 22).
    bodyparts:
        `BOTH_EYES_BODYPARTS`.
    """
    from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    eye_bodyparts = [bp.removeprefix("left_") for bp in BOTH_EYES_BODYPARTS[:11]]
    assert BOTH_EYES_BODYPARTS == [f"left_{bp}" for bp in eye_bodyparts] + [
        f"right_{bp}" for bp in eye_bodyparts
    ]

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}

    for rf in folders:
        if rf.skull_kinematics is None:
            raise FileNotFoundError(f"skull_kinematics not found for '{rf.recording_name}'")
        skull_reference_geometry = ReferenceGeometry.from_json_file(
            rf.skull_kinematics / _SKULL_REFERENCE_GEOMETRY_FILENAME
        )

        side_coords = []
        for side, csv_path in [
            ("left", rf.left_eye_resampled_trajectories),
            ("right", rf.right_eye_resampled_trajectories),
        ]:
            if csv_path is None:
                raise FileNotFoundError(
                    f"{side} eye resampled trajectories not found for '{rf.recording_name}'"
                )
            side_coords.append(
                _eye_points_in_skull_frame(
                    csv_path,
                    side,
                    eye_bodyparts,
                    skull_reference_geometry.get_keypoint_position(f"{side}_eye"),
                )
            )

        left_coords, right_coords = side_coords
        if left_coords.shape[0] != right_coords.shape[0]:
            raise ValueError(
                f"'{rf.recording_name}': left eye has {left_coords.shape[0]} frames, "
                f"right eye has {right_coords.shape[0]}"
            )
        coords = np.concatenate([left_coords, right_coords], axis=1) * coordinate_scale
        n_frames = coords.shape[0]

        if quality_level is None:
            confs = np.ones((n_frames, len(BOTH_EYES_BODYPARTS)))
        else:
            left_good, right_good = load_eye_data_quality(rf, quality_level)
            if len(left_good) != n_frames:
                raise ValueError(
                    f"'{rf.recording_name}': eye_data_quality.csv has {len(left_good)} "
                    f"frames but eye trajectories have {n_frames}"
                )
            confs = np.concatenate(
                [
                    np.repeat(left_good[:, None], len(eye_bodyparts), axis=1),
                    np.repeat(right_good[:, None], len(eye_bodyparts), axis=1),
                ],
                axis=1,
            ).astype(np.float64)
            print(
                f"{rf.recording_name}: {quality_level}-quality frames "
                f"left={left_good.mean():.1%}, right={right_good.mean():.1%}"
            )

        name = rf.recording_name
        if name in coordinates:
            raise ValueError(
                f"Duplicate recording name '{name}'. Each RecordingFolder must "
                "have a unique recording_name."
            )
        coordinates[name] = coords
        confidences[name] = confs

    if recenter_static_shape:
        coordinates = replace_static_shape(coordinates, confidences)
    return coordinates, confidences, BOTH_EYES_BODYPARTS


def load_skull_and_gaze_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load resampled skull/spine trajectories combined with both eyes' gaze
    targets for Keypoint MoSeq.

    Reads ``analyzable_output/skull_and_spine_trajectories_resampled.csv``
    and both ``analyzable_output/gaze_kinematics/left_gaze_kinematics.csv``
    and ``right_gaze_kinematics.csv`` from each RecordingFolder. Only the
    ``keypoint__gaze_target`` trajectory (x, y, z) is extracted from the gaze
    files; it is renamed to ``left_gaze_target`` / ``right_gaze_target`` before
    merging with the skull data.

    Parameters
    ----------
    recording_folder:
        A single RecordingFolder or a list of them.

    Returns
    -------
    coordinates:
        Dict mapping ``recording_folder.recording_name`` to array of shape
        (n_frames, n_keypoints, 3).
    confidences:
        Same keys; values are (n_frames, n_keypoints) arrays of ones.
    bodyparts:
        Ordered list of keypoint names. Consistent across all folders.
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}
    bodyparts: list[str] | None = None

    for rf in folders:
        # --- skull/spine ---
        skull_path = rf.skull_and_spine_resampled_trajectories
        if skull_path is None:
            raise FileNotFoundError(
                f"skull_and_spine_resampled_trajectories not found for '{rf.recording_name}'"
            )

        skull_df = pd.read_csv(skull_path)
        missing = _SKULL_GAZE_REQUIRED_COLUMNS - set(skull_df.columns)
        if missing:
            raise ValueError(f"{skull_path}: missing required columns {missing}")

        wide_skull = _long_to_wide_xyz(skull_df, skull_path)

        # --- gaze targets ---
        wide_parts = [wide_skull]
        for side, csv_path in [
            ("left_gaze_target", rf.left_gaze_kinematics_csv),
            ("right_gaze_target", rf.right_gaze_kinematics_csv),
        ]:
            if csv_path is None:
                raise FileNotFoundError(
                    f"{side} gaze kinematics CSV not found for '{rf.recording_name}'"
                )

            gaze_df = pd.read_csv(csv_path)
            missing = _GAZE_REQUIRED_COLUMNS - set(gaze_df.columns)
            if missing:
                raise ValueError(f"{csv_path}: missing required columns {missing}")

            gaze_df = gaze_df[gaze_df["trajectory"] == _GAZE_TRAJECTORY_FILTER].copy()
            if gaze_df.empty:
                raise ValueError(
                    f"{csv_path}: no rows with trajectory == '{_GAZE_TRAJECTORY_FILTER}'"
                )
            gaze_df["trajectory"] = side
            wide_parts.append(_long_to_wide_xyz(gaze_df, csv_path))

        combined = pd.concat(wide_parts, ignore_index=True)
        coords, confs, file_bodyparts = _pivot_to_kpms_arrays(combined, keypoint_col="trajectory")

        if bodyparts is None:
            bodyparts = file_bodyparts
        elif file_bodyparts != bodyparts:
            raise ValueError(
                f"Bodyparts mismatch for recording '{rf.recording_name}'.\n"
                f"Expected: {bodyparts}\n"
                f"Got:      {file_bodyparts}"
            )

        name = rf.recording_name
        if name in coordinates:
            raise ValueError(
                f"Duplicate recording name '{name}'. Each RecordingFolder must "
                "have a unique recording_name."
            )

        coordinates[name] = coords
        confidences[name] = confs

    assert bodyparts is not None
    return coordinates, confidences, bodyparts


_SKULL_KINEMATICS_FILENAME = "skull_kinematics.parquet"
_SKULL_KINEMATICS_CSV_FILENAME = "skull_kinematics.csv"
_HEAD_KEYPOINT_PREFIX = "keypoint__"
_TRACKED_PUPIL_PREFIX = "tracked_pupil__"


def _polars_tidy_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a tidy-format kinematics polars DataFrame to pandas, with
    `trajectory`/`component` as plain strings (they are stored as Categorical)."""
    return df.with_columns(
        pl.col("trajectory").cast(pl.String),
        pl.col("component").cast(pl.String),
    ).to_pandas()


def _tidy_trajectories_to_wide_xyz(
    df: pd.DataFrame,
    source_path: Path,
    trajectory_prefix: str,
    rename_prefix: str = "",
) -> pd.DataFrame:
    """
    Filter a tidy-format kinematics DataFrame (frame, trajectory, component,
    value, ...) to trajectories starting with `trajectory_prefix`, strip that
    prefix (optionally substituting `rename_prefix`), and reshape to wide
    format: frame, trajectory, x, y, z.
    """
    df = df[df["trajectory"].str.startswith(trajectory_prefix)].copy()
    if df.empty:
        raise ValueError(
            f"{source_path}: no rows with trajectory starting with '{trajectory_prefix}'"
        )
    df["trajectory"] = rename_prefix + df["trajectory"].str.removeprefix(trajectory_prefix)
    return _long_to_wide_xyz(df, source_path)


def load_head_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
    recenter_static_shape: bool = False,
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Head-only counterpart of `load_head_with_pupil_points_kpms`: the 8 skull
    keypoints from ``skull_kinematics.parquet`` on the same resampled
    common-timestamp grid, with no pupil points (so gaze kinematics aren't
    required). Returns (n_frames, 8, 3) coordinates per recording.

    With `recenter_static_shape`, each session's skull geometry is replaced
    by the across-session mean (`replace_static_shape_in_skull_frame`). The
    head keypoints are rigid, so what's left for kpms after its own
    translation + yaw alignment is head pitch and roll only.
    """
    coordinates, confidences, bodyparts = load_head_with_pupil_points_kpms(
        recording_folder, include_pupil_points=False
    )
    if recenter_static_shape:
        coordinates = replace_static_shape_in_skull_frame(
            _as_folder_list(recording_folder), coordinates, confidences, bodyparts
        )
    return coordinates, confidences, bodyparts


def _as_folder_list(recording_folder: RecordingFolder | list[RecordingFolder]) -> list[RecordingFolder]:
    return [recording_folder] if isinstance(recording_folder, RecordingFolder) else recording_folder


def replace_static_shape_in_skull_frame(
    folders: list[RecordingFolder],
    coordinates: dict[str, NDArray],
    confidences: dict[str, NDArray],
    bodyparts: list[str],
) -> dict[str, NDArray]:
    """
    `replace_static_shape` for world-space keypoint sets that include the 8
    head keypoints (`HEAD_BODYPARTS`, in any order).

    Per frame, fits the skull pose to the session's skull reference geometry
    (`_fit_rigid_poses` on the head keypoints), moves every keypoint into the
    skull frame, replaces each session's static configuration there with the
    across-session mean, and moves back with the same pose. Head motion is
    unchanged; per-session differences in skull/headmount geometry (base and
    cam-tip markers vary by ~4 mm SD across sessions) and in resting eye
    position are removed.
    """
    from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

    missing = set(HEAD_BODYPARTS) - set(bodyparts)
    if missing:
        raise ValueError(f"need all head keypoints to fit the skull pose, missing {sorted(missing)}")
    head_idxs = [bodyparts.index(bp) for bp in HEAD_BODYPARTS]

    poses, in_skull = {}, {}
    for rf in folders:
        name = rf.recording_name
        geometry = ReferenceGeometry.from_json_file(rf.skull_kinematics / _SKULL_REFERENCE_GEOMETRY_FILENAME)
        reference = np.stack([geometry.get_keypoint_position(bp) for bp in HEAD_BODYPARTS])
        rotations, translations = _fit_rigid_poses(reference, coordinates[name][:, head_idxs])
        poses[name] = rotations, translations
        # Inverse of world = R @ p + t, for every keypoint.
        in_skull[name] = np.einsum("nji,nkj->nki", rotations, coordinates[name] - translations[:, None])

    recentered = replace_static_shape(in_skull, confidences)
    return {
        name: np.einsum("nij,nkj->nki", rotations, recentered[name]) + translations[:, None]
        for name, (rotations, translations) in poses.items()
    }


def load_head_with_pupil_points_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
    include_pupil_points: bool = True,
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load head (skull) keypoints and both eyes' tracked pupil boundary points,
    combined into a single keypoint set per recording, for Keypoint MoSeq.

    Reads ``analyzable_output/skull_kinematics/skull_kinematics.parquet``
    (falling back to the CSV sibling, which has identical content) for 8 head
    keypoints, and ``analyzable_output/gaze_kinematics/{left,right}_gaze_kinematics.parquet``
    for each eye's 9 tracked pupil points (``pupil_center``, ``p1``..``p8``) —
    real per-frame pupil detections already projected into world space. These
    tracked-pupil trajectories exist only in the parquet, not the CSV, so the
    gaze files are read directly with no CSV fallback.

    Both sources are on the same resampled common-timestamp grid (see
    ``ferret_data_resampler.resample_ferret_data``, which saves the resampled
    skull kinematics and asserts its timestamps match those used to compute
    gaze), so they are merged by frame index directly with no further
    resampling.

    Parameters
    ----------
    recording_folder:
        A single RecordingFolder or a list of them.
    include_pupil_points:
        If False, load only the 8 head keypoints (see `load_head_kpms`).

    Returns
    -------
    coordinates:
        Dict mapping ``recording_folder.recording_name`` to array of shape
        (n_frames, 26, 3).
    confidences:
        Same keys; values are (n_frames, 26) arrays of ones.
    bodyparts:
        Ordered list of keypoint names: 8 head keypoints, then 9 left-eye
        tracked pupil points, then 9 right-eye tracked pupil points.
        Consistent across all folders.
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}
    bodyparts: list[str] | None = None

    for rf in folders:
        if rf.skull_kinematics is None:
            raise FileNotFoundError(
                f"skull_kinematics does not exist for recording '{rf.recording_name}'"
            )
        skull_parquet_path = rf.skull_kinematics / _SKULL_KINEMATICS_FILENAME
        skull_csv_path = rf.skull_kinematics / _SKULL_KINEMATICS_CSV_FILENAME
        if not skull_parquet_path.exists() and not skull_csv_path.exists():
            raise FileNotFoundError(f"Skull kinematics not found: {skull_parquet_path}")

        skull_df = _polars_tidy_to_pandas(
            read_parquet_or_csv(parquet_path=skull_parquet_path, csv_path=skull_csv_path)
        )
        wide_parts = [
            _tidy_trajectories_to_wide_xyz(
                skull_df, skull_parquet_path, trajectory_prefix=_HEAD_KEYPOINT_PREFIX
            )
        ]

        if include_pupil_points and rf.gaze_kinematics is None:
            raise FileNotFoundError(
                f"gaze_kinematics does not exist for recording '{rf.recording_name}'"
            )
        for side in ("left", "right") if include_pupil_points else ():
            gaze_parquet_path = rf.gaze_kinematics / f"{side}_gaze_kinematics.parquet"
            if not gaze_parquet_path.exists():
                raise FileNotFoundError(
                    f"{gaze_parquet_path} not found. load_head_with_pupil_points_kpms requires the "
                    "gaze kinematics parquet output — tracked pupil points are parquet-only "
                    "and not available in the CSV."
                )
            gaze_df = _polars_tidy_to_pandas(pl.read_parquet(gaze_parquet_path))
            wide_parts.append(
                _tidy_trajectories_to_wide_xyz(
                    gaze_df,
                    gaze_parquet_path,
                    trajectory_prefix=_TRACKED_PUPIL_PREFIX,
                    rename_prefix=f"{side}_",
                )
            )

        combined = pd.concat(wide_parts, ignore_index=True)
        coords, confs, file_bodyparts = _pivot_to_kpms_arrays(combined, keypoint_col="trajectory")

        if bodyparts is None:
            bodyparts = file_bodyparts
        elif file_bodyparts != bodyparts:
            raise ValueError(
                f"Bodyparts mismatch for recording '{rf.recording_name}'.\n"
                f"Expected: {bodyparts}\n"
                f"Got:      {file_bodyparts}"
            )

        name = rf.recording_name
        if name in coordinates:
            raise ValueError(
                f"Duplicate recording name '{name}'. Each RecordingFolder must "
                "have a unique recording_name."
            )

        coordinates[name] = coords
        confidences[name] = confs

    assert bodyparts is not None
    return coordinates, confidences, bodyparts


# Default eye-in-head amplification for `load_head_with_scaled_eyes_kpms`.
HEAD_WITH_SCALED_EYES_DEFAULT_SCALE = 10.0

_PUPIL_POINT_NAMES = ["pupil_center", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]


def _fit_rigid_poses(
    reference_points: NDArray[np.float64],
    world_points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Per-frame least-squares rigid transform (Kabsch) taking `reference_points`
    (k, 3), body frame, to `world_points` (n_frames, k, 3).

    Returns rotations (n_frames, 3, 3) and translations (n_frames, 3) with
    `world ~= reference @ R.T + t`. Frames with any non-finite point get NaN.
    """
    n_frames = world_points.shape[0]
    rotations = np.full((n_frames, 3, 3), np.nan)
    translations = np.full((n_frames, 3), np.nan)
    valid = np.isfinite(world_points).all(axis=(1, 2))

    ref_mean = reference_points.mean(axis=0)
    ref_centered = reference_points - ref_mean
    world = world_points[valid]
    world_mean = world.mean(axis=1)
    world_centered = world - world_mean[:, None]

    covariance = np.einsum("nki,kj->nij", world_centered, ref_centered)  # sum_k w_k r_k^T
    u, _, vt = np.linalg.svd(covariance)
    correction = np.tile(np.eye(3), (len(world), 1, 1))
    correction[:, 2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ correction @ vt
    rotations[valid] = rot
    translations[valid] = world_mean - np.einsum("nij,j->ni", rot, ref_mean)
    return rotations, translations


def load_head_with_scaled_eyes_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
    eye_scale: float = HEAD_WITH_SCALED_EYES_DEFAULT_SCALE,
    quality_level: str | None = "medium",
    return_rigid_fit_error: bool = False,
    recenter_static_shape: bool = False,
):
    """
    Head keypoints plus both eyes' pupil points, with the eye-in-head part of
    the pupil-point motion amplified by `eye_scale`.

    Same 26 keypoints as `load_head_with_pupil_points_kpms` (so the same
    bodypart groups, skeleton, and diagnostics apply), but built so that eye
    rotation isn't swamped by head motion. kpms's egocentric alignment only
    removes translation + yaw; head pitch/roll still moves the pupil points
    rigidly by several mm, while eye rotation moves them ~1 mm, so scaling
    the raw world-space pupil points would amplify both by the same factor
    and change nothing (see `eye_movement_in_pcs.md`). Instead, each pupil
    point is built as

        p_world = R_skull @ (c + eye_scale * (p_skull - c)) + t_skull

    where `p_skull` is the pupil point in the skull frame
    (`_eye_points_in_skull_frame`, from the resampled eye CSVs -- no gaze
    parquet needed), `c` is that eye's center in the skull reference
    geometry, and `R_skull, t_skull` are fit per frame to the 8 head
    keypoints. Head pitch/roll still moves the eye points rigidly (as for any
    head-fixed point), but only the eye-in-head part is amplified. With
    `eye_scale=1` this reproduces the gaze pipeline's world-space tracked
    pupil points. Equivalent to an eyeball `eye_scale` times larger, so
    pupil-size changes (p1..p8 relative to pupil_center) are amplified too.

    Parameters
    ----------
    eye_scale:
        Eye-in-head amplification factor (alpha).
    quality_level:
        `load_eye_data_quality` level: each eye's 9 points get confidence 0
        on frames where that eye fails (head keypoints stay 1). None for
        all-ones confidences.
    return_rigid_fit_error:
        Also return `{recording_name: (n_frames,)}` RMS residual (mm) of the
        per-frame rigid skull fit, as a sanity check that the head keypoints
        are rigid.
    recenter_static_shape:
        Replace each session's skull geometry and resting (scaled) pupil
        positions with the across-session mean, in the skull frame
        (`replace_static_shape_in_skull_frame`). Applied after scaling, so
        eye-in-head motion stays amplified.

    Returns
    -------
    coordinates, confidences, bodyparts (and rigid fit error if requested):
        As `load_head_with_pupil_points_kpms`, in mm.
    """
    from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )
    head_coords, _, head_bodyparts = load_head_kpms(folders)
    bodyparts = head_bodyparts + [
        f"{side}_{name}" for side in ("left", "right") for name in _PUPIL_POINT_NAMES
    ]

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}
    rigid_fit_error: dict[str, NDArray] = {}

    for rf in folders:
        name = rf.recording_name
        skull_reference_geometry = ReferenceGeometry.from_json_file(
            rf.skull_kinematics / _SKULL_REFERENCE_GEOMETRY_FILENAME
        )
        head_world = head_coords[name]  # (n_frames, 8, 3)
        head_reference = np.stack(
            [skull_reference_geometry.get_keypoint_position(bp) for bp in head_bodyparts]
        )
        rotations, translations = _fit_rigid_poses(head_reference, head_world)
        fitted = np.einsum("nij,kj->nki", rotations, head_reference) + translations[:, None]
        rigid_fit_error[name] = np.sqrt(((fitted - head_world) ** 2).sum(-1).mean(-1))

        eye_parts = []
        for side, csv_path in [
            ("left", rf.left_eye_resampled_trajectories),
            ("right", rf.right_eye_resampled_trajectories),
        ]:
            if csv_path is None:
                raise FileNotFoundError(f"{side} eye resampled trajectories not found for '{name}'")
            eye_center = skull_reference_geometry.get_keypoint_position(f"{side}_eye")
            points_skull = _eye_points_in_skull_frame(csv_path, side, _PUPIL_POINT_NAMES, eye_center)
            if points_skull.shape[0] != head_world.shape[0]:
                raise ValueError(
                    f"'{name}': {side} eye has {points_skull.shape[0]} frames, "
                    f"skull kinematics has {head_world.shape[0]}"
                )
            points_scaled = eye_center + eye_scale * (points_skull - eye_center)
            eye_parts.append(
                np.einsum("nij,nkj->nki", rotations, points_scaled) + translations[:, None]
            )

        coords = np.concatenate([head_world, *eye_parts], axis=1)
        n_frames = coords.shape[0]
        confs = np.ones((n_frames, len(bodyparts)))
        if quality_level is not None:
            left_good, right_good = load_eye_data_quality(rf, quality_level)
            if len(left_good) != n_frames:
                raise ValueError(
                    f"'{name}': eye_data_quality.csv has {len(left_good)} frames "
                    f"but skull kinematics has {n_frames}"
                )
            for side, good in (("left", left_good), ("right", right_good)):
                if good.mean() < 0.5:
                    # An eye with (almost) no good frames can't be interpolated
                    # by kpms and ends up dominating the top PCs as a rigid
                    # shift of that eye's points -- drop such recordings.
                    print(
                        f"WARNING '{name}': only {good.mean():.1%} {quality_level}-quality "
                        f"{side} eye frames -- consider excluding this recording."
                    )
            n_head, n_eye = len(head_bodyparts), len(_PUPIL_POINT_NAMES)
            confs[:, n_head : n_head + n_eye] = left_good[:, None]
            confs[:, n_head + n_eye :] = right_good[:, None]

        coordinates[name] = coords
        confidences[name] = confs

    if recenter_static_shape:
        coordinates = replace_static_shape_in_skull_frame(folders, coordinates, confidences, bodyparts)
    if return_rigid_fit_error:
        return coordinates, confidences, bodyparts, rigid_fit_error
    return coordinates, confidences, bodyparts
