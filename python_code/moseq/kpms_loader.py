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
from python_code.moseq.utils.bodyparts import BOTH_EYES_BODYPARTS
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
_EYE_IN_HEAD_REQUIRED_COLUMNS = {"frame", "trajectory", "component", "value"}


def load_eye_in_head_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
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

    Returns
    -------
    eye_in_head_left, eye_in_head_right:
        Each a dict mapping `recording_folder.recording_name` to an array of
        shape (n_frames, 2) with columns [adduction_deg, elevation_deg].
    """
    folders: list[RecordingFolder] = (
        [recording_folder]
        if isinstance(recording_folder, RecordingFolder)
        else recording_folder
    )

    eye_in_head_left: dict[str, NDArray] = {}
    eye_in_head_right: dict[str, NDArray] = {}

    for rf in folders:
        for side, csv_path, out in [
            ("left", rf.left_eye_kinematics_csv, eye_in_head_left),
            ("right", rf.right_eye_kinematics_csv, eye_in_head_right),
        ]:
            if csv_path is None:
                raise FileNotFoundError(
                    f"{side}_eye_kinematics.csv not found for recording '{rf.recording_name}'"
                )

            df = pd.read_csv(csv_path)
            missing = _EYE_IN_HEAD_REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path}: missing required columns {missing}")

            df = df[df["trajectory"] == _EYE_IN_HEAD_TRAJECTORY]
            if df.empty:
                raise ValueError(
                    f"{csv_path}: no rows with trajectory == '{_EYE_IN_HEAD_TRAJECTORY}'"
                )

            wide = df.pivot_table(
                index="frame", columns="component", values="value", aggfunc="first"
            ).sort_index()
            missing_components = set(_EYE_IN_HEAD_COMPONENTS) - set(wide.columns)
            if missing_components:
                raise ValueError(
                    f"{csv_path}: missing eye_in_head components {missing_components}"
                )

            angles_deg = np.degrees(wide[_EYE_IN_HEAD_COMPONENTS].to_numpy())

            name = rf.recording_name
            if name in out:
                raise ValueError(
                    f"Duplicate recording name '{name}'. Each RecordingFolder must "
                    "have a unique recording_name."
                )
            out[name] = angles_deg

    return eye_in_head_left, eye_in_head_right


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


def load_head_with_pupil_points_kpms(
    recording_folder: RecordingFolder | list[RecordingFolder],
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

        if rf.gaze_kinematics is None:
            raise FileNotFoundError(
                f"gaze_kinematics does not exist for recording '{rf.recording_name}'"
            )
        for side in ("left", "right"):
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
