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

All return data in the format expected by kpms.load_keypoints():
    coordinates: dict[str, NDArray[(N, K, 3)]]
    confidences: dict[str, NDArray[(N, K)]]
    bodyparts: list[str]
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
