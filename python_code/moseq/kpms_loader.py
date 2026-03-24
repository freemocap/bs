"""
Keypoint MoSeq Loader
=====================

Custom data loaders for Keypoint MoSeq (kpms).

load_3d_data_kpms
    Reads the project's 3D triangulated keypoint CSV format:
        frame, keypoint, x, y, z, model, trajectory, reprojection_error
    Only rows where trajectory == 'rigid_3d_xyz' are used.

load_solver_output_kpms
    Reads the mocap solver tidy output:
        frame, timestamp, marker, data_type, x, y, z
    Only rows where data_type == 'optimized' are used.
    Accepts a single RecordingFolder or a list of them.

Both return data in the format expected by kpms.load_keypoints():
    coordinates: dict[str, NDArray[(N, K, 3)]]
    confidences: dict[str, NDArray[(N, K)]]
    bodyparts: list[str]
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


TRAJECTORY_FILTER = "rigid_3d_xyz"
REQUIRED_COLUMNS = {"frame", "keypoint", "x", "y", "z", "trajectory"}


def _glob_csv_files(filepath_pattern: str | Path | list[str | Path], recursive: bool) -> list[Path]:
    """Resolve filepath_pattern to a sorted list of CSV paths."""
    if isinstance(filepath_pattern, list):
        paths = []
        for p in filepath_pattern:
            paths.extend(_glob_csv_files(p, recursive))
        return paths

    filepath_pattern = Path(filepath_pattern)

    if filepath_pattern.is_file():
        return [filepath_pattern]

    if filepath_pattern.is_dir():
        pattern = "**/*.csv" if recursive else "*.csv"
        return sorted(filepath_pattern.glob(pattern))

    # Treat as a glob pattern
    parent = filepath_pattern.parent
    glob_str = filepath_pattern.name
    if recursive:
        return sorted(parent.rglob(glob_str))
    return sorted(parent.glob(glob_str))


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
    filepath_pattern: str | Path | list[str | Path],
    recursive: bool = True,
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load 3D triangulated keypoint CSVs for Keypoint MoSeq.

    Parameters
    ----------
    filepath_pattern:
        A single file path, directory, glob pattern string, or list of any of
        the above. CSV files are searched recursively when a directory is given.
    recursive:
        Whether to search directories recursively (default True).

    Returns
    -------
    coordinates:
        Dict mapping recording name (file stem) to array of shape
        (n_frames, n_keypoints, 3).
    confidences:
        Dict with the same keys, values are (n_frames, n_keypoints) arrays of
        ones (reprojection_error is not bounded 0-1 so it cannot be used
        directly as a confidence score).
    bodyparts:
        Ordered list of keypoint names. Consistent across all files.
    """
    csv_files = _glob_csv_files(filepath_pattern, recursive)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for pattern: {filepath_pattern}")

    coordinates: dict[str, NDArray] = {}
    confidences: dict[str, NDArray] = {}
    bodyparts: list[str] | None = None

    for csv_path in csv_files:
        name = csv_path.stem
        df = pd.read_csv(csv_path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing required columns {missing}")

        df = df[df["trajectory"] == TRAJECTORY_FILTER]
        if df.empty:
            raise ValueError(
                f"{csv_path}: no rows with trajectory == '{TRAJECTORY_FILTER}'"
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

        if name in coordinates:
            raise ValueError(
                f"Duplicate recording name '{name}' from {csv_path}. "
                "Set a glob pattern that selects unique file stems, or rename files."
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
