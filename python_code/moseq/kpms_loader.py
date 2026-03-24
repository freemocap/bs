"""
Keypoint MoSeq Loader
=====================

Custom data loader for Keypoint MoSeq (kpms) that reads the project's 3D
triangulated keypoint CSV format:

    frame, keypoint, x, y, z, model, trajectory, reprojection_error

Only rows where trajectory == 'rigid_3d_xyz' are used.

Returns data in the format expected by kpms.load_keypoints():
    coordinates: dict[str, NDArray[(N, K, 3)]]
    confidences: dict[str, NDArray[(N, K)]]   -- all ones (reprojection_error
                                                  is not bounded 0-1)
    bodyparts: list[str]

Usage:
    from python_code.utilities.kpms_loader import load_keypoints_for_kpms

    coordinates, confidences, bodyparts = load_keypoints_for_kpms(
        "/path/to/session/mocap_data/output_data/dlc/*.csv"
    )
    # Pass directly to kpms:
    # kpms.fit_pca(coordinates, confidences, bodyparts, ...)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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


def _load_single_file(csv_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """
    Load one CSV file and return (coords_NKD, confs_NK, bodyparts).

    coords_NKD: (N_frames, N_keypoints, 3)
    confs_NK:   (N_frames, N_keypoints) — all ones
    bodyparts:  list of keypoint names in column order
    """
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")

    df = df[df["trajectory"] == TRAJECTORY_FILTER]
    if df.empty:
        raise ValueError(
            f"{csv_path}: no rows with trajectory == '{TRAJECTORY_FILTER}'"
        )

    # Pivot: index=frame, columns=keypoint, values=[x, y, z]
    # Result MultiIndex columns: (coord, keypoint)
    pivoted = df.pivot(index="frame", columns="keypoint", values=["x", "y", "z"])
    pivoted = pivoted.sort_index()  # ensure frames are ascending

    bodyparts: list[str] = pivoted["x"].columns.tolist()
    n_frames = len(pivoted)
    n_keypoints = len(bodyparts)

    coords = np.stack(
        [pivoted[coord][bodyparts].to_numpy() for coord in ("x", "y", "z")],
        axis=-1,
    )  # (N_frames, N_keypoints, 3)
    assert coords.shape == (n_frames, n_keypoints, 3)

    confs = np.ones((n_frames, n_keypoints), dtype=np.float64)

    return coords, confs, bodyparts


def load_keypoints_for_kpms(
    filepath_pattern: str | Path | list[str | Path],
    recursive: bool = True,
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """
    Load 3D keypoint CSVs for Keypoint MoSeq.

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
        coords, confs, file_bodyparts = _load_single_file(csv_path)

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
