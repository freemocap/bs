"""Data loading utilities for eye tracking - computes pupil centers from DLC points."""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_deeplabcut_csv(
    *,
    filepath: Path
) -> pd.DataFrame:
    """
    Load DeepLabCut CSV with multi-row header.

    DLC CSVs have 3 header rows:
    - Row 0: scorer name
    - Row 1: bodypart names
    - Row 2: coordinates (x, y, likelihood)

    Args:
        filepath: Path to DLC CSV file

    Returns:
        DataFrame with flattened column names like 'p1_x', 'p1_y', 'p1_likelihood'
    """
    logger.info(f"Loading DeepLabCut CSV: {filepath.name}")

    df = pd.read_csv(filepath_or_buffer=filepath, header=[0, 1, 2], index_col=0)

    # Flatten the multi-index columns
    new_columns = []
    for col in df.columns:
        scorer, bodypart, coord = col
        new_columns.append(f"{bodypart}_{coord}")

    df.columns = new_columns

    logger.info(f"  Loaded {len(df)} frames")
    logger.info(f"  Columns: {df.columns.tolist()}")

    return df


def load_pupil_centers(
    *,
    filepath: Path,
    point_columns: list[str] | None = None
) -> dict[str, np.ndarray]:
    """
    Load pupil tracking data and compute centers from ellipse points.

    Expected format (DeepLabCut):
        - Multi-row header with scorer, bodypart, coords
        - Bodyparts: p1, p2, p3, ..., p8 (8 points on ellipse)
        - Each bodypart has x, y, likelihood columns

    Args:
        filepath: Path to CSV file
        point_columns: Column names for points (e.g., ['p1', 'p2', ...])
                      If None, auto-detects columns starting with 'p'

    Returns:
        Dictionary with:
        - 'pupil_centers': (n_frames, 2) array of center positions
        - 'frame_indices': (n_frames,) array of frame numbers
        - 'n_valid_points': (n_frames,) number of valid points used per frame
        - 'raw_data': Original DataFrame
    """
    logger.info(f"Loading pupil centers from: {filepath.name}")

    # Try to detect if this is a DLC CSV
    with open(file=filepath, mode='r') as f:
        first_line = f.readline()
        is_dlc = 'scorer' in first_line.lower() or first_line.count(',') > 20

    if is_dlc:
        df = load_deeplabcut_csv(filepath=filepath)
    else:
        df = pd.read_csv(filepath_or_buffer=filepath)

    # Auto-detect point columns if not provided
    if point_columns is None:
        x_cols = [col for col in df.columns if col.endswith('_x') and col.startswith('p')]
        point_columns = [col[:-2] for col in x_cols]
        try:
            point_columns = sorted(point_columns, key=lambda x: int(x[1:]))
        except ValueError:
            point_columns = sorted(point_columns)

    logger.info(f"  Detected {len(point_columns)} points: {point_columns}")

    n_frames = len(df)
    n_points = len(point_columns)

    # Extract all points
    pupil_points = np.zeros(shape=(n_frames, n_points, 2))

    for i, point_name in enumerate(point_columns):
        x_col = f"{point_name}_x"
        y_col = f"{point_name}_y"

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Missing columns for point {point_name}: expected {x_col} and {y_col}")

        pupil_points[:, i, 0] = df[x_col].values
        pupil_points[:, i, 1] = df[y_col].values

    # Compute centers as mean of valid points
    pupil_centers = np.zeros(shape=(n_frames, 2))
    n_valid_points = np.zeros(shape=n_frames, dtype=int)

    for frame_idx in range(n_frames):
        points = pupil_points[frame_idx]
        valid_mask = ~np.isnan(points[:, 0])
        valid_points = points[valid_mask]

        n_valid_points[frame_idx] = len(valid_points)

        if len(valid_points) > 0:
            pupil_centers[frame_idx] = np.mean(valid_points, axis=0)
        else:
            pupil_centers[frame_idx] = np.nan

    # Get frame indices
    frame_indices = df.index.values

    # Check for NaN values
    nan_count = np.isnan(pupil_centers).sum()
    if nan_count > 0:
        logger.warning(f"  Found {nan_count // 2} frames with no valid points")

    logger.info(f"  Computed pupil centers for {n_frames} frames")
    logger.info(f"  Valid points per frame: mean={np.mean(n_valid_points):.1f}, min={np.min(n_valid_points)}, max={np.max(n_valid_points)}")

    return {
        'pupil_centers': pupil_centers,
        'frame_indices': frame_indices,
        'n_valid_points': n_valid_points,
        'raw_data': df
    }


def filter_invalid_frames(
    *,
    pupil_centers: np.ndarray,
    frame_indices: np.ndarray,
    n_valid_points: np.ndarray,
    min_valid_points: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out invalid frames based on number of valid points.

    Args:
        pupil_centers: (n_frames, 2) pupil centers
        frame_indices: (n_frames,) frame numbers
        n_valid_points: (n_frames,) number of valid points per frame
        min_valid_points: Minimum number of valid points required

    Returns:
        Tuple of:
        - filtered_pupil_centers: (n_valid_frames, 2)
        - filtered_frame_indices: (n_valid_frames,)
        - valid_mask: (n_frames,) boolean mask
    """
    # Check for NaN centers
    valid_mask = ~np.isnan(pupil_centers[:, 0])

    # Check for minimum valid points
    valid_mask &= (n_valid_points >= min_valid_points)

    n_valid = np.sum(valid_mask)
    n_invalid = len(valid_mask) - n_valid

    logger.info(f"\nFrame filtering:")
    logger.info(f"  Valid frames: {n_valid}")
    logger.info(f"  Invalid frames: {n_invalid}")

    if n_invalid > 0:
        logger.warning(f"  Removed {n_invalid} invalid frames")

        low_points = np.sum(n_valid_points < min_valid_points)
        nan_centers = np.sum(np.isnan(pupil_centers[:, 0]))

        if low_points > 0:
            logger.info(f"    - {low_points} frames with < {min_valid_points} valid points")
        if nan_centers > 0:
            logger.info(f"    - {nan_centers} frames with NaN centers")

    return (
        pupil_centers[valid_mask],
        frame_indices[valid_mask],
        valid_mask
    )