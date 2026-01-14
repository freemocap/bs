"""
Body Trajectories DataFrame Conversion

Converts body keypoint trajectory dictionary to tidy-formatted DataFrame.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from numpy.typing import NDArray


def body_trajectories_to_dataframe(
    trajectory_data: dict[str, NDArray[np.float64]],
) -> pd.DataFrame:
    """Convert body trajectory dictionary to tidy-formatted DataFrame.

    Args:
        trajectory_data: Dictionary with "timestamps" key mapping to (N,) array,
            and marker name keys mapping to (N, 3) position arrays.

    Returns:
        DataFrame with columns: timestamp, marker, x_mm, y_mm, z_mm
        Each row represents one marker at one timestamp.
        Total rows = n_frames * n_markers

    Raises:
        ValueError: If timestamps missing, no markers present, or shapes inconsistent
    """
    if "timestamps" not in trajectory_data:
        raise ValueError("trajectory_data must contain 'timestamps' key")

    timestamps = trajectory_data["timestamps"]

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be 1D, got shape {timestamps.shape}")

    n_frames = len(timestamps)
    if n_frames == 0:
        raise ValueError("timestamps array is empty")

    marker_names = [k for k in trajectory_data.keys() if k != "timestamps"]
    if len(marker_names) == 0:
        raise ValueError("trajectory_data contains no marker data (only timestamps)")

    # Validate all marker arrays have correct shape
    for marker_name in marker_names:
        positions = trajectory_data[marker_name]
        if positions.shape != (n_frames, 3):
            raise ValueError(
                f"Marker '{marker_name}' has shape {positions.shape}, "
                f"expected ({n_frames}, 3)"
            )

    # Build tidy DataFrame
    rows: list[dict[str, float | str]] = []

    for frame_idx in range(n_frames):
        timestamp = float(timestamps[frame_idx])
        for marker_name in marker_names:
            positions = trajectory_data[marker_name]
            rows.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "marker": marker_name,
                "x_mm": float(positions[frame_idx, 0]),
                "y_mm": float(positions[frame_idx, 1]),
                "z_mm": float(positions[frame_idx, 2]),
            })

    return pd.DataFrame(rows)


def load_body_trajectory_data(trajectory_csv_path: Path) -> dict[str, NDArray[np.float64]]:
    """Load trajectory data from CSV into arrays of shape (n_frames, 3).

    Args:
        trajectory_csv_path: Path to tidy_trajectory_data.csv

    Returns:
        Dictionary mapping marker name to position array of shape (n_frames, 3),
        plus a "timestamps" key with shape (n_frames,)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If frames are not contiguous or data is invalid
    """
    trajectory_csv_path = Path(trajectory_csv_path)
    if not trajectory_csv_path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {trajectory_csv_path}")

    df = pd.read_csv(trajectory_csv_path)

    if len(df) == 0:
        raise ValueError(f"Empty CSV file: {trajectory_csv_path}")

    # Filter to optimized data only
    optimized_df = df[df["data_type"] == "optimized"]
    if len(optimized_df) == 0:
        raise ValueError(f"No 'optimized' data_type found in {trajectory_csv_path}")

    # Get frame info
    n_frames = optimized_df["frame"].nunique()
    frame_indices = optimized_df["frame"].unique()

    # Verify frames are contiguous starting from 0
    expected_frames = np.arange(n_frames)
    if not np.array_equal(np.sort(frame_indices), expected_frames):
        raise ValueError(
            f"Frame indices must be contiguous from 0 to {n_frames - 1}, "
            f"got: {sorted(frame_indices)[:10]}..."
        )

    marker_names = optimized_df["marker"].unique()
    if len(marker_names) == 0:
        raise ValueError(f"No markers found in {trajectory_csv_path}")

    # Extract timestamps - one per frame (vectorized)
    timestamp_df = optimized_df.groupby("frame")["timestamp"].first().sort_index()
    if len(timestamp_df) != n_frames:
        raise ValueError(f"Timestamp count {len(timestamp_df)} doesn't match frame count {n_frames}")
    timestamps = timestamp_df.values.astype(np.float64)

    # Pivot the data: rows=frames, columns=marker_coord combinations
    # This is O(n_rows) instead of O(n_markers * n_frames * n_rows)
    optimized_df = optimized_df.sort_values("frame")

    body_trajectories: dict[str, NDArray[np.float64]] = {"timestamps": timestamps}

    # Group by marker and extract coordinates in one pass per marker
    grouped = optimized_df.groupby("marker")

    for marker_name in marker_names:
        marker_df = grouped.get_group(marker_name).sort_values("frame")

        # Validate we have exactly one row per frame
        if len(marker_df) != n_frames:
            raise ValueError(
                f"Marker '{marker_name}' has {len(marker_df)} rows, expected {n_frames} (one per frame)"
            )

        # Check for duplicate frames
        if marker_df["frame"].nunique() != n_frames:
            duplicate_frames = marker_df[marker_df["frame"].duplicated()]["frame"].unique()
            raise ValueError(
                f"Marker '{marker_name}' has duplicate entries at frames: {duplicate_frames[:10].tolist()}"
            )

        # Extract xyz as contiguous array - this is fast
        positions = marker_df[["x", "y", "z"]].values.astype(np.float64)
        body_trajectories[str(marker_name)] = positions

    return body_trajectories
