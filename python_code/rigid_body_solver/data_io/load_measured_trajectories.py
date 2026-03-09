"""Data loading utilities for various CSV formats."""

from pathlib import Path
import csv
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_measured_trajectories_csv(
        *,
        filepath: Path,
        scale_factor: float = 1.0
) -> dict[str, np.ndarray]:
    """
    Load trajectories from tidy/long-format CSV.

    Expected format:
        frame, keypoint, x, y, z
        0, keypoint1, 1.0, 2.0, 3.0
        0, keypoint2, 4.0, 5.0, 6.0
        ...

    Args:
        filepath: Path to tidy CSV file
        scale_factor: Multiplier for coordinates (e.g., 0.001 for mm to m)

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) arrays
    """
    logger.info(f"Loading tidy CSV: {filepath.name}")

    trajectories: dict[str, list[list[float]]] = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            keypoint = row['keypoint'] if 'keypoint' in row else row['keypoint']  # Support both 'keypoint' and 'keypoint' columns
            x = float(row['x']) if row['x'] != '' else np.nan
            y = float(row['y']) if row['y'] != '' else np.nan
            z = float(row['z']) if 'z' in row and row['z'] else 0.0

            if keypoint not in trajectories:
                trajectories[keypoint] = []

            if row["trajectory"] == "3d_xyz":
                trajectories[keypoint].append([x, y, z])

    # Convert to numpy arrays
    result = {
        name: np.array(coords, dtype=np.float64) * scale_factor
        for name, coords in trajectories.items()
    }

    n_keypoints = len(result)
    n_frames = len(next(iter(result.values())))
    for value in result.values():
        assert value.shape == (n_frames, 3)
    logger.info(f"  Loaded {n_keypoints} keypoints × {n_frames} frames")

    return result
