"""Data loading utilities for various CSV formats."""

from pathlib import Path
import csv
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_tidy_csv(
        *,
        filepath: Path,
        scale_factor: float = 1.0
) -> dict[str, np.ndarray]:
    """
    Load trajectories from tidy/long-format CSV.

    Expected format:
        frame, keypoint, x, y, z
        0, marker1, 1.0, 2.0, 3.0
        0, marker2, 4.0, 5.0, 6.0
        ...

    Args:
        filepath: Path to tidy CSV file
        scale_factor: Multiplier for coordinates (e.g., 0.001 for mm to m)

    Returns:
        Dictionary mapping marker names to (n_frames, 3) arrays
    """
    logger.info(f"Loading tidy CSV: {filepath.name}")

    trajectories: dict[str, list[list[float]]] = {}

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            keypoint = row['keypoint']
            x = float(row['x']) if row['x'] != '' else -999
            y = float(row['y']) if row['y'] != '' else -999
            z = float(row['z']) if 'z' in row and row['z'] else 0.0

            if keypoint not in trajectories:
                trajectories[keypoint] = []

            trajectories[keypoint].append([x, y, z])

    # Convert to numpy arrays
    result = {
        name: np.array(coords, dtype=np.float64) * scale_factor
        for name, coords in trajectories.items()
    }

    n_markers = len(result)
    n_frames = len(next(iter(result.values())))
    logger.info(f"  Loaded {n_markers} markers × {n_frames} frames")

    return result


def load_wide_csv(
        *,
        filepath: Path,
        scale_factor: float = 1.0,
        z_value: float = 0.0
) -> dict[str, np.ndarray]:
    """
    Load 2D trajectories from wide-format CSV.

    Expected format:
        frame, marker1_x, marker1_y, marker2_x, marker2_y, ...
        0, 1.0, 2.0, 3.0, 4.0, ...
        1, 1.1, 2.1, 3.1, 4.1, ...

    Args:
        filepath: Path to wide CSV file
        scale_factor: Multiplier for coordinates
        z_value: Default z-coordinate for 2D data

    Returns:
        Dictionary mapping marker names to (n_frames, 3) arrays
    """
    logger.info(f"Loading wide CSV: {filepath.name}")

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError("CSV has no headers")

        # Find marker names from _x suffixes
        marker_names: set[str] = set()
        for header in headers:
            if header.endswith('_x'):
                marker_names.add(header[:-2])

        if not marker_names:
            raise ValueError("No markers found (expected columns ending in '_x')")

        # Read data
        trajectories: dict[str, list[list[float]]] = {
            name: [] for name in marker_names
        }

        f.seek(0)
        reader = csv.DictReader(f)

        for row in reader:
            for marker_name in marker_names:
                try:
                    x = float(row[f"{marker_name}_x"]) if row[f"{marker_name}_x"] else np.nan
                    y = float(row[f"{marker_name}_y"]) if row[f"{marker_name}_y"] else np.nan
                    # Try to read z column if it exists
                    z_col = f"{marker_name}_z"
                    if z_col in row and row[z_col]:
                        z = float(row[z_col])
                    else:
                        z = z_value  # Fall back to default only if column doesn't exist
                except (KeyError, ValueError):
                    x, y, z = np.nan, np.nan, z_value

                trajectories[marker_name].append([x, y, z])

    # Convert to numpy
    result = {
        name: np.array(coords, dtype=np.float64) * scale_factor
        for name, coords in trajectories.items()
    }

    n_markers = len(result)
    n_frames = len(next(iter(result.values())))
    logger.info(f"  Loaded {n_markers} markers × {n_frames} frames")

    return result


def load_dlc_csv(
        *,
        filepath: Path,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
        likelihood_threshold: float | None = None
) -> dict[str, np.ndarray]:
    """
    Load trajectories from DeepLabCut CSV with 3-row header.

    Expected format:
        scorer, scorer, scorer, ...           (row 0)
        bodypart1, bodypart1, bodypart2, ...  (row 1)
        x, y, likelihood, x, y, likelihood    (row 2)
        1.0, 2.0, 0.95, 3.0, 4.0, 0.92, ...   (row 3+)

    Args:
        filepath: Path to DLC CSV file
        scale_factor: Multiplier for coordinates
        z_value: Default z-coordinate for 2D data
        likelihood_threshold: Filter points below this confidence

    Returns:
        Dictionary mapping bodypart names to (n_frames, 3) arrays
    """
    logger.info(f"Loading DLC CSV: {filepath.name}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError("DLC CSV must have at least 4 rows")

    # Parse 3-row header
    bodypart_row = lines[1].strip().split(',')
    coords_row = lines[2].strip().split(',')

    # Build column mapping
    column_map: dict[str, dict[str, int]] = {}

    for col_idx, (bodypart, coord_type) in enumerate(zip(bodypart_row, coords_row)):
        bodypart = bodypart.strip()
        coord_type = coord_type.strip()

        if not bodypart or bodypart == 'scorer':
            continue

        if bodypart not in column_map:
            column_map[bodypart] = {}

        column_map[bodypart][coord_type] = col_idx

    # Validate
    valid_bodyparts = [
        bp for bp, coords in column_map.items()
        if 'x' in coords and 'y' in coords
    ]

    if not valid_bodyparts:
        raise ValueError("No valid bodyparts with x and y coordinates")

    # Read data
    trajectories: dict[str, list[list[float]]] = {
        name: [] for name in valid_bodyparts
    }

    for line in lines[3:]:
        values = line.strip().split(',')

        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]

            try:
                x_str = values[coords['x']].strip()
                y_str = values[coords['y']].strip()

                x = float(x_str) if x_str else np.nan
                y = float(y_str) if y_str else np.nan

                # Apply likelihood threshold
                if likelihood_threshold is not None and 'likelihood' in coords:
                    likelihood_str = values[coords['likelihood']].strip()
                    likelihood = float(likelihood_str) if likelihood_str else 0.0

                    if likelihood < likelihood_threshold:
                        x, y = np.nan, np.nan

            except (ValueError, IndexError):
                x, y = np.nan, np.nan

            trajectories[bodypart].append([x, y, z_value])

    # Convert to numpy
    result = {
        name: np.array(coords, dtype=np.float64) * scale_factor
        for name, coords in trajectories.items()
    }

    n_markers = len(result)
    n_frames = len(next(iter(result.values())))
    logger.info(f"  Loaded {n_markers} bodyparts × {n_frames} frames")

    return result


def detect_csv_format(*, filepath: Path) -> str:
    """
    Auto-detect CSV format.

    Returns:
        One of: 'tidy', 'wide', 'dlc'
    """
    with open(filepath, 'r') as f:
        lines = [f.readline().strip() for _ in range(3)]

    if len(lines) < 1:
        raise ValueError("CSV file is empty")

    # Check for DLC format
    if len(lines) >= 3:
        row3_values = lines[2].split(',')
        if any(val.strip() in ['x', 'y', 'likelihood', 'coords'] for val in row3_values):
            return 'dlc'

    # Check tidy vs wide
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError("CSV has no headers")

        # Tidy format
        if 'keypoint' in headers and 'x' in headers and 'y' in headers:
            return 'tidy'

        # Wide format
        if any(h.endswith('_x') for h in headers) and any(h.endswith('_y') for h in headers):
            return 'wide'

    raise ValueError(
        "Unknown CSV format. Expected:\n"
        "  - Tidy: columns 'frame', 'keypoint', 'x', 'y', 'z'\n"
        "  - Wide: columns 'frame', '{marker}_x', '{marker}_y'\n"
        "  - DLC: 3-row header with scorer/bodyparts/coords"
    )


def load_trajectories(
        *,
        filepath: Path,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
        likelihood_threshold: float | None = None,
        format: str | None = None
) -> dict[str, np.ndarray]:
    """
    Load trajectories with automatic format detection.

    Args:
        filepath: Path to CSV file
        scale_factor: Multiplier for coordinates (e.g., 0.001 for mm to m)
        z_value: Default z-coordinate for 2D data
        likelihood_threshold: For DLC format, filter low-confidence points
        format: Force specific format ('tidy', 'wide', 'dlc'), or None for auto

    Returns:
        Dictionary mapping marker names to (n_frames, 3) arrays
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect or use specified format
    if format is None:
        format = detect_csv_format(filepath=filepath)

    logger.info(f"CSV format: {format}")

    if format == 'tidy':
        return load_tidy_csv(filepath=filepath, scale_factor=scale_factor)
    elif format == 'wide':
        return load_wide_csv(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value
        )
    elif format == 'dlc':
        return load_dlc_csv(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value,
            likelihood_threshold=likelihood_threshold
        )
    else:
        raise ValueError(f"Unknown format: {format}")