"""
Toy Trajectory Loading
======================

Functions for loading toy trajectory data from CSV files.
Automatically detects the CSV format based on column headers.

Supported formats:
1. DLC format (legacy): frame, keypoint, x, y, z
2. Extended DLC format: frame, keypoint, x, y, z, model, trajectory, reprojection_error

When multiple trajectory types are present, the user must specify which to use.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class ToyCSVFormat(Enum):
    """Detected format of toy trajectory CSV file."""

    DLC_BASIC = "dlc_basic"
    """Basic DLC format: frame, keypoint, x, y, z"""

    DLC_EXTENDED = "dlc_extended"
    """Extended DLC format: frame, keypoint, x, y, z, model, trajectory, reprojection_error"""


@dataclass(frozen=True)
class ToyCSVMetadata:
    """Metadata extracted from a toy trajectory CSV file."""

    format: ToyCSVFormat
    """Detected CSV format."""

    keypoint_names: tuple[str, ...]
    """Unique keypoint names found in the file."""

    trajectory_types: tuple[str, ...]
    """Unique trajectory types found (empty tuple for DLC_BASIC format)."""

    model_names: tuple[str, ...]
    """Unique model names found (empty tuple for DLC_BASIC format)."""

    n_frames: int
    """Number of frames (max frame index + 1)."""


def detect_toy_csv_format(csv_path: Path) -> ToyCSVMetadata:
    """
    Detect the format of a toy trajectory CSV file by examining column headers.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        ToyCSVMetadata with format details and available options.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Toy trajectory CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        header_line = f.readline().strip()
        columns = [col.strip() for col in header_line.split(",")]

    # Check for required columns
    required_basic = {"frame", "keypoint", "x", "y", "z"}
    missing_basic = required_basic - set(columns)
    if missing_basic:
        raise ValueError(
            f"CSV missing required columns: {missing_basic}. "
            f"Found columns: {columns}"
        )

    # Check for extended format columns
    extended_columns = {"model", "trajectory"}
    has_extended = extended_columns.issubset(set(columns))

    if has_extended:
        csv_format = ToyCSVFormat.DLC_EXTENDED
    else:
        csv_format = ToyCSVFormat.DLC_BASIC

    # Scan file to collect metadata
    keypoints: set[str] = set()
    trajectories: set[str] = set()
    models: set[str] = set()
    max_frame: int = -1

    col_indices = {col: i for i, col in enumerate(columns)}
    frame_idx = col_indices["frame"]
    keypoint_idx = col_indices["keypoint"]
    trajectory_idx = col_indices.get("trajectory")
    model_idx = col_indices.get("model")

    with open(csv_path, "r") as f:
        f.readline()  # Skip header
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")

            frame = int(parts[frame_idx])
            max_frame = max(max_frame, frame)

            keypoints.add(parts[keypoint_idx])

            if trajectory_idx is not None:
                trajectories.add(parts[trajectory_idx])

            if model_idx is not None:
                models.add(parts[model_idx])

    return ToyCSVMetadata(
        format=csv_format,
        keypoint_names=tuple(sorted(keypoints)),
        trajectory_types=tuple(sorted(trajectories)),
        model_names=tuple(sorted(models)),
        n_frames=max_frame + 1,
    )


def load_toy_trajectories_from_dlc_csv(
    csv_path: Path,
    reference_timestamps: NDArray[np.float64],
    trajectory_type: str | None = None,
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    """
    Load toy trajectories from a DLC-format CSV file.

    Automatically detects the CSV format and handles both basic and extended formats.
    For extended format CSVs with multiple trajectory types, the trajectory_type
    parameter must be specified.

    Args:
        csv_path: Path to the toy trajectory CSV file.
        reference_timestamps: Reference timestamps from mocap data (same acquisition).
            The toy data uses the same timestamps as the mocap body/skull/spine data.
        trajectory_type: Which trajectory type to load (required for extended format
            with multiple trajectory types, ignored for basic format).

    Returns:
        Tuple of (trajectories_array, marker_names, timestamps):
            - trajectories_array: (n_frames, n_markers, 3) array
            - marker_names: list of marker names in canonical order
            - timestamps: Same as reference_timestamps

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing, trajectory_type is needed but not
            provided, or frame counts don't match.
    """
    logger.info(f"Loading toy trajectories from: {csv_path}")

    # Detect format
    metadata = detect_toy_csv_format(csv_path)
    logger.info(f"  Detected format: {metadata.format.value}")
    logger.info(f"  Keypoints: {metadata.keypoint_names}")

    if metadata.format == ToyCSVFormat.DLC_EXTENDED:
        logger.info(f"  Trajectory types: {metadata.trajectory_types}")
        logger.info(f"  Models: {metadata.model_names}")

    # Determine which trajectory type to use for extended format
    selected_trajectory: str | None = None
    if metadata.format == ToyCSVFormat.DLC_EXTENDED:
        if len(metadata.trajectory_types) == 0:
            raise ValueError("Extended format CSV has no trajectory types")
        elif len(metadata.trajectory_types) == 1:
            selected_trajectory = metadata.trajectory_types[0]
            logger.info(f"  Using only available trajectory type: {selected_trajectory}")
        elif trajectory_type is None:
            raise ValueError(
                f"CSV contains multiple trajectory types {metadata.trajectory_types}. "
                f"You must specify trajectory_type parameter."
            )
        elif trajectory_type not in metadata.trajectory_types:
            raise ValueError(
                f"Requested trajectory_type '{trajectory_type}' not found in CSV. "
                f"Available types: {metadata.trajectory_types}"
            )
        else:
            selected_trajectory = trajectory_type
            logger.info(f"  Using requested trajectory type: {selected_trajectory}")

    # Parse CSV into structured data
    data: dict[str, dict[int, dict[str, float]]] = {}

    with open(csv_path, "r") as f:
        header_line = f.readline().strip()
        columns = [col.strip() for col in header_line.split(",")]

        col_indices = {col: i for i, col in enumerate(columns)}
        frame_idx = col_indices["frame"]
        keypoint_idx = col_indices["keypoint"]
        x_idx = col_indices["x"]
        y_idx = col_indices["y"]
        z_idx = col_indices["z"]
        trajectory_idx = col_indices.get("trajectory")

        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")

            # Skip rows that don't match selected trajectory type (extended format)
            if trajectory_idx is not None and selected_trajectory is not None:
                row_trajectory = parts[trajectory_idx]
                if row_trajectory != selected_trajectory:
                    continue

            frame = int(parts[frame_idx])
            keypoint = parts[keypoint_idx]
            x = float(parts[x_idx])
            y = float(parts[y_idx])
            z = float(parts[z_idx])

            if keypoint not in data:
                data[keypoint] = {}
            data[keypoint][frame] = {"x": x, "y": y, "z": z}

    # Canonical marker names for toy data
    canonical_marker_names = ["toy_face", "toy_top", "toy_tail"]

    # Verify we have all expected markers
    missing_markers = set(canonical_marker_names) - set(data.keys())
    if missing_markers:
        raise ValueError(f"Missing toy markers in CSV: {missing_markers}")

    # Get frame count from data
    all_frames: set[int] = set()
    for keypoint_data in data.values():
        all_frames.update(keypoint_data.keys())
    n_frames_data = max(all_frames) + 1

    n_frames_ref = len(reference_timestamps)
    if n_frames_data != n_frames_ref:
        raise ValueError(
            f"Toy frame count ({n_frames_data}) differs from reference ({n_frames_ref})."
        )

    n_frames = n_frames_ref
    n_markers = len(canonical_marker_names)

    # Build trajectories array
    trajectories = np.full((n_frames, n_markers, 3), np.nan, dtype=np.float64)

    for marker_idx, marker_name in enumerate(canonical_marker_names):
        marker_data = data[marker_name]
        for frame in range(n_frames):
            if frame in marker_data:
                trajectories[frame, marker_idx, 0] = marker_data[frame]["x"]
                trajectories[frame, marker_idx, 1] = marker_data[frame]["y"]
                trajectories[frame, marker_idx, 2] = marker_data[frame]["z"]

    # Check for NaN values
    nan_count = int(np.sum(np.isnan(trajectories)))
    if nan_count > 0:
        raise ValueError(f"Toy trajectories contain {nan_count} NaN values")

    logger.info(f"  Loaded {n_markers} markers ({canonical_marker_names}), {n_frames} frames")
    return trajectories, canonical_marker_names, reference_timestamps.copy()


def load_toy_trajectories_dict_from_dlc_csv(
    csv_path: Path,
    trajectory_type: str | None = None,
) -> tuple[dict[str, NDArray[np.float64]], int]:
    """
    Load toy trajectories from a DLC-format CSV into a dictionary format.

    This is an alternative interface that returns trajectories as a dictionary
    mapping keypoint names to (n_frames, 3) arrays, without requiring reference
    timestamps.

    Args:
        csv_path: Path to the toy trajectory CSV file.
        trajectory_type: Which trajectory type to load (required for extended format
            with multiple trajectory types, ignored for basic format).

    Returns:
        Tuple of (trajectories_dict, n_frames):
            - trajectories_dict: dict mapping keypoint name to (n_frames, 3) array
            - n_frames: number of frames

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or trajectory_type is needed but
            not provided.
    """
    logger.info(f"Loading toy trajectories (dict format) from: {csv_path}")

    # Detect format
    metadata = detect_toy_csv_format(csv_path)
    logger.info(f"  Detected format: {metadata.format.value}")

    # Determine which trajectory type to use for extended format
    selected_trajectory: str | None = None
    if metadata.format == ToyCSVFormat.DLC_EXTENDED:
        if len(metadata.trajectory_types) == 0:
            raise ValueError("Extended format CSV has no trajectory types")
        elif len(metadata.trajectory_types) == 1:
            selected_trajectory = metadata.trajectory_types[0]
            logger.info(f"  Using only available trajectory type: {selected_trajectory}")
        elif trajectory_type is None:
            raise ValueError(
                f"CSV contains multiple trajectory types {metadata.trajectory_types}. "
                f"You must specify trajectory_type parameter."
            )
        elif trajectory_type not in metadata.trajectory_types:
            raise ValueError(
                f"Requested trajectory_type '{trajectory_type}' not found in CSV. "
                f"Available types: {metadata.trajectory_types}"
            )
        else:
            selected_trajectory = trajectory_type
            logger.info(f"  Using requested trajectory type: {selected_trajectory}")

    # Parse CSV into structured data
    data: dict[str, dict[int, dict[str, float]]] = {}

    with open(csv_path, "r") as f:
        header_line = f.readline().strip()
        columns = [col.strip() for col in header_line.split(",")]

        col_indices = {col: i for i, col in enumerate(columns)}
        frame_idx = col_indices["frame"]
        keypoint_idx = col_indices["keypoint"]
        x_idx = col_indices["x"]
        y_idx = col_indices["y"]
        z_idx = col_indices["z"]
        trajectory_idx = col_indices.get("trajectory")

        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")

            # Skip rows that don't match selected trajectory type (extended format)
            if trajectory_idx is not None and selected_trajectory is not None:
                row_trajectory = parts[trajectory_idx]
                if row_trajectory != selected_trajectory:
                    continue

            frame = int(parts[frame_idx])
            keypoint = parts[keypoint_idx]
            x = float(parts[x_idx])
            y = float(parts[y_idx])
            z = float(parts[z_idx])

            if keypoint not in data:
                data[keypoint] = {}
            data[keypoint][frame] = {"x": x, "y": y, "z": z}

    # Get frame count from data
    all_frames: set[int] = set()
    for keypoint_data in data.values():
        all_frames.update(keypoint_data.keys())
    n_frames = max(all_frames) + 1

    # Build trajectories dict
    trajectories_dict: dict[str, NDArray[np.float64]] = {}
    for keypoint_name, frame_data in data.items():
        traj = np.full((n_frames, 3), np.nan, dtype=np.float64)
        for frame, coords in frame_data.items():
            traj[frame, 0] = coords["x"]
            traj[frame, 1] = coords["y"]
            traj[frame, 2] = coords["z"]
        trajectories_dict[keypoint_name] = traj

    logger.info(f"  Loaded {len(trajectories_dict)} keypoints, {n_frames} frames")
    return trajectories_dict, n_frames


def get_available_trajectory_types(csv_path: Path) -> list[str]:
    """
    Get the list of available trajectory types in a toy CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of trajectory type names. Empty list for basic DLC format.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    metadata = detect_toy_csv_format(csv_path)
    return list(metadata.trajectory_types)