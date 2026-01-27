"""
Ferret Data Resampler
=====================

Loads skull kinematics, left/right eye kinematics, trajectory data, and videos,
resamples all to common timestamps, and saves the results.

All output files will have EXACTLY the same number of frames and identical timestamps.
Videos are resampled to the 'display_videos' folder at the same level as the output directory.
"""
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from numpy.typing import NDArray

from python_code.ferret_gaze.data_resampling.data_resampling_helpers import (
    ResamplingStrategy,
    resample_to_common_timestamps,
)
from python_code.ferret_gaze.data_resampling.toy_trajectory_loader import (
    ToyCSVFormat,
    detect_toy_csv_format,
    get_available_trajectory_types,
    load_toy_trajectories_from_dlc_csv,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.ferret_eyeball_reference_geometry import NUM_PUPIL_POINTS
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.stick_figure_topology_model import StickFigureTopology

logger = logging.getLogger(__name__)


# =============================================================================
# VIDEO RESAMPLING
# =============================================================================


@dataclass
class VideoConfig:
    """Configuration for a video to be resampled."""

    path: Path
    timestamps_path: Path
    name: str


def load_video_timestamps(timestamps_path: Path) -> NDArray[np.float64]:
    """
    Load video timestamps from a file.

    Supports:
    - .npy: numpy array file
    - .csv: CSV with 'timestamp' or 'timestamp_s' column
    - .txt: plain text, one timestamp per line

    Args:
        timestamps_path: Path to timestamps file

    Returns:
        Array of timestamps (n_frames,) in RAW units (not converted)
    """
    if not timestamps_path.exists():
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")

    suffix = timestamps_path.suffix.lower()

    if suffix == ".npy":
        timestamps = np.load(timestamps_path)
    elif suffix == ".csv":
        import csv

        with open(timestamps_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"Empty CSV file: {timestamps_path}")

            ts_col = None
            for col in ["timestamp", "timestamp_s", "time", "time_s", "timestamps"]:
                if col in fieldnames:
                    ts_col = col
                    break

            if ts_col is None:
                raise ValueError(
                    f"No timestamp column found in {timestamps_path}. "
                    f"Expected one of: timestamp, timestamp_s, time, time_s"
                )

            timestamps = np.array([float(row[ts_col]) for row in reader], dtype=np.float64)
    elif suffix == ".txt":
        timestamps = np.loadtxt(timestamps_path, dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported timestamps file format: {suffix}. Use .npy, .csv, or .txt"
        )

    return timestamps.astype(np.float64)


def detect_timestamp_unit_and_convert_to_seconds(
    timestamps: NDArray[np.float64],
    min_fps: float = 10.0,
    max_fps: float = 1000.0,
) -> NDArray[np.float64]:
    """
    Detect timestamp units and convert to seconds.

    Uses frame duration statistics to detect if timestamps are in:
    - seconds (frame duration 0.001-0.1s for 10-1000 fps)
    - milliseconds (frame duration 1-100 for 10-1000 fps)
    - microseconds (frame duration 1000-100000 for 10-1000 fps)
    - nanoseconds (frame duration 1e6-1e8 for 10-1000 fps)

    Args:
        timestamps: Raw timestamps array
        min_fps: Minimum expected framerate (default 10 fps)
        max_fps: Maximum expected framerate (default 1000 fps)

    Returns:
        Timestamps converted to seconds
    """
    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to detect units")

    # Compute median frame duration
    frame_durations = np.diff(timestamps)
    median_duration = float(np.median(frame_durations))

    # Expected frame duration range in seconds
    min_duration_s = 1.0 / max_fps  # e.g., 0.001s at 1000fps
    max_duration_s = 1.0 / min_fps  # e.g., 0.1s at 10fps

    # Define conversion factors and their expected duration ranges
    unit_specs = [
        ("seconds", 1.0, min_duration_s, max_duration_s),
        ("milliseconds", 1e-3, min_duration_s * 1e3, max_duration_s * 1e3),
        ("microseconds", 1e-6, min_duration_s * 1e6, max_duration_s * 1e6),
        ("nanoseconds", 1e-9, min_duration_s * 1e9, max_duration_s * 1e9),
    ]

    detected_unit = None
    conversion_factor = 1.0

    for unit_name, factor, min_expected, max_expected in unit_specs:
        if min_expected <= median_duration <= max_expected:
            detected_unit = unit_name
            conversion_factor = factor
            break

    if detected_unit is None:
        if median_duration > 1e6:
            detected_unit = "nanoseconds"
            conversion_factor = 1e-9
        elif median_duration > 1e3:
            detected_unit = "microseconds"
            conversion_factor = 1e-6
        elif median_duration > 1:
            detected_unit = "milliseconds"
            conversion_factor = 1e-3
        else:
            detected_unit = "seconds"
            conversion_factor = 1.0

        raise ValueError(
            f"Could not definitively detect timestamp unit from frame duration {median_duration:.2e}. "
            f"Assuming {detected_unit}."
        )

    logger.info(f"  Detected timestamp unit: {detected_unit} (median frame duration: {median_duration:.2e})")

    converted = timestamps * conversion_factor
    duration_s = converted[-1] - converted[0]
    effective_fps = (len(converted) - 1) / duration_s if duration_s > 0 else 0

    logger.info(f"  Converted duration: {duration_s:.2f}s, effective FPS: {effective_fps:.1f}")

    return converted


def normalize_video_timestamps(
    video_timestamps: NDArray[np.float64],
    target_timestamps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Normalize video timestamps to align with target timestamps.

    This function:
    1. Detects the unit of video timestamps and converts to seconds
    2. Zeros the video timestamps to start from 0 (matching zeroed target timestamps)

    Args:
        video_timestamps: Raw video timestamps (may be in ns, us, ms, or s)
        target_timestamps: Target timestamps (assumed to be in seconds, starting from 0)

    Returns:
        Video timestamps in seconds, zeroed to start from 0
    """
    # Convert to seconds
    video_ts_seconds = detect_timestamp_unit_and_convert_to_seconds(video_timestamps)

    # Zero the timestamps (subtract the first timestamp)
    video_ts_zeroed = video_ts_seconds - video_ts_seconds[0]

    logger.info(f"  Zeroed video timestamps: 0.0s to {video_ts_zeroed[-1]:.4f}s")

    return video_ts_zeroed


def compute_frame_mapping(
    video_timestamps: NDArray[np.float64],
    target_timestamps: NDArray[np.float64],
) -> NDArray[np.int64]:
    """
    Compute mapping from target timestamps to closest video frames.

    Args:
        video_timestamps: Original video timestamps (n_video_frames,)
        target_timestamps: Target timestamps to resample to (n_target_frames,)

    Returns:
        Array of video frame indices (n_target_frames,) - closest frame for each target timestamp
    """
    n_target = len(target_timestamps)
    n_video = len(video_timestamps)

    # Use searchsorted to find insertion points
    indices = np.searchsorted(video_timestamps, target_timestamps, side="left")

    # Refine to find closest frame
    frame_mapping = np.zeros(n_target, dtype=np.int64)

    for i in range(n_target):
        target_time = target_timestamps[i]
        idx = indices[i]

        # Check if previous frame is closer
        if idx > 0:
            if idx >= n_video:
                idx = n_video - 1
            elif abs(video_timestamps[idx - 1] - target_time) < abs(video_timestamps[idx] - target_time):
                idx = idx - 1

        frame_mapping[i] = idx

    return frame_mapping


def draw_frame_label(
    frame: NDArray[np.uint8],
    original_frame_number: int,
    font_scale: float,
    thickness: int,
    margin: int = 10,
) -> NDArray[np.uint8]:
    """
    Draw the original frame number on the frame.

    Args:
        frame: BGR image array (H, W, 3)
        original_frame_number: Frame number to display
        font_scale: Font scale factor
        thickness: Line thickness
        margin: Margin from corner in pixels

    Returns:
        Frame with label drawn
    """
    frame_copy = frame.copy()

    label = f"F:{original_frame_number}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Position in top-left corner
    x = margin
    y = margin + text_height

    # Draw background rectangle for readability
    padding = 4
    cv2.rectangle(
        frame_copy,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (0, 0, 0),
        -1,
    )

    # Draw text in white
    cv2.putText(
        frame_copy,
        label,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return frame_copy


def resample_single_video(
    video_path: Path,
    video_timestamps: NDArray[np.float64],
    target_timestamps: NDArray[np.float64],
    output_path: Path,
    video_name: str,
    target_fps: float,
) -> None:
    """
    Resample a single video to match target timestamps.

    Args:
        video_path: Path to input video
        video_timestamps: Original video timestamps (will be normalized to start from 0)
        target_timestamps: Target timestamps (assumed to start from 0)
        output_path: Path to output video
        video_name: Name for logging
        target_fps: FPS for output video
    """
    logger.info(f"Resampling video: {video_name}")
    logger.info(f"  Input: {video_path}")
    logger.info(f"  Output: {output_path}")

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"  Original video: {frame_width}x{frame_height}, {original_frame_count} frames")
    logger.info(f"  Original timestamps: {len(video_timestamps)} entries")
    logger.info(f"  Target frames: {len(target_timestamps)}")

    # Normalize timestamps to start from 0 for alignment
    # This handles the case where video timestamps are in UTC (large numbers)
    # and target timestamps are zeroed (starting from 0)
    video_timestamps_normalized = video_timestamps - video_timestamps[0]
    target_timestamps_normalized = target_timestamps - target_timestamps[0]

    logger.info(f"  Video duration: {video_timestamps_normalized[-1]:.4f}s")
    logger.info(f"  Target duration: {target_timestamps_normalized[-1]:.4f}s")

    # Compute frame mapping using normalized timestamps
    frame_mapping = compute_frame_mapping(
        video_timestamps=video_timestamps_normalized,
        target_timestamps=target_timestamps_normalized,
    )

    logger.info(f"  Frame mapping range: {frame_mapping.min()} to {frame_mapping.max()}")

    # Determine font scale based on frame size
    min_dimension = min(frame_width, frame_height)
    font_scale = max(0.4, min_dimension / 500.0)
    thickness = max(1, int(min_dimension / 300))

    # Create output video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        target_fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")

    # Cache for frames (to avoid re-reading the same frame multiple times)
    frame_cache: dict[int, NDArray[np.uint8]] = {}
    cache_size_limit = 100

    n_target = len(target_timestamps)

    for target_idx in range(n_target):
        original_frame_idx = int(frame_mapping[target_idx])

        # Check cache
        if original_frame_idx in frame_cache:
            frame = frame_cache[original_frame_idx]
        else:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = cap.read()

            if not ret:
                raise ValueError(
                    f"  Failed to read frame {original_frame_idx}, using black frame"
                )
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # Cache the frame
            if len(frame_cache) >= cache_size_limit:
                oldest_key = next(iter(frame_cache))
                del frame_cache[oldest_key]

            frame_cache[original_frame_idx] = frame

        # Draw frame label
        labeled_frame = draw_frame_label(
            frame=frame,
            original_frame_number=original_frame_idx,
            font_scale=font_scale,
            thickness=thickness,
        )

        writer.write(labeled_frame)

        # Progress logging
        if (target_idx + 1) % 1000 == 0 or target_idx == n_target - 1:
            logger.info(f"  Progress: {target_idx + 1}/{n_target} frames")

    cap.release()
    writer.release()

    logger.info(f"  Saved: {output_path}")


def get_video_frame_count(video_path: Path) -> int:
    """
    Get the number of frames in a video file.

    Args:
        video_path: Path to video file

    Returns:
        Number of frames in the video

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video can't be opened or frame count is invalid
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count ({frame_count}) for video: {video_path}")

    return frame_count


def resample_videos(
    video_configs: list[VideoConfig],
    common_timestamps_original: NDArray[np.float64],
    output_dir: Path,
    target_fps: float,
    recreate_videos: bool = False,
) -> list[Path]:
    """
    Resample multiple videos to match common timestamps.

    Creates resampled videos where each frame corresponds to a common timestamp,
    showing the closest frame from the original video with the original frame
    number drawn on the frame.

    Args:
        video_configs: List of VideoConfig objects specifying videos to resample
        common_timestamps_original: Common timestamps in the ORIGINAL domain
            (before zeroing). Must be the same timestamps used for kinematics resampling.
        output_dir: Directory to save resampled videos
        target_fps: FPS for output videos
        recreate_videos: If False, skip videos that already exist with correct frame count.
            If True, recreate all videos regardless.

    Returns:
        List of paths to the created resampled videos
    """
    logger.info("\n" + "=" * 80)
    logger.info("RESAMPLING VIDEOS")
    logger.info("=" * 80)
    logger.info(f"  Number of videos: {len(video_configs)}")
    logger.info(f"  Common timestamps: {len(common_timestamps_original)} frames")
    logger.info(f"  Target FPS: {target_fps}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Recreate existing: {recreate_videos}")

    expected_frames = len(common_timestamps_original)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for config in video_configs:
        logger.info(f"\n--- Processing: {config.name} ---")

        # Check if output already exists with correct frame count
        output_path = output_dir / f"{config.name}_resampled.mp4"

        if not recreate_videos and output_path.exists():
            existing_frames = get_video_frame_count(output_path)
            if existing_frames == expected_frames:
                logger.info(f"  SKIPPING: Output already exists with correct frame count ({existing_frames})")
                logger.info(f"  Path: {output_path}")
                output_paths.append(output_path)
                continue
            else:
                logger.info(f"  Existing output has wrong frame count ({existing_frames} != {expected_frames}), recreating...")

        # Validate paths
        if not config.path.exists():
            raise FileNotFoundError(f"Video file not found: {config.path}")

        if not config.timestamps_path.exists():
            raise FileNotFoundError(f"Timestamps file not found: {config.timestamps_path}")

        # Load raw video timestamps
        raw_video_timestamps = load_video_timestamps(config.timestamps_path)
        logger.info(f"  Raw video timestamps: {len(raw_video_timestamps)} frames")

        # Normalize video timestamps: convert to seconds and zero
        video_timestamps = normalize_video_timestamps(
            video_timestamps=raw_video_timestamps,
            target_timestamps=common_timestamps_original,
        )

        logger.info(
            f"  Normalized video time range: {video_timestamps[0]:.4f}s to {video_timestamps[-1]:.4f}s"
        )
        logger.info(
            f"  Target time range: {common_timestamps_original[0]:.4f}s to {common_timestamps_original[-1]:.4f}s"
        )

        # Resample the video
        resample_single_video(
            video_path=config.path,
            video_timestamps=video_timestamps,
            target_timestamps=common_timestamps_original,
            output_path=output_path,
            video_name=config.name,
            target_fps=target_fps,
        )

        output_paths.append(output_path)

    logger.info("\n" + "=" * 80)
    logger.info("VIDEO RESAMPLING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Resampled {len(output_paths)} videos")
    logger.info(f"  All videos have exactly {len(common_timestamps_original)} frames")

    return output_paths


def resample_videos_from_dicts(
    video_dicts: list[dict[str, str]],
    common_timestamps_original: NDArray[np.float64],
    output_dir: Path,
    target_fps: float,
    recreate_videos: bool = False,
) -> list[Path]:
    """
    Resample videos using dictionary configuration (matching VIDEOS format from blender script).

    Args:
        video_dicts: List of dicts with 'path', 'timestamps_path', and optionally 'name'
        common_timestamps_original: Common timestamps in original domain
        output_dir: Directory to save resampled videos
        target_fps: FPS for output videos
        recreate_videos: If False, skip videos that already exist with correct frame count.

    Returns:
        List of paths to created videos
    """
    configs: list[VideoConfig] = []

    for video_dict in video_dicts:
        if "path" not in video_dict:
            raise ValueError("Video dict missing required 'path' field")
        if "timestamps_path" not in video_dict:
            raise ValueError(f"Video dict for {video_dict['path']} missing required 'timestamps_path' field")

        path = Path(video_dict["path"])
        timestamps_path = Path(video_dict["timestamps_path"])
        name = video_dict.get("name", path.stem)

        configs.append(VideoConfig(path=path, timestamps_path=timestamps_path, name=name))

    return resample_videos(
        video_configs=configs,
        common_timestamps_original=common_timestamps_original,
        output_dir=output_dir,
        target_fps=target_fps,
        recreate_videos=recreate_videos,
    )


# =============================================================================
# TRAJECTORY LOADING AND SAVING
# =============================================================================


def load_skull_and_spine_trajectories(
    trajectories_csv_path: Path,
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    """
    Load skull and spine trajectories from tidy CSV.

    Returns:
        Tuple of (trajectories_array, marker_names, timestamps)
        - trajectories_array: (n_frames, n_markers, 3) array
        - marker_names: list of marker names
        - timestamps: (n_frames,) array
    """
    logger.info(f"Loading skull and spine trajectories from: {trajectories_csv_path}")

    # Parse CSV
    data: dict[str, dict[int, dict[str, float]]] = {}
    timestamps_dict: dict[int, float] = {}

    with open(trajectories_csv_path, "r") as f:
        header = f.readline().strip().split(",")
        frame_idx = header.index("frame")
        timestamp_idx = header.index("timestamp")
        traj_idx = header.index("trajectory")
        comp_idx = header.index("component")
        val_idx = header.index("value")

        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[frame_idx])
            timestamp = float(parts[timestamp_idx])
            traj_name = parts[traj_idx]
            component = parts[comp_idx]
            value = float(parts[val_idx])

            timestamps_dict[frame] = timestamp

            if traj_name not in data:
                data[traj_name] = {}
            if frame not in data[traj_name]:
                data[traj_name][frame] = {}
            data[traj_name][frame][component] = value

    marker_names = list(data.keys())
    n_frames = max(timestamps_dict.keys()) + 1
    n_markers = len(marker_names)

    # Build arrays
    timestamps = np.array([timestamps_dict[i] for i in range(n_frames)], dtype=np.float64)
    trajectories = np.zeros((n_frames, n_markers, 3), dtype=np.float64)

    for marker_idx, marker_name in enumerate(marker_names):
        marker_data = data[marker_name]
        for frame in range(n_frames):
            if frame in marker_data:
                trajectories[frame, marker_idx, 0] = marker_data[frame].get("x", 0.0)
                trajectories[frame, marker_idx, 1] = marker_data[frame].get("y", 0.0)
                trajectories[frame, marker_idx, 2] = marker_data[frame].get("z", 0.0)

    logger.info(f"  Loaded {n_markers} markers, {n_frames} frames")
    return trajectories, marker_names, timestamps


def save_skull_and_spine_trajectories_csv(
    trajectories: NDArray[np.float64],
    marker_names: list[str],
    timestamps: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Save skull and spine trajectories to tidy CSV format."""
    logger.info(f"Saving skull and spine trajectories to: {output_path}")

    n_frames = len(timestamps)
    n_markers = len(marker_names)

    rows: list[dict[str, int | float | str]] = []
    for frame in range(n_frames):
        for marker_idx in range(n_markers):
            for comp_idx, comp_name in enumerate(["x", "y", "z"]):
                rows.append({
                    "frame": frame,
                    "timestamp": timestamps[frame],
                    "trajectory": marker_names[marker_idx],
                    "component": comp_name,
                    "value": trajectories[frame, marker_idx, comp_idx],
                    "units": "mm",
                })

    df = pl.DataFrame(rows)
    df.write_csv(output_path)


def create_eye_topology(eye_name: str) -> StickFigureTopology:
    """
    Create a StickFigureTopology for an eye.

    Marker names: tear_duct, outer_eye, pupil_center, p1-p8
    Rigid edges: tear_duct <-> outer_eye (fixed socket landmarks)
    Display edges: tear_duct-outer_eye line, pupil boundary ring (p1->p2->...->p8->p1)
    """
    marker_names = [
        "tear_duct",
        "outer_eye",
        "pupil_center",
    ] + [f"p{i}" for i in range(1, NUM_PUPIL_POINTS + 1)]

    # Rigid edges: only the socket landmarks maintain fixed distance
    rigid_edges: list[tuple[str, str]] = [
        ("tear_duct", "outer_eye"),
    ]

    # Display edges: socket line + pupil boundary ring
    display_edges: list[tuple[str, str]] = [
        ("tear_duct", "outer_eye"),
    ]
    # Add pupil boundary connections: p1->p2, p2->p3, ..., p8->p1
    for i in range(1, NUM_PUPIL_POINTS + 1):
        next_i = (i % NUM_PUPIL_POINTS) + 1
        display_edges.append((f"p{i}", f"p{next_i}"))

    return StickFigureTopology(
        name=eye_name,
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        display_edges=display_edges,
    )


def extract_eye_trajectories(
    eye_kinematics: FerretEyeKinematics,
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Extract eye landmark trajectories from FerretEyeKinematics.

    Returns:
        Tuple of (trajectories_array, marker_names)
        - trajectories_array: (n_frames, n_markers, 3) array
        - marker_names: list of marker names in order
    """
    n_frames = eye_kinematics.n_frames

    marker_names = [
        "tear_duct",
        "outer_eye",
        "pupil_center",
    ] + [f"p{i}" for i in range(1, NUM_PUPIL_POINTS + 1)]

    n_markers = len(marker_names)
    trajectories = np.zeros((n_frames, n_markers, 3), dtype=np.float64)

    # Socket landmarks
    trajectories[:, 0, :] = eye_kinematics.tear_duct_mm
    trajectories[:, 1, :] = eye_kinematics.outer_eye_mm

    # Tracked pupil center
    trajectories[:, 2, :] = eye_kinematics.tracked_pupil_center

    # Tracked pupil boundary points p1-p8
    for i in range(NUM_PUPIL_POINTS):
        trajectories[:, 3 + i, :] = eye_kinematics.tracked_pupil_points[:, i, :]

    return trajectories, marker_names


def save_eye_trajectories_csv(
    trajectories: NDArray[np.float64],
    marker_names: list[str],
    timestamps: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Save eye trajectories to tidy CSV format."""
    logger.info(f"Saving eye trajectories to: {output_path}")

    n_frames = len(timestamps)
    n_markers = len(marker_names)

    rows: list[dict[str, int | float | str]] = []
    for frame in range(n_frames):
        for marker_idx in range(n_markers):
            for comp_idx, comp_name in enumerate(["x", "y", "z"]):
                rows.append({
                    "frame": frame,
                    "timestamp": timestamps[frame],
                    "trajectory": marker_names[marker_idx],
                    "component": comp_name,
                    "value": trajectories[frame, marker_idx, comp_idx],
                    "units": "mm",
                })

    df = pl.DataFrame(rows)
    df.write_csv(output_path)


# =============================================================================
# TOY TRAJECTORY LOADING AND SAVING
# =============================================================================

# Canonical marker names for toy data
TOY_MARKER_NAMES = ["toy_face", "toy_top", "toy_tail"]


def load_toy_trajectories(
    toy_csv_path: Path,
    reference_timestamps: NDArray[np.float64],
    trajectory_type: str | None = None,
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    """
    Load toy trajectories from DLC-format CSV.

    Automatically detects the CSV format from column headers:
    - Basic format: frame, keypoint, x, y, z
    - Extended format: frame, keypoint, x, y, z, model, trajectory, reprojection_error

    For extended format CSVs with multiple trajectory types (e.g., '3d_xyz' and
    'rigid_3d_xyz'), you must specify which trajectory_type to load.

    The toy data uses the same timestamps as the mocap body/skull/spine data,
    so we use the reference timestamps directly.

    Args:
        toy_csv_path: Path to toy trajectory CSV file (DLC format)
        reference_timestamps: Reference timestamps from mocap data (same acquisition)
        trajectory_type: Which trajectory type to load. Required for extended format
            CSVs with multiple trajectory types. Use get_available_trajectory_types()
            to see available options. Ignored for basic format CSVs.

    Returns:
        Tuple of (trajectories_array, marker_names, timestamps)
        - trajectories_array: (n_frames, 3, 3) array for [toy_face, toy_top, toy_tail]
        - marker_names: ["toy_face", "toy_top", "toy_tail"]
        - timestamps: Same as reference_timestamps
    """
    # Check what format and trajectory types are available
    metadata = detect_toy_csv_format(toy_csv_path)

    if metadata.format == ToyCSVFormat.DLC_EXTENDED and len(metadata.trajectory_types) > 1:
        if trajectory_type is None:
            available = get_available_trajectory_types(toy_csv_path)
            raise ValueError(
                f"Toy CSV contains multiple trajectory types: {available}. "
                f"You must specify trajectory_type parameter."
            )
        logger.info(f"Loading toy trajectories with trajectory_type='{trajectory_type}'")
    elif metadata.format == ToyCSVFormat.DLC_EXTENDED and len(metadata.trajectory_types) == 1:
        logger.info(f"Loading toy trajectories (auto-selected trajectory_type='{metadata.trajectory_types[0]}')")
    else:
        logger.info("Loading toy trajectories (basic DLC format)")

    # Delegate to the loader module
    return load_toy_trajectories_from_dlc_csv(
        csv_path=toy_csv_path,
        reference_timestamps=reference_timestamps,
        trajectory_type=trajectory_type,
    )


def save_toy_trajectories_csv(
    trajectories: NDArray[np.float64],
    marker_names: list[str],
    timestamps: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Save toy trajectories to tidy CSV format (matching skull_and_spine format)."""
    logger.info(f"Saving toy trajectories to: {output_path}")

    n_frames = len(timestamps)
    n_markers = len(marker_names)

    rows: list[dict[str, int | float | str]] = []
    for frame in range(n_frames):
        for marker_idx in range(n_markers):
            for comp_idx, comp_name in enumerate(["x", "y", "z"]):
                rows.append({
                    "frame": frame,
                    "timestamp": timestamps[frame],
                    "trajectory": marker_names[marker_idx],
                    "component": comp_name,
                    "value": trajectories[frame, marker_idx, comp_idx],
                    "units": "mm",
                })

    df = pl.DataFrame(rows)
    df.write_csv(output_path)
    logger.info(f"  Saved Toy trajectory with  {n_markers} markers, {n_frames} frames")


def create_toy_topology() -> StickFigureTopology:
    """
    Create a StickFigureTopology for the toy.

    Marker names: toy_face, toy_top, toy_tail
    No rigid edges (toy can deform/move)
    Display edges: toy_top -> toy_face, toy_top -> toy_tail (radial from top)
    """
    # Display edges radiate from toy_top to face and tail
    display_edges: list[tuple[str, str]] = [
        ("toy_top", "toy_face"),
        ("toy_top", "toy_tail"),
    ]

    return StickFigureTopology(
        name="toy",
        marker_names=TOY_MARKER_NAMES,
        rigid_edges=[],  # No rigid edges - toy can deform
        display_edges=display_edges,
    )


# =============================================================================
# MAIN RESAMPLING FUNCTION
# =============================================================================


def resample_ferret_data(
    skull_solver_output_dir: Path,
    eye_kinematics_dir: Path,
    resampled_data_output_dir: Path,
    toy_trajectories_csv: Path,
    toy_trajectory_type: str | None = None,
    resampling_strategy: ResamplingStrategy = ResamplingStrategy.FASTEST,
    video_configs: list[VideoConfig] | None = None,
    recreate_videos: bool = False,
) -> NDArray[np.float64]:
    """
    Load all ferret data, resample to common timestamps, and save.

    Args:
        skull_solver_output_dir: Directory containing skull_kinematics.csv, skull_reference_geometry.json,
            skull_and_spine_trajectories.csv, skull_and_spine_topology.json
        eye_kinematics_dir: Directory containing left_eye/ and right_eye/ subdirectories
        resampled_data_output_dir: Output directory for resampled data (e.g., 'analyzable_output')
        toy_trajectories_csv: Path to toy trajectory CSV file (DLC format).
            Supports both basic format (frame, keypoint, x, y, z) and extended format
            (with model, trajectory, reprojection_error columns).
        toy_trajectory_type: Which trajectory type to load from the toy CSV.
            Required for extended format CSVs with multiple trajectory types
            (e.g., '3d_xyz' or 'rigid_3d_xyz'). Use get_available_trajectory_types()
            to see available options. Ignored for basic format CSVs.
        resampling_strategy: Strategy for selecting target framerate
        video_configs: Optional list of VideoConfig for videos to resample.
            If provided, videos will be saved to 'display_videos' folder at the
            same level as resampled_data_output_dir.
        recreate_videos: If False, skip videos that already exist with correct frame count.
            If True, recreate all videos regardless.

    Returns:
        Common timestamps array (zeroed to start at 0)
    """
    resampled_data_output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD ALL DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)

    # Load skull kinematics
    skull_kinematics_csv = skull_solver_output_dir / "skull_kinematics.csv"
    skull_reference_geometry_json = skull_solver_output_dir / "skull_reference_geometry.json"
    skull_kinematics = RigidBodyKinematics.load_from_disk(
        kinematics_csv_path=skull_kinematics_csv,
        reference_geometry_json_path=skull_reference_geometry_json,
    )
    logger.info(f"  Skull kinematics: {skull_kinematics.n_frames} frames, {skull_kinematics.framerate_hz:.2f} Hz")
    logger.info(f"    Timestamp range: {skull_kinematics.timestamps[0]:.4f}s to {skull_kinematics.timestamps[-1]:.4f}s")

    # Load skull and spine trajectories
    skull_and_spine_csv = skull_solver_output_dir / "skull_and_spine_trajectories.csv"
    skull_and_spine_trajectories, skull_and_spine_marker_names, skull_and_spine_timestamps = (
        load_skull_and_spine_trajectories(skull_and_spine_csv)
    )
    logger.info(f"  Skull+spine trajectories: {skull_and_spine_trajectories.shape[0]} frames")
    logger.info(f"    Timestamp range: {skull_and_spine_timestamps[0]:.4f}s to {skull_and_spine_timestamps[-1]:.4f}s")

    # Load left eye kinematics
    left_eye_kinematics = FerretEyeKinematics.load_from_directory(
        eye_name="left_eye",
        input_directory=eye_kinematics_dir,
    )
    logger.info(f"  Left eye kinematics: {left_eye_kinematics.n_frames} frames, {left_eye_kinematics.framerate_hz:.2f} Hz")
    logger.info(f"    Timestamp range: {left_eye_kinematics.timestamps[0]:.4f}s to {left_eye_kinematics.timestamps[-1]:.4f}s")

    # Load right eye kinematics
    right_eye_kinematics = FerretEyeKinematics.load_from_directory(
        eye_name="right_eye",
        input_directory=eye_kinematics_dir,
    )
    logger.info(f"  Right eye kinematics: {right_eye_kinematics.n_frames} frames, {right_eye_kinematics.framerate_hz:.2f} Hz")
    logger.info(f"    Timestamp range: {right_eye_kinematics.timestamps[0]:.4f}s to {right_eye_kinematics.timestamps[-1]:.4f}s")

    # Load toy trajectories (uses same timestamps as mocap)
    if not toy_trajectories_csv.exists():
        raise FileNotFoundError(f"Toy trajectories file not found: {toy_trajectories_csv}")

    # Log available trajectory types if extended format
    metadata = detect_toy_csv_format(toy_trajectories_csv)
    if metadata.format == ToyCSVFormat.DLC_EXTENDED:
        logger.info(f"  Toy CSV format: extended (trajectory types: {metadata.trajectory_types})")
    else:
        logger.info(f"  Toy CSV format: basic")

    toy_trajectories, toy_marker_names, _ = load_toy_trajectories(
        toy_csv_path=toy_trajectories_csv,
        reference_timestamps=skull_and_spine_timestamps,
        trajectory_type=toy_trajectory_type,
    )
    logger.info(f"  Toy trajectories: {toy_trajectories.shape[0]} frames")
    logger.info(f"    Markers: {toy_marker_names}")

    # =========================================================================
    # RESAMPLE TO COMMON TIMESTAMPS
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("RESAMPLING TO COMMON TIMESTAMPS")
    logger.info("=" * 40)
    logger.info(f"  Strategy: {resampling_strategy.value}")

    # Use the resample_to_common_timestamps function
    kinematics_list = [
        skull_kinematics,
        left_eye_kinematics.eyeball,
        right_eye_kinematics.eyeball,
    ]

    trajectories_list = [
        (skull_and_spine_trajectories, skull_and_spine_timestamps),
        (toy_trajectories, skull_and_spine_timestamps),
    ]

    # CRITICAL: zero_timestamps=False here! We'll zero later.
    resampled_kinematics, resampled_trajectories = resample_to_common_timestamps(
        kinematics_list=kinematics_list,
        trajectories=trajectories_list,
        strategy=resampling_strategy,
        zero_timestamps=False,
    )

    # Extract resampled data (still in original timestamp domain)
    resampled_skull_kinematics = resampled_kinematics[0]
    resampled_skull_and_spine_trajectories = resampled_trajectories[0]
    resampled_toy_trajectories = resampled_trajectories[1]

    # Get the common timestamps IN THE ORIGINAL DOMAIN (not zeroed yet!)
    common_timestamps_original = resampled_skull_kinematics.timestamps.copy()

    logger.info(f"  Common timestamps (original domain): {len(common_timestamps_original)} frames")
    logger.info(f"  Time range: {common_timestamps_original[0]:.4f}s to {common_timestamps_original[-1]:.4f}s")
    logger.info(f"  Duration: {common_timestamps_original[-1] - common_timestamps_original[0]:.4f}s")
    logger.info(f"  Framerate: {resampled_skull_kinematics.framerate_hz:.2f} Hz")

    # Resample the full eye kinematics using the ORIGINAL (non-zeroed) timestamps
    resampled_left_eye_kinematics = left_eye_kinematics.resample(common_timestamps_original)
    resampled_right_eye_kinematics = right_eye_kinematics.resample(common_timestamps_original)

    # =========================================================================
    # RESAMPLE VIDEOS (before zeroing timestamps!)
    # =========================================================================
    if video_configs:
        display_videos_dir = resampled_data_output_dir.parent / "display_videos"
        resample_videos(
            video_configs=video_configs,
            common_timestamps_original=common_timestamps_original,
            output_dir=display_videos_dir,
            target_fps=resampled_skull_kinematics.framerate_hz,
            recreate_videos=recreate_videos,
        )

    # =========================================================================
    # NOW ZERO ALL TIMESTAMPS
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("ZEROING TIMESTAMPS")
    logger.info("=" * 40)

    time_offset = common_timestamps_original[0]
    common_timestamps = common_timestamps_original - time_offset
    logger.info(f"  Subtracted offset: {time_offset:.4f}s")
    logger.info(f"  New time range: {common_timestamps[0]:.4f}s to {common_timestamps[-1]:.4f}s")

    # Zero the skull kinematics timestamps
    resampled_skull_kinematics = RigidBodyKinematics.from_pose_arrays(
        name=resampled_skull_kinematics.name,
        timestamps=common_timestamps,
        position_xyz=resampled_skull_kinematics.position_xyz,
        quaternions_wxyz=resampled_skull_kinematics.quaternions_wxyz,
        reference_geometry=resampled_skull_kinematics.reference_geometry,
    )

    # Zero the eye kinematics timestamps
    resampled_left_eye_kinematics = resampled_left_eye_kinematics.shift_timestamps(offset=-time_offset)
    resampled_right_eye_kinematics = resampled_right_eye_kinematics.shift_timestamps(offset=-time_offset)

    # =========================================================================
    # VERIFY FRAME COUNTS MATCH
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("VERIFYING FRAME COUNTS")
    logger.info("=" * 40)

    n_frames = len(common_timestamps)
    assert resampled_skull_kinematics.n_frames == n_frames, "Skull kinematics frame count mismatch"
    assert resampled_left_eye_kinematics.n_frames == n_frames, "Left eye kinematics frame count mismatch"
    assert resampled_right_eye_kinematics.n_frames == n_frames, "Right eye kinematics frame count mismatch"
    assert resampled_skull_and_spine_trajectories.shape[0] == n_frames, "Skull+spine trajectories frame count mismatch"
    assert resampled_toy_trajectories.shape[0] == n_frames, "Toy trajectories frame count mismatch"

    # Verify timestamps are identical
    assert np.allclose(resampled_skull_kinematics.timestamps, common_timestamps), "Skull timestamps mismatch"
    assert np.allclose(resampled_left_eye_kinematics.timestamps, common_timestamps), "Left eye timestamps mismatch"
    assert np.allclose(resampled_right_eye_kinematics.timestamps, common_timestamps), "Right eye timestamps mismatch"

    logger.info(f"  All data have exactly {n_frames} frames with identical timestamps")

    # =========================================================================
    # SAVE RESAMPLED DATA
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("SAVING RESAMPLED DATA")
    logger.info("=" * 40)

    # Save common timestamps
    timestamps_path = resampled_data_output_dir / "common_timestamps.npy"
    np.save(timestamps_path, common_timestamps)
    logger.info(f"  Saved: {timestamps_path.name}")

    # Also save the original (non-zeroed) timestamps for video alignment
    timestamps_original_path = resampled_data_output_dir / "common_timestamps_original.npy"
    np.save(timestamps_original_path, common_timestamps_original)
    logger.info(f"  Saved: {timestamps_original_path.name}")

    # Save skull kinematics
    resampled_skull_kinematics.save_to_disk(
        output_directory=resampled_data_output_dir / "skull_kinematics",
    )

    # Copy skull reference geometry (unchanged)
    shutil.copy(
        skull_reference_geometry_json,
        resampled_data_output_dir / "skull_kinematics" / "skull_reference_geometry.json",
    )
    logger.info("  Copied: skull_reference_geometry.json")

    # Copy skull and spine topology (unchanged)
    skull_and_spine_topology_json = skull_solver_output_dir / "skull_and_spine_topology.json"
    shutil.copy(
        skull_and_spine_topology_json,
        resampled_data_output_dir / "skull_kinematics" / "skull_and_spine_topology.json",
    )
    logger.info("  Copied: skull_and_spine_topology.json")

    # Save skull and spine trajectories (with zeroed timestamps)
    save_skull_and_spine_trajectories_csv(
        trajectories=resampled_skull_and_spine_trajectories,
        marker_names=skull_and_spine_marker_names,
        timestamps=common_timestamps,
        output_path=resampled_data_output_dir / "skull_and_spine_trajectories_resampled.csv",
    )

    # Save eye kinematics
    resampled_left_eye_kinematics.save_to_disk(resampled_data_output_dir / "left_eye_kinematics")
    logger.info("  Saved: left_eye_kinematics/")

    resampled_right_eye_kinematics.save_to_disk(resampled_data_output_dir / "right_eye_kinematics")
    logger.info("  Saved: right_eye_kinematics/")

    # =========================================================================
    # SAVE RESAMPLED EYE TRAJECTORIES
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("SAVING RESAMPLED EYE TRAJECTORIES")
    logger.info("=" * 40)

    # Extract and save left eye trajectories
    left_eye_trajectories, left_eye_marker_names = extract_eye_trajectories(resampled_left_eye_kinematics)
    save_eye_trajectories_csv(
        trajectories=left_eye_trajectories,
        marker_names=left_eye_marker_names,
        timestamps=common_timestamps,
        output_path=resampled_data_output_dir / "left_eye_kinematics" / "left_eye_trajectories_resampled.csv",
    )

    # Extract and save right eye trajectories
    right_eye_trajectories, right_eye_marker_names = extract_eye_trajectories(resampled_right_eye_kinematics)
    save_eye_trajectories_csv(
        trajectories=right_eye_trajectories,
        marker_names=right_eye_marker_names,
        timestamps=common_timestamps,
        output_path=resampled_data_output_dir / "right_eye_kinematics" / "right_eye_trajectories_resampled.csv",
    )

    # Create and save eye topologies
    left_eye_topology = create_eye_topology("left_eye")
    left_eye_topology.save_json(resampled_data_output_dir / "left_eye_kinematics" / "left_eye_topology.json")
    logger.info("  Saved: left_eye_topology.json")

    right_eye_topology = create_eye_topology("right_eye")
    right_eye_topology.save_json(resampled_data_output_dir / "right_eye_kinematics" / "right_eye_topology.json")
    logger.info("  Saved: right_eye_topology.json")

    # =========================================================================
    # SAVE RESAMPLED TOY TRAJECTORIES
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("SAVING RESAMPLED TOY TRAJECTORIES")
    logger.info("=" * 40)

    save_toy_trajectories_csv(
        trajectories=resampled_toy_trajectories,
        marker_names=toy_marker_names,
        timestamps=common_timestamps,
        output_path=resampled_data_output_dir / "toy_trajectories_resampled.csv",
    )

    toy_topology = create_toy_topology()
    toy_topology.save_json(resampled_data_output_dir / "toy_topology.json")
    logger.info("  Saved: toy_topology.json")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RESAMPLING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {resampled_data_output_dir}")
    logger.info(f"All files have exactly {n_frames} frames")
    logger.info(f"Timestamp range: {common_timestamps[0]:.4f}s to {common_timestamps[-1]:.4f}s")
    logger.info(f"Framerate: {resampled_skull_kinematics.framerate_hz:.2f} Hz")
    if video_configs:
        logger.info(f"Display videos saved to: {resampled_data_output_dir.parent / 'display_videos'}")

    return common_timestamps


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Example paths based on the visualization scripts
    _skull_solver_output_dir = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output"
    )
    _eye_kinematics_dir = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye_kinematics"
    )
    _resampled_data_output_dir = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\analyzable_output"
    )

    # Video configurations (matching blender script format)
    _video_configs = [
        VideoConfig(
            path=Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\annotated_videos\24676894_synchronized_corrected_clipped_3377_8754.mp4"),
            timestamps_path=Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\synchronized_videos\24676894_synchronized_corrected_synchronized_timestamps_utc_clipped_3377_8754.npy"),
            name="top_down_mocap",
        ),
        VideoConfig(
            path=Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\right_eye_stabilized.mp4"),
            timestamps_path=Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_videos\eye0_timestamps_utc_clipped_4354_11523.npy"),
            name="left_eye",
        ),
        VideoConfig(
            path=Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\left_eye_stabilized.mp4"),
            timestamps_path=Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"),
            name="right_eye",
        ),
    ]

    # Toy trajectories (DLC format - same timestamps as mocap)
    # For extended format with multiple trajectory types, specify which one:
    _toy_trajectories_csv = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\dlc\toy_body_3d_xyz.csv"
    )

    # Check available trajectory types before running
    _available_types = get_available_trajectory_types(_toy_trajectories_csv)
    if _available_types:
        print(f"Available trajectory types in toy CSV: {_available_types}")
        # Choose one - e.g., 'rigid_3d_xyz' for rigidified trajectories
        _toy_trajectory_type: str | None = "rigid_3d_xyz"
    else:
        _toy_trajectory_type = None

    resample_ferret_data(
        skull_solver_output_dir=_skull_solver_output_dir,
        eye_kinematics_dir=_eye_kinematics_dir,
        resampled_data_output_dir=_resampled_data_output_dir,
        resampling_strategy=ResamplingStrategy.FASTEST,
        video_configs=_video_configs,
        toy_trajectories_csv=_toy_trajectories_csv,
        toy_trajectory_type=_toy_trajectory_type,
    )