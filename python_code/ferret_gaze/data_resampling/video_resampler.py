"""
Video Resampler
===============

Resamples videos to match common timestamps from resampled kinematics data.
Each output video frame shows the closest frame from the original video,
with the original frame number drawn as a label.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for a video to be resampled."""

    path: Path
    timestamps_path: Path
    name: str


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
    # (unit_name, conversion_to_seconds, min_expected, max_expected)
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
        # Try to infer from magnitude
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

        logger.warning(
            f"Could not definitively detect timestamp unit from frame duration {median_duration:.2e}. "
            f"Assuming {detected_unit}."
        )

    logger.info(f"  Detected timestamp unit: {detected_unit} (median frame duration: {median_duration:.2e})")

    converted = timestamps * conversion_factor
    duration_s = converted[-1] - converted[0]
    effective_fps = (len(converted) - 1) / duration_s if duration_s > 0 else 0

    logger.info(f"  Converted duration: {duration_s:.2f}s, effective FPS: {effective_fps:.1f}")

    return converted


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
    font_scale: float = 1.0,
    thickness: int = 2,
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
                logger.warning(
                    f"  Failed to read frame {original_frame_idx}, using black frame"
                )
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # Cache the frame
            if len(frame_cache) >= cache_size_limit:
                # Remove oldest entry (simple approach)
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
    target_fps: float = 90.0,
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
        target_fps: FPS for output videos (should match the effective framerate
            of the common timestamps)
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
    target_fps: float = 90.0,
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


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Example video configs (matching the blender script format)
    VIDEOS: list[dict[str, str]] = [
        {
            "path": r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\annotated_videos\24676894_synchronized_corrected_clipped_3377_8754.mp4",
            "timestamps_path": r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\synchronized_videos\24676894_synchronized_corrected_synchronized_timestamps_utc_clipped_3377_8754.npy",
            "name": "top_down_mocap",
        },
        {
            "path": r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\right_eye_stabilized.mp4",
            "timestamps_path": r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_videos\eye0_timestamps_utc_clipped_4354_11523.npy",
            "name": "left_eye",
        },
        {
            "path": r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\left_eye_stabilized.mp4",
            "timestamps_path": r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy",
            "name": "right_eye",
        },
    ]

    # Load common timestamps from resampled data
    analyzable_output_dir = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\analyzable_output"
    )
    common_timestamps = np.load(analyzable_output_dir / "common_timestamps.npy")

    # The common_timestamps from the resampler are already zeroed.
    # We need the ORIGINAL timestamps for proper alignment with video timestamps.
    # In a real workflow, you would pass these from resample_ferret_data before zeroing.
    # For now, we'll demonstrate with zeroed timestamps (which means video timestamps
    # would also need to be zeroed/aligned the same way).

    # Output directory parallel to analyzable_output
    display_videos_dir = analyzable_output_dir.parent / "display_videos"

    # Resample videos
    output_paths = resample_videos_from_dicts(
        video_dicts=VIDEOS,
        common_timestamps_original=common_timestamps,
        output_dir=display_videos_dir,
        target_fps=90.0,
    )

    print(f"\nCreated {len(output_paths)} resampled videos:")
    for p in output_paths:
        print(f"  {p}")