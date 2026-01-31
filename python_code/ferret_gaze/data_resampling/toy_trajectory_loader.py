"""
Toy Trajectory Loader
=====================

Loads toy trajectory data from CSV format:
    frame,keypoint,x,y,z
    0,toy_face,-8.03,-38.96,538.42
    0,toy_top,-15.06,-47.38,509.89
    ...

Returns data in the same format as load_skull_and_spine_trajectories:
    (trajectories_array, keypoint_names)
where trajectories_array has shape (n_frames, n_keypoints, 3).
"""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_toy_trajectories(csv_path: Path) -> tuple[NDArray[np.float64], list[str]]:
    """
    Load toy trajectories from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (trajectories_array, keypoint_names)
        - trajectories_array: (n_frames, n_keypoints, 3) array
        - keypoint_names: list of keypoint names in order
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Toy trajectory CSV not found: {csv_path}")

    data: dict[str, dict[int, tuple[float, float, float]]] = {}
    frames_seen: set[int] = set()

    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")

        required_cols = {"frame", "keypoint", "x", "y", "z"}
        missing = required_cols - set(header)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        frame_idx = header.index("frame")
        keypoint_idx = header.index("keypoint")
        x_idx = header.index("x")
        y_idx = header.index("y")
        z_idx = header.index("z")

        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != len(header):
                raise ValueError(
                    f"Line {line_num}: expected {len(header)} columns, got {len(parts)}"
                )

            frame = int(parts[frame_idx])
            keypoint = parts[keypoint_idx]
            x = float(parts[x_idx])
            y = float(parts[y_idx])
            z = float(parts[z_idx])

            frames_seen.add(frame)

            if keypoint not in data:
                data[keypoint] = {}

            if frame in data[keypoint]:
                raise ValueError(
                    f"Line {line_num}: duplicate entry for keypoint '{keypoint}' at frame {frame}"
                )

            data[keypoint][frame] = (x, y, z)

    if not data:
        raise ValueError(f"CSV file is empty: {csv_path}")

    min_frame = min(frames_seen)
    max_frame = max(frames_seen)
    n_frames = max_frame - min_frame + 1

    # Check contiguous
    if len(frames_seen) != n_frames:
        missing_frames = set(range(min_frame, max_frame + 1)) - frames_seen
        raise ValueError(f"Missing frames: {sorted(missing_frames)[:10]}...")

    # Build output in array format (n_frames, n_keypoints, 3)
    keypoint_names = list(data.keys())
    n_keypoints = len(keypoint_names)
    trajectories = np.zeros((n_frames, n_keypoints, 3), dtype=np.float64)

    for keypoint_idx, keypoint_name in enumerate(keypoint_names):
        keypoint_data = data[keypoint_name]
        for frame, xyz in keypoint_data.items():
            trajectories[frame - min_frame, keypoint_idx, :] = xyz

    return trajectories, keypoint_names