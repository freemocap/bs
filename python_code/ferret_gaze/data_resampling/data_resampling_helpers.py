"""
Resampling Utilities
====================

Functions for resampling RigidBodyKinematics and trajectory data to common timestamps.
"""

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics


# Type alias for trajectory data: either (frames, keypoints, dims) array
# or dict mapping keypoint names to (frames, dims) arrays
TrajectoryData = NDArray[np.float64] | dict[str, NDArray[np.float64]]


class ResamplingStrategy(Enum):
    """Strategy for selecting target framerate when resampling multiple data sources."""

    FASTEST = "fastest"
    """Use the highest framerate among all inputs."""

    SLOWEST = "slowest"
    """Use the lowest framerate among all inputs."""

    MEAN = "mean"
    """Use the mean framerate across all inputs."""

    MEDIAN = "median"
    """Use the median framerate across all inputs."""


def resample_to_common_timestamps(
    kinematics_list: list[RigidBodyKinematics],
    trajectories: list[tuple[TrajectoryData, NDArray[np.float64]]],
    strategy: ResamplingStrategy | NDArray[np.float64],
        zero_timestamps:bool=True
) -> tuple[list[RigidBodyKinematics], list[TrajectoryData]]:
    """
    Resample multiple RigidBodyKinematics and trajectory arrays to share identical timestamps.

    The time range is computed as the intersection of all input time ranges
    (latest start time to earliest end time) to ensure all data sources
    have valid data throughout.

    Args:
        kinematics_list: List of RigidBodyKinematics to resample. Can be empty if
            trajectories is non-empty.
        trajectories: List of (trajectory_data, timestamps) tuples. Each trajectory_data
            is either:
            - NDArray of shape (n_frames, n_keypoints, n_dims) where n_dims is 2 or 3
            - dict mapping keypoint names to NDArray of shape (n_frames, n_dims)
            The timestamps array must have shape (n_frames,) matching the trajectory.
            Can be empty if kinematics_list is non-empty.
        strategy: How to determine target timestamps. Either:
            - ResamplingStrategy.FASTEST: Use the highest framerate among all inputs
            - ResamplingStrategy.SLOWEST: Use the lowest framerate among all inputs
            - ResamplingStrategy.MEAN: Use the mean framerate across all inputs
            - ResamplingStrategy.MEDIAN: Use the median framerate across all inputs
            - NDArray[np.float64]: Explicit target timestamps (used directly)
        zero_timestamps: Whether to subtract the first frame so timestamps count up from zero

    Returns:
        Tuple of (resampled_kinematics, resampled_trajectories).
        All outputs have EXACTLY the same number of frames.
        Order matches the input lists.

    Raises:
        ValueError: If both kinematics_list and trajectories are empty
        ValueError: If time ranges don't overlap
        ValueError: If trajectory data has invalid shape
    """
    if len(kinematics_list) == 0 and len(trajectories) == 0:
        raise ValueError("At least one of kinematics_list or trajectories must be non-empty")

    # Validate trajectory shapes before computing timestamps
    for i, (traj_data, traj_timestamps) in enumerate(trajectories):
        _validate_trajectory_data(traj_data, traj_timestamps, index=i)

    # Compute or use provided target timestamps
    if isinstance(strategy, np.ndarray):
        target_timestamps = np.asarray(strategy, dtype=np.float64)
    else:
        target_timestamps = _compute_common_timestamps_with_trajectories(
            kinematics_list=kinematics_list,
            trajectories=trajectories,
            strategy=strategy,
        )


    # Resample all kinematics
    resampled_kinematics = [k.resample(target_timestamps=target_timestamps) for k in kinematics_list]

    # Resample all trajectories
    resampled_trajectories = [
        _resample_trajectory_data(traj_data, traj_timestamps, target_timestamps)
        for traj_data, traj_timestamps in trajectories
    ]

    # Verify all outputs have exactly the same number of frames
    n_target = len(target_timestamps)
    for i, k in enumerate(resampled_kinematics):
        if k.n_frames != n_target:
            raise RuntimeError(
                f"Frame count mismatch: kinematics[{i}] has {k.n_frames} frames, "
                f"expected {n_target}"
            )
    for i, traj in enumerate(resampled_trajectories):
        n_traj = _get_trajectory_n_frames(traj)
        if n_traj != n_target:
            raise RuntimeError(
                f"Frame count mismatch: trajectory[{i}] has {n_traj} frames, "
                f"expected {n_target}"
            )

    return resampled_kinematics, resampled_trajectories


def _validate_trajectory_data(
    traj_data: TrajectoryData,
    timestamps: NDArray[np.float64],
    index: int,
) -> None:
    """Validate trajectory data shape and timestamps alignment."""
    if timestamps.ndim != 1:
        raise ValueError(
            f"trajectory[{index}] timestamps must be 1D, got shape {timestamps.shape}"
        )

    n_frames = len(timestamps)

    if isinstance(traj_data, dict):
        for kp_name, kp_data in traj_data.items():
            if kp_data.ndim != 2:
                raise ValueError(
                    f"trajectory[{index}]['{kp_name}'] must be 2D (frames, dims), "
                    f"got shape {kp_data.shape}"
                )
            if kp_data.shape[0] != n_frames:
                raise ValueError(
                    f"trajectory[{index}]['{kp_name}'] has {kp_data.shape[0]} frames, "
                    f"but timestamps has {n_frames}"
                )
            if kp_data.shape[1] not in (2, 3):
                raise ValueError(
                    f"trajectory[{index}]['{kp_name}'] must have 2 or 3 dimensions, "
                    f"got {kp_data.shape[1]}"
                )
    else:
        if traj_data.ndim != 3:
            raise ValueError(
                f"trajectory[{index}] array must be 3D (frames, keypoints, dims), "
                f"got shape {traj_data.shape}"
            )
        if traj_data.shape[0] != n_frames:
            raise ValueError(
                f"trajectory[{index}] has {traj_data.shape[0]} frames, "
                f"but timestamps has {n_frames}"
            )
        if traj_data.shape[2] not in (2, 3):
            raise ValueError(
                f"trajectory[{index}] must have 2 or 3 dimensions (last axis), "
                f"got {traj_data.shape[2]}"
            )


def _get_trajectory_n_frames(traj_data: TrajectoryData) -> int:
    """Get number of frames from trajectory data."""
    if isinstance(traj_data, dict):
        first_key = next(iter(traj_data))
        return traj_data[first_key].shape[0]
    else:
        return traj_data.shape[0]


def _compute_framerate_from_timestamps(timestamps: NDArray[np.float64]) -> float:
    """Compute framerate from timestamps array."""
    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to compute framerate")
    return float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))


def _compute_common_timestamps_with_trajectories(
    kinematics_list: list[RigidBodyKinematics],
    trajectories: list[tuple[TrajectoryData, NDArray[np.float64]]],
    strategy: ResamplingStrategy,
) -> NDArray[np.float64]:
    """
    Compute common timestamps considering both kinematics and trajectory data.

    Args:
        kinematics_list: List of RigidBodyKinematics
        trajectories: List of (trajectory_data, timestamps) tuples
        strategy: ResamplingStrategy enum value

    Returns:
        Uniformly spaced timestamps covering the intersection of all time ranges
    """
    # Collect framerates from all sources
    framerates: list[float] = []
    for k in kinematics_list:
        framerates.append(k.framerate_hz)
    for _, timestamps in trajectories:
        framerates.append(_compute_framerate_from_timestamps(timestamps))

    framerates_array = np.array(framerates, dtype=np.float64)

    # Select target framerate based on strategy
    if strategy == ResamplingStrategy.FASTEST:
        target_framerate = float(np.max(framerates_array))
    elif strategy == ResamplingStrategy.SLOWEST:
        target_framerate = float(np.min(framerates_array))
    elif strategy == ResamplingStrategy.MEAN:
        target_framerate = float(np.mean(framerates_array))
    elif strategy == ResamplingStrategy.MEDIAN:
        target_framerate = float(np.median(framerates_array))

    # Collect time ranges from all sources
    start_times: list[float] = []
    end_times: list[float] = []

    for k in kinematics_list:
        start_times.append(float(k.timestamps[0]))
        end_times.append(float(k.timestamps[-1]))

    for _, timestamps in trajectories:
        start_times.append(float(timestamps[0]))
        end_times.append(float(timestamps[-1]))

    # Compute time range intersection
    start_time = max(start_times)
    end_time = min(end_times)

    if end_time <= start_time:
        raise ValueError(
            f"Time ranges do not overlap. "
            f"Latest start: {start_time:.4f}s, earliest end: {end_time:.4f}s"
        )

    # Generate uniform timestamps
    frame_duration = 1.0 / target_framerate
    n_frames = int(np.floor((end_time - start_time) / frame_duration)) + 1

    if n_frames < 2:
        raise ValueError(
            f"Time range too short for target framerate. "
            f"Duration: {end_time - start_time:.4f}s, framerate: {target_framerate:.2f}Hz"
        )

    return np.linspace(start_time, start_time + (n_frames - 1) * frame_duration, n_frames)


def _resample_trajectory_data(
    traj_data: TrajectoryData,
    original_timestamps: NDArray[np.float64],
    target_timestamps: NDArray[np.float64],
) -> TrajectoryData:
    """
    Resample trajectory data to target timestamps using linear interpolation.

    Args:
        traj_data: Either (n_frames, n_keypoints, n_dims) array or
            dict mapping keypoint names to (n_frames, n_dims) arrays
        original_timestamps: Original timestamps (n_frames,)
        target_timestamps: Target timestamps (n_target,)

    Returns:
        Resampled trajectory data in the same format as input
    """
    n_target = len(target_timestamps)

    if isinstance(traj_data, dict):
        resampled: dict[str, NDArray[np.float64]] = {}
        for kp_name, kp_data in traj_data.items():
            n_dims = kp_data.shape[1]
            resampled_kp = np.zeros((n_target, n_dims), dtype=np.float64)
            for dim in range(n_dims):
                resampled_kp[:, dim] = np.interp(
                    target_timestamps,
                    original_timestamps,
                    kp_data[:, dim],
                )
            resampled[kp_name] = resampled_kp
        return resampled
    else:
        n_keypoints = traj_data.shape[1]
        n_dims = traj_data.shape[2]
        resampled_array = np.zeros((n_target, n_keypoints, n_dims), dtype=np.float64)
        for kp in range(n_keypoints):
            for dim in range(n_dims):
                resampled_array[:, kp, dim] = np.interp(
                    target_timestamps,
                    original_timestamps,
                    traj_data[:, kp, dim],
                )
        return resampled_array