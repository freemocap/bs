"""
Ferret Data Resampler
=====================

Loads skull kinematics, left/right eye kinematics, and trajectory data,
resamples all to common timestamps, and saves the results.

All output files will have EXACTLY the same number of frames and identical timestamps.
"""
import json
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from python_code.ferret_gaze.data_resampling.data_resampling_helpers import ResamplingStrategy, \
    resample_to_common_timestamps
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics

logger = logging.getLogger(__name__)



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


def resample_ferret_data(
    skull_solver_output_dir: Path,
    eye_kinematics_dir: Path,
    resampled_data_output_dir: Path,
    resampling_strategy: ResamplingStrategy = ResamplingStrategy.FASTEST,
) -> NDArray[np.float64]:
    """
    Load, resample, and save all ferret tracking data to common timestamps.

    Args:
        skull_solver_output_dir: Directory containing skull_kinematics.csv,
            skull_reference_geometry.json, and skull_and_spine_trajectories.csv
        eye_kinematics_dir: Directory containing left_eye and right_eye
            kinematics subdirectories
        resampled_data_output_dir: Directory to save all resampled data
        resampling_strategy: Strategy for selecting target framerate

    Returns:
        The common timestamps array used for all resampled data
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    logger.info("=" * 80)
    logger.info("FERRET DATA RESAMPLING PIPELINE")
    logger.info("=" * 80)

    resampled_data_output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD ALL DATA
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("LOADING DATA")
    logger.info("=" * 40)

    # Load skull kinematics
    skull_kinematics_csv = skull_solver_output_dir / "skull_kinematics.csv"
    skull_reference_geometry_json = skull_solver_output_dir / "skull_reference_geometry.json"
    skull_kinematics = RigidBodyKinematics.load_from_disk(
        kinematics_csv_path=skull_kinematics_csv,
        reference_geometry_json_path=skull_reference_geometry_json,
    )
    logger.info(f"  Skull kinematics: {skull_kinematics.n_frames} frames, {skull_kinematics.framerate_hz:.2f} Hz")

    # Load skull and spine trajectories
    skull_and_spine_csv = skull_solver_output_dir / "skull_and_spine_trajectories.csv"
    skull_and_spine_trajectories, skull_and_spine_marker_names, skull_and_spine_timestamps = (
        load_skull_and_spine_trajectories(skull_and_spine_csv)
    )
    logger.info(f"  Skull+spine trajectories: {skull_and_spine_trajectories.shape[0]} frames")

    # Load left eye kinematics
    left_eye_kinematics = FerretEyeKinematics.load_from_directory(
        eye_name="left_eye",
        input_directory=eye_kinematics_dir,
    )
    logger.info(f"  Left eye kinematics: {left_eye_kinematics.n_frames} frames, {left_eye_kinematics.framerate_hz:.2f} Hz")

    # Load right eye kinematics
    right_eye_kinematics = FerretEyeKinematics.load_from_directory(
        eye_name="right_eye",
        input_directory=eye_kinematics_dir,
    )
    logger.info(f"  Right eye kinematics: {right_eye_kinematics.n_frames} frames, {right_eye_kinematics.framerate_hz:.2f} Hz")

    # =========================================================================
    # RESAMPLE TO COMMON TIMESTAMPS
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("RESAMPLING TO COMMON TIMESTAMPS")
    logger.info("=" * 40)
    logger.info(f"  Strategy: {resampling_strategy.value}")

    # Use the resample_to_common_timestamps function
    # We need to pass in the kinematics and trajectories separately
    kinematics_list = [
        skull_kinematics,
        left_eye_kinematics.eyeball,
        right_eye_kinematics.eyeball,
    ]

    trajectories_list = [
        (skull_and_spine_trajectories, skull_and_spine_timestamps),
    ]

    resampled_kinematics, resampled_trajectories = resample_to_common_timestamps(
        kinematics_list=kinematics_list,
        trajectories=trajectories_list,
        strategy=resampling_strategy,
        zero_timestamps=True
    )

    # Extract resampled data
    resampled_skull_kinematics = resampled_kinematics[0]
    resampled_left_eye_kinematics_eyeball = resampled_kinematics[1]
    resampled_right_eye_kinematics_eyeball = resampled_kinematics[2]
    resampled_skull_and_spine_trajectories = resampled_trajectories[0]

    common_timestamps = resampled_skull_kinematics.timestamps

    logger.info(f"  Common timestamps: {len(common_timestamps)} frames")
    logger.info(f"  Time range: {common_timestamps[0]:.4f}s to {common_timestamps[-1]:.4f}s")
    logger.info(f"  Duration: {common_timestamps[-1] - common_timestamps[0]:.4f}s")
    logger.info(f"  Framerate: {resampled_skull_kinematics.framerate_hz:.2f} Hz")

    # Now resample the full eye kinematics (socket landmarks and tracked pupil too)
    resampled_left_eye_kinematics = left_eye_kinematics.resample(common_timestamps)
    resampled_right_eye_kinematics = right_eye_kinematics.resample(common_timestamps)

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

    # Save skull kinematics
    resampled_skull_kinematics.save_to_disk(
        output_directory=resampled_data_output_dir,
    )

    # Copy skull reference geometry (unchanged)
    shutil.copy(
        skull_reference_geometry_json,
        resampled_data_output_dir / "skull_reference_geometry.json",
    )
    logger.info(f"  Copied: skull_reference_geometry.json")

    # Copy skull and spine topology (unchanged)
    skull_and_spine_topology_json = skull_solver_output_dir / "skull_and_spine_topology.json"
    shutil.copy(
        skull_and_spine_topology_json,
        resampled_data_output_dir / "skull_and_spine_topology.json",
    )
    logger.info(f"  Copied: skull_and_spine_topology.json")

    # Save skull and spine trajectories
    save_skull_and_spine_trajectories_csv(
        trajectories=resampled_skull_and_spine_trajectories,
        marker_names=skull_and_spine_marker_names,
        timestamps=common_timestamps,
        output_path=resampled_data_output_dir / "skull_and_spine_trajectories_resampled.csv",
    )

    # Save eye kinematics
    resampled_left_eye_kinematics.save_to_disk(resampled_data_output_dir / "left_eye_kinematics")
    logger.info(f"  Saved: left_eye_kinematics/")

    resampled_right_eye_kinematics.save_to_disk(resampled_data_output_dir / "right_eye_kinematics")
    logger.info(f"  Saved: right_eye_kinematics/")

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

    return common_timestamps


if __name__ == "__main__":
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

    resample_ferret_data(
        skull_solver_output_dir=_skull_solver_output_dir,
        eye_kinematics_dir=_eye_kinematics_dir,
        resampled_data_output_dir=_resampled_data_output_dir,
        resampling_strategy=ResamplingStrategy.FASTEST,
    )