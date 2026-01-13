"""Save optimization results to various formats."""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def save_simple_csv(
        *,
        filepath: Path,
        data: np.ndarray,
        marker_names: list[str]
) -> None:
    """
    Save trajectory data in simple format (for input files).

    Creates CSV with columns: frame, {marker}_x, {marker}_y, {marker}_z
    No prefixes like "original_" or "optimized_".

    Args:
        filepath: Output CSV path
        data: (n_frames, n_markers, 3) trajectory data
        marker_names: List of marker names
    """
    n_frames, n_markers, _ = data.shape

    csv_data: dict[str, np.ndarray | range] = {"frame": range(n_frames)}

    for idx, marker_name in enumerate(marker_names):
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            csv_data[f"{marker_name}_{coord_name}"] = data[:, idx, coord_idx]

    df = pd.DataFrame(data=csv_data)
    df.to_csv(path_or_buf=filepath, index=False)

    logger.info(f"Saved simple CSV: {filepath} ({len(df)} frames)")


def save_trajectory_csv(
        *,
        filepath: Path,
        original_data: np.ndarray,
        optimized_data: np.ndarray,
        marker_names: list[str],
        ground_truth_data: np.ndarray | None = None
) -> None:
    """
    Save trajectory data for visualization.

    Creates CSV with columns:
        frame, original_{marker}_x/y/z, optimized_{marker}_x/y/z, [gt_{marker}_x/y/z]

    Args:
        filepath: Output CSV path
        original_data: (n_frames, n_markers, 3) original measurements
        optimized_data: (n_frames, n_markers, 3) optimized trajectory
        marker_names: List of marker names
        ground_truth_data: Optional (n_frames, n_markers, 3) ground truth
    """
    n_frames, n_markers, _ = original_data.shape

    data: dict[str, np.ndarray | range] = {"frame": range(n_frames)}

    # Add original data
    for idx, marker_name in enumerate(marker_names):
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            data[f"original_{marker_name}_{coord_name}"] = original_data[:, idx, coord_idx]

    # Add optimized data
    for idx, marker_name in enumerate(marker_names):
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            data[f"optimized_{marker_name}_{coord_name}"] = optimized_data[:, idx, coord_idx]

    # Add ground truth if provided
    if ground_truth_data is not None:
        for idx, marker_name in enumerate(marker_names):
            for coord_idx, coord_name in enumerate(["x", "y", "z"]):
                data[f"gt_{marker_name}_{coord_name}"] = ground_truth_data[:, idx, coord_idx]

    # Add centroids
    original_center = np.mean(original_data, axis=1)
    optimized_center = np.mean(optimized_data, axis=1)

    for coord_idx, coord_name in enumerate(["x", "y", "z"]):
        data[f"original_center_{coord_name}"] = original_center[:, coord_idx]
        data[f"optimized_center_{coord_name}"] = optimized_center[:, coord_idx]

    if ground_truth_data is not None:
        gt_center = np.mean(ground_truth_data, axis=1)
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            data[f"gt_center_{coord_name}"] = gt_center[:, coord_idx]

    # Save
    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=filepath, index=False)

    logger.info(f"Saved trajectory CSV: {filepath} ({len(df)} frames)")


def save_tidy_trajectory_csv(
        *,
        filepath: Path,
        original_data: np.ndarray,
        optimized_data: np.ndarray,
        marker_names: list[str],
        origin_markers: list[str],
        timestamps: np.ndarray,
) -> None:
    """
    Save trajectory data for visualization.

    Creates CSV with columns:
        frame, marker, data_type, x/y/z

    Args:
        filepath: Output CSV path
        original_data: (n_frames, n_markers, 3) original measurements
        optimized_data: (n_frames, n_markers, 3) optimized trajectory
        marker_names: List of marker names
    """
    n_frames, n_markers, _ = original_data.shape
    df_list: list[pd.DataFrame] = []

    # Add original data
    for idx, marker_name in enumerate(marker_names):
        df = pd.DataFrame()
        df["frame"] = range(n_frames)
        df["timestamp"] = timestamps
        df["marker"] = marker_name
        df["data_type"] = "original"
        df["x"] = original_data[:, idx, 0]
        df["y"] = original_data[:, idx, 1]
        df["z"] = original_data[:, idx, 2]
        df_list.append(df)

    # Add optimized data
    for idx, marker_name in enumerate(marker_names):
        df = pd.DataFrame()
        df["frame"] = range(n_frames)
        df["timestamp"] = timestamps
        df["marker"] = marker_name
        df["data_type"] = "optimized"
        df["x"] = optimized_data[:, idx, 0]
        df["y"] = optimized_data[:, idx, 1]
        df["z"] = optimized_data[:, idx, 2]
        df_list.append(df)

    # Add local origin trajectory
    origin_indices = [marker_names.index(name) for name in origin_markers if name in marker_names]
    original_origin = np.mean(original_data[:, origin_indices, :], axis=1)
    optimized_origin = np.mean(optimized_data[:, origin_indices, :], axis=1)

    original_origin_df = pd.DataFrame()
    original_origin_df["frame"] = range(n_frames)
    original_origin_df["timestamp"] = timestamps
    original_origin_df["marker"] = "local_origin"
    original_origin_df["data_type"] = "original"
    original_origin_df["x"] = original_origin[:, 0]
    original_origin_df["y"] = original_origin[:, 1]
    original_origin_df["z"] = original_origin[:, 2]
    df_list.append(original_origin_df)

    optimized_origin_df = pd.DataFrame()
    optimized_origin_df["frame"] = range(n_frames)
    optimized_origin_df["timestamp"] = timestamps
    optimized_origin_df["marker"] = "local_origin"
    optimized_origin_df["data_type"] = "optimized"
    optimized_origin_df["x"] = optimized_origin[:, 0]
    optimized_origin_df["y"] = optimized_origin[:, 1]
    optimized_origin_df["z"] = optimized_origin[:, 2]
    df_list.append(optimized_origin_df)

    # Save
    full_df = pd.concat(df_list).sort_values(["frame", "marker", "data_type"])
    full_df.to_csv(path_or_buf=filepath, index=False)

    logger.info(f"Saved tidy trajectory CSV: {filepath} ({len(full_df)} frames)")

    plot_origin_trajectories(df=full_df)

    print('ddd')


def plot_origin_trajectories(df: pd.DataFrame) -> None:
    """Plot original vs optimized local_origin trajectory as stacked x/y/z timeseries."""

    markers_to_plot = ["local_origin", "nose"]

    # Style config: (color, linestyle, marker)
    styles = {
        "original": { "linestyle": "-", "marker": "o", "markerfacecolor": "none", "markersize": 2},
        "optimized": { "linestyle": "-", "marker": ".", "markersize": 2},
    }

    # Extract data for each marker/data_type combination
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for marker in markers_to_plot:
        data[marker] = {}
        for data_type in ["original", "optimized"]:
            subset = df[(df["marker"] == marker) & (df["data_type"] == data_type)].sort_values("frame")
            if len(subset) == 0:
                raise ValueError(f"No '{marker}' marker with data_type='{data_type}' found in DataFrame")
            data[marker][data_type] = subset

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)

    for idx, coord in enumerate(["x", "y", "z"]):
        ax = axes[idx]

        for marker in markers_to_plot:
            for data_type, style in styles.items():
                subset = data[marker][data_type]
                label = f"{marker} ({data_type})"
                ax.plot(
                    subset["timestamp"].values,
                    subset[coord].values,
                    label=label,
                    **style,
                )

        ax.set_ylabel(coord)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("timestamp")
    fig.suptitle("Local Origin Trajectory: Original vs Optimized")
    plt.tight_layout()
    plt.show()

def save_rigid_body_pose_csv(
        *,
        filepath: Path,
        rotation_data: np.ndarray,
        quaternion_data: np.ndarray,
        translation_data: np.ndarray,
        timestamps: np.ndarray
):
    """
    Save rotation and translation data for visualization.

    Creates CSV with columns:
        frame, rotation_r{0-2}_c{0-2}, quaternion_w/x/y/z, translation_x/y/z
    """
    n_frames = rotation_data.shape[0]

    data = {
        "frame": range(n_frames),
        "timestamp": timestamps,
        "object": ["ferret_head"] * n_frames,
        "rotation_r0_c0": rotation_data[:, 0, 0],
        "rotation_r0_c1": rotation_data[:, 0, 1],
        "rotation_r0_c2": rotation_data[:, 0, 2],
        "rotation_r1_c0": rotation_data[:, 1, 0],
        "rotation_r1_c1": rotation_data[:, 1, 1],
        "rotation_r1_c2": rotation_data[:, 1, 2],
        "rotation_r2_c0": rotation_data[:, 2, 0],
        "rotation_r2_c1": rotation_data[:, 2, 1],
        "rotation_r2_c2": rotation_data[:, 2, 2],
        "translation_x": translation_data[:, 0],
        "translation_y": translation_data[:, 1],
        "translation_z": translation_data[:, 2],
        "quaternion_w": quaternion_data[:, 0],
        "quaternion_x": quaternion_data[:, 1],
        "quaternion_y": quaternion_data[:, 2],
        "quaternion_z": quaternion_data[:, 3],
    }

    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=filepath, index=False)

    logger.info(f"Saved rotation and translation CSV: {filepath} ({len(df)} frames)")


def save_topology_json(
        *,
        filepath: Path,
        topology_dict: dict[str, object],
        marker_names: list[str],
        n_frames: int,
) -> None:
    """
    Save topology metadata for viewer.

    Args:
        filepath: Output JSON path
        topology_dict: Topology dictionary from RigidBodyTopology.to_dict()
        marker_names: List of marker names
        n_frames: Number of frames
    """
    # Add soft_edges to topology dict if provided

    metadata = {
        "topology": topology_dict,
        "marker_names": marker_names,
        "n_frames": n_frames,
        "n_markers": len(marker_names),
    }

    with open(filepath, "w") as f:
        json.dump(obj=metadata, fp=f, indent=2)

    logger.info(f"Saved topology JSON: {filepath}")


def save_results(
        *,
        output_dir: Path,
        original_trajectories: np.ndarray,
        optimized_trajectories: np.ndarray,
        origin_markers_names:list[str],
        marker_names: list[str],
        rigid_body_name: str,
        topology_dict: dict[str, object],
        rotations: np.ndarray,
        quaternions: np.ndarray,
        translations: np.ndarray,
        timestamps: np.ndarray,
        viewer_source: Path | None = None
) -> None:
    """
    Save complete optimization results with viewer.

    Creates:
        output_dir/
            trajectory_data.csv
            topology.json

    Args:
        output_dir: Directory to save results
        original_trajectories: (n_frames, n_markers, 3)
        optimized_trajectories: (n_frames, n_markers, 3)
        marker_names: List of marker names
        topology_dict: Topology dictionary
        viewer_source: Path to viewer HTML file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = original_trajectories.shape[0]

    print(f"original data shape: {original_trajectories.shape}")
    print(f"optimized data shape: {optimized_trajectories.shape}")



    save_tidy_trajectory_csv(
        filepath=output_dir / f"{rigid_body_name}_trajectory_data.csv",
        original_data=original_trajectories,
        optimized_data=optimized_trajectories,
        marker_names=marker_names,
        origin_markers=origin_markers_names,
        timestamps=timestamps,
    )

    save_rigid_body_pose_csv(
        filepath=output_dir / f"{rigid_body_name}_pose_data.csv",
        rotation_data=rotations,
        quaternion_data=quaternions,
        translation_data=translations,
        timestamps=timestamps
    )

    save_topology_json(
        filepath=output_dir / f"{rigid_body_name}_topology.json",
        topology_dict=topology_dict,
        marker_names=marker_names,
        n_frames=n_frames,
    )

    logger.info(f"\n✓ Results saved to: {output_dir}")

