"""Save optimization results to various formats."""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
import shutil

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
    No prefixes like "noisy_" or "optimized_".

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
    noisy_data: np.ndarray,
    optimized_data: np.ndarray,
    marker_names: list[str],
    ground_truth_data: np.ndarray | None = None
) -> None:
    """
    Save trajectory data for visualization.

    Creates CSV with columns:
        frame, noisy_{marker}_x/y/z, optimized_{marker}_x/y/z, [gt_{marker}_x/y/z]

    Args:
        filepath: Output CSV path
        noisy_data: (n_frames, n_markers, 3) noisy measurements
        optimized_data: (n_frames, n_markers, 3) optimized trajectory
        marker_names: List of marker names
        ground_truth_data: Optional (n_frames, n_markers, 3) ground truth
    """
    n_frames, n_markers, _ = noisy_data.shape

    data: dict[str, np.ndarray | range] = {"frame": range(n_frames)}

    # Add noisy data
    for idx, marker_name in enumerate(marker_names):
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            data[f"noisy_{marker_name}_{coord_name}"] = noisy_data[:, idx, coord_idx]

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
    noisy_center = np.mean(noisy_data, axis=1)
    optimized_center = np.mean(optimized_data, axis=1)

    for coord_idx, coord_name in enumerate(["x", "y", "z"]):
        data[f"noisy_center_{coord_name}"] = noisy_center[:, coord_idx]
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
    noisy_data: np.ndarray,
    optimized_data: np.ndarray,
    marker_names: list[str],
    timestamps: np.ndarray,
    ground_truth_data: np.ndarray | None = None
) -> None:
    """
    Save trajectory data for visualization.

    Creates CSV with columns:
        frame, marker, data_type, x/y/z

    Args:
        filepath: Output CSV path
        noisy_data: (n_frames, n_markers, 3) noisy measurements
        optimized_data: (n_frames, n_markers, 3) optimized trajectory
        marker_names: List of marker names
        ground_truth_data: Optional (n_frames, n_markers, 3) ground truth
    """
    n_frames, n_markers, _ = noisy_data.shape
    df_list: list[pd.DataFrame] = []

    # Add noisy data
    for idx, marker_name in enumerate(marker_names):
        df = pd.DataFrame()
        df["frame"] = range(n_frames)
        df["timestamp"] = timestamps
        df["marker"] = marker_name
        df["data_type"] = "noisy"
        df["x"] = noisy_data[:, idx, 0]
        df["y"] = noisy_data[:, idx, 1]
        df["z"] = noisy_data[:, idx, 2]
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

    # Add ground truth if provided
    if ground_truth_data is not None:
        for idx, marker_name in enumerate(marker_names):
            df = pd.DataFrame()
            df["frame"] = range(n_frames)
            df["timestamp"] = timestamps
            df["marker"] = marker_name
            df["data_type"] = "ground_truth"
            df["x"] = ground_truth_data[:, idx, 0]
            df["y"] = ground_truth_data[:, idx, 1]
            df["z"] = ground_truth_data[:, idx, 2]
            df_list.append(df)

    # Add centroids
    noisy_center = np.mean(noisy_data, axis=1)
    optimized_center = np.mean(optimized_data, axis=1)

    df = pd.DataFrame()
    df["frame"] = range(n_frames)
    df["timestamp"] = timestamps
    df["marker"] = "center"
    df["data_type"] = "noisy"
    df["x"] = noisy_center[:, 0]
    df["y"] = noisy_center[:, 1]
    df["z"] = noisy_center[:, 2]
    df_list.append(df)

    df = pd.DataFrame()
    df["frame"] = range(n_frames)
    df["timestamp"] = timestamps
    df["marker"] = marker_name
    df["data_type"] = "optimized"
    df["x"] = optimized_center[:, 0]
    df["y"] = optimized_center[:, 1]
    df["z"] = optimized_center[:, 2]
    df_list.append(df)

    if ground_truth_data is not None:
        gt_center = np.mean(ground_truth_data, axis=1)
        df = pd.DataFrame()
        df["frame"] = range(n_frames)
        df["timestamp"] = timestamps
        df["marker"] = marker_name
        df["data_type"] = "ground_truth"
        df["x"] = gt_center[:, 0]
        df["y"] = gt_center[:, 1]
        df["z"] = gt_center[:, 2]
        df_list.append(df)

    # Save
    tidy_df = pd.concat(df_list).sort_values(["frame", "marker", "data_type"])
    tidy_df.to_csv(path_or_buf=filepath, index=False)

    logger.info(f"Saved tidy trajectory CSV: {filepath} ({len(df)} frames)")

def save_rotation_translation_csv(
    *,
    filepath: Path,
    rotation_data: np.ndarray,
    translation_data: np.ndarray,
    timestamps: np.ndarray
):
    """
    Save rotation and translation data for visualization.

    Creates CSV with columns:
        frame, rotation_r{0-2}_c{0-2}, translation_x/y/z
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
    has_ground_truth: bool = False,
    soft_edges: list[tuple[int, int]] | None = None
) -> None:
    """
    Save topology metadata for viewer.

    Args:
        filepath: Output JSON path
        topology_dict: Topology dictionary from RigidBodyTopology.to_dict()
        marker_names: List of marker names
        n_frames: Number of frames
        has_ground_truth: Whether ground truth data is included
        soft_edges: Optional list of soft edges to include
    """
    # Add soft_edges to topology dict if provided
    topology_with_soft = topology_dict.copy()
    if soft_edges is not None:
        topology_with_soft["soft_edges"] = soft_edges

    metadata = {
        "topology": topology_with_soft,
        "marker_names": marker_names,
        "n_frames": n_frames,
        "n_markers": len(marker_names),
        "has_ground_truth": has_ground_truth,
    }

    with open(filepath, "w") as f:
        json.dump(obj=metadata, fp=f, indent=2)

    logger.info(f"Saved topology JSON: {filepath}")


def save_results(
    *,
    output_dir: Path,
    noisy_data: np.ndarray,
    optimized_data: np.ndarray,
    marker_names: list[str],
    topology_dict: dict[str, object],
    rotations: np.ndarray,
    translations: np.ndarray,
    timestamps: np.ndarray,
    ground_truth_data: np.ndarray | None = None,
    soft_edges: list[tuple[int, int]] | None = None,
    copy_viewer: bool = True,
    viewer_source: Path | None = None
) -> None:
    """
    Save complete optimization results with viewer.

    Creates:
        output_dir/
            trajectory_data.csv
            topology.json
            rigid_body_viewer.html  (if copy_viewer=True)

    Args:
        output_dir: Directory to save results
        noisy_data: (n_frames, n_markers, 3)
        optimized_data: (n_frames, n_markers, 3)
        marker_names: List of marker names
        topology_dict: Topology dictionary
        ground_truth_data: Optional ground truth
        soft_edges: Optional list of soft edges
        copy_viewer: Whether to copy viewer HTML
        viewer_source: Path to viewer HTML file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = noisy_data.shape[0]

    # Save trajectory data
    save_trajectory_csv(
        filepath=output_dir / "trajectory_data.csv",
        noisy_data=noisy_data,
        optimized_data=optimized_data,
        marker_names=marker_names,
        ground_truth_data=ground_truth_data
    )

    save_tidy_trajectory_csv(
        filepath=output_dir / "tidy_trajectory_data.csv",
        noisy_data=noisy_data,
        optimized_data=optimized_data,
        marker_names=marker_names, 
        timestamps=timestamps,
    )

    # TODO: convert this to quaternions at some point
    save_rotation_translation_csv(
        filepath=output_dir / "rotation_translation_data.csv",
        rotation_data=rotations,
        translation_data=translations,
        timestamps=timestamps
    )

    # Save topology with soft edges
    save_topology_json(
        filepath=output_dir / "topology.json",
        topology_dict=topology_dict,
        marker_names=marker_names,
        n_frames=n_frames,
        has_ground_truth=ground_truth_data is not None,
        soft_edges=soft_edges
    )

    # Copy viewer
    if copy_viewer:
        if viewer_source is None:
            # Try to find viewer in common locations
            possible_locations = [
                Path(__file__).parent.parent / "viewer" / "rigid_body_viewer.html",
                Path("viewer/rigid_body_viewer.html"),
                Path("rigid_body_viewer.html"),
            ]

            for location in possible_locations:
                if location.exists():
                    viewer_source = location
                    break

        if viewer_source and viewer_source.exists():
            shutil.copy(
                src=viewer_source,
                dst=output_dir / "rigid_body_viewer.html"
            )
            logger.info(f"Copied viewer: {output_dir / 'rigid_body_viewer.html'}")
        else:
            logger.warning("Viewer HTML not found, skipping copy")

    logger.info(f"\n✓ Results saved to: {output_dir}")
    logger.info(f"  Open {output_dir / 'rigid_body_viewer.html'} to visualize")


def save_evaluation_report(
    *,
    filepath: Path,
    metrics: dict[str, float],
    config: dict[str, object]
) -> None:
    """
    Save evaluation metrics and configuration to JSON.

    Args:
        filepath: Output JSON path
        metrics: Dictionary of evaluation metrics
        config: Configuration dictionary
    """
    report = {
        "metrics": metrics,
        "config": config,
    }

    with open(filepath, "w") as f:
        json.dump(obj=report, fp=f, indent=2)

    logger.info(f"Saved evaluation report: {filepath}")


def print_summary(
    *,
    noisy_data: np.ndarray,
    optimized_data: np.ndarray,
    ground_truth_data: np.ndarray | None = None
) -> None:
    """
    Print summary statistics to console.

    Args:
        noisy_data: (n_frames, n_markers, 3)
        optimized_data: (n_frames, n_markers, 3)
        ground_truth_data: Optional ground truth
    """
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    if ground_truth_data is not None:
        noisy_errors = np.linalg.norm(noisy_data - ground_truth_data, axis=2)
        opt_errors = np.linalg.norm(optimized_data - ground_truth_data, axis=2)

        logger.info("\nReconstruction accuracy (vs ground truth):")
        logger.info(f"  Noisy:     mean={np.mean(noisy_errors)*1000:.2f}mm, max={np.max(noisy_errors)*1000:.2f}mm")
        logger.info(f"  Optimized: mean={np.mean(opt_errors)*1000:.2f}mm, max={np.max(opt_errors)*1000:.2f}mm")

        improvement = (np.mean(noisy_errors) - np.mean(opt_errors)) / np.mean(noisy_errors) * 100
        logger.info(f"  Improvement: {improvement:.1f}%")

    # Compute edge length consistency
    n_frames, n_markers, _ = noisy_data.shape

    if n_markers >= 2:
        # Check first edge as example
        noisy_dists = np.linalg.norm(noisy_data[:, 0, :] - noisy_data[:, 1, :], axis=1)
        opt_dists = np.linalg.norm(optimized_data[:, 0, :] - optimized_data[:, 1, :], axis=1)

        logger.info(f"\nEdge length consistency (marker 0-1):")
        logger.info(f"  Noisy:     {np.mean(noisy_dists):.4f}m ± {np.std(noisy_dists)*1000:.2f}mm")
        logger.info(f"  Optimized: {np.mean(opt_dists):.4f}m ± {np.std(opt_dists)*1000:.2f}mm")
