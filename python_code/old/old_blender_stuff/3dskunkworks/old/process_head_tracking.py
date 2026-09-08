"""Process head tracking data with PyCeres rigid body optimization."""

from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import json
import logging

from main_pyceres import (
    optimize_rigid_body_pyceres,
)
from python_code.blender_stuff.load_trajectories import load_trajectories_auto
from sba_utils import estimate_reference_geometry, compute_reference_distances

logger = logging.getLogger(__name__)


@dataclass
class RigidBodyTopology:
    """Define which keypoints form a rigid body and how they're connected."""
    
    keypoint_names: list[str]
    """Names of keypoints that belong to this rigid body"""
    
    rigid_edges: list[tuple[int, int]]
    """Pairs of keypoint indices that should maintain fixed distance"""
    
    display_edges: list[tuple[int, int]] | None = None
    """Edges to display in visualization (defaults to rigid_edges)"""
    
    name: str = "rigid_body"
    """Name for this rigid body"""
    
    def __post_init__(self) -> None:
        if self.display_edges is None:
            self.display_edges = self.rigid_edges.copy()
    
    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "keypoint_names": self.keypoint_names,
            "rigid_edges": self.rigid_edges,
            "display_edges": self.display_edges,
        }
    
    @classmethod
    def from_dict(cls, *, data: dict[str, object]) -> "RigidBodyTopology":
        """Create from dict."""
        return cls(
            name=str(data["name"]),
            keypoint_names=list(data["keypoint_names"]),
            rigid_edges=list(data["rigid_edges"]),
            display_edges=list(data.get("display_edges")),
        )


@dataclass
class HeadTrackingConfig:
    """Configuration for head tracking processing."""
    
    # Input/output paths
    input_csv: Path
    output_dir: Path
    
    # Rigid body topology
    topology: RigidBodyTopology
    
    # CSV loading options
    scale_factor: float = 0.001  # mm to meters
    z_value: float = 0.0
    likelihood_threshold: float | None = None
    
    # PyCeres optimization parameters
    max_iter: int = 300
    lambda_data: float = 100.0
    lambda_rigid: float = 100.0
    lambda_rot_smooth: float = 500.0
    lambda_trans_smooth: float = 200.0
    
    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HEAD MARKER TOPOLOGY EXAMPLES
# =============================================================================

def create_ferret_head_topology() -> RigidBodyTopology:
    """
    Example topology for ferret head tracking.
    
    Update keypoint_names and rigid_edges for your actual setup!
    """
    # TODO: Update these with your actual keypoint names
    keypoint_names = [
        "head_top",
        "head_left", 
        "head_right",
        "head_front",
        "head_back",
        # Add more keypoints here
    ]
    
    # Define which pairs should maintain rigid distance
    # Use indices corresponding to keypoint_names list above
    rigid_edges = [
        (0, 1),  # head_top to head_left
        (0, 2),  # head_top to head_right
        (0, 3),  # head_top to head_front
        (1, 2),  # head_left to head_right
        (1, 4),  # head_left to head_back
        (2, 4),  # head_right to head_back
        # Add more rigid connections
    ]
    
    # Optional: display edges (for visualization only, not constraints)
    # If None, will use rigid_edges
    display_edges = rigid_edges.copy()
    
    return RigidBodyTopology(
        keypoint_names=keypoint_names,
        rigid_edges=rigid_edges,
        display_edges=display_edges,
        name="ferret_head",
    )


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data_from_trajectories(
    *,
    trajectory_dict: dict[str, np.ndarray],
    topology: RigidBodyTopology,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract and order trajectories according to topology.
    
    Args:
        trajectory_dict: Maps keypoint names to (n_frames, 3) arrays
        topology: Defines which keypoints to use and in what order
        
    Returns:
        - original_data: (n_frames, n_keypoints, 3) array
        - keypoint_names: Ordered list of keypoint names
    """
    logger.info(f"Preparing data for {len(topology.keypoint_names)} keypoints...")
    
    # Validate all keypoints exist
    missing_keypoints = set(topology.keypoint_names) - set(trajectory_dict.keys())
    if missing_keypoints:
        raise ValueError(f"Missing keypoints in data: {missing_keypoints}")
    
    # Extract trajectories in order specified by topology
    trajectories: list[np.ndarray] = []
    for keypoint_name in topology.keypoint_names:
        traj = trajectory_dict[keypoint_name]
        trajectories.append(traj)
        logger.info(f"  {keypoint_name}: {traj.shape}")
    
    # Stack into (n_frames, n_keypoints, 3)
    original_data = np.stack(trajectories, axis=1)
    
    logger.info(f"Data shape: {original_data.shape}")
    return original_data, topology.keypoint_names


# =============================================================================
# OPTIMIZATION
# =============================================================================

def optimize_head_tracking(
    *,
    original_data: np.ndarray,
    topology: RigidBodyTopology,
    max_iter: int = 300,
    lambda_data: float = 100.0,
    lambda_rigid: float = 100.0,
    lambda_rot_smooth: float = 500.0,
    lambda_trans_smooth: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run PyCeres optimization on head tracking data.
    
    Args:
        original_data: (n_frames, n_keypoints, 3)
        topology: Rigid body topology
        max_iter: Maximum iterations
        lambda_data: Weight for data fitting
        lambda_rigid: Weight for rigid constraints
        lambda_rot_smooth: Weight for rotation smoothness
        lambda_trans_smooth: Weight for translation smoothness
        
    Returns:
        - rotations: (n_frames, 3, 3)
        - translations: (n_frames, 3)
        - optimized_data: (n_frames, n_keypoints, 3)
    """
    logger.info("\n" + "="*80)
    logger.info(f"OPTIMIZING: {topology.name}")
    logger.info("="*80)
    
    # Estimate reference geometry
    logger.info("\nEstimating reference geometry...")
    reference_geometry = estimate_reference_geometry(original_data=original_data)
    reference_distances = compute_reference_distances(reference=reference_geometry)
    
    # Run optimization with custom edge topology
    rotations, translations, optimized_data = optimize_rigid_body_pyceres(
        original_data=original_data,
        reference_geometry=reference_geometry,
        reference_distances=reference_distances,
        max_iter=max_iter,
        lambda_data=lambda_data,
        lambda_rigid=lambda_rigid,
        lambda_rot_smooth=lambda_rot_smooth,
        lambda_trans_smooth=lambda_trans_smooth,
    )
    
    return rotations, translations, optimized_data


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    *,
    output_dir: Path,
    original_data: np.ndarray,
    optimized_data: np.ndarray,
    keypoint_names: list[str],
    topology: RigidBodyTopology,
    ground_truth_data: np.ndarray | None = None,
) -> None:
    """
    Save optimization results with topology metadata.
    
    Args:
        output_dir: Directory to save results
        original_data: (n_frames, n_keypoints, 3) original data
        optimized_data: (n_frames, n_keypoints, 3) optimized data
        keypoint_names: Ordered list of keypoint names
        topology: Rigid body topology
        ground_truth_data: Optional ground truth for comparison
    """
    logger.info("\nSaving results...")
    
    n_frames, n_keypoints, _ = original_data.shape
    
    # Save trajectory CSV
    csv_data: dict[str, np.ndarray | range] = {"frame": range(n_frames)}
    
    # Add original data
    for idx, keypoint_name in enumerate(keypoint_names):
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            csv_data[f"original_{keypoint_name}_{coord_name}"] = original_data[:, idx, coord_idx]
    
    # Add optimized data
    for idx, keypoint_name in enumerate(keypoint_names):
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            csv_data[f"optimized_{keypoint_name}_{coord_name}"] = optimized_data[:, idx, coord_idx]
    
    # Add ground truth if provided
    if ground_truth_data is not None:
        for idx, keypoint_name in enumerate(keypoint_names):
            for coord_idx, coord_name in enumerate(["x", "y", "z"]):
                csv_data[f"gt_{keypoint_name}_{coord_name}"] = ground_truth_data[:, idx, coord_idx]
    
    # Add centroids
    original_center = np.mean(original_data, axis=1)
    optimized_center = np.mean(optimized_data, axis=1)
    
    for coord_idx, coord_name in enumerate(["x", "y", "z"]):
        csv_data[f"original_center_{coord_name}"] = original_center[:, coord_idx]
        csv_data[f"optimized_center_{coord_name}"] = optimized_center[:, coord_idx]
    
    if ground_truth_data is not None:
        gt_center = np.mean(ground_truth_data, axis=1)
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            csv_data[f"gt_center_{coord_name}"] = gt_center[:, coord_idx]
    
    # Save CSV
    df = pd.DataFrame(data=csv_data)
    csv_path = output_dir / "trajectory_data.csv"
    df.to_csv(path_or_buf=csv_path, index=False)
    logger.info(f"  Saved trajectory CSV: {csv_path}")
    
    # Save topology metadata as JSON
    topology_data = {
        "topology": topology.to_dict(),
        "keypoint_names": keypoint_names,
        "n_frames": n_frames,
        "n_keypoints": n_keypoints,
        "has_ground_truth": ground_truth_data is not None,
    }
    
    json_path = output_dir / "topology.json"
    with open(json_path, "w") as f:
        json.dump(obj=topology_data, fp=f, indent=2)
    logger.info(f"  Saved topology JSON: {json_path}")
    
    logger.info(f"\n Results saved to: {output_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_head_tracking_pipeline(*, config: HeadTrackingConfig) -> None:
    """
    Complete pipeline: load, optimize, save.
    
    Args:
        config: HeadTrackingConfig with all parameters
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )
    
    logger.info("="*80)
    logger.info("HEAD TRACKING PROCESSING PIPELINE")
    logger.info("="*80)
    
    # Load data
    logger.info(f"\n[1/3] Loading data from {config.input_csv.name}")
    trajectory_dict = load_trajectories_auto(
        filepath=config.input_csv,
        scale_factor=config.scale_factor,
        z_value=config.z_value,
        likelihood_threshold=config.likelihood_threshold,
    )
    
    # Prepare data according to topology
    original_data, keypoint_names = prepare_data_from_trajectories(
        trajectory_dict=trajectory_dict,
        topology=config.topology,
    )
    
    # Optimize
    logger.info("\n[2/3] Running PyCeres optimization")
    _, _, optimized_data = optimize_head_tracking(
        original_data=original_data,
        topology=config.topology,
        max_iter=config.max_iter,
        lambda_data=config.lambda_data,
        lambda_rigid=config.lambda_rigid,
        lambda_rot_smooth=config.lambda_rot_smooth,
        lambda_trans_smooth=config.lambda_trans_smooth,
    )
    
    # Save results
    logger.info("\n[3/3] Saving results")
    save_results(
        output_dir=config.output_dir,
        original_data=original_data,
        optimized_data=optimized_data,
        keypoint_names=keypoint_names,
        topology=config.topology,
        ground_truth_data=None,  # No ground truth for real data
    )
    
    logger.info("\n" + "="*80)
    logger.info(" PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOpen {config.output_dir / 'rigid_body_viewer.html'} to visualize!")

