"""Evaluation metrics for rigid body tracking."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def evaluate_reconstruction(
    *,
    original_data: np.ndarray,
    optimized_data: np.ndarray,
    reference_distances: np.ndarray,
    topology: 'RigidBodyTopology',
    ground_truth_data: np.ndarray | None = None
) -> dict[str, float]:
    """
    Evaluate reconstruction quality.
    
    Args:
        original_data: (n_frames, n_markers, 3)
        optimized_data: (n_frames, n_markers, 3)
        reference_distances: (n_markers, n_markers) target distances
        topology: RigidBodyTopology
        ground_truth_data: Optional ground truth
        
    Returns:
        Dictionary of metrics
    """
    n_frames, n_markers, _ = original_data.shape
    metrics: dict[str, float] = {}
    
    # =========================================================================
    # RIGIDITY METRICS
    # =========================================================================
    logger.info("\nRigidity metrics:")
    
    # Compute edge length errors for rigid edges
    original_edge_errors = []
    optimized_edge_errors = []
    
    for i, j in topology.rigid_edges:
        target_dist = reference_distances[i, j]
        
        original_dists = np.linalg.norm(original_data[:, i, :] - original_data[:, j, :], axis=1)
        opt_dists = np.linalg.norm(optimized_data[:, i, :] - optimized_data[:, j, :], axis=1)
        
        original_edge_errors.extend(np.abs(original_dists - target_dist))
        optimized_edge_errors.extend(np.abs(opt_dists - target_dist))
    
    metrics['original_edge_error_mean_mm'] = float(np.mean(original_edge_errors) * 1000)
    metrics['original_edge_error_max_mm'] = float(np.max(original_edge_errors) * 1000)
    metrics['optimized_edge_error_mean_mm'] = float(np.mean(optimized_edge_errors) * 1000)
    metrics['optimized_edge_error_max_mm'] = float(np.max(optimized_edge_errors) * 1000)
    
    logger.info(f"  Original edge errors:     {metrics['original_edge_error_mean_mm']:.2f}mm (max: {metrics['original_edge_error_max_mm']:.2f}mm)")
    logger.info(f"  Optimized edge errors: {metrics['optimized_edge_error_mean_mm']:.2f}mm (max: {metrics['optimized_edge_error_max_mm']:.2f}mm)")
    
    # Edge length consistency (std dev across time)
    original_consistency = []
    optimized_consistency = []
    
    for i, j in topology.rigid_edges:
        original_dists = np.linalg.norm(original_data[:, i, :] - original_data[:, j, :], axis=1)
        opt_dists = np.linalg.norm(optimized_data[:, i, :] - optimized_data[:, j, :], axis=1)
        
        original_consistency.append(np.std(original_dists))
        optimized_consistency.append(np.std(opt_dists))
    
    metrics['original_edge_std_mm'] = float(np.mean(original_consistency) * 1000)
    metrics['optimized_edge_std_mm'] = float(np.mean(optimized_consistency) * 1000)
    
    logger.info(f"  Original consistency:     {metrics['original_edge_std_mm']:.2f}mm std")
    logger.info(f"  Optimized consistency: {metrics['optimized_edge_std_mm']:.2f}mm std")
    
    # =========================================================================
    # ACCURACY METRICS (if ground truth available)
    # =========================================================================
    if ground_truth_data is not None:
        logger.info("\nAccuracy metrics (vs ground truth):")
        
        original_errors = np.linalg.norm(original_data - ground_truth_data, axis=2)
        opt_errors = np.linalg.norm(optimized_data - ground_truth_data, axis=2)
        
        metrics['original_position_error_mean_mm'] = float(np.mean(original_errors) * 1000)
        metrics['original_position_error_max_mm'] = float(np.max(original_errors) * 1000)
        metrics['optimized_position_error_mean_mm'] = float(np.mean(opt_errors) * 1000)
        metrics['optimized_position_error_max_mm'] = float(np.max(opt_errors) * 1000)
        
        logger.info(f"  Original position error:     {metrics['original_position_error_mean_mm']:.2f}mm (max: {metrics['original_position_error_max_mm']:.2f}mm)")
        logger.info(f"  Optimized position error: {metrics['optimized_position_error_mean_mm']:.2f}mm (max: {metrics['optimized_position_error_max_mm']:.2f}mm)")
        
        # Improvement
        improvement = (metrics['original_position_error_mean_mm'] - metrics['optimized_position_error_mean_mm']) / metrics['original_position_error_mean_mm'] * 100
        metrics['improvement_percent'] = float(improvement)
        
        logger.info(f"  Improvement: {improvement:.1f}%")
    
    # =========================================================================
    # SMOOTHNESS METRICS
    # =========================================================================
    logger.info("\nSmoothness metrics:")
    
    # Centroid acceleration (measure of jitter)
    original_centroids = np.mean(original_data, axis=1)
    opt_centroids = np.mean(optimized_data, axis=1)
    
    original_vel = np.diff(original_centroids, axis=0)
    opt_vel = np.diff(opt_centroids, axis=0)
    
    original_accel = np.diff(original_vel, axis=0)
    opt_accel = np.diff(opt_vel, axis=0)
    
    metrics['original_centroid_jitter_mm'] = float(np.mean(np.linalg.norm(original_accel, axis=1)) * 1000)
    metrics['optimized_centroid_jitter_mm'] = float(np.mean(np.linalg.norm(opt_accel, axis=1)) * 1000)
    
    logger.info(f"  Original jitter:     {metrics['original_centroid_jitter_mm']:.3f}mm")
    logger.info(f"  Optimized jitter: {metrics['optimized_centroid_jitter_mm']:.3f}mm")
    
    return metrics
