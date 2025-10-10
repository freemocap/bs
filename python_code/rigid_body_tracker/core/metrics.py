"""Evaluation metrics for rigid body tracking."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def evaluate_reconstruction(
    *,
    noisy_data: np.ndarray,
    optimized_data: np.ndarray,
    reference_distances: np.ndarray,
    topology: 'RigidBodyTopology',
    ground_truth_data: np.ndarray | None = None
) -> dict[str, float]:
    """
    Evaluate reconstruction quality.
    
    Args:
        noisy_data: (n_frames, n_markers, 3)
        optimized_data: (n_frames, n_markers, 3)
        reference_distances: (n_markers, n_markers) target distances
        topology: RigidBodyTopology
        ground_truth_data: Optional ground truth
        
    Returns:
        Dictionary of metrics
    """
    n_frames, n_markers, _ = noisy_data.shape
    metrics: dict[str, float] = {}
    
    # =========================================================================
    # RIGIDITY METRICS
    # =========================================================================
    logger.info("\nRigidity metrics:")
    
    # Compute edge length errors for rigid edges
    noisy_edge_errors = []
    optimized_edge_errors = []
    
    for i, j in topology.rigid_edges:
        target_dist = reference_distances[i, j]
        
        noisy_dists = np.linalg.norm(noisy_data[:, i, :] - noisy_data[:, j, :], axis=1)
        opt_dists = np.linalg.norm(optimized_data[:, i, :] - optimized_data[:, j, :], axis=1)
        
        noisy_edge_errors.extend(np.abs(noisy_dists - target_dist))
        optimized_edge_errors.extend(np.abs(opt_dists - target_dist))
    
    metrics['noisy_edge_error_mean_mm'] = float(np.mean(noisy_edge_errors) * 1000)
    metrics['noisy_edge_error_max_mm'] = float(np.max(noisy_edge_errors) * 1000)
    metrics['optimized_edge_error_mean_mm'] = float(np.mean(optimized_edge_errors) * 1000)
    metrics['optimized_edge_error_max_mm'] = float(np.max(optimized_edge_errors) * 1000)
    
    logger.info(f"  Noisy edge errors:     {metrics['noisy_edge_error_mean_mm']:.2f}mm (max: {metrics['noisy_edge_error_max_mm']:.2f}mm)")
    logger.info(f"  Optimized edge errors: {metrics['optimized_edge_error_mean_mm']:.2f}mm (max: {metrics['optimized_edge_error_max_mm']:.2f}mm)")
    
    # Edge length consistency (std dev across time)
    noisy_consistency = []
    optimized_consistency = []
    
    for i, j in topology.rigid_edges:
        noisy_dists = np.linalg.norm(noisy_data[:, i, :] - noisy_data[:, j, :], axis=1)
        opt_dists = np.linalg.norm(optimized_data[:, i, :] - optimized_data[:, j, :], axis=1)
        
        noisy_consistency.append(np.std(noisy_dists))
        optimized_consistency.append(np.std(opt_dists))
    
    metrics['noisy_edge_std_mm'] = float(np.mean(noisy_consistency) * 1000)
    metrics['optimized_edge_std_mm'] = float(np.mean(optimized_consistency) * 1000)
    
    logger.info(f"  Noisy consistency:     {metrics['noisy_edge_std_mm']:.2f}mm std")
    logger.info(f"  Optimized consistency: {metrics['optimized_edge_std_mm']:.2f}mm std")
    
    # =========================================================================
    # ACCURACY METRICS (if ground truth available)
    # =========================================================================
    if ground_truth_data is not None:
        logger.info("\nAccuracy metrics (vs ground truth):")
        
        noisy_errors = np.linalg.norm(noisy_data - ground_truth_data, axis=2)
        opt_errors = np.linalg.norm(optimized_data - ground_truth_data, axis=2)
        
        metrics['noisy_position_error_mean_mm'] = float(np.mean(noisy_errors) * 1000)
        metrics['noisy_position_error_max_mm'] = float(np.max(noisy_errors) * 1000)
        metrics['optimized_position_error_mean_mm'] = float(np.mean(opt_errors) * 1000)
        metrics['optimized_position_error_max_mm'] = float(np.max(opt_errors) * 1000)
        
        logger.info(f"  Noisy position error:     {metrics['noisy_position_error_mean_mm']:.2f}mm (max: {metrics['noisy_position_error_max_mm']:.2f}mm)")
        logger.info(f"  Optimized position error: {metrics['optimized_position_error_mean_mm']:.2f}mm (max: {metrics['optimized_position_error_max_mm']:.2f}mm)")
        
        # Improvement
        improvement = (metrics['noisy_position_error_mean_mm'] - metrics['optimized_position_error_mean_mm']) / metrics['noisy_position_error_mean_mm'] * 100
        metrics['improvement_percent'] = float(improvement)
        
        logger.info(f"  Improvement: {improvement:.1f}%")
    
    # =========================================================================
    # SMOOTHNESS METRICS
    # =========================================================================
    logger.info("\nSmoothness metrics:")
    
    # Centroid acceleration (measure of jitter)
    noisy_centroids = np.mean(noisy_data, axis=1)
    opt_centroids = np.mean(optimized_data, axis=1)
    
    noisy_vel = np.diff(noisy_centroids, axis=0)
    opt_vel = np.diff(opt_centroids, axis=0)
    
    noisy_accel = np.diff(noisy_vel, axis=0)
    opt_accel = np.diff(opt_vel, axis=0)
    
    metrics['noisy_centroid_jitter_mm'] = float(np.mean(np.linalg.norm(noisy_accel, axis=1)) * 1000)
    metrics['optimized_centroid_jitter_mm'] = float(np.mean(np.linalg.norm(opt_accel, axis=1)) * 1000)
    
    logger.info(f"  Noisy jitter:     {metrics['noisy_centroid_jitter_mm']:.3f}mm")
    logger.info(f"  Optimized jitter: {metrics['optimized_centroid_jitter_mm']:.3f}mm")
    
    return metrics
