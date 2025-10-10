"""Reference geometry estimation from noisy measurements."""

import numpy as np
import logging
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def estimate_pairwise_distances_robust(
        *,
        noisy_data: np.ndarray
) -> np.ndarray:
    """
    Estimate true pairwise distances by aggregating measurements across all frames.

    For each pair of points, compute distance in each frame and take the median.
    This is robust to noise and preserves rigid body structure.

    Args:
        noisy_data: (n_frames, n_points, 3) noisy measurements

    Returns:
        (n_points, n_points) estimated distance matrix
    """
    n_frames, n_points, _ = noisy_data.shape
    distances = np.zeros((n_points, n_points))

    logger.info(f"Estimating pairwise distances from {n_frames} frames...")

    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Compute distance in each frame
            frame_distances = np.linalg.norm(
                noisy_data[:, i, :] - noisy_data[:, j, :],
                axis=1
            )

            # Use median as robust estimate
            estimated_dist = np.median(frame_distances)
            distances[i, j] = estimated_dist
            distances[j, i] = estimated_dist

    return distances


def reconstruct_geometry_from_distances(
        *,
        distances: np.ndarray
) -> np.ndarray:
    """
    Reconstruct 3D geometry from pairwise distances using SMACOF.

    SMACOF (Scaling by MAjorizing a COmplicated Function) is more robust
    than Classical MDS and better preserves 3D structure.

    Args:
        distances: (n_points, n_points) distance matrix

    Returns:
        (n_points, 3) reconstructed geometry centered at origin
    """
    n_points = distances.shape[0]

    logger.info("Reconstructing geometry using SMACOF MDS...")

    # Initialize with Classical MDS
    D_squared = distances ** 2
    J = np.eye(n_points) - np.ones((n_points, n_points)) / n_points
    B = -0.5 * J @ D_squared @ J
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_3d = np.maximum(eigenvalues[idx[:3]], 0)
    X = eigenvectors[:, idx[:3]] @ np.diag(np.sqrt(eigenvalues_3d))

    # SMACOF iterations for refinement
    max_iter = 300
    eps = 1e-6
    stress_old = np.inf

    for iteration in range(max_iter):
        # Compute current distances
        D_current = squareform(pdist(X))

        # Avoid division by zero
        D_current_safe = np.where(D_current < 1e-10, 1e-10, D_current)

        # Compute B matrix (Guttman transform)
        ratio = distances / D_current_safe
        np.fill_diagonal(ratio, 0)
        B_iter = -ratio
        B_iter[np.diag_indices(n_points)] = -np.sum(B_iter, axis=1)

        # Update X
        X_new = (1.0 / n_points) * B_iter @ X

        # Compute stress
        diff = distances - D_current
        stress = np.sqrt(np.sum(diff ** 2))

        # Check convergence
        if abs(stress_old - stress) < eps:
            logger.info(f"  Converged at iteration {iteration}")
            break

        stress_old = stress
        X = X_new

    # Center at origin
    geometry = X - np.mean(X, axis=0)

    # Validate reconstruction
    reconstruction_errors = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            recon_dist = np.linalg.norm(geometry[i] - geometry[j])
            target_dist = distances[i, j]
            reconstruction_errors.append(abs(recon_dist - target_dist))

    avg_error = np.mean(reconstruction_errors)
    max_error = np.max(reconstruction_errors)

    logger.info(f"  Reconstruction error: mean={avg_error * 1000:.2f}mm, max={max_error * 1000:.2f}mm")

    return geometry


def estimate_reference_geometry(
        *,
        noisy_data: np.ndarray,
        method: str = "median"
) -> np.ndarray:
    """
    Estimate reference geometry from noisy temporal measurements.

    Args:
        noisy_data: (n_frames, n_points, 3) noisy measurements
        method: Estimation method:
            - "median": Robust distance estimation + SMACOF MDS (recommended)
            - "mean": Simple mean of all frames (fast but less accurate)

    Returns:
        (n_points, 3) reference geometry centered at origin
    """
    if method == "mean":
        # Simple approach: average all frames
        geometry = np.mean(noisy_data, axis=0)
        geometry = geometry - np.mean(geometry, axis=0)
        logger.info("Estimated reference using temporal mean")
        return geometry

    elif method == "median":
        # Robust approach: estimate distances then reconstruct with SMACOF
        distances = estimate_pairwise_distances_robust(noisy_data=noisy_data)
        geometry = reconstruct_geometry_from_distances(distances=distances)
        return geometry

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_all_pairwise_distances(
        *,
        reference: np.ndarray
) -> np.ndarray:
    """
    Compute all pairwise distances in reference geometry.

    Args:
        reference: (n_points, 3) reference positions

    Returns:
        (n_points, n_points) symmetric distance matrix
    """
    n_points = len(reference)
    distances = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(reference[i] - reference[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def analyze_reference_quality(
        *,
        reference_geometry: np.ndarray,
        noisy_data: np.ndarray
) -> dict[str, float]:
    """
    Analyze quality of estimated reference geometry.

    Args:
        reference_geometry: (n_points, 3) estimated reference
        noisy_data: (n_frames, n_points, 3) original noisy data

    Returns:
        Dictionary of quality metrics
    """
    n_frames, n_points, _ = noisy_data.shape

    # Compute reference distances
    ref_distances = compute_all_pairwise_distances(reference=reference_geometry)

    # Compute distance variation across frames
    distance_variations = []

    for i in range(n_points):
        for j in range(i + 1, n_points):
            frame_distances = np.linalg.norm(
                noisy_data[:, i, :] - noisy_data[:, j, :],
                axis=1
            )
            variation = np.std(frame_distances)
            distance_variations.append(variation)

    metrics = {
        "mean_distance_m": float(np.mean(ref_distances[ref_distances > 0])),
        "max_distance_m": float(np.max(ref_distances)),
        "mean_distance_variation_mm": float(np.mean(distance_variations) * 1000),
        "max_distance_variation_mm": float(np.max(distance_variations) * 1000),
    }

    logger.info("\nReference geometry quality:")
    logger.info(f"  Mean distance: {metrics['mean_distance_m']:.3f}m")
    logger.info(f"  Max distance: {metrics['max_distance_m']:.3f}m")
    logger.info(
        f"  Distance variation: {metrics['mean_distance_variation_mm']:.1f}mm ± {metrics['max_distance_variation_mm']:.1f}mm")

    return metrics