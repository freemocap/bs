"""Reference geometry estimation from noisy measurements with visualization."""

import numpy as np
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)





def align_reference_to_data_procrustes(
    *,
    reference: np.ndarray,
    noisy_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align reference to FIRST FRAME ONLY using Procrustes.

    Uses EXACT same method as optimization initialization to ensure
    consistent coordinate frames throughout the pipeline.

    Args:
        reference: (n_markers, 3) reference from MDS (arbitrary orientation)
        noisy_data: (n_frames, n_markers, 3) noisy measurements

    Returns:
        (unaligned_reference, aligned_reference) tuple
    """
    logger.info(f"  Aligning reference to first frame...")

    # Center both (exactly as optimization init does)
    ref_centered = reference - np.mean(reference, axis=0)
    data_centered = noisy_data[0] - np.mean(noisy_data[0], axis=0)

    # Procrustes: H = data^T @ ref (MUST match optimization init)
    H = data_centered.T @ ref_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation to centered reference
    aligned_ref = (R @ ref_centered.T).T

    logger.info(f"  ✓ Reference aligned to frame 0")

    return ref_centered, aligned_ref


def estimate_reference_geometry(
    *,
    noisy_data: np.ndarray,
    show_n_frames: int | None = None
) -> tuple[np.ndarray]:
    """
    Estimate reference geometry from noisy temporal measurements.

    Args:
        noisy_data: (n_frames, n_points, 3) noisy measurements
        show_n_frames: Limit visualization to N frames (None = all)

    Returns:
        (reference_geometry)
        - reference_geometry: Aligned reference
    """
    distances = estimate_pairwise_distances_robust(noisy_data=noisy_data)
    geometry = reconstruct_geometry_from_distances(distances=distances)

    # Align to noisy data coordinate frame using Procrustes
    unaligned_ref, aligned_ref = align_reference_to_data_procrustes(
        reference=geometry,
        noisy_data=noisy_data
    )


    return aligned_ref


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