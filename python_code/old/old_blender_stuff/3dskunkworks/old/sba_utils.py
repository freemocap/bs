"""
 Sparse Bundle Adjustment for Rigid Body Tracking
========================================================

Core idea: Optimize ALL frames globally using only:
1. Data term: fit original measurements
2. Rigid body constraint: all pairwise distances are fixed
3. Temporal smoothness: positions/rotations change smoothly

"""

import numpy as np
import torch
import logging
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SBAConfig:
    """Configuration for sparse bundle adjustment."""

    # Optimization weights
    lambda_data: float = 1.0
    lambda_rigid: float = 10000.0  # Strong pairwise distance constraints
    lambda_smooth_pos: float = 1.0  # Acceleration smoothness
    lambda_smooth_rot: float = 0.5  # Rotation acceleration smoothness
    lambda_velocity_pos: float = 0.0  # Velocity smoothness for position
    lambda_velocity_rot: float = 0.0  # Velocity smoothness for rotation

    # Optimizer settings
    max_iter: int = 500
    learning_rate: float = 0.01

    # Distance constraint tolerance
    distance_tolerance: float = 0.01  # 1cm slack in rigid constraint

    # Post-processing smoothing
    apply_post_smoothing: bool = False
    post_smooth_window: int = 5
    post_smooth_iterations: int = 3


# =============================================================================
# REFERENCE GEOMETRY ESTIMATION (IMPROVED)
# =============================================================================

def estimate_pairwise_distances_robust(*, original_data: np.ndarray) -> np.ndarray:
    """
    Estimate true pairwise distances by looking at measurements across all frames.

    For each pair of points, compute distance in each frame and take the median.
    This is robust to noise and preserves the rigid body structure.

    Args:
        original_data: (n_frames, n_points, 3) original measurements

    Returns:
        (n_points, n_points) estimated distance matrix
    """
    logger.info("Estimating pairwise distances from temporal data...")

    n_frames, n_points, _ = original_data.shape
    distances = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Compute distance in each frame
            frame_distances = np.linalg.norm(
                original_data[:, i, :] - original_data[:, j, :],
                axis=1
            )

            # Use median as robust estimate
            estimated_dist = np.median(frame_distances)
            distances[i, j] = estimated_dist
            distances[j, i] = estimated_dist

            # Log statistics for debugging
            dist_std = np.std(frame_distances)
            if j == i + 1:  # Log first few pairs
                logger.info(f"  Point {i}-{j}: {estimated_dist:.4f}m (σ={dist_std:.4f}m)")

    return distances


def reconstruct_geometry_from_distances(
    *,
    distances: np.ndarray,
    n_points: int
) -> np.ndarray:
    """
    Reconstruct 3D geometry from pairwise distances using Classical MDS.

    Uses eigenvalue decomposition for globally optimal solution.

    Args:
        distances: (n_points, n_points) distance matrix
        n_points: Number of points

    Returns:
        (n_points, 3) reconstructed geometry
    """
    logger.info("Reconstructing reference geometry from distances (Classical MDS)...")

    # Classical Multidimensional Scaling algorithm
    # Step 1: Convert distance matrix to squared distances
    D_squared = distances ** 2

    # Step 2: Apply double centering
    # J = I - (1/n)*11^T (centering matrix)
    J = np.eye(n_points) - np.ones((n_points, n_points)) / n_points

    # B = -0.5 * J * D^2 * J
    B = -0.5 * J @ D_squared @ J

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 4: Take top 3 dimensions
    # X = V_k * Lambda_k^(1/2)
    k = 3  # We want 3D coordinates

    # Handle negative eigenvalues (due to numerical errors)
    eigenvalues_k = eigenvalues[:k]
    eigenvalues_k = np.maximum(eigenvalues_k, 0)

    geometry = eigenvectors[:, :k] @ np.diag(np.sqrt(eigenvalues_k))

    # Center at origin (should already be centered, but ensure it)
    geometry = geometry - np.mean(geometry, axis=0)

    # Verify reconstruction quality
    reconstruction_error = 0.0
    max_error = 0.0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            recon_dist = np.linalg.norm(geometry[i] - geometry[j])
            target_dist = distances[i, j]
            error = abs(recon_dist - target_dist)
            reconstruction_error += error
            max_error = max(max_error, error)

    avg_error = reconstruction_error / (n_points * (n_points - 1) / 2)
    logger.info(f"  Reconstruction error: avg={avg_error * 1000:.3f}mm, max={max_error * 1000:.3f}mm")
    logger.info(f"  Geometry span: {np.ptp(geometry):.3f}m")
    logger.info(f"  Top 3 eigenvalues: {eigenvalues[:3]}")

    return geometry


def estimate_reference_geometry(*, original_data: np.ndarray) -> np.ndarray:
    """
    Estimate reference geometry from original measurements.

    Two-step process:
    1. Robustly estimate pairwise distances from temporal data
    2. Reconstruct geometry that matches those distances

    Args:
        original_data: (n_frames, n_points, 3) original measurements

    Returns:
        (n_points, 3) reference geometry
    """
    logger.info("Estimating reference geometry from original data...")

    n_points = original_data.shape[1]

    # Step 1: Get robust distance estimates
    distances = estimate_pairwise_distances_robust(original_data=original_data)

    # Step 2: Reconstruct geometry from distances
    reference = reconstruct_geometry_from_distances(
        distances=distances,
        n_points=n_points
    )

    return reference


def compute_reference_distances(*, reference: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise distances in reference geometry.

    Args:
        reference: (n_points, 3)

    Returns:
        (n_points, n_points) distance matrix
    """
    n_points = len(reference)
    distances = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(reference[i] - reference[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


# =============================================================================
# PYTORCH ROTATION UTILITIES (BATCHED)
# =============================================================================

def rotation_matrices_from_rotvecs_batched(*, rotvecs: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of rotation vectors to rotation matrices.

    Args:
        rotvecs: (n_frames, 3)

    Returns:
        (n_frames, 3, 3) rotation matrices
    """
    angles = torch.linalg.norm(rotvecs, dim=1, keepdim=True)  # (n_frames, 1)
    small_angle = angles < 1e-8

    # Normalized axes
    axes = torch.where(
        small_angle.expand(-1, 3),
        torch.zeros_like(rotvecs),
        rotvecs / (angles + 1e-10)
    )

    # Rodrigues formula (vectorized)
    n_frames = rotvecs.shape[0]

    # Skew-symmetric matrices
    K = torch.zeros(n_frames, 3, 3, dtype=rotvecs.dtype, device=rotvecs.device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    I = torch.eye(3, dtype=rotvecs.dtype, device=rotvecs.device).unsqueeze(0).expand(n_frames, -1, -1)

    sin_angle = torch.sin(angles).unsqueeze(2)
    cos_angle = torch.cos(angles).unsqueeze(2)

    R = I + sin_angle * K + (1 - cos_angle) * torch.bmm(K, K)
    R = torch.where(small_angle.unsqueeze(2).expand(-1, 3, 3), I, R)

    return R


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def sba_objective(
    *,
    x: torch.Tensor,
    original_data: torch.Tensor,
    reference_geometry: torch.Tensor,
    reference_distances: torch.Tensor,
    config: SBAConfig
) -> torch.Tensor:
    """
    Sparse bundle adjustment objective function.

    Optimizes all frames globally with:
    1. Data fitting term
    2. Rigid body constraints (all pairwise distances)
    3. Temporal smoothness (velocity and acceleration)

    Args:
        x: Flattened parameters (n_frames * 6,)
        original_data: (n_frames, n_points, 3)
        reference_geometry: (n_points, 3)
        reference_distances: (n_points, n_points)
        config: SBAConfig

    Returns:
        Total cost
    """
    n_frames = original_data.shape[0]
    n_points = reference_geometry.shape[0]

    # Reshape parameters
    x_reshaped = x.reshape(n_frames, 6)
    rotvecs = x_reshaped[:, :3]  # (n_frames, 3)
    translations = x_reshaped[:, 3:]  # (n_frames, 3)

    total_cost = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    # ===== 1. DATA FITTING TERM =====
    R_batch = rotation_matrices_from_rotvecs_batched(rotvecs=rotvecs)  # (n_frames, 3, 3)

    # Transform reference geometry for all frames
    # (n_frames, n_points, 3) = (n_points, 3) @ (n_frames, 3, 3)^T + (n_frames, 1, 3)
    reconstructed = torch.matmul(
        reference_geometry.unsqueeze(0).expand(n_frames, -1, -1),
        R_batch.transpose(1, 2)
    ) + translations.unsqueeze(1)

    # Data residuals
    residuals = reconstructed - original_data
    data_cost = torch.sum(residuals ** 2)
    total_cost += config.lambda_data * data_cost

    # ===== 2. RIGID BODY CONSTRAINTS =====
    # For each frame, compute all pairwise distances and compare to reference
    rigid_cost = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Distance in reconstructed frames
            diff = reconstructed[:, i, :] - reconstructed[:, j, :]  # (n_frames, 3)
            distances = torch.linalg.norm(diff, dim=1)  # (n_frames,)

            # Reference distance
            ref_dist = reference_distances[i, j]

            # Soft constraint with tolerance
            violations = torch.abs(distances - ref_dist)
            rigid_cost += torch.sum(violations ** 2)

    total_cost += config.lambda_rigid * rigid_cost

    # ===== 3. VELOCITY SMOOTHNESS - POSITIONS =====
    if n_frames > 1 and config.lambda_velocity_pos > 0:
        pos_vel = translations[1:] - translations[:-1]
        vel_smooth_cost = torch.sum(pos_vel ** 2)
        total_cost += config.lambda_velocity_pos * vel_smooth_cost

    # ===== 4. ACCELERATION SMOOTHNESS - POSITIONS =====
    if n_frames > 2 and config.lambda_smooth_pos > 0:
        pos_vel = translations[1:] - translations[:-1]
        pos_accel = pos_vel[1:] - pos_vel[:-1]
        smooth_pos_cost = torch.sum(pos_accel ** 2)
        total_cost += config.lambda_smooth_pos * smooth_pos_cost

    # ===== 5. VELOCITY SMOOTHNESS - ROTATIONS =====
    if n_frames > 1 and config.lambda_velocity_rot > 0:
        rot_vel = rotvecs[1:] - rotvecs[:-1]
        rot_vel_smooth_cost = torch.sum(rot_vel ** 2)
        total_cost += config.lambda_velocity_rot * rot_vel_smooth_cost

    # ===== 6. ACCELERATION SMOOTHNESS - ROTATIONS =====
    if n_frames > 2 and config.lambda_smooth_rot > 0:
        rot_vel = rotvecs[1:] - rotvecs[:-1]
        rot_accel = rot_vel[1:] - rot_vel[:-1]
        smooth_rot_cost = torch.sum(rot_accel ** 2)
        total_cost += config.lambda_smooth_rot * smooth_rot_cost

    return total_cost


# =============================================================================
# INITIALIZATION (IMPROVED)
# =============================================================================

def initialize_parameters(
    *,
    original_data: np.ndarray,
    reference_geometry: np.ndarray
) -> np.ndarray:
    """
    Better initialization using Procrustes alignment for each frame.

    This gives a much better starting point than identity rotation.

    Args:
        original_data: (n_frames, n_points, 3)
        reference_geometry: (n_points, 3)

    Returns:
        (n_frames * 6,) initial parameters
    """
    logger.info("Initializing parameters with Procrustes alignment...")

    n_frames = original_data.shape[0]
    x0 = np.zeros(n_frames * 6)

    for i in range(n_frames):
        # Center both point sets
        ref_centered = reference_geometry - np.mean(reference_geometry, axis=0)
        data_centered = original_data[i] - np.mean(original_data[i], axis=0)

        # Simple Procrustes: H = data^T @ ref
        H = data_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        R_init = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_init) < 0:
            Vt[-1, :] *= -1
            R_init = Vt.T @ U.T

        # Convert to rotation vector
        rotvec = Rotation.from_matrix(R_init).as_rotvec()
        x0[i * 6:i * 6 + 3] = rotvec

        # Translation
        centroid = np.mean(original_data[i], axis=0)
        x0[i * 6 + 3:i * 6 + 6] = centroid

    logger.info("  Initialization complete")

    return x0


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def optimize_rigid_body_sba(
    *,
    original_data: np.ndarray,
    config: SBAConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize rigid body trajectory using sparse bundle adjustment.

    Args:
        original_data: (n_frames, n_points, 3) original measurements
        config: Optimization configuration

    Returns:
        - rotations: (n_frames, 3, 3)
        - translations: (n_frames, 3)
        - reconstructed: (n_frames, n_points, 3)
    """
    if config is None:
        config = SBAConfig()

    logger.info("=" * 80)
    logger.info("SPARSE BUNDLE ADJUSTMENT - SIMPLE RIGID BODY")
    logger.info("=" * 80)

    n_frames, n_points, _ = original_data.shape

    # Step 1: Estimate reference geometry
    reference_geometry = estimate_reference_geometry(original_data=original_data)
    reference_distances = compute_reference_distances(reference=reference_geometry)

    logger.info("\nReference distances:")
    for i in range(min(3, n_points)):
        for j in range(i + 1, min(4, n_points)):
            logger.info(f"  Point {i}-{j}: {reference_distances[i, j]:.4f}m")

    # Step 2: Initialize parameters
    logger.info("\nInitializing parameters...")
    x0 = initialize_parameters(
        original_data=original_data,
        reference_geometry=reference_geometry
    )

    # Step 3: Prepare torch tensors for gradient computation
    logger.info(f"\nUsing device for gradients: {DEVICE}")
    original_data_torch = torch.tensor(original_data, dtype=torch.float64, device=DEVICE)
    reference_geometry_torch = torch.tensor(reference_geometry, dtype=torch.float64, device=DEVICE)
    reference_distances_torch = torch.tensor(reference_distances, dtype=torch.float64, device=DEVICE)

    # Step 4: Display optimization setup
    logger.info(f"\nOptimization setup:")
    logger.info(f"  Frames: {n_frames}")
    logger.info(f"  Points: {n_points}")
    logger.info(f"  Parameters: {len(x0)}")
    logger.info(f"  Max iterations: {config.max_iter}")
    logger.info(f"\nWeights:")
    logger.info(f"  λ_data:     {config.lambda_data:8.1f}")
    logger.info(f"  λ_rigid:    {config.lambda_rigid:8.1f}")
    logger.info(f"  λ_vel_pos:  {config.lambda_velocity_pos:8.1f}")
    logger.info(f"  λ_acc_pos:  {config.lambda_smooth_pos:8.1f}")
    logger.info(f"  λ_vel_rot:  {config.lambda_velocity_rot:8.1f}")
    logger.info(f"  λ_acc_rot:  {config.lambda_smooth_rot:8.1f}")

    # Step 5: Define objective and gradient function for scipy
    def objective_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute objective and gradient using PyTorch autodiff."""
        x_torch = torch.tensor(x, dtype=torch.float64, device=DEVICE, requires_grad=True)

        cost = sba_objective(
            x=x_torch,
            original_data=original_data_torch,
            reference_geometry=reference_geometry_torch,
            reference_distances=reference_distances_torch,
            config=config
        )

        cost.backward()

        return float(cost.item()), x_torch.grad.cpu().numpy()

    # Step 6: Optimize with scipy L-BFGS-B
    logger.info("\n" + "=" * 80)
    logger.info("SCIPY L-BFGS-B OPTIMIZATION")
    logger.info("=" * 80 + "\n")

    result = minimize(
        fun=objective_and_gradient,
        x0=x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': config.max_iter,
            'ftol': 1e-9,
            'gtol': 1e-7,
            'disp': True,
            'iprint': 1,
            'maxls': 20,
        }
    )

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Optimization complete!")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Message: {result.message}")
    logger.info(f"  Final cost: {result.fun:.2f}")
    logger.info(f"  Iterations: {result.nit}")
    logger.info(f"  Function evaluations: {result.nfev}")
    logger.info("=" * 80)

    # Step 7: Extract results
    x_final = result.x

    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))
    reconstructed = np.zeros((n_frames, n_points, 3))

    for i in range(n_frames):
        rotvec = x_final[i * 6:i * 6 + 3]
        rotations[i] = Rotation.from_rotvec(rotvec).as_matrix()
        translations[i] = x_final[i * 6 + 3:i * 6 + 6]
        reconstructed[i] = (rotations[i] @ reference_geometry.T).T + translations[i]

    # Step 8: Apply post-smoothing if requested
    if config.apply_post_smoothing:
        rotations, translations, reconstructed = smooth_with_rigidity_preservation(
            rotations=rotations,
            translations=translations,
            reference_geometry=reference_geometry,
            window_size=config.post_smooth_window,
            iterations=config.post_smooth_iterations
        )

    return rotations, translations, reconstructed


# =============================================================================
# POST-PROCESSING: RIGID-BODY-PRESERVING SMOOTHING
# =============================================================================

def smooth_with_rigidity_preservation(
    *,
    rotations: np.ndarray,
    translations: np.ndarray,
    reference_geometry: np.ndarray,
    window_size: int = 5,
    iterations: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply smoothing while strictly preserving rigid body constraints.

    Uses iterative refinement: smooth poses, then project back to maintain
    exact pairwise distances in the reference geometry.

    Args:
        rotations: (n_frames, 3, 3) rotation matrices
        translations: (n_frames, 3) translation vectors
        reference_geometry: (n_points, 3) reference configuration
        window_size: Smoothing window size
        iterations: Number of refinement iterations

    Returns:
        - smoothed_rotations: (n_frames, 3, 3)
        - smoothed_translations: (n_frames, 3)
        - reconstructed: (n_frames, n_points, 3)
    """
    logger.info(f"\nPost-processing: Rigid-body-preserving smooth (window={window_size}, iter={iterations})...")

    from scipy.ndimage import uniform_filter1d

    n_frames = len(rotations)
    n_points = len(reference_geometry)

    # Convert rotations to rotation vectors for smoothing
    rotvecs = np.array([Rotation.from_matrix(R).as_rotvec() for R in rotations])

    for iter_idx in range(iterations):
        # Smooth rotation vectors and translations
        rotvecs_smooth = np.zeros_like(rotvecs)
        translations_smooth = np.zeros_like(translations)

        for dim in range(3):
            rotvecs_smooth[:, dim] = uniform_filter1d(
                input=rotvecs[:, dim],
                size=window_size,
                mode='nearest'
            )
            translations_smooth[:, dim] = uniform_filter1d(
                input=translations[:, dim],
                size=window_size,
                mode='nearest'
            )

        # Convert back to rotation matrices
        rotations_smooth = np.array([
            Rotation.from_rotvec(rv).as_matrix() for rv in rotvecs_smooth
        ])

        # Reconstruct points
        reconstructed = np.zeros((n_frames, n_points, 3))
        for i in range(n_frames):
            reconstructed[i] = (rotations_smooth[i] @ reference_geometry.T).T + translations_smooth[i]

        # Update for next iteration
        rotvecs = rotvecs_smooth
        translations = translations_smooth
        rotations = rotations_smooth

    logger.info("  ✓ Smoothing complete")

    return rotations, translations, reconstructed


