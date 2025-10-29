import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple
from dataclasses import dataclass
import time


@dataclass
class OptimizationParams:
    """Parameters for joint optimization - MUCH WEAKER regularization."""
    lambda_translation_accel: float = 0.1  # Reduced 100x from 1e1
    lambda_translation_jerk: float = 1.0  # Reduced 100x from 1e2
    lambda_rotation_accel: float = 0.1  # Reduced 100x from 1e1
    lambda_rotation_jerk: float = 0.5  # Reduced 100x from 5e1
    lambda_geometry_regularization: float = 0.01  # Reduced 100x from 1e0
    robust_loss: str = 'huber'
    robust_loss_delta: float = 0.010  # 10mm


def fit_rigid_transform(
        points_ref: np.ndarray,
        points_measured: np.ndarray,
        weights: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Find optimal rigid transformation using weighted Kabsch algorithm."""
    valid_mask = ~(np.isnan(points_measured).any(axis=1) | np.isnan(points_ref).any(axis=1))
    points_ref = points_ref[valid_mask]
    points_measured = points_measured[valid_mask]

    if weights is not None:
        weights = weights[valid_mask]
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(points_ref)) / len(points_ref)

    centroid_ref = np.sum(points_ref * weights[:, None], axis=0)
    centroid_measured = np.sum(points_measured * weights[:, None], axis=0)

    points_ref_centered = points_ref - centroid_ref
    points_measured_centered = points_measured - centroid_measured

    H = points_ref_centered.T @ (points_measured_centered * weights[:, None])
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_measured - R @ centroid_ref
    return R, t


def estimate_marker_geometry_from_distances(
        marker_trajectories: np.ndarray
) -> np.ndarray:
    """Estimate marker geometry using median pairwise distances and MDS."""
    n_frames, n_markers, _ = marker_trajectories.shape
    pairwise_distances = np.zeros((n_frames, n_markers, n_markers))

    for t in range(n_frames):
        for i in range(n_markers):
            for j in range(n_markers):
                pairwise_distances[t, i, j] = np.linalg.norm(
                    marker_trajectories[t, i] - marker_trajectories[t, j]
                )

    median_distances = np.median(pairwise_distances, axis=0)
    n = n_markers
    H = np.eye(n) - np.ones((n, n)) / n
    D_squared = median_distances ** 2
    B = -0.5 * H @ D_squared @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1][:3]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    marker_geometry = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    return marker_geometry


def initialize_trajectory(
        marker_trajectories: np.ndarray,
        marker_geometry: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize trajectory using per-frame rigid fitting."""
    n_frames = marker_trajectories.shape[0]
    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))

    for i in range(n_frames):
        R, t = fit_rigid_transform(
            points_ref=marker_geometry,
            points_measured=marker_trajectories[i]
        )
        rotations[i] = R
        translations[i] = t

    rotation_vectors = Rotation.from_matrix(rotations).as_rotvec()
    return rotation_vectors, translations


def pack_parameters(
        rotation_vectors: np.ndarray,
        translations: np.ndarray,
        marker_geometry: np.ndarray,
        fix_geometry: bool = False
) -> np.ndarray:
    """Pack optimization parameters into a single vector."""
    if fix_geometry:
        return np.concatenate([
            rotation_vectors.ravel(),
            translations.ravel()
        ])
    else:
        marker_geometry_free = marker_geometry[2:].copy()
        return np.concatenate([
            rotation_vectors.ravel(),
            translations.ravel(),
            marker_geometry_free.ravel()
        ])


def unpack_parameters(
        params: np.ndarray,
        n_frames: int,
        n_markers: int,
        fix_geometry: bool = False,
        initial_geometry: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack optimization parameters from vector."""
    idx = 0
    rotation_vectors = params[idx:idx + n_frames * 3].reshape(n_frames, 3)
    idx += n_frames * 3
    translations = params[idx:idx + n_frames * 3].reshape(n_frames, 3)
    idx += n_frames * 3

    if fix_geometry:
        marker_geometry = initial_geometry.copy()
    else:
        marker_geometry = np.zeros((n_markers, 3))
        marker_geometry[0] = [0, 0, 0]
        marker_geometry[1, 0] = initial_geometry[1, 0]
        marker_geometry[1, 1:] = 0
        marker_geometry_free = params[idx:].reshape(-1, 3)
        marker_geometry[2:] = marker_geometry_free

    return rotation_vectors, translations, marker_geometry


def compute_jacobian_sparsity(
        n_frames: int,
        n_markers: int,
        fix_geometry: bool = False
) -> csr_matrix:
    """Compute sparsity pattern of Jacobian matrix."""
    n_rot_params = n_frames * 3
    n_trans_params = n_frames * 3
    n_geom_params = 0 if fix_geometry else (n_markers - 2) * 3
    n_params = n_rot_params + n_trans_params + n_geom_params

    n_data_residuals = n_frames * n_markers * 3
    n_accel_residuals = (n_frames - 2) * 3 * 2
    n_jerk_residuals = (n_frames - 3) * 3 * 2
    n_geom_residuals = n_geom_params
    n_residuals = n_data_residuals + n_accel_residuals + n_jerk_residuals + n_geom_residuals

    sparsity = lil_matrix((n_residuals, n_params), dtype=bool)
    residual_idx = 0

    # Data fidelity residuals
    for t in range(n_frames):
        for m in range(n_markers):
            for xyz in range(3):
                rot_start = t * 3
                sparsity[residual_idx, rot_start:rot_start + 3] = True
                trans_start = n_rot_params + t * 3
                sparsity[residual_idx, trans_start:trans_start + 3] = True
                if not fix_geometry and m >= 2:
                    geom_start = n_rot_params + n_trans_params + (m - 2) * 3
                    sparsity[residual_idx, geom_start:geom_start + 3] = True
                residual_idx += 1

    # Translation acceleration
    for t in range(n_frames - 2):
        for xyz in range(3):
            trans_start = n_rot_params + t * 3
            sparsity[residual_idx, trans_start:trans_start + 9] = True
            residual_idx += 1

    # Translation jerk
    for t in range(n_frames - 3):
        for xyz in range(3):
            trans_start = n_rot_params + t * 3
            sparsity[residual_idx, trans_start:trans_start + 12] = True
            residual_idx += 1

    # Rotation acceleration
    for t in range(n_frames - 2):
        for xyz in range(3):
            rot_start = t * 3
            sparsity[residual_idx, rot_start:rot_start + 9] = True
            residual_idx += 1

    # Rotation jerk
    for t in range(n_frames - 3):
        for xyz in range(3):
            rot_start = t * 3
            sparsity[residual_idx, rot_start:rot_start + 12] = True
            residual_idx += 1

    # Geometry regularization
    if not fix_geometry:
        for i in range(n_geom_params):
            geom_start = n_rot_params + n_trans_params + i
            sparsity[residual_idx, geom_start] = True
            residual_idx += 1

    return sparsity.tocsr()[:residual_idx, :]


def apply_robust_loss_manual(
        residuals: np.ndarray,
        loss_type: str,
        delta: float
) -> np.ndarray:
    """Manually apply robust loss to residuals - ONLY for data fidelity."""
    if loss_type == 'linear':
        return residuals
    elif loss_type == 'huber':
        abs_residuals = np.abs(residuals)
        mask = abs_residuals <= delta
        result = residuals.copy()
        result[~mask] = delta * np.sign(residuals[~mask]) * np.sqrt(2 * abs_residuals[~mask] / delta - 1)
        return result
    elif loss_type == 'soft_l1':
        return residuals / np.sqrt(1 + (residuals / delta) ** 2)
    elif loss_type == 'cauchy':
        return residuals / (1 + (residuals / delta) ** 2)
    else:
        return residuals


def compute_residuals_fixed(
        params: np.ndarray,
        measured_trajectories: np.ndarray,
        n_frames: int,
        n_markers: int,
        dt: float,
        opt_params: OptimizationParams,
        fix_geometry: bool = False,
        initial_geometry: np.ndarray | None = None
) -> np.ndarray:
    """Compute residuals with robust loss ONLY on data residuals."""
    rotation_vectors, translations, marker_geometry = unpack_parameters(
        params=params,
        n_frames=n_frames,
        n_markers=n_markers,
        fix_geometry=fix_geometry,
        initial_geometry=initial_geometry
    )

    residuals_list = []

    # 1. Data fidelity - APPLY ROBUST LOSS HERE
    data_residuals = []
    for t in range(n_frames):
        R = Rotation.from_rotvec(rotation_vectors[t]).as_matrix()
        predicted = (R @ marker_geometry.T).T + translations[t]
        diff = measured_trajectories[t] - predicted
        data_residuals.append(diff.ravel())

    data_residuals = np.concatenate(data_residuals)

    # Apply robust loss to data residuals
    data_residuals_robust = apply_robust_loss_manual(
        residuals=data_residuals,
        loss_type=opt_params.robust_loss,
        delta=opt_params.robust_loss_delta
    )
    residuals_list.append(data_residuals_robust)

    # 2-6. Regularization terms - NO ROBUST LOSS

    if n_frames > 2 and opt_params.lambda_translation_accel > 0:
        trans_accel = np.diff(translations, n=2, axis=0) / (dt ** 2)
        weight = np.sqrt(opt_params.lambda_translation_accel)
        residuals_list.append((weight * trans_accel).ravel())

    if n_frames > 3 and opt_params.lambda_translation_jerk > 0:
        trans_jerk = np.diff(translations, n=3, axis=0) / (dt ** 3)
        weight = np.sqrt(opt_params.lambda_translation_jerk)
        residuals_list.append((weight * trans_jerk).ravel())

    if n_frames > 2 and opt_params.lambda_rotation_accel > 0:
        rot_accel = np.diff(rotation_vectors, n=2, axis=0) / (dt ** 2)
        weight = np.sqrt(opt_params.lambda_rotation_accel)
        residuals_list.append((weight * rot_accel).ravel())

    if n_frames > 3 and opt_params.lambda_rotation_jerk > 0:
        rot_jerk = np.diff(rotation_vectors, n=3, axis=0) / (dt ** 3)
        weight = np.sqrt(opt_params.lambda_rotation_jerk)
        residuals_list.append((weight * rot_jerk).ravel())

    if not fix_geometry and opt_params.lambda_geometry_regularization > 0:
        geom_diff = marker_geometry[2:] - initial_geometry[2:]
        weight = np.sqrt(opt_params.lambda_geometry_regularization)
        residuals_list.append((weight * geom_diff).ravel())

    return np.concatenate(residuals_list)


def diagnose_residuals(
        params: np.ndarray,
        measured_trajectories: np.ndarray,
        n_frames: int,
        n_markers: int,
        dt: float,
        opt_params: OptimizationParams,
        fix_geometry: bool,
        initial_geometry: np.ndarray
) -> None:
    """Detailed diagnostics of residual contributions."""
    rotation_vectors, translations, marker_geometry = unpack_parameters(
        params=params,
        n_frames=n_frames,
        n_markers=n_markers,
        fix_geometry=fix_geometry,
        initial_geometry=initial_geometry
    )

    print("=" * 70)
    print("RESIDUAL DIAGNOSTICS")
    print("=" * 70)

    # Data residuals (before robust loss)
    data_residuals = []
    for t in range(n_frames):
        R = Rotation.from_rotvec(rotation_vectors[t]).as_matrix()
        predicted = (R @ marker_geometry.T).T + translations[t]
        diff = measured_trajectories[t] - predicted
        data_residuals.append(diff.ravel())
    data_residuals = np.concatenate(data_residuals)

    print(f"\n1. Data residuals (raw, before robust loss):")
    print(f"   Count: {len(data_residuals)}")
    print(f"   Min: {np.min(data_residuals):.6e} m")
    print(f"   Max: {np.max(data_residuals):.6e} m")
    print(f"   Mean: {np.mean(np.abs(data_residuals)):.6e} m ({np.mean(np.abs(data_residuals)) * 1000:.2f} mm)")
    print(
        f"   RMS: {np.sqrt(np.mean(data_residuals ** 2)):.6e} m ({np.sqrt(np.mean(data_residuals ** 2)) * 1000:.2f} mm)")
    print(f"   Cost contribution: {np.sum(data_residuals ** 2):.6e}")

    # Data residuals after robust loss
    data_residuals_robust = apply_robust_loss_manual(
        residuals=data_residuals,
        loss_type=opt_params.robust_loss,
        delta=opt_params.robust_loss_delta
    )
    print(f"\n2. Data residuals (after {opt_params.robust_loss} loss):")
    print(f"   Cost contribution: {np.sum(data_residuals_robust ** 2):.6e}")

    # Translation acceleration
    if n_frames > 2:
        trans_accel = np.diff(translations, n=2, axis=0) / (dt ** 2)
        trans_accel_weighted = np.sqrt(opt_params.lambda_translation_accel) * trans_accel.ravel()
        print(f"\n3. Translation acceleration:")
        print(f"   Count: {len(trans_accel_weighted)}")
        print(f"   Lambda: {opt_params.lambda_translation_accel:.2e}")
        print(f"   RMS (unweighted): {np.sqrt(np.mean(trans_accel.ravel() ** 2)):.2f} m/s²")
        print(f"   Cost contribution: {np.sum(trans_accel_weighted ** 2):.6e}")

    # Translation jerk
    if n_frames > 3:
        trans_jerk = np.diff(translations, n=3, axis=0) / (dt ** 3)
        trans_jerk_weighted = np.sqrt(opt_params.lambda_translation_jerk) * trans_jerk.ravel()
        print(f"\n4. Translation jerk:")
        print(f"   Count: {len(trans_jerk_weighted)}")
        print(f"   Lambda: {opt_params.lambda_translation_jerk:.2e}")
        print(f"   RMS (unweighted): {np.sqrt(np.mean(trans_jerk.ravel() ** 2)):.2f} m/s³")
        print(f"   Cost contribution: {np.sum(trans_jerk_weighted ** 2):.6e}")

    # Rotation terms
    if n_frames > 2:
        rot_accel = np.diff(rotation_vectors, n=2, axis=0) / (dt ** 2)
        rot_accel_weighted = np.sqrt(opt_params.lambda_rotation_accel) * rot_accel.ravel()
        print(f"\n5. Rotation acceleration:")
        print(f"   Cost contribution: {np.sum(rot_accel_weighted ** 2):.6e}")

    if n_frames > 3:
        rot_jerk = np.diff(rotation_vectors, n=3, axis=0) / (dt ** 3)
        rot_jerk_weighted = np.sqrt(opt_params.lambda_rotation_jerk) * rot_jerk.ravel()
        print(f"\n6. Rotation jerk:")
        print(f"   Cost contribution: {np.sum(rot_jerk_weighted ** 2):.6e}")

    # Total cost
    all_residuals = compute_residuals_fixed(
        params=params,
        measured_trajectories=measured_trajectories,
        n_frames=n_frames,
        n_markers=n_markers,
        dt=dt,
        opt_params=opt_params,
        fix_geometry=fix_geometry,
        initial_geometry=initial_geometry
    )
    total_cost = np.sum(all_residuals ** 2)
    print(f"\n7. TOTAL COST: {total_cost:.6e}")
    print("=" * 70)
    print()


def joint_optimization(
        measured_trajectories: np.ndarray,
        dt: float,
        opt_params: OptimizationParams,
        optimize_geometry: bool = True,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Jointly optimize 6DOF trajectory and marker geometry."""
    n_frames, n_markers, _ = measured_trajectories.shape

    if verbose:
        print("=" * 70)
        print("PROPERLY FIXED JOINT OPTIMIZATION")
        print("=" * 70)
        print(f"Frames: {n_frames}, Markers: {n_markers}")
        print(f"Optimize geometry: {optimize_geometry}")
        print(f"Robust loss: {opt_params.robust_loss} (delta={opt_params.robust_loss_delta * 1000:.1f} mm)")
        print()

    # Step 1: Estimate initial geometry
    if verbose:
        print("Step 1: Estimating initial marker geometry...")
    initial_geometry = estimate_marker_geometry_from_distances(
        marker_trajectories=measured_trajectories
    )
    initial_geometry = initial_geometry - initial_geometry[0]

    if verbose:
        print("Initial marker geometry:")
        for i, pos in enumerate(initial_geometry):
            print(f"  Marker {i}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}] m")
        print()

    # Step 2: Initialize trajectory
    if verbose:
        print("Step 2: Initializing trajectory...")
    rotation_vectors_init, translations_init = initialize_trajectory(
        marker_trajectories=measured_trajectories,
        marker_geometry=initial_geometry
    )

    # Step 3: Compute Jacobian sparsity
    if verbose:
        print("Step 3: Computing sparse Jacobian structure...")
    t_jac_start = time.time()
    jac_sparsity = compute_jacobian_sparsity(
        n_frames=n_frames,
        n_markers=n_markers,
        fix_geometry=not optimize_geometry
    )
    t_jac_end = time.time()

    if verbose:
        sparsity_percent = 100.0 * jac_sparsity.nnz / (jac_sparsity.shape[0] * jac_sparsity.shape[1])
        print(f"  Jacobian: {jac_sparsity.shape}, {sparsity_percent:.4f}% non-zero")
        print(f"  Time: {t_jac_end - t_jac_start:.3f} s")
        print()

    # Step 4: Pack parameters
    params_init = pack_parameters(
        rotation_vectors=rotation_vectors_init,
        translations=translations_init,
        marker_geometry=initial_geometry,
        fix_geometry=not optimize_geometry
    )

    if verbose:
        print(f"Step 4: Total parameters: {len(params_init):,}")
        print()

        # DIAGNOSTICS: Check initial residuals
        diagnose_residuals(
            params=params_init,
            measured_trajectories=measured_trajectories,
            n_frames=n_frames,
            n_markers=n_markers,
            dt=dt,
            opt_params=opt_params,
            fix_geometry=not optimize_geometry,
            initial_geometry=initial_geometry
        )

    # Step 5: Optimize
    if verbose:
        print("Step 5: Running optimization...")
        print(f"  Regularization weights (MUCH WEAKER):")
        print(f"    - Translation accel: {opt_params.lambda_translation_accel:.2e}")
        print(f"    - Translation jerk: {opt_params.lambda_translation_jerk:.2e}")
        print(f"    - Rotation accel: {opt_params.lambda_rotation_accel:.2e}")
        print(f"    - Rotation jerk: {opt_params.lambda_rotation_jerk:.2e}")
        if optimize_geometry:
            print(f"    - Geometry reg: {opt_params.lambda_geometry_regularization:.2e}")
        print()

    t_opt_start = time.time()

    result = least_squares(
        fun=compute_residuals_fixed,
        x0=params_init,
        jac_sparsity=jac_sparsity,
        args=(
            measured_trajectories,
            n_frames,
            n_markers,
            dt,
            opt_params,
            not optimize_geometry,
            initial_geometry
        ),
        method='trf',
        loss='linear',  # We handle robust loss manually
        verbose=2 if verbose else 0,
        max_nfev=1000,  # Allow more iterations
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10
    )

    t_opt_end = time.time()

    if verbose:
        print()
        print("=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Message: {result.message}")
        print(f"Final cost: {result.cost:.6e}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Optimization time: {t_opt_end - t_opt_start:.2f} s")
        print(f"Optimality: {result.optimality:.2e}")

        # Show if we actually converged
        if result.optimality < 1e-4:
            print("✓ CONVERGED (optimality < 1e-4)")
        elif result.optimality < 1e-2:
            print("~ PARTIALLY CONVERGED (optimality < 1e-2)")
        else:
            print("✗ NOT CONVERGED (optimality > 1e-2)")
        print()

    # Unpack results
    rotation_vectors_opt, translations_opt, marker_geometry_opt = unpack_parameters(
        params=result.x,
        n_frames=n_frames,
        n_markers=n_markers,
        fix_geometry=not optimize_geometry,
        initial_geometry=initial_geometry
    )

    rotations_opt = Rotation.from_rotvec(rotation_vectors_opt).as_matrix()

    if optimize_geometry and verbose:
        print("Optimized marker geometry:")
        for i, pos in enumerate(marker_geometry_opt):
            print(f"  Marker {i}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}] m")
        print()
        print("Geometry change from initial:")
        for i, (init, opt) in enumerate(zip(initial_geometry, marker_geometry_opt)):
            change = np.linalg.norm(opt - init) * 1000
            print(f"  Marker {i}: {change:.2f} mm")
        print()

    info = {
        'success': result.success,
        'cost': result.cost,
        'nfev': result.nfev,
        'optimality': result.optimality,
        'optimization_time': t_opt_end - t_opt_start,
        'initial_geometry': initial_geometry,
    }

    return rotations_opt, translations_opt, marker_geometry_opt, info


def reconstruct_marker_trajectories(
        rotations: np.ndarray,
        translations: np.ndarray,
        marker_geometry: np.ndarray
) -> np.ndarray:
    """Reconstruct marker trajectories from rigid body motion."""
    n_frames = rotations.shape[0]
    n_markers = marker_geometry.shape[0]
    trajectories = np.zeros((n_frames, n_markers, 3))

    for i in range(n_frames):
        trajectories[i] = (rotations[i] @ marker_geometry.T).T + translations[i]

    return trajectories


def compute_metrics(
        measured: np.ndarray,
        reconstructed: np.ndarray,
        translations: np.ndarray,
        dt: float
) -> dict[str, float]:
    """Compute quality metrics."""
    diff = measured - reconstructed
    per_marker_errors = np.linalg.norm(diff, axis=2)

    trans_vel = np.diff(translations, n=1, axis=0) / dt
    trans_accel = np.diff(translations, n=2, axis=0) / (dt ** 2)
    trans_jerk = np.diff(translations, n=3, axis=0) / (dt ** 3)

    return {
        'mean_error_mm': np.mean(per_marker_errors) * 1000,
        'median_error_mm': np.median(per_marker_errors) * 1000,
        'max_error_mm': np.max(per_marker_errors) * 1000,
        'rms_error_mm': np.sqrt(np.mean(per_marker_errors ** 2)) * 1000,
        'p95_error_mm': np.percentile(per_marker_errors, 95) * 1000,
        'mean_velocity_m_s': np.mean(np.linalg.norm(trans_vel, axis=1)),
        'mean_accel_m_s2': np.mean(np.linalg.norm(trans_accel, axis=1)),
        'mean_jerk_m_s3': np.mean(np.linalg.norm(trans_jerk, axis=1)),
        'total_jerk': np.sum(np.linalg.norm(trans_jerk, axis=1))
    }


def run_demo(
        n_frames: int = 300,
        n_markers: int = 6,
        noise_level_mm: float = 4.0,
        outlier_probability: float = 0.02,
        outlier_magnitude_mm: float = 20.0
) -> None:
    """Run a complete demo comparing different methods."""
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("RIGID BODY TRACKING DEMO - PROPERLY FIXED VERSION")
    print("=" * 70)
    print(f"Generating synthetic data:")
    print(f"  - Frames: {n_frames}")
    print(f"  - Markers: {n_markers}")
    print(f"  - Noise: {noise_level_mm:.1f} mm RMS")
    print(f"  - Outliers: {outlier_probability * 100:.1f}% probability, {outlier_magnitude_mm:.1f} mm magnitude")
    print()

    # Create TRUE marker geometry
    true_marker_geometry = np.array([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.05, 0.08, 0.00],
        [0.02, 0.03, 0.07],
        [0.09, 0.07, 0.02],
        [0.03, 0.08, 0.08]
    ])[:n_markers]

    # Generate smooth ground truth trajectory
    dt = 1.0 / 90.0  # 90 Hz
    times = np.arange(n_frames) * dt

    # Complex motion
    angles = np.column_stack([
        0.6 * np.sin(2 * np.pi * 0.4 * times),
        0.4 * np.sin(2 * np.pi * 0.5 * times + 0.5),
        0.3 * np.sin(2 * np.pi * 0.3 * times + 1.0)
    ])

    true_rotations = Rotation.from_rotvec(angles).as_matrix()

    true_translations = np.column_stack([
        0.3 * np.sin(2 * np.pi * 0.3 * times),
        0.2 * np.cos(2 * np.pi * 0.4 * times),
        0.15 * times + 0.1 * np.sin(2 * np.pi * 0.2 * times)
    ])

    # Generate clean marker trajectories
    clean_markers = reconstruct_marker_trajectories(
        rotations=true_rotations,
        translations=true_translations,
        marker_geometry=true_marker_geometry
    )

    # Add realistic noise
    noise_level = noise_level_mm / 1000.0
    noisy_markers = clean_markers + np.random.randn(*clean_markers.shape) * noise_level

    # Add outliers
    n_outliers = int(n_frames * n_markers * outlier_probability)
    outlier_indices = np.random.choice(n_frames * n_markers, size=n_outliers, replace=False)

    for idx in outlier_indices:
        frame_idx = idx // n_markers
        marker_idx = idx % n_markers
        outlier_direction = np.random.randn(3)
        outlier_direction /= np.linalg.norm(outlier_direction)
        noisy_markers[frame_idx, marker_idx] += outlier_direction * (outlier_magnitude_mm / 1000.0)

    print(f"Added {n_outliers} outliers to data")
    print()

    # Method 1: Per-frame fitting only (baseline)
    print("-" * 70)
    print("METHOD 1: Per-frame fitting (no optimization)")
    print("-" * 70)

    rotation_vectors_init, translations_init = initialize_trajectory(
        marker_trajectories=noisy_markers,
        marker_geometry=true_marker_geometry
    )
    rotations_init = Rotation.from_rotvec(rotation_vectors_init).as_matrix()

    reconstructed_init = reconstruct_marker_trajectories(
        rotations=rotations_init,
        translations=translations_init,
        marker_geometry=true_marker_geometry
    )

    metrics_init = compute_metrics(
        measured=noisy_markers,
        reconstructed=reconstructed_init,
        translations=translations_init,
        dt=dt
    )

    print("Results:")
    for key, value in metrics_init.items():
        print(f"  {key}: {value:.4f}")
    print()

    # Method 2: PROPERLY FIXED joint optimization WITHOUT geometry estimation
    print("-" * 70)
    print("METHOD 2: PROPERLY FIXED joint optimization (fixed geometry)")
    print("-" * 70)

    opt_params_fixed_geom = OptimizationParams(
        lambda_translation_accel=0.1,
        lambda_translation_jerk=1.0,
        lambda_rotation_accel=0.1,
        lambda_rotation_jerk=0.5,
        robust_loss='huber',
        robust_loss_delta=0.010
    )

    rotations_opt_fixed, translations_opt_fixed, _, info_fixed = joint_optimization(
        measured_trajectories=noisy_markers,
        dt=dt,
        opt_params=opt_params_fixed_geom,
        optimize_geometry=False,
        verbose=True
    )

    reconstructed_opt_fixed = reconstruct_marker_trajectories(
        rotations=rotations_opt_fixed,
        translations=translations_opt_fixed,
        marker_geometry=true_marker_geometry
    )

    metrics_opt_fixed = compute_metrics(
        measured=noisy_markers,
        reconstructed=reconstructed_opt_fixed,
        translations=translations_opt_fixed,
        dt=dt
    )

    print("Results:")
    for key, value in metrics_opt_fixed.items():
        print(f"  {key}: {value:.4f}")
    print()

    # Method 3: PROPERLY FIXED FULL joint optimization WITH geometry estimation
    print("-" * 70)
    print("METHOD 3: PROPERLY FIXED FULL joint optimization (with geometry estimation)")
    print("-" * 70)

    opt_params_full = OptimizationParams(
        lambda_translation_accel=0.1,
        lambda_translation_jerk=1.0,
        lambda_rotation_accel=0.1,
        lambda_rotation_jerk=0.5,
        lambda_geometry_regularization=0.01,
        robust_loss='huber',
        robust_loss_delta=0.010
    )

    rotations_opt_full, translations_opt_full, geometry_opt_full, info_full = joint_optimization(
        measured_trajectories=noisy_markers,
        dt=dt,
        opt_params=opt_params_full,
        optimize_geometry=True,
        verbose=True
    )

    reconstructed_opt_full = reconstruct_marker_trajectories(
        rotations=rotations_opt_full,
        translations=translations_opt_full,
        marker_geometry=geometry_opt_full
    )

    metrics_opt_full = compute_metrics(
        measured=noisy_markers,
        reconstructed=reconstructed_opt_full,
        translations=translations_opt_full,
        dt=dt
    )

    print("Results:")
    for key, value in metrics_opt_full.items():
        print(f"  {key}: {value:.4f}")
    print()

    # Compare geometry estimates
    print("Geometry estimation quality:")
    geom_errors = np.linalg.norm(true_marker_geometry - geometry_opt_full, axis=1) * 1000
    for i, err in enumerate(geom_errors):
        print(f"  Marker {i}: {err:.2f} mm error")
    print(f"  Mean geometry error: {np.mean(geom_errors):.2f} mm")
    print()

    # Summary comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'Method 1':<15} {'Method 2':<15} {'Method 3':<15}")
    print("-" * 70)

    for key in ['mean_error_mm', 'rms_error_mm', 'max_error_mm', 'mean_jerk_m_s3']:
        v1 = metrics_init[key]
        v2 = metrics_opt_fixed[key]
        v3 = metrics_opt_full[key]
        print(f"{key:<30} {v1:>14.4f} {v2:>14.4f} {v3:>14.4f}")

    print(
        f"{'Optimization time (s)':<30} {0.0:>14.2f} {info_fixed['optimization_time']:>14.2f} {info_full['optimization_time']:>14.2f}")
    print(f"{'Final optimality':<30} {'N/A':>14} {info_fixed['optimality']:>14.2e} {info_full['optimality']:>14.2e}")
    print()

    # Show improvements
    print("IMPROVEMENTS OVER BASELINE:")
    if metrics_opt_fixed['mean_error_mm'] < metrics_init['mean_error_mm']:
        improvement = (1 - metrics_opt_fixed['mean_error_mm'] / metrics_init['mean_error_mm']) * 100
        print(f"  Method 2: {improvement:.1f}% better mean error ✓")
    else:
        worsening = (metrics_opt_fixed['mean_error_mm'] / metrics_init['mean_error_mm'] - 1) * 100
        print(f"  Method 2: {worsening:.1f}% WORSE mean error ✗")

    if metrics_opt_full['mean_error_mm'] < metrics_init['mean_error_mm']:
        improvement = (1 - metrics_opt_full['mean_error_mm'] / metrics_init['mean_error_mm']) * 100
        print(f"  Method 3: {improvement:.1f}% better mean error ✓")
    else:
        worsening = (metrics_opt_full['mean_error_mm'] / metrics_init['mean_error_mm'] - 1) * 100
        print(f"  Method 3: {worsening:.1f}% WORSE mean error ✗")
    print()


if __name__ == "__main__":
    run_demo(
        n_frames=300,
        n_markers=6,
        noise_level_mm=4.0,
        outlier_probability=0.02,
        outlier_magnitude_mm=20.0
    )