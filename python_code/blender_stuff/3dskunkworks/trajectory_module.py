"""
Trajectory Optimization Module - QUATERNION SIGN CONSISTENCY FIX
================================================================

Key improvements:
1. Quaternion sign consistency penalty
2. Post-optimization unwrapping
3. Stronger temporal coherence

Author: AI Assistant
Date: 2025
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from scipy.spatial.transform import Rotation
from scipy.linalg import svd
from scipy.optimize import minimize
import logging
import time
from typing import Optional

from geometry_module import (
    rotation_matrix_from_rotvec_jax,
    unwrap_rotation_vectors_advanced,
    apply_butterworth_filter,
    slerp_smooth_rotations
)

logger = logging.getLogger(__name__)


# =============================================================================
# KABSCH ALGORITHM (unchanged)
# =============================================================================

def fit_rigid_transform_kabsch(
    *,
    reference: np.ndarray,
    measured: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit optimal rigid transformation using Kabsch algorithm."""
    ref_centroid = np.mean(reference, axis=0)
    meas_centroid = np.mean(measured, axis=0)

    ref_centered = reference - ref_centroid
    meas_centered = measured - meas_centroid

    H = ref_centered.T @ meas_centered
    U, S, Vt = svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = meas_centroid - R @ ref_centroid
    return R, t


def kabsch_initialization(
    *,
    noisy_measurements: np.ndarray,
    reference_geometry: np.ndarray,
    apply_slerp: bool = True,
    slerp_window: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize trajectory with enhanced quaternion consistency."""
    logger.info("Kabsch initialization with quaternion consistency...")

    n_frames = noisy_measurements.shape[0]
    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))

    for i in range(n_frames):
        measured_vertices = noisy_measurements[i, :8, :]
        R, t = fit_rigid_transform_kabsch(
            reference=reference_geometry,
            measured=measured_vertices
        )
        rotations[i] = R
        translations[i] = t

    # Convert to quaternions for consistency check
    logger.info("  Enforcing quaternion sign consistency...")
    quats = np.array([Rotation.from_matrix(R).as_quat() for R in rotations])

    # Enforce same hemisphere: q·q_prev > 0
    for i in range(1, n_frames):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]

    # Convert back to rotation matrices
    rotations = np.array([Rotation.from_quat(q).as_matrix() for q in quats])

    # Now do rotation vector unwrapping
    logger.info("  Unwrapping rotation vectors...")
    rotvecs = np.array([Rotation.from_matrix(R).as_rotvec() for R in rotations])
    rotvecs_unwrapped = unwrap_rotation_vectors_advanced(rotvecs=rotvecs)
    rotations = np.array([Rotation.from_rotvec(rv).as_matrix() for rv in rotvecs_unwrapped])

    if apply_slerp:
        logger.info(f"  Applying SLERP smoothing (window={slerp_window})...")
        rotations = slerp_smooth_rotations(
            rotations=rotations,
            window_size=slerp_window
        )
        translations = apply_butterworth_filter(
            data=translations,
            cutoff_freq=0.2,
            sampling_rate=1.0,
            order=4
        )

    return rotations, translations


# =============================================================================
# POSE VECTOR CONVERSION
# =============================================================================

def poses_to_vector(
    *,
    rotations: np.ndarray,
    translations: np.ndarray
) -> np.ndarray:
    """Convert rotations and translations to optimization vector."""
    n_frames = rotations.shape[0]
    x = np.zeros(n_frames * 6)

    rotvecs = np.array([
        Rotation.from_matrix(R).as_rotvec() for R in rotations
    ])

    rotvecs = unwrap_rotation_vectors_advanced(rotvecs=rotvecs)

    for i in range(n_frames):
        x[i * 6:i * 6 + 3] = rotvecs[i]
        x[i * 6 + 3:i * 6 + 6] = translations[i]

    return x


def vector_to_poses_numpy(
    *,
    x: np.ndarray,
    n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert optimization vector back to rotations and translations."""
    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))

    for i in range(n_frames):
        rotvec = x[i * 6:i * 6 + 3]
        rotations[i] = Rotation.from_rotvec(rotvec).as_matrix()
        translations[i] = x[i * 6 + 3:i * 6 + 6]

    return rotations, translations


# =============================================================================
# JAX OPTIMIZATION - ENHANCED
# =============================================================================

@jit
def rotvec_to_quat_jax(*, rotvec: jnp.ndarray) -> jnp.ndarray:
    """
    Convert rotation vector to quaternion (w, x, y, z).

    Args:
        rotvec: Rotation vector (3,)

    Returns:
        Quaternion (4,) in [w, x, y, z] order
    """
    angle = jnp.linalg.norm(rotvec)
    small_angle = angle < 1e-8

    half_angle = angle / 2.0
    sinc_half = jnp.where(
        small_angle,
        0.5,  # lim(sin(x/2)/(x/2)) as x->0 = 1, so sin(x/2)/x = 0.5
        jnp.sin(half_angle) / angle
    )

    w = jnp.cos(half_angle)
    xyz = rotvec * sinc_half

    return jnp.array([w, xyz[0], xyz[1], xyz[2]])


@jit
def quaternion_dot_product_jax(*, q1: jnp.ndarray, q2: jnp.ndarray) -> float:
    """Compute dot product of two quaternions."""
    return jnp.dot(q1, q2)


@jit
def rotation_geodesic_distance_jax(
    *,
    rotvec1: jnp.ndarray,
    rotvec2: jnp.ndarray
) -> float:
    """Compute TRUE geodesic distance on SO(3)."""
    R1 = rotation_matrix_from_rotvec_jax(rotvec=rotvec1)
    R2 = rotation_matrix_from_rotvec_jax(rotvec=rotvec2)

    R_rel = R1.T @ R2
    trace = jnp.trace(R_rel)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    angle = jnp.arccos(cos_angle)

    return angle


@jit
def trajectory_objective_jax(
    *,
    x: jnp.ndarray,
    noisy_measurements: jnp.ndarray,
    reference_geometry: jnp.ndarray,
    lambda_data: float,
    lambda_smooth_pos: float,
    lambda_smooth_rot: float,
    lambda_accel: float,
    lambda_rot_geodesic: float,
    lambda_max_rotation: float,
    lambda_quat_consistency: float,
    max_rotation_per_frame: float
) -> float:
    """
    🔧 ENHANCED: Added quaternion sign consistency term!

    New term penalizes when consecutive quaternions are in opposite hemispheres,
    preventing the 360° spin artifact.
    """
    n_frames = noisy_measurements.shape[0]
    total_cost = 0.0

    # ===== DATA TERM =====
    data_cost = 0.0
    for i in range(n_frames):
        rotvec = x[i * 6:i * 6 + 3]
        trans = x[i * 6 + 3:i * 6 + 6]

        R = rotation_matrix_from_rotvec_jax(rotvec=rotvec)
        reconstructed = jnp.dot(reference_geometry, R.T) + trans
        residuals = reconstructed - noisy_measurements[i, :8, :]
        data_cost += jnp.sum(residuals ** 2)

    total_cost += lambda_data * data_cost

    # ===== POSITION SMOOTHNESS =====
    if n_frames > 2:
        translations = x.reshape(n_frames, 6)[:, 3:6]
        pos_vel = translations[1:] - translations[:-1]
        pos_accel = pos_vel[1:] - pos_vel[:-1]
        total_cost += lambda_smooth_pos * jnp.sum(pos_accel ** 2)

    # ===== GEODESIC ROTATION SMOOTHNESS =====
    if n_frames > 1:
        geodesic_cost = 0.0
        for i in range(n_frames - 1):
            rotvec1 = x[i * 6:i * 6 + 3]
            rotvec2 = x[(i + 1) * 6:(i + 1) * 6 + 3]

            geodesic_dist = rotation_geodesic_distance_jax(
                rotvec1=rotvec1,
                rotvec2=rotvec2
            )
            geodesic_cost += geodesic_dist ** 2

        total_cost += lambda_rot_geodesic * geodesic_cost

    # ===== 🔧 NEW: QUATERNION SIGN CONSISTENCY =====
    if n_frames > 1:
        quat_consistency_cost = 0.0
        for i in range(n_frames - 1):
            rotvec1 = x[i * 6:i * 6 + 3]
            rotvec2 = x[(i + 1) * 6:(i + 1) * 6 + 3]

            q1 = rotvec_to_quat_jax(rotvec=rotvec1)
            q2 = rotvec_to_quat_jax(rotvec=rotvec2)

            # Dot product should be positive (same hemisphere)
            # Penalize when negative (opposite hemisphere = potential flip)
            dot = quaternion_dot_product_jax(q1=q1, q2=q2)

            # Smooth penalty function:
            # - If dot > 0: small penalty
            # - If dot < 0: large penalty (quadratic)
            penalty = jnp.maximum(0.0, -dot) ** 2
            quat_consistency_cost += penalty

        total_cost += lambda_quat_consistency * quat_consistency_cost

    # ===== LARGE ROTATION PENALTY =====
    if n_frames > 1:
        spin_penalty = 0.0
        for i in range(n_frames - 1):
            rotvec1 = x[i * 6:i * 6 + 3]
            rotvec2 = x[(i + 1) * 6:(i + 1) * 6 + 3]

            geodesic_dist = rotation_geodesic_distance_jax(
                rotvec1=rotvec1,
                rotvec2=rotvec2
            )

            excess = jnp.maximum(0.0, geodesic_dist - max_rotation_per_frame)
            spin_penalty += excess ** 2

        total_cost += lambda_max_rotation * spin_penalty

    # ===== ROTATION TANGENT SMOOTHNESS =====
    if n_frames > 2:
        rotvecs = x.reshape(n_frames, 6)[:, 0:3]
        rot_vel = rotvecs[1:] - rotvecs[:-1]
        rot_accel = rot_vel[1:] - rot_vel[:-1]
        total_cost += lambda_smooth_rot * jnp.sum(rot_accel ** 2)

    # ===== POSITION JERK =====
    if n_frames > 3:
        translations = x.reshape(n_frames, 6)[:, 3:6]
        pos_vel = translations[1:] - translations[:-1]
        pos_acc = pos_vel[1:] - pos_vel[:-1]
        pos_jerk = pos_acc[1:] - pos_acc[:-1]
        total_cost += lambda_accel * jnp.sum(pos_jerk ** 2)

    return total_cost


def post_optimization_unwrap(
    *,
    rotations: np.ndarray,
    translations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    🔧 NEW: Post-optimization unwrapping pass to fix any remaining flips.

    This is CRITICAL - even after optimization, there might be residual
    quaternion flips that cause 360° spins.
    """
    logger.info("Post-optimization unwrapping...")

    # Convert to quaternions
    quats = np.array([Rotation.from_matrix(R).as_quat() for R in rotations])

    # Enforce quaternion consistency
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]

    # Convert back to rotation matrices
    rotations_fixed = np.array([Rotation.from_quat(q).as_matrix() for q in quats])

    # Also unwrap rotation vectors
    rotvecs = np.array([Rotation.from_matrix(R).as_rotvec() for R in rotations_fixed])
    rotvecs_unwrapped = unwrap_rotation_vectors_advanced(rotvecs=rotvecs)
    rotations_fixed = np.array([Rotation.from_rotvec(rv).as_matrix() for rv in rotvecs_unwrapped])

    logger.info("  Post-unwrap complete")

    return rotations_fixed, translations


def optimize_trajectory_jax(
    *,
    noisy_measurements: np.ndarray,
    reference_geometry: np.ndarray,
    lambda_data: float = 1.0,
    lambda_smooth_pos: float = 30.0,
    lambda_smooth_rot: float = 15.0,
    lambda_accel: float = 3.0,
    lambda_rot_geodesic: float = 300.0,  # INCREASED from 200
    lambda_max_rotation: float = 2000.0,  # INCREASED from 1000
    lambda_quat_consistency: float = 500.0,  # 🔧 NEW!
    max_rotation_per_frame_deg: float = 20.0,  # REDUCED from 25
    apply_slerp_smoothing: bool = True,
    slerp_window: int = 5,
    max_iter: int = 400
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🔧 FINAL FIX: Added quaternion consistency term and post-unwrapping!
    """
    n_frames = noisy_measurements.shape[0]

    logger.info("=" * 80)
    logger.info("TRAJECTORY OPTIMIZATION (QUATERNION CONSISTENCY FIX)")
    logger.info("=" * 80)

    init_rotations, init_translations = kabsch_initialization(
        noisy_measurements=noisy_measurements,
        reference_geometry=reference_geometry,
        apply_slerp=True,
        slerp_window=7
    )

    x0 = poses_to_vector(rotations=init_rotations, translations=init_translations)

    noisy_measurements_jax = jnp.array(noisy_measurements)
    reference_geometry_jax = jnp.array(reference_geometry)
    max_rotation_per_frame_rad = np.deg2rad(max_rotation_per_frame_deg)

    logger.info(f"\nOptimization configuration:")
    logger.info(f"  Frames:      {n_frames}")
    logger.info(f"  Parameters:  {len(x0)} (6 per frame)")
    logger.info(f"  Max iters:   {max_iter}")
    logger.info(f"\n🔧 ENHANCED regularization weights:")
    logger.info(f"  λ_data:          {lambda_data:6.1f}")
    logger.info(f"  λ_pos:           {lambda_smooth_pos:6.1f}")
    logger.info(f"  λ_rot_tang:      {lambda_smooth_rot:6.1f}")
    logger.info(f"  λ_rot_geo:       {lambda_rot_geodesic:6.1f}  ⭐ (INCREASED)")
    logger.info(f"  λ_max_rot:       {lambda_max_rotation:6.1f}  🚫 (DOUBLED)")
    logger.info(f"  λ_quat_cons:     {lambda_quat_consistency:6.1f}  🆕 (flip killer)")
    logger.info(f"  λ_accel:         {lambda_accel:6.1f}")
    logger.info(f"  max_rot/frame:   {max_rotation_per_frame_deg:.1f}° 🎯 (STRICTER)")

    @jit
    def objective_fn(x: jnp.ndarray) -> float:
        return trajectory_objective_jax(
            x=x,
            noisy_measurements=noisy_measurements_jax,
            reference_geometry=reference_geometry_jax,
            lambda_data=lambda_data,
            lambda_smooth_pos=lambda_smooth_pos,
            lambda_smooth_rot=lambda_smooth_rot,
            lambda_accel=lambda_accel,
            lambda_rot_geodesic=lambda_rot_geodesic,
            lambda_max_rotation=lambda_max_rotation,
            lambda_quat_consistency=lambda_quat_consistency,
            max_rotation_per_frame=max_rotation_per_frame_rad
        )

    gradient_fn = jit(grad(objective_fn))

    logger.info("\nJIT compilation...")
    warmup_start = time.time()
    _ = objective_fn(jnp.array(x0))
    obj_time = time.time() - warmup_start
    logger.info(f"  Objective: {obj_time:.1f}s")

    warmup_start = time.time()
    _ = gradient_fn(jnp.array(x0))
    grad_time = time.time() - warmup_start
    logger.info(f"  Gradient:  {grad_time:.1f}s")

    iteration_count = [0]
    last_print = [0]

    def callback(xk: np.ndarray) -> None:
        iteration_count[0] += 1
        if iteration_count[0] - last_print[0] >= 10:
            cost = float(objective_fn(jnp.array(xk)))
            logger.info(f"  Iter {iteration_count[0]:3d}: cost = {cost:.2f}")
            last_print[0] = iteration_count[0]

    def objective_and_grad_wrapper(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_jax = jnp.array(x)
        cost = float(objective_fn(x_jax))
        grad_jax = gradient_fn(x_jax)
        return cost, np.array(grad_jax)

    logger.info("\nRunning L-BFGS-B optimization...")

    opt_start = time.time()
    result = minimize(
        fun=objective_and_grad_wrapper,
        x0=x0,
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': max_iter,
            'ftol': 1e-9,
            'gtol': 1e-7,
            'maxcor': 20,
            'maxls': 50
        },
        callback=callback
    )
    opt_time = time.time() - opt_start

    logger.info(f"\n{'Converged' if result.success else 'Completed'}!")
    logger.info(f"  Final cost:  {result.fun:.2f}")
    logger.info(f"  Iterations:  {iteration_count[0]}")
    logger.info(f"  Time:        {opt_time:.1f}s")

    opt_rotations, opt_translations = vector_to_poses_numpy(
        x=result.x,
        n_frames=n_frames
    )

    # 🔧 NEW: Post-optimization unwrapping!
    opt_rotations, opt_translations = post_optimization_unwrap(
        rotations=opt_rotations,
        translations=opt_translations
    )

    # Save pre-filter version
    opt_rotations_no_filter = opt_rotations.copy()
    opt_translations_no_filter = opt_translations.copy()

    # Reconstruct without filter
    reconstructed_no_filter = np.zeros((n_frames, 9, 3))
    for i in range(n_frames):
        vertices = (opt_rotations_no_filter[i] @ reference_geometry.T).T + \
                   opt_translations_no_filter[i]
        reconstructed_no_filter[i, :8, :] = vertices
        reconstructed_no_filter[i, 8, :] = np.mean(vertices, axis=0)

    # Apply smoothing
    if apply_slerp_smoothing:
        logger.info("\nPost-optimization smoothing...")
        opt_rotations = slerp_smooth_rotations(
            rotations=opt_rotations,
            window_size=slerp_window
        )
        opt_translations = apply_butterworth_filter(
            data=opt_translations,
            cutoff_freq=0.15,
            sampling_rate=1.0,
            order=4
        )

    # Reconstruct with smoothing
    reconstructed = np.zeros((n_frames, 9, 3))
    for i in range(n_frames):
        vertices = (opt_rotations[i] @ reference_geometry.T).T + opt_translations[i]
        reconstructed[i, :8, :] = vertices
        reconstructed[i, 8, :] = np.mean(vertices, axis=0)

    return (opt_rotations, opt_translations, reconstructed,
            opt_rotations_no_filter, opt_translations_no_filter, reconstructed_no_filter)