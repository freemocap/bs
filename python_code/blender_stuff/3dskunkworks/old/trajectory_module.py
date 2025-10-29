"""
Trajectory Optimization Module
"""

import numpy as np
import torch
import logging
import time

from scipy.spatial.transform._rotation import Rotation

from geometry_module import (
    unwrap_rotation_vectors_advanced,
    apply_butterworth_filter,
    slerp_smooth_rotations
)

logger = logging.getLogger(__name__)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"PyTorch using device: {DEVICE}")


# =============================================================================
# KABSCH ALGORITHM (FIXED!)
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
    U, S, Vt = np.linalg.svd(H)  # ✅ FIXED: Use numpy's svd, not torch
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
    """
    Initialize trajectory with enhanced quaternion consistency.

    🔧 FIXED: Now uses conservative unwrapping after quaternion sign fixing
    to prevent spinning artifacts from the Kabsch initialization!
    """
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

    # Enforce same hemisphere
    for i in range(1, n_frames):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]

    # Convert back to rotation matrices
    rotations = np.array([Rotation.from_quat(q).as_matrix() for q in quats])

    # Conservative unwrapping (after quaternion fix, just handle 2π wraps)
    logger.info("  Conservative unwrapping (post-quat-fix)...")
    rotvecs = np.array([Rotation.from_matrix(R).as_rotvec() for R in rotations])
    rotvecs_unwrapped = unwrap_rotation_vectors_advanced(
        rotvecs=rotvecs,
        after_quat_fix=True  # 🔧 Use conservative mode - this is the key fix!
    )
    rotations = np.array([Rotation.from_rotvec(rv).as_matrix() for rv in rotvecs_unwrapped])

    if apply_slerp:
        logger.info(f"  Applying SLERP smoothing (window={slerp_window})...")
        rotations = slerp_smooth_rotations(
            rotations=rotations,
            window_size=slerp_window
        )
        translations = apply_butterworth_filter(
            data=translations,
            cutoff_freq_hz=10.0,
            sampling_rate_hz=100.0,
            order=4
        )

    return rotations, translations



def rotation_matrix_from_rotvec_torch_batched(*, rotvecs: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of rotation vectors to rotation matrices (VECTORIZED!)

    Args:
        rotvecs: (n_frames, 3) rotation vectors

    Returns:
        (n_frames, 3, 3) rotation matrices
    """
    angles = torch.linalg.norm(rotvecs, dim=1, keepdim=True)  # (n_frames, 1)
    small_angle = angles < 1e-8

    # Normalized axes with safe fallback
    axes = torch.where(
        small_angle.expand(-1, 3),
        torch.zeros_like(rotvecs),
        rotvecs / (angles + 1e-10)
    )

    # Rodrigues formula vectorized
    n_frames = rotvecs.shape[0]

    # Cross product matrices K (vectorized)
    K = torch.zeros(n_frames, 3, 3, dtype=rotvecs.dtype, device=rotvecs.device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    I = torch.eye(3, dtype=rotvecs.dtype, device=rotvecs.device).unsqueeze(0).expand(n_frames, -1, -1)

    sin_angle = torch.sin(angles).unsqueeze(2)  # (n_frames, 1, 1)
    cos_angle = torch.cos(angles).unsqueeze(2)

    R = I + sin_angle * K + (1 - cos_angle) * torch.bmm(K, K)

    # Handle small angles
    R = torch.where(small_angle.unsqueeze(2).expand(-1, 3, 3), I, R)

    return R


def rotvec_to_quat_torch_batched(*, rotvecs: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of rotation vectors to quaternions (VECTORIZED!)

    Args:
        rotvecs: (n_frames, 3)

    Returns:
        (n_frames, 4) quaternions in [w, x, y, z] order
    """
    angles = torch.linalg.norm(rotvecs, dim=1)  # (n_frames,)
    small_angle = angles < 1e-8

    half_angles = angles / 2.0
    sinc_half = torch.where(
        small_angle,
        torch.tensor(0.5, dtype=rotvecs.dtype, device=rotvecs.device),
        torch.sin(half_angles) / (angles + 1e-10)
    )

    w = torch.cos(half_angles)
    xyz = rotvecs * sinc_half.unsqueeze(1)

    return torch.cat([w.unsqueeze(1), xyz], dim=1)


def trajectory_objective_vectorized(
    *,
    x: torch.Tensor,
    noisy_measurements: torch.Tensor,
    reference_geometry: torch.Tensor,
    reference_rotvec: torch.Tensor,
    lambda_data: float,
    lambda_smooth_pos: float,
    lambda_smooth_rot: float,
    lambda_accel: float,
    lambda_rot_geodesic: float,
    lambda_max_rotation: float,
    lambda_quat_consistency: float,
    lambda_orientation_anchor: float,
    lambda_rot_jerk: float,
    max_rotation_per_frame: float
) -> torch.Tensor:
    """
    🚀 VECTORIZED objective function - NO LOOPS!
    """
    n_frames = noisy_measurements.shape[0]

    # Reshape x into rotations and translations
    x_reshaped = x.reshape(n_frames, 6)
    rotvecs = x_reshaped[:, :3]  # (n_frames, 3)
    translations = x_reshaped[:, 3:]  # (n_frames, 3)

    total_cost = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    # ===== DATA TERM (VECTORIZED) =====
    R_batch = rotation_matrix_from_rotvec_torch_batched(rotvecs=rotvecs)  # (n_frames, 3, 3)

    # Reconstruct all frames at once: (n_frames, 8, 3) = (8, 3) @ (n_frames, 3, 3)^T + (n_frames, 1, 3)
    reconstructed = torch.matmul(
        reference_geometry.unsqueeze(0).expand(n_frames, -1, -1),  # (n_frames, 8, 3)
        R_batch.transpose(1, 2)  # (n_frames, 3, 3)
    ) + translations.unsqueeze(1)  # (n_frames, 1, 3) broadcasts to (n_frames, 8, 3)

    residuals = reconstructed - noisy_measurements[:, :8, :]
    data_cost = torch.sum(residuals ** 2)
    total_cost += lambda_data * data_cost

    # ===== ABSOLUTE ORIENTATION ANCHORING (VECTORIZED) =====
    if lambda_orientation_anchor > 0:
        # Geodesic distance from reference orientation for all frames
        R_ref = rotation_matrix_from_rotvec_torch_batched(
            rotvecs=reference_rotvec.unsqueeze(0).expand(n_frames, -1)
        )  # (n_frames, 3, 3)

        # Relative rotation: R_ref^T @ R_batch
        R_rel = torch.bmm(R_ref.transpose(1, 2), R_batch)  # (n_frames, 3, 3)

        # Trace for all frames
        traces = torch.diagonal(R_rel, dim1=1, dim2=2).sum(dim=1)  # (n_frames,)
        cos_angles = torch.clamp((traces - 1.0) / 2.0, -1.0, 1.0)
        angles = torch.acos(cos_angles)

        orientation_cost = torch.sum(angles ** 2)
        total_cost += lambda_orientation_anchor * orientation_cost

    # ===== POSITION SMOOTHNESS (VECTORIZED) =====
    if n_frames > 2:
        pos_vel = translations[1:] - translations[:-1]  # (n_frames-1, 3)
        pos_accel = pos_vel[1:] - pos_vel[:-1]  # (n_frames-2, 3)
        total_cost += lambda_smooth_pos * torch.sum(pos_accel ** 2)

    # ===== GEODESIC ROTATION SMOOTHNESS (VECTORIZED) =====
    if n_frames > 1:
        R_current = R_batch[:-1]  # (n_frames-1, 3, 3)
        R_next = R_batch[1:]  # (n_frames-1, 3, 3)

        R_rel = torch.bmm(R_current.transpose(1, 2), R_next)  # (n_frames-1, 3, 3)
        traces = torch.diagonal(R_rel, dim1=1, dim2=2).sum(dim=1)  # (n_frames-1,)
        cos_angles = torch.clamp((traces - 1.0) / 2.0, -1.0, 1.0)
        geodesic_dists = torch.acos(cos_angles)

        geodesic_cost = torch.sum(geodesic_dists ** 2)
        total_cost += lambda_rot_geodesic * geodesic_cost

        # LARGE ROTATION PENALTY
        excess = torch.maximum(
            torch.zeros_like(geodesic_dists),
            geodesic_dists - max_rotation_per_frame
        )
        spin_penalty = torch.sum(excess ** 2)
        total_cost += lambda_max_rotation * spin_penalty

    # ===== ROTATION JERK (VECTORIZED) =====
    if n_frames > 3 and lambda_rot_jerk > 0:
        vel1 = rotvecs[1:-2] - rotvecs[:-3]
        vel2 = rotvecs[2:-1] - rotvecs[1:-2]
        vel3 = rotvecs[3:] - rotvecs[2:-1]

        accel1 = vel2 - vel1
        accel2 = vel3 - vel2

        jerk = accel2 - accel1
        jerk_cost = torch.sum(jerk ** 2)
        total_cost += lambda_rot_jerk * jerk_cost

    # ===== QUATERNION SIGN CONSISTENCY (VECTORIZED) =====
    if n_frames > 1:
        quats = rotvec_to_quat_torch_batched(rotvecs=rotvecs)  # (n_frames, 4)

        # Dot products between consecutive quaternions
        q_current = quats[:-1]  # (n_frames-1, 4)
        q_next = quats[1:]  # (n_frames-1, 4)

        dots = torch.sum(q_current * q_next, dim=1)  # (n_frames-1,)
        penalties = torch.maximum(torch.zeros_like(dots), -dots) ** 2
        quat_consistency_cost = torch.sum(penalties)
        total_cost += lambda_quat_consistency * quat_consistency_cost

    # ===== ROTATION TANGENT SMOOTHNESS (VECTORIZED) =====
    if n_frames > 2:
        rot_vel = rotvecs[1:] - rotvecs[:-1]
        rot_accel = rot_vel[1:] - rot_vel[:-1]
        total_cost += lambda_smooth_rot * torch.sum(rot_accel ** 2)

    # ===== POSITION JERK (VECTORIZED) =====
    if n_frames > 3:
        pos_vel = translations[1:] - translations[:-1]
        pos_acc = pos_vel[1:] - pos_vel[:-1]
        pos_jerk = pos_acc[1:] - pos_acc[:-1]
        total_cost += lambda_accel * torch.sum(pos_jerk ** 2)

    return total_cost


def post_optimization_unwrap(
    *,
    rotations: np.ndarray,
    translations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Post-optimization unwrapping pass."""
    logger.info("Post-optimization unwrapping...")

    quats = np.array([Rotation.from_matrix(R).as_quat() for R in rotations])

    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]

    rotations_fixed = np.array([Rotation.from_quat(q).as_matrix() for q in quats])

    rotvecs = np.array([Rotation.from_matrix(R).as_rotvec() for R in rotations_fixed])
    rotvecs_unwrapped = unwrap_rotation_vectors_advanced(
        rotvecs=rotvecs,
        after_quat_fix=True  # Use conservative mode
    )
    rotations_fixed = np.array([Rotation.from_rotvec(rv).as_matrix() for rv in rotvecs_unwrapped])

    logger.info("  Post-unwrap complete")

    return rotations_fixed, translations


def optimize_trajectory_torch(
    *,
    noisy_measurements: np.ndarray,
    reference_geometry: np.ndarray,
    lambda_data: float = 1.0,
    lambda_smooth_pos: float = 30.0,
    lambda_smooth_rot: float = 15.0,
    lambda_accel: float = 3.0,
    lambda_rot_geodesic: float = 1000.0,
    lambda_max_rotation: float = 5000.0,
    lambda_quat_consistency: float = 2000.0,
    lambda_orientation_anchor: float = 100.0,
    lambda_rot_jerk: float = 50.0,
    max_rotation_per_frame_deg: float = 15.0,
    apply_slerp_smoothing: bool = True,
    slerp_window: int = 5,
    max_iter: int = 500,
    learning_rate: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🚀 BLAZING FAST pure PyTorch optimization!
    """
    n_frames = noisy_measurements.shape[0]

    logger.info("=" * 80)
    logger.info("TRAJECTORY OPTIMIZATION - BLAZING FAST GPU VERSION 🚀")
    logger.info("=" * 80)

    # Initialize
    init_rotations, init_translations = kabsch_initialization(
        noisy_measurements=noisy_measurements,
        reference_geometry=reference_geometry,
        apply_slerp=True,
        slerp_window=7
    )

    # Convert to rotation vectors
    init_rotvecs = np.array([
        Rotation.from_matrix(R).as_rotvec() for R in init_rotations
    ])
    init_rotvecs = unwrap_rotation_vectors_advanced(
        rotvecs=init_rotvecs,
        after_quat_fix=True  # Already unwrapped in Kabsch
    )

    # Pack into single vector
    x0 = np.zeros(n_frames * 6)
    for i in range(n_frames):
        x0[i * 6:i * 6 + 3] = init_rotvecs[i]
        x0[i * 6 + 3:i * 6 + 6] = init_translations[i]

    # Reference orientation
    reference_rotvec = np.mean(init_rotvecs, axis=0)

    logger.info(f"\n🎯 Reference orientation:")
    logger.info(f"   Angle: {np.rad2deg(np.linalg.norm(reference_rotvec)):.1f}°")

    # Move everything to GPU
    x_torch = torch.tensor(x0, dtype=torch.float64, device=DEVICE, requires_grad=True)
    noisy_measurements_torch = torch.tensor(
        noisy_measurements, dtype=torch.float64, device=DEVICE
    )
    reference_geometry_torch = torch.tensor(
        reference_geometry, dtype=torch.float64, device=DEVICE
    )
    reference_rotvec_torch = torch.tensor(
        reference_rotvec, dtype=torch.float64, device=DEVICE
    )
    max_rotation_per_frame_rad = np.deg2rad(max_rotation_per_frame_deg)

    logger.info(f"\nOptimization configuration:")
    logger.info(f"  Device:      {DEVICE}")
    logger.info(f"  Frames:      {n_frames}")
    logger.info(f"  Parameters:  {len(x0)} (6 per frame)")
    logger.info(f"  Max iters:   {max_iter}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"\n🔥 Regularization weights:")
    logger.info(f"  λ_data:          {lambda_data:6.1f}")
    logger.info(f"  λ_pos:           {lambda_smooth_pos:6.1f}")
    logger.info(f"  λ_rot_tang:      {lambda_smooth_rot:6.1f}")
    logger.info(f"  λ_rot_geo:       {lambda_rot_geodesic:6.1f}")
    logger.info(f"  λ_max_rot:       {lambda_max_rotation:6.1f}")
    logger.info(f"  λ_quat_cons:     {lambda_quat_consistency:6.1f}")
    logger.info(f"  λ_orient_anchor: {lambda_orientation_anchor:6.1f}")
    logger.info(f"  λ_rot_jerk:      {lambda_rot_jerk:6.1f}")
    logger.info(f"  λ_accel:         {lambda_accel:6.1f}")

    # Define objective with closure over parameters
    def objective_fn(x: torch.Tensor) -> torch.Tensor:
        return trajectory_objective_vectorized(
            x=x,
            noisy_measurements=noisy_measurements_torch,
            reference_geometry=reference_geometry_torch,
            reference_rotvec=reference_rotvec_torch,
            lambda_data=lambda_data,
            lambda_smooth_pos=lambda_smooth_pos,
            lambda_smooth_rot=lambda_smooth_rot,
            lambda_accel=lambda_accel,
            lambda_rot_geodesic=lambda_rot_geodesic,
            lambda_max_rotation=lambda_max_rotation,
            lambda_quat_consistency=lambda_quat_consistency,
            lambda_orientation_anchor=lambda_orientation_anchor,
            lambda_rot_jerk=lambda_rot_jerk,
            max_rotation_per_frame=max_rotation_per_frame_rad
        )

    # Try torch.compile for extra speed (requires PyTorch 2.0+ and Triton)
    logger.info("\n⚡ Attempting torch.compile (requires Triton - Linux only)...")

    compile_enabled = False
    objective_fn_compiled = None

    try:
        objective_fn_compiled = torch.compile(objective_fn, mode='default')
        compile_enabled = True
        logger.info("   torch.compile created, testing during warmup...")
    except Exception as e:
        logger.info(f"   ⚠️ torch.compile setup failed: {type(e).__name__}")
        logger.info(f"   → Using eager mode (still fast on GPU!)")
        compile_enabled = False

    # Warm up - this is where compilation actually happens
    logger.info("\nWarming up...")
    warmup_start = time.time()

    if compile_enabled:
        try:
            # Try the compiled version
            _ = objective_fn_compiled(x_torch)
            warmup_time = time.time() - warmup_start
            logger.info(f"  ✅ torch.compile SUCCESS! First call: {warmup_time:.2f}s")
            objective_fn = objective_fn_compiled  # Use compiled version
        except Exception as e:
            # Compilation failed, fall back to eager mode
            logger.info(f"  ⚠️ torch.compile FAILED during execution: {type(e).__name__}")
            if "triton" in str(e).lower() or "Triton" in str(e):
                logger.info(f"  → Missing Triton (not available on Windows)")
            logger.info(f"  → Falling back to eager mode (still fast!)")
            compile_enabled = False

            # Now warmup with eager mode
            _ = objective_fn(x_torch)
            warmup_time = time.time() - warmup_start
            logger.info(f"  First call (eager): {warmup_time:.2f}s")
    else:
        # Already in eager mode
        _ = objective_fn(x_torch)
        warmup_time = time.time() - warmup_start
        logger.info(f"  First call (eager): {warmup_time:.2f}s")

    # 🚀 Use PyTorch L-BFGS optimizer (stays on GPU!)
    optimizer = torch.optim.LBFGS(
        [x_torch],
        lr=learning_rate,
        max_iter=20,  # Per step
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    logger.info("\nRunning L-BFGS optimization...")
    opt_start = time.time()

    iteration_count = [0]
    costs = []

    def closure():
        optimizer.zero_grad()
        cost = objective_fn(x_torch)
        cost.backward()

        iteration_count[0] += 1
        cost_val = float(cost.item())
        costs.append(cost_val)

        # Log every 10 iterations
        if iteration_count[0] % 10 == 0:
            logger.info(f"  Iter {iteration_count[0]:3d}: cost = {cost_val:.2f}")

        return cost

    # Run optimization
    n_steps = max_iter // 20  # Each step does up to 20 iterations
    for step in range(n_steps):
        optimizer.step(closure)

        # Check convergence
        if len(costs) > 1 and abs(costs[-1] - costs[-2]) < 1e-6:
            logger.info(f"\nConverged at iteration {iteration_count[0]}")
            break

    opt_time = time.time() - opt_start

    logger.info(f"\nCompleted!")
    logger.info(f"  Final cost:  {costs[-1]:.2f}")
    logger.info(f"  Iterations:  {iteration_count[0]}")
    logger.info(f"  Time:        {opt_time:.1f}s")
    logger.info(f"  Speed:       {iteration_count[0] / opt_time:.1f} iter/sec 🚀")

    # Convert back to numpy
    x_final = x_torch.detach().cpu().numpy()

    # Extract rotations and translations
    opt_rotations = np.zeros((n_frames, 3, 3))
    opt_translations = np.zeros((n_frames, 3))

    for i in range(n_frames):
        rotvec = x_final[i * 6:i * 6 + 3]
        opt_rotations[i] = Rotation.from_rotvec(rotvec).as_matrix()
        opt_translations[i] = x_final[i * 6 + 3:i * 6 + 6]

    # Post-optimization unwrapping
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
            cutoff_freq_hz=10.0,
            sampling_rate_hz=100.0,
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