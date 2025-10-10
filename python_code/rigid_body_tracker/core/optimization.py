"""PyCeres-based rigid body pose optimization."""

import numpy as np
import logging
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import pyceres

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for PyCeres optimization."""

    max_iter: int = 300
    """Maximum number of optimization iterations"""

    lambda_data: float = 100.0
    """Weight for data fitting term (higher = fit measurements more closely)"""

    lambda_rigid: float = 500.0
    """Weight for rigid body constraints (higher = enforce rigidity more strictly)"""

    lambda_rot_smooth: float = 200.0
    """Weight for rotation smoothness across frames"""

    lambda_trans_smooth: float = 200.0
    """Weight for translation smoothness across frames"""

    function_tolerance: float = 1e-9
    gradient_tolerance: float = 1e-11
    parameter_tolerance: float = 1e-10


# =============================================================================
# PYCERES COST FUNCTIONS
# =============================================================================

class MeasurementFactor(pyceres.CostFunction):
    """Data fitting term: residual between measured and predicted point positions."""

    def __init__(
            self,
            *,
            measured_point: np.ndarray,
            reference_point: np.ndarray,
            weight: float = 100.0
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.reference_point = reference_point.copy()
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3])

    def Evaluate(
            self,
            parameters: list[np.ndarray],
            residuals: np.ndarray,
            jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]  # [qw, qx, qy, qz]
        translation = parameters[1]

        # Convert to scipy format [qx, qy, qz, qw]
        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        predicted = R @ self.reference_point + translation
        residuals[:] = self.weight * (self.measured_point - predicted)

        if jacobians is not None:
            if jacobians[0] is not None:
                # Jacobian w.r.t. quaternion (finite differences)
                eps = 1e-8
                J_quat = np.zeros((3, 4))
                for i in range(4):
                    quat_plus = quat.copy()
                    quat_plus[i] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    predicted_plus = R_plus @ self.reference_point + translation
                    residual_plus = self.weight * (self.measured_point - predicted_plus)
                    J_quat[:, i] = (residual_plus - residuals) / eps
                jacobians[0][:] = J_quat.ravel()

            if jacobians[1] is not None:
                # Jacobian w.r.t. translation
                J_trans = -self.weight * np.eye(3)
                jacobians[1][:] = J_trans.ravel()

        return True


class RigidBodyFactor(pyceres.CostFunction):
    """Rigid body constraint: enforce fixed distance between two points."""

    def __init__(
            self,
            *,
            ref_point_i: np.ndarray,
            ref_point_j: np.ndarray,
            reference_distance: float,
            weight: float = 100.0
    ) -> None:
        super().__init__()
        self.ref_i = ref_point_i.copy()
        self.ref_j = ref_point_j.copy()
        self.ref_dist = reference_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([4, 3])

    def Evaluate(
            self,
            parameters: list[np.ndarray],
            residuals: np.ndarray,
            jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        p_i = R @ self.ref_i + translation
        p_j = R @ self.ref_j + translation
        diff = p_i - p_j
        current_dist = np.linalg.norm(diff)
        residuals[0] = self.weight * (current_dist - self.ref_dist)

        if jacobians is not None:
            if jacobians[0] is not None:
                eps = 1e-8
                J_quat = np.zeros(4)
                for i in range(4):
                    quat_plus = quat.copy()
                    quat_plus[i] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    p_i_plus = R_plus @ self.ref_i + translation
                    p_j_plus = R_plus @ self.ref_j + translation
                    dist_plus = np.linalg.norm(p_i_plus - p_j_plus)
                    residual_plus = self.weight * (dist_plus - self.ref_dist)
                    J_quat[i] = (residual_plus - residuals[0]) / eps
                jacobians[0][:] = J_quat

            if jacobians[1] is not None:
                # Translation cancels out in distance
                jacobians[1][:] = 0.0

        return True


class RotationSmoothnessFactor(pyceres.CostFunction):
    """Rotation smoothness via quaternion distance."""

    def __init__(self, *, weight: float = 500.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(4)
        self.set_parameter_block_sizes([4, 4])

    def Evaluate(
            self,
            parameters: list[np.ndarray],
            residuals: np.ndarray,
            jacobians: list[np.ndarray] | None
    ) -> bool:
        quat_t = parameters[0]
        quat_t1 = parameters[1]

        # Ensure same hemisphere
        if np.dot(quat_t, quat_t1) < 0:
            quat_t1_corrected = -quat_t1.copy()
            flip_sign = -1.0
        else:
            quat_t1_corrected = quat_t1.copy()
            flip_sign = 1.0

        residuals[:] = self.weight * (quat_t1_corrected - quat_t)

        if jacobians is not None:
            if jacobians[0] is not None:
                J0 = -self.weight * np.eye(4)
                jacobians[0][:] = J0.ravel()

            if jacobians[1] is not None:
                J1 = self.weight * flip_sign * np.eye(4)
                jacobians[1][:] = J1.ravel()

        return True


class TranslationSmoothnessFactor(pyceres.CostFunction):
    """Translation smoothness."""

    def __init__(self, *, weight: float = 200.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3])

    def Evaluate(
            self,
            parameters: list[np.ndarray],
            residuals: np.ndarray,
            jacobians: list[np.ndarray] | None
    ) -> bool:
        trans_t = parameters[0]
        trans_t1 = parameters[1]
        residuals[:] = self.weight * (trans_t1 - trans_t)

        if jacobians is not None:
            if jacobians[0] is not None:
                J0 = -self.weight * np.eye(3)
                jacobians[0][:] = J0.ravel()

            if jacobians[1] is not None:
                J1 = self.weight * np.eye(3)
                jacobians[1][:] = J1.ravel()

        return True


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_poses_procrustes(
        *,
        noisy_data: np.ndarray,
        reference_geometry: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Initialize poses using Procrustes alignment for each frame.

    Args:
        noisy_data: (n_frames, n_points, 3) noisy measurements
        reference_geometry: (n_points, 3) reference shape

    Returns:
        List of (quaternion, translation) tuples for each frame
    """
    n_frames = noisy_data.shape[0]
    ref_centered = reference_geometry - np.mean(reference_geometry, axis=0)
    poses: list[tuple[np.ndarray, np.ndarray]] = []

    for frame_idx in range(n_frames):
        data_centered = noisy_data[frame_idx] - np.mean(noisy_data[frame_idx], axis=0)

        # Procrustes: H = data^T @ ref
        H = data_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Convert to quaternion (Ceres format: [qw, qx, qy, qz])
        quat_scipy = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
        quat_ceres = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

        translation = np.mean(noisy_data[frame_idx], axis=0)
        poses.append((quat_ceres, translation))

    # Unwrap quaternions for consistency
    for i in range(1, n_frames):
        if np.dot(poses[i][0], poses[i - 1][0]) < 0:
            poses[i] = (-poses[i][0], poses[i][1])

    return poses


# =============================================================================
# POST-OPTIMIZATION ALIGNMENT
# =============================================================================

def align_reconstructed_to_noisy(
        *,
        noisy_data: np.ndarray,
        reconstructed_data: np.ndarray,
        rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find global rotation to align reconstructed data with noisy data.

    This fixes the common issue where optimization produces a rigid solution
    that's rotationally offset from the input data.

    Args:
        noisy_data: (n_frames, n_points, 3) original noisy measurements
        reconstructed_data: (n_frames, n_points, 3) optimized reconstruction
        rotations: (n_frames, 3, 3) rotation matrices from optimization

    Returns:
        Tuple of (aligned_reconstructed, aligned_rotations)
    """
    logger.info("\nPost-optimization alignment:")

    # Flatten all frames into single point clouds
    noisy_flat = noisy_data.reshape(-1, 3)
    recon_flat = reconstructed_data.reshape(-1, 3)

    # Center both
    noisy_centered = noisy_flat - np.mean(noisy_flat, axis=0)
    recon_centered = recon_flat - np.mean(recon_flat, axis=0)

    # Find optimal rotation using Procrustes (SVD)
    H = recon_centered.T @ noisy_centered
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T

    # Compute alignment error before and after
    error_before = np.mean(np.linalg.norm(recon_flat - noisy_flat, axis=1))
    recon_aligned_flat = (R_align @ recon_centered.T).T + np.mean(noisy_flat, axis=0)
    error_after = np.mean(np.linalg.norm(recon_aligned_flat - noisy_flat, axis=1))

    logger.info(f"  Alignment error: {error_before*1000:.2f}mm → {error_after*1000:.2f}mm")

    # Apply global rotation to all reconstructed points
    n_frames, n_points, _ = noisy_data.shape
    aligned_reconstructed = np.zeros_like(reconstructed_data)

    for t in range(n_frames):
        # Center, rotate, uncenter
        center = np.mean(reconstructed_data[t], axis=0)
        aligned_reconstructed[t] = (R_align @ (reconstructed_data[t] - center).T).T + center

    # Apply global rotation to all rotation matrices: R_new[t] = R_align @ R[t]
    aligned_rotations = np.zeros_like(rotations)
    for t in range(n_frames):
        aligned_rotations[t] = R_align @ rotations[t]

    return aligned_reconstructed, aligned_rotations


# =============================================================================
# POST-OPTIMIZATION ALIGNMENT
# =============================================================================

def align_reconstructed_to_noisy(
        *,
        noisy_data: np.ndarray,
        reconstructed_data: np.ndarray,
        rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find global rotation to align reconstructed data with noisy data.

    This fixes rotational offset by finding a single global rotation
    that aligns the entire optimized trajectory to the input data.

    Args:
        noisy_data: (n_frames, n_points, 3) original noisy measurements
        reconstructed_data: (n_frames, n_points, 3) optimized reconstruction
        rotations: (n_frames, 3, 3) rotation matrices from optimization

    Returns:
        Tuple of (aligned_reconstructed, aligned_rotations)
    """
    logger.info("\nAligning optimized output to input coordinate frame...")

    # Flatten all frames into single point clouds
    noisy_flat = noisy_data.reshape(-1, 3)
    recon_flat = reconstructed_data.reshape(-1, 3)

    # Center both
    noisy_centered = noisy_flat - np.mean(noisy_flat, axis=0)
    recon_centered = recon_flat - np.mean(recon_flat, axis=0)

    # Find optimal rotation using Procrustes (SVD)
    H = recon_centered.T @ noisy_centered
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T

    # Compute alignment error before and after
    error_before = np.mean(np.linalg.norm(recon_flat - noisy_flat, axis=1))
    recon_aligned_flat = (R_align @ recon_centered.T).T + np.mean(noisy_flat, axis=0)
    error_after = np.mean(np.linalg.norm(recon_aligned_flat - noisy_flat, axis=1))

    logger.info(f"  Alignment error: {error_before*1000:.2f}mm → {error_after*1000:.2f}mm")

    # Apply global rotation to all reconstructed points
    n_frames, n_points, _ = noisy_data.shape
    aligned_reconstructed = np.zeros_like(reconstructed_data)

    for t in range(n_frames):
        # Center, rotate, uncenter
        center = np.mean(reconstructed_data[t], axis=0)
        aligned_reconstructed[t] = (R_align @ (reconstructed_data[t] - center).T).T + center

    # Apply global rotation to all rotation matrices: R_new[t] = R_align @ R[t]
    aligned_rotations = np.zeros_like(rotations)
    for t in range(n_frames):
        aligned_rotations[t] = R_align @ rotations[t]

    logger.info(f"  ✓ Applied global rotation alignment")

    return aligned_reconstructed, aligned_rotations


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

@dataclass
class OptimizationResult:
    """Results from rigid body optimization."""

    rotations: np.ndarray  # (n_frames, 3, 3)
    translations: np.ndarray  # (n_frames, 3)
    reconstructed: np.ndarray  # (n_frames, n_points, 3)
    initial_cost: float
    final_cost: float
    success: bool
    iterations: int
    time_seconds: float


def optimize_rigid_body(
        *,
        noisy_data: np.ndarray,
        reference_geometry: np.ndarray,
        rigid_edges: list[tuple[int, int]],
        reference_distances: np.ndarray,
        config: OptimizationConfig | None = None
) -> OptimizationResult:
    """
    Optimize rigid body trajectory using PyCeres pose graph optimization.

    Args:
        noisy_data: (n_frames, n_points, 3) noisy measurements
        reference_geometry: (n_points, 3) reference shape
        rigid_edges: List of (i, j) pairs defining rigid constraints
        reference_distances: (n_points, n_points) distance matrix
        config: Optimization configuration

    Returns:
        OptimizationResult with optimized poses and reconstructed points
    """
    if config is None:
        config = OptimizationConfig()

    n_frames, n_points, _ = noisy_data.shape

    logger.info("=" * 80)
    logger.info("PYCERES RIGID BODY OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"\nFrames: {n_frames}, Markers: {n_points}, Rigid edges: {len(rigid_edges)}")
    logger.info(f"\nWeights:")
    logger.info(f"  λ_data:        {config.lambda_data:8.1f}")
    logger.info(f"  λ_rigid:       {config.lambda_rigid:8.1f}")
    logger.info(f"  λ_rot_smooth:  {config.lambda_rot_smooth:8.1f}")
    logger.info(f"  λ_trans_smooth: {config.lambda_trans_smooth:8.1f}")

    # Initialize poses
    logger.info("\nInitializing poses with Procrustes...")
    poses = initialize_poses_procrustes(
        noisy_data=noisy_data,
        reference_geometry=reference_geometry
    )

    # Build problem
    logger.info("Building optimization problem...")
    problem = pyceres.Problem()
    pose_params: list[tuple[np.ndarray, np.ndarray]] = []

    for quat, trans in poses:
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(trans, 3)
        problem.set_manifold(quat, pyceres.QuaternionManifold())
        pose_params.append((quat, trans))

    # Add measurement factors
    logger.info("  Adding measurement factors...")
    for frame_idx in range(n_frames):
        quat, trans = pose_params[frame_idx]
        for point_idx in range(n_points):
            cost = MeasurementFactor(
                measured_point=noisy_data[frame_idx, point_idx],
                reference_point=reference_geometry[point_idx],
                weight=config.lambda_data
            )
            problem.add_residual_block(cost, None, [quat, trans])

    # Add rigid body constraints
    logger.info(f"  Adding {len(rigid_edges)} rigid body constraints...")
    for frame_idx in range(n_frames):
        quat, trans = pose_params[frame_idx]
        for i, j in rigid_edges:
            cost = RigidBodyFactor(
                ref_point_i=reference_geometry[i],
                ref_point_j=reference_geometry[j],
                reference_distance=reference_distances[i, j],
                weight=config.lambda_rigid
            )
            problem.add_residual_block(cost, None, [quat, trans])

    # Add smoothness factors
    logger.info("  Adding smoothness factors...")
    for frame_idx in range(n_frames - 1):
        quat_t, trans_t = pose_params[frame_idx]
        quat_t1, trans_t1 = pose_params[frame_idx + 1]

        rot_cost = RotationSmoothnessFactor(weight=config.lambda_rot_smooth)
        problem.add_residual_block(rot_cost, None, [quat_t, quat_t1])

        trans_cost = TranslationSmoothnessFactor(weight=config.lambda_trans_smooth)
        problem.add_residual_block(trans_cost, None, [trans_t, trans_t1])

    logger.info(f"\n  Total residual blocks: {problem.num_residual_blocks()}")
    logger.info(f"  Total parameters: {problem.num_parameters()}")

    # Solve
    logger.info("\n" + "=" * 80)
    logger.info("SOLVING")
    logger.info("=" * 80)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = config.max_iter
    options.function_tolerance = config.function_tolerance
    options.gradient_tolerance = config.gradient_tolerance
    options.parameter_tolerance = config.parameter_tolerance

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Status: {summary.termination_type}")
    logger.info(f"  Initial cost: {summary.initial_cost:.2f}")
    logger.info(f"  Final cost: {summary.final_cost:.2f}")
    logger.info(f"  Iterations: {summary.num_successful_steps}")
    logger.info(f"  Time: {summary.total_time_in_seconds:.2f}s")

    # Extract results
    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))
    reconstructed = np.zeros((n_frames, n_points, 3))

    for frame_idx in range(n_frames):
        quat_ceres = pose_params[frame_idx][0]
        trans = pose_params[frame_idx][1]

        quat_scipy = np.array([quat_ceres[1], quat_ceres[2], quat_ceres[3], quat_ceres[0]])
        R = Rotation.from_quat(quat_scipy).as_matrix()

        rotations[frame_idx] = R
        translations[frame_idx] = trans
        reconstructed[frame_idx] = (R @ reference_geometry.T).T + trans

    return OptimizationResult(
        rotations=rotations,
        translations=translations,
        reconstructed=reconstructed,
        initial_cost=summary.initial_cost,
        final_cost=summary.final_cost,
        success=summary.termination_type == pyceres.TerminationType.CONVERGENCE,
        iterations=summary.num_successful_steps,
        time_seconds=summary.total_time_in_seconds
    )