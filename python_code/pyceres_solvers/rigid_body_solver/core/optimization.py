"""Bundle adjustment optimization - jointly optimize reference geometry AND poses."""

import numpy as np
import logging
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import pyceres

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    max_iter: int = 300
    lambda_data: float = 100.0
    lambda_rigid: float = 500.0
    lambda_rot_smooth: float = 200.0
    lambda_trans_smooth: float = 200.0
    function_tolerance: float = 1e-9
    gradient_tolerance: float = 1e-11
    parameter_tolerance: float = 1e-10


@dataclass
class OptimizationResult:
    """Results from optimization."""
    rotations: np.ndarray  # (n_frames, 3, 3)
    translations: np.ndarray  # (n_frames, 3)
    reconstructed: np.ndarray  # (n_frames, n_markers, 3)
    reference_geometry: np.ndarray  # (n_markers, 3)
    initial_cost: float
    final_cost: float
    success: bool
    iterations: int
    time_seconds: float


# =============================================================================
# COST FUNCTIONS
# =============================================================================

class MeasurementFactorBA(pyceres.CostFunction):
    """Data fitting: measured point should match transformed reference."""

    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        marker_idx: int,
        n_markers: int,
        weight: float = 100.0
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.marker_idx = marker_idx
        self.n_markers = n_markers
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3, n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]
        reference_flat = parameters[2]

        start_idx = self.marker_idx * 3
        reference_point = reference_flat[start_idx:start_idx + 3]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        predicted = R @ reference_point + translation
        residuals[:] = self.weight * (self.measured_point - predicted)

        if jacobians is not None:
            eps = 1e-8

            if jacobians[0] is not None:
                for j in range(4):
                    quat_plus = quat.copy()
                    quat_plus[j] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    predicted_plus = R_plus @ reference_point + translation
                    residual_plus = self.weight * (self.measured_point - predicted_plus)
                    for i in range(3):
                        jacobians[0][i * 4 + j] = (residual_plus[i] - residuals[i]) / eps

            if jacobians[1] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[1][i * 3 + j] = -self.weight if i == j else 0.0

            if jacobians[2] is not None:
                jacobians[2][:] = 0.0
                start_idx = self.marker_idx * 3
                for i in range(3):
                    for j in range(3):
                        jacobians[2][i * (self.n_markers * 3) + (start_idx + j)] = -self.weight * R[i, j]

        return True


class RigidBodyFactorBA(pyceres.CostFunction):
    """Rigid body constraint: edge length in reference geometry."""

    def __init__(
        self,
        *,
        marker_i: int,
        marker_j: int,
        n_markers: int,
        target_distance: float,
        weight: float = 100.0
    ) -> None:
        super().__init__()
        self.marker_i = marker_i
        self.marker_j = marker_j
        self.n_markers = n_markers
        self.target_dist = target_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        reference_flat = parameters[0]

        ref_i = reference_flat[self.marker_i * 3:(self.marker_i + 1) * 3]
        ref_j = reference_flat[self.marker_j * 3:(self.marker_j + 1) * 3]

        diff = ref_i - ref_j
        current_dist = np.linalg.norm(diff)
        residuals[0] = self.weight * (current_dist - self.target_dist)

        if jacobians is not None and jacobians[0] is not None:
            eps = 1e-8
            jacobians[0][:] = 0.0

            for k in range(3):
                ref_flat_plus = reference_flat.copy()
                ref_flat_plus[self.marker_i * 3 + k] += eps
                ref_i_plus = ref_flat_plus[self.marker_i * 3:(self.marker_i + 1) * 3]
                ref_j_plus = ref_flat_plus[self.marker_j * 3:(self.marker_j + 1) * 3]
                diff_plus = ref_i_plus - ref_j_plus
                dist_plus = np.linalg.norm(diff_plus)
                residual_plus = self.weight * (dist_plus - self.target_dist)
                jacobians[0][self.marker_i * 3 + k] = (residual_plus - residuals[0]) / eps

            for k in range(3):
                ref_flat_plus = reference_flat.copy()
                ref_flat_plus[self.marker_j * 3 + k] += eps
                ref_i_plus = ref_flat_plus[self.marker_i * 3:(self.marker_i + 1) * 3]
                ref_j_plus = ref_flat_plus[self.marker_j * 3:(self.marker_j + 1) * 3]
                diff_plus = ref_i_plus - ref_j_plus
                dist_plus = np.linalg.norm(diff_plus)
                residual_plus = self.weight * (dist_plus - self.target_dist)
                jacobians[0][self.marker_j * 3 + k] = (residual_plus - residuals[0]) / eps

        return True


class SoftDistanceFactorBA(pyceres.CostFunction):
    """Soft distance constraint between measured point and reference point."""

    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        marker_idx_on_body: int,
        n_markers: int,
        median_distance: float,
        weight: float = 10.0
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.marker_idx = marker_idx_on_body
        self.n_markers = n_markers
        self.median_dist = median_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([4, 3, n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]
        reference_flat = parameters[2]

        start_idx = self.marker_idx * 3
        ref_point = reference_flat[start_idx:start_idx + 3]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        transformed_ref = R @ ref_point + translation
        diff = self.measured_point - transformed_ref
        current_dist = np.linalg.norm(diff)
        residuals[0] = self.weight * (current_dist - self.median_dist)

        if jacobians is not None:
            eps = 1e-8

            if jacobians[0] is not None:
                for j in range(4):
                    quat_plus = quat.copy()
                    quat_plus[j] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    transformed_plus = R_plus @ ref_point + translation
                    diff_plus = self.measured_point - transformed_plus
                    dist_plus = np.linalg.norm(diff_plus)
                    residual_plus = self.weight * (dist_plus - self.median_dist)
                    jacobians[0][j] = (residual_plus - residuals[0]) / eps

            if jacobians[1] is not None:
                if current_dist > 1e-10:
                    grad = -self.weight * diff / current_dist
                    jacobians[1][:] = grad
                else:
                    jacobians[1][:] = 0.0

            if jacobians[2] is not None:
                jacobians[2][:] = 0.0
                if current_dist > 1e-10:
                    start_idx = self.marker_idx * 3
                    grad_ref = self.weight * (diff / current_dist) @ R
                    for i in range(3):
                        jacobians[2][start_idx + i] = grad_ref[i]

        return True


class RotationSmoothnessFactor(pyceres.CostFunction):
    """Rotation smoothness between consecutive frames."""

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

        if np.dot(quat_t, quat_t1) < 0:
            quat_t1_corrected = -quat_t1.copy()
        else:
            quat_t1_corrected = quat_t1.copy()

        residuals[:] = self.weight * (quat_t1_corrected - quat_t)

        if jacobians is not None:
            if jacobians[0] is not None:
                for i in range(4):
                    for j in range(4):
                        jacobians[0][i * 4 + j] = -self.weight if i == j else 0.0
            if jacobians[1] is not None:
                sign = -1.0 if np.dot(quat_t, quat_t1) < 0 else 1.0
                for i in range(4):
                    for j in range(4):
                        jacobians[1][i * 4 + j] = self.weight * sign if i == j else 0.0

        return True


class TranslationSmoothnessFactor(pyceres.CostFunction):
    """Translation smoothness between consecutive frames."""

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
                for i in range(3):
                    for j in range(3):
                        jacobians[0][i * 3 + j] = -self.weight if i == j else 0.0
            if jacobians[1] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[1][i * 3 + j] = self.weight if i == j else 0.0

        return True


class ReferenceAnchorFactor(pyceres.CostFunction):
    """Soft anchor to prevent reference from drifting too far."""

    def __init__(self, *, initial_reference: np.ndarray, weight: float = 10.0) -> None:
        super().__init__()
        self.initial_ref = initial_reference.copy()
        self.weight = weight
        n_params = len(initial_reference)
        self.set_num_residuals(n_params)
        self.set_parameter_block_sizes([n_params])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        reference = parameters[0]
        residuals[:] = self.weight * (reference - self.initial_ref)

        if jacobians is not None and jacobians[0] is not None:
            n = len(residuals)
            for i in range(n):
                for j in range(n):
                    jacobians[0][i * n + j] = self.weight if i == j else 0.0

        return True


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def optimize_rigid_body(
    *,
    noisy_data: np.ndarray,
    rigid_edges: list[tuple[int, int]],
    reference_distances: np.ndarray,
    config: OptimizationConfig,
    soft_edges: list[tuple[int, int]] | None = None,
    soft_distances: np.ndarray | None = None,
    lambda_soft: float = 10.0
) -> OptimizationResult:
    """
    Bundle adjustment: jointly optimize reference geometry AND poses.

    Args:
        noisy_data: (n_frames, n_markers, 3) measured positions
        rigid_edges: List of (i, j) pairs that should remain rigid
        reference_distances: (n_markers, n_markers) initial distance estimates
        config: OptimizationConfig
        soft_edges: Optional list of soft edges
        soft_distances: Optional soft edge distances
        lambda_soft: Weight for soft constraints

    Returns:
        OptimizationResult with optimized reference geometry and poses
    """
    n_frames, n_markers, _ = noisy_data.shape

    logger.info("="*80)
    logger.info("BUNDLE ADJUSTMENT OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Frames: {n_frames}, Markers: {n_markers}, Rigid edges: {len(rigid_edges)}")

    # =========================================================================
    # INITIALIZE REFERENCE GEOMETRY
    # =========================================================================
    logger.info("\nInitializing reference geometry from median frame...")
    median_frame = np.median(noisy_data, axis=0)
    reference_geometry = median_frame - np.mean(median_frame, axis=0)
    reference_params = reference_geometry.flatten().copy()

    # =========================================================================
    # INITIALIZE POSES
    # =========================================================================
    logger.info("Initializing poses...")
    poses: list[tuple[np.ndarray, np.ndarray]] = []

    for frame_idx in range(n_frames):
        quat_ceres = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        translation = np.mean(noisy_data[frame_idx], axis=0)
        poses.append((quat_ceres, translation))

    # =========================================================================
    # BUILD OPTIMIZATION PROBLEM
    # =========================================================================
    logger.info("\nBuilding optimization problem...")
    problem = pyceres.Problem()

    # Add reference geometry as parameters
    problem.add_parameter_block(reference_params, n_markers * 3)

    # Add pose parameters
    pose_params: list[tuple[np.ndarray, np.ndarray]] = []
    for quat, trans in poses:
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(trans, 3)
        problem.set_manifold(quat, pyceres.QuaternionManifold())
        pose_params.append((quat, trans))

    # DATA FITTING
    logger.info("  Adding measurement factors...")
    for frame_idx in range(n_frames):
        quat, trans = pose_params[frame_idx]
        for point_idx in range(n_markers):
            cost = MeasurementFactorBA(
                measured_point=noisy_data[frame_idx, point_idx],
                marker_idx=point_idx,
                n_markers=n_markers,
                weight=config.lambda_data
            )
            problem.add_residual_block(cost, None, [quat, trans, reference_params])

    # RIGID BODY CONSTRAINTS
    logger.info(f"  Adding {len(rigid_edges)} rigid body constraints...")
    for i, j in rigid_edges:
        cost = RigidBodyFactorBA(
            marker_i=i,
            marker_j=j,
            n_markers=n_markers,
            target_distance=reference_distances[i, j],
            weight=config.lambda_rigid
        )
        problem.add_residual_block(cost, None, [reference_params])

    # SOFT EDGES
    if soft_edges is not None and soft_distances is not None:
        logger.info(f"  Adding {len(soft_edges)} soft constraints...")
        for frame_idx in range(n_frames):
            quat, trans = pose_params[frame_idx]
            for i, j in soft_edges:
                cost = SoftDistanceFactorBA(
                    measured_point=noisy_data[frame_idx, j],
                    marker_idx_on_body=i,
                    n_markers=n_markers,
                    median_distance=soft_distances[i, j],
                    weight=lambda_soft
                )
                problem.add_residual_block(cost, None, [quat, trans, reference_params])

    # SMOOTHNESS
    logger.info("  Adding smoothness factors...")
    for frame_idx in range(n_frames - 1):
        quat_t, trans_t = pose_params[frame_idx]
        quat_t1, trans_t1 = pose_params[frame_idx + 1]

        rot_cost = RotationSmoothnessFactor(weight=config.lambda_rot_smooth)
        problem.add_residual_block(rot_cost, None, [quat_t, quat_t1])

        trans_cost = TranslationSmoothnessFactor(weight=config.lambda_trans_smooth)
        problem.add_residual_block(trans_cost, None, [trans_t, trans_t1])

    # REFERENCE ANCHOR
    logger.info("  Adding reference anchor...")
    anchor_cost = ReferenceAnchorFactor(
        initial_reference=reference_params,
        weight=config.lambda_data * 0.1
    )
    problem.add_residual_block(anchor_cost, None, [reference_params])

    logger.info(f"\n  Total residual blocks: {problem.num_residual_blocks()}")
    logger.info(f"  Total parameters: {problem.num_parameters()}")

    # =========================================================================
    # SOLVE
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("SOLVING")
    logger.info("="*80)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = config.max_iter
    options.function_tolerance = config.function_tolerance
    options.gradient_tolerance = config.gradient_tolerance
    options.parameter_tolerance = config.parameter_tolerance

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"  Status: {summary.termination_type}")
    logger.info(f"  Initial cost: {summary.initial_cost:.2f}")
    logger.info(f"  Final cost: {summary.final_cost:.2f}")
    logger.info(f"  Iterations: {summary.num_successful_steps}")
    logger.info(f"  Time: {summary.total_time_in_seconds:.2f}s")

    # =========================================================================
    # EXTRACT RESULTS
    # =========================================================================
    optimized_reference = reference_params.reshape(n_markers, 3)

    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))
    reconstructed = np.zeros((n_frames, n_markers, 3))

    for frame_idx in range(n_frames):
        quat_ceres = pose_params[frame_idx][0]
        trans = pose_params[frame_idx][1]

        quat_scipy = np.array([quat_ceres[1], quat_ceres[2], quat_ceres[3], quat_ceres[0]])
        R = Rotation.from_quat(quat_scipy).as_matrix()

        rotations[frame_idx] = R
        translations[frame_idx] = trans
        reconstructed[frame_idx] = (R @ optimized_reference.T).T + trans

    return OptimizationResult(
        rotations=rotations,
        translations=translations,
        reconstructed=reconstructed,
        reference_geometry=optimized_reference,
        initial_cost=summary.initial_cost,
        final_cost=summary.final_cost,
        success=summary.termination_type == pyceres.TerminationType.CONVERGENCE,
        iterations=summary.num_successful_steps,
        time_seconds=summary.total_time_in_seconds
    )