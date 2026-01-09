"""Bundle adjustment optimization - jointly optimize reference geometry AND poses."""

import logging
from dataclasses import dataclass

import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

from python_code.pyceres_solvers.rigid_body_solver.core.reference_geometry import \
    estimate_rigid_body_reference_geometry, \
    plot_reference_geometry, define_body_frame

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
    quaternions: np.ndarray  # (n_frames, 4) [w, x, y, z]
    rotations: np.ndarray  # (n_frames, 3, 3)
    translations: np.ndarray  # (n_frames, 3)
    reconstructed_keypoints: np.ndarray  # (n_frames, n_markers, 3)
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
    original_data: np.ndarray,
    rigid_edges: list[tuple[str, str]],
    reference_distances: np.ndarray,
    config: OptimizationConfig,
    marker_names: list[str] | None = None,
    display_edges: list[tuple[int, int]] | None = None,
    body_frame_origin_markers: list[str] | None = None,
    body_frame_x_axis_marker: str | None = None,
    body_frame_y_axis_marker: str | None = None
) -> OptimizationResult:
    """
    Bundle adjustment: jointly optimize reference geometry AND poses.

    Args:
        original_data: (n_frames, n_markers, 3) measured positions
        rigid_edges: List of (i, j) pairs that should remain rigid
        reference_distances: (n_markers, n_markers) initial distance estimates
        config: OptimizationConfig
        marker_names: Marker names for visualization
        display_edges: Edges to show in visualization (defaults to rigid_edges)
        body_frame_origin_markers: Marker names whose mean defines the origin
        body_frame_x_axis_marker: Marker name that defines X-axis direction
        body_frame_y_axis_marker: Marker name that defines Y-axis direction

    Returns:
        OptimizationResult with optimized reference geometry and poses
    """
    def name_to_index(name: str) -> int:
        if marker_names is None:
            raise ValueError("marker_names must be provided to use name_to_index")
        return marker_names.index(name)

    n_frames, n_markers, _ = original_data.shape

    # Set defaults
    if marker_names is None:
        marker_names = [f"M{i}" for i in range(n_markers)]
    if display_edges is None:
        display_edges = rigid_edges
    if body_frame_origin_markers is None:
        body_frame_origin_markers = [marker_names[0]]
    if body_frame_x_axis_marker is None:
        body_frame_x_axis_marker = marker_names[1] if len(marker_names) > 1 else marker_names[0]
    if body_frame_y_axis_marker is None:
        body_frame_y_axis_marker = marker_names[2] if len(marker_names) > 2 else marker_names[0]

    logger.info("="*80)
    logger.info("BUNDLE ADJUSTMENT OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Frames: {n_frames}, Markers: {n_markers}, Rigid edges: {len(rigid_edges)}")

    # =========================================================================
    # INITIALIZE REFERENCE GEOMETRY USING DISTANCE MATRIX + MDS
    # =========================================================================
    logger.info("\nInitializing reference geometry from rigid distance matrix...")
    reference_geometry, basis_vectors, _ = estimate_rigid_body_reference_geometry(
        original_data=original_data,
        marker_names=marker_names,
        display_edges=display_edges,
        origin_markers=body_frame_origin_markers,
        x_axis_marker=body_frame_x_axis_marker,
        y_axis_marker=body_frame_y_axis_marker,
        show_alignment_plots=True
    )
    # reference_geometry is now in body-fixed frame:
    # - Origin at (0,0,0) [head_center]
    # - +X towards nose, +Y towards left, +Z up
    # This transformed geometry is used for optimization below



    # =========================================================================
    # INITIALIZE POSES
    # =========================================================================
    logger.info("Initializing poses...")
    poses: list[tuple[np.ndarray, np.ndarray]] = []

    for frame_idx in range(n_frames):
        quat_ceres = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        translation = np.mean(original_data[frame_idx], axis=0)
        poses.append((quat_ceres, translation))

    # =========================================================================
    # BUILD OPTIMIZATION PROBLEM
    # =========================================================================
    logger.info("\nBuilding optimization problem...")
    problem = pyceres.Problem()

    # flatten reference geometry for optimization
    reference_params = reference_geometry.flatten().copy()

    # Add reference geometry as parameters
    problem.add_parameter_block(reference_params, n_markers * 3)

    # Add pose parameters
    for quat, trans in poses:
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(trans, 3)
        problem.set_manifold(quat, pyceres.QuaternionManifold())

    # DATA FITTING
    logger.info("  Adding measurement factors...")
    for frame_idx in range(n_frames):
        quat, trans = poses[frame_idx]
        for point_idx in range(n_markers):
            cost = MeasurementFactorBA(
                measured_point=original_data[frame_idx, point_idx],
                marker_idx=point_idx,
                n_markers=n_markers,
                weight=config.lambda_data
            )
            problem.add_residual_block(cost, None, [quat, trans, reference_params])

    # RIGID BODY CONSTRAINTS
    logger.info(f"  Adding {len(rigid_edges)} rigid body constraints...")
    for i, j in rigid_edges:
        cost = RigidBodyFactorBA(
            marker_i=name_to_index(i),
            marker_j=name_to_index(j),
            n_markers=n_markers,
            target_distance=reference_distances[name_to_index(i),name_to_index(j)],
            weight=config.lambda_rigid
        )
        problem.add_residual_block(cost, None, [reference_params])


    # SMOOTHNESS
    logger.info("  Adding smoothness factors...")
    for frame_idx in range(n_frames - 1):
        quat_t, trans_t = poses[frame_idx]
        quat_t1, trans_t1 = poses[frame_idx + 1]

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
    reconstructed_keypoints = np.zeros((n_frames, n_markers, 3))
    quaternions = np.zeros((n_frames, 4))

    for frame_idx in range(n_frames):
        quat_ceres = poses[frame_idx][0]
        trans = poses[frame_idx][1]

        quat_scipy = np.array([quat_ceres[1], quat_ceres[2], quat_ceres[3], quat_ceres[0]])
        R = Rotation.from_quat(quat_scipy).as_matrix()

        quaternions[frame_idx] = quat_scipy
        rotations[frame_idx] = R
        translations[frame_idx] = trans
        reconstructed_keypoints[frame_idx] = (R @ optimized_reference.T).T + trans

    # Plot the estimated reference geometry
    logger.info("\nPlotting reference geometry (close window to continue)...")
    display_edges_as_indices = [(name_to_index(i), name_to_index(j)) for i, j in display_edges]

    optimized_basis_vectors, optimized_origin_point = define_body_frame(
        reference_geometry=reference_geometry,
        marker_names=marker_names,
        origin_markers=body_frame_origin_markers,
        x_axis_marker=body_frame_x_axis_marker,
        y_axis_marker=body_frame_y_axis_marker
    )
    plot_reference_geometry(
        original_geometry=reference_geometry,
        original_basis_vectors=basis_vectors,
        original_origin_point=np.mean(reference_geometry[[name_to_index(m) for m in body_frame_origin_markers]], axis=0),

        aligned_geometry=optimized_reference,
        aligned_basis_vectors=optimized_basis_vectors,
        aligned_origin_point=optimized_origin_point,


        marker_names=marker_names,
        display_edges=display_edges_as_indices,
        origin_markers=body_frame_origin_markers,
        x_axis_marker=body_frame_x_axis_marker,
        y_axis_marker=body_frame_y_axis_marker
    )


    return OptimizationResult(
        quaternions=np.array([poses[i][0] for i in range(n_frames)]),
        rotations=rotations,
        translations=translations,
        reconstructed_keypoints=reconstructed_keypoints,
        reference_geometry=optimized_reference,
        initial_cost=summary.initial_cost,
        final_cost=summary.final_cost,
        success=summary.termination_type == pyceres.TerminationType.CONVERGENCE,
        iterations=summary.num_successful_steps,
        time_seconds=summary.total_time_in_seconds
    )