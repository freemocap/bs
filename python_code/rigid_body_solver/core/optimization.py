"""Bundle adjustment optimization - optimize poses with FIXED reference geometry."""

import logging
from dataclasses import dataclass

import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

from python_code.rigid_body_solver.core.reference_geometry import (
    estimate_rigid_body_reference_geometry,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    max_iter: int = 300
    lambda_data: float = 100.0
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
    keypoint_trajectories: np.ndarray  # (n_frames, n_markers, 3)
    reference_geometry: np.ndarray  # (n_markers, 3)
    initial_cost: float
    final_cost: float
    success: bool
    iterations: int
    time_seconds: float


# =============================================================================
# COST FUNCTIONS
# =============================================================================


class MeasurementFactor(pyceres.CostFunction):
    """Data fitting: measured point should match transformed reference point.

    """

    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        reference_point: np.ndarray,
        weight: float = 100.0,
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.reference_point = reference_point.copy()
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3])  # [quat, trans] only

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None,
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        predicted = R @ self.reference_point + translation
        residuals[:] = self.weight * (self.measured_point - predicted)

        if jacobians is not None:
            eps = 1e-8

            # Jacobian w.r.t. quaternion
            if jacobians[0] is not None:
                for j in range(4):
                    quat_plus = quat.copy()
                    quat_plus[j] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    predicted_plus = R_plus @ self.reference_point + translation
                    residual_plus = self.weight * (self.measured_point - predicted_plus)
                    for i in range(3):
                        jacobians[0][i * 4 + j] = (residual_plus[i] - residuals[i]) / eps

            # Jacobian w.r.t. translation
            if jacobians[1] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[1][i * 3 + j] = -self.weight if i == j else 0.0

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
        jacobians: list[np.ndarray] | None,
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
        jacobians: list[np.ndarray] | None,
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


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================


def optimize_rigid_body(
    *,
    original_trajectories: np.ndarray,
    rigid_edges: list[tuple[str, str]],
    config: OptimizationConfig,
    marker_names: list[str] | None = None,
    display_edges: list[tuple[str, str]] | None = None,
    body_frame_origin_markers: list[str] | None = None,
    body_frame_x_axis_marker: str | None = None,
    body_frame_y_axis_marker: str | None = None,
) -> OptimizationResult:
    """
    Optimize poses with FIXED reference geometry.

    Reference geometry is estimated once from the data (median distances + MDS)
    and then held constant. Only per-frame poses (rotation, translation) are optimized.

    Args:
        original_trajectories: (n_frames, n_markers, 3) measured positions (may contain NaN for missing markers)
        rigid_edges: List of (name_i, name_j) pairs defining rigid body structure (used for display only)
        config: OptimizationConfig
        marker_names: Marker names for visualization
        display_edges: Edges to show in visualization (defaults to rigid_edges)
        body_frame_origin_markers: Marker names whose mean defines the origin
        body_frame_x_axis_marker: Marker name that defines X-axis direction
        body_frame_y_axis_marker: Marker name that defines Y-axis direction

    Returns:
        OptimizationResult with fixed reference geometry and optimized poses
    """

    def name_to_index(name: str) -> int:
        if marker_names is None:
            raise ValueError("marker_names must be provided to use name_to_index")
        return marker_names.index(name)

    n_frames, n_markers, _ = original_trajectories.shape

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

    logger.info("=" * 80)
    logger.info("POSE OPTIMIZATION (FIXED REFERENCE GEOMETRY)")
    logger.info("=" * 80)
    logger.info(f"Frames: {n_frames}, Markers: {n_markers}")

    # =========================================================================
    # INITIALIZE REFERENCE GEOMETRY USING DISTANCE MATRIX + MDS
    # =========================================================================
    logger.info("\nEstimating reference geometry from rigid distance matrix...")
    reference_geometry, basis_vectors, _ = estimate_rigid_body_reference_geometry(
        original_data=original_trajectories,
        marker_names=marker_names,
        display_edges=display_edges,
        origin_markers=body_frame_origin_markers,
        x_axis_marker=body_frame_x_axis_marker,
        y_axis_marker=body_frame_y_axis_marker,
        show_alignment_plots=True,
    )
    # reference_geometry is now in body-fixed frame:
    # - Origin at (0,0,0) [head_center]
    # - +X towards nose, +Y towards left, +Z up
    # This geometry is FIXED and will not be optimized.

    logger.info("Reference geometry is FIXED (will not be optimized)")

    # =========================================================================
    # COUNT VALID MEASUREMENTS
    # =========================================================================
    valid_mask = ~np.isnan(original_trajectories).any(axis=2)  # (n_frames, n_markers)
    n_valid_measurements = valid_mask.sum()
    n_total_measurements = n_frames * n_markers
    n_missing = n_total_measurements - n_valid_measurements

    logger.info(f"\nMeasurement statistics:")
    logger.info(f"  Total possible: {n_total_measurements}")
    logger.info(f"  Valid:          {n_valid_measurements} ({100 * n_valid_measurements / n_total_measurements:.1f}%)")
    logger.info(f"  Missing (NaN):  {n_missing} ({100 * n_missing / n_total_measurements:.1f}%)")

    # =========================================================================
    # INITIALIZE POSES
    # =========================================================================
    logger.info("\nInitializing poses...")
    poses: list[tuple[np.ndarray, np.ndarray]] = []

    for frame_idx in range(n_frames):
        quat_ceres = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        # Use nanmean to handle missing markers in initial translation estimate
        translation = np.nanmean(original_trajectories[frame_idx], axis=0)
        # If all markers are NaN for this frame, fall back to zeros
        if np.isnan(translation).any():
            translation = np.zeros(3)
        poses.append((quat_ceres, translation))

    # =========================================================================
    # BUILD OPTIMIZATION PROBLEM
    # =========================================================================
    logger.info("\nBuilding optimization problem...")
    problem = pyceres.Problem()

    # Add pose parameters (NO reference geometry parameters!)
    for quat, trans in poses:
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(trans, 3)
        problem.set_manifold(quat, pyceres.QuaternionManifold())

    # DATA FITTING (skip NaN measurements)
    logger.info("  Adding measurement factors (skipping NaN)...")
    n_measurement_factors = 0
    for frame_idx in range(n_frames):
        quat, trans = poses[frame_idx]
        for marker_idx in range(n_markers):
            measured_point = original_trajectories[frame_idx, marker_idx]

            # Skip if this measurement is NaN
            if np.isnan(measured_point).any():
                continue

            cost = MeasurementFactor(
                measured_point=measured_point,
                reference_point=reference_geometry[marker_idx],
                weight=config.lambda_data,
            )
            problem.add_residual_block(cost, None, [quat, trans])
            n_measurement_factors += 1

    logger.info(f"    Added {n_measurement_factors} measurement factors")

    # SMOOTHNESS
    logger.info("  Adding smoothness factors...")
    for frame_idx in range(n_frames - 1):
        quat_t, trans_t = poses[frame_idx]
        quat_t1, trans_t1 = poses[frame_idx + 1]

        rot_cost = RotationSmoothnessFactor(weight=config.lambda_rot_smooth)
        problem.add_residual_block(rot_cost, None, [quat_t, quat_t1])

        trans_cost = TranslationSmoothnessFactor(weight=config.lambda_trans_smooth)
        problem.add_residual_block(trans_cost, None, [trans_t, trans_t1])

    logger.info(f"\n  Total residual blocks: {problem.num_residual_blocks()}")
    logger.info(f"  Total parameters: {problem.num_parameters()}")

    # =========================================================================
    # SOLVE
    # =========================================================================
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

    # =========================================================================
    # EXTRACT RESULTS
    # =========================================================================
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
        reconstructed_keypoints[frame_idx] = (R @ reference_geometry.T).T + trans


    return OptimizationResult(
        quaternions=np.array([poses[i][0] for i in range(n_frames)]),
        rotations=rotations,
        translations=translations,
        keypoint_trajectories=reconstructed_keypoints,
        reference_geometry=reference_geometry,  # Return the FIXED reference geometry
        initial_cost=summary.initial_cost,
        final_cost=summary.final_cost,
        success=summary.termination_type == pyceres.TerminationType.CONVERGENCE,
        iterations=summary.num_successful_steps,
        time_seconds=summary.total_time_in_seconds,
    )