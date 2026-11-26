"""Bundle adjustment optimization - optimize poses with FIXED reference geometry."""

import logging

import numpy as np
import pyceres
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.spatial.transform import Rotation

from python_code.kinematics_core.quaternion_model import Quaternion
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.rigid_body_solver.core.calculate_reference_geometry import estimate_reference_geometry

logger = logging.getLogger(__name__)


def _estimate_initial_pose(
    measured_points: NDArray[np.float64],
    reference_points: NDArray[np.float64],
    fallback_pose: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate an initial (quat_wxyz, translation) pose for one frame via Kabsch alignment
    of whichever keypoints are valid in that frame against the fixed reference geometry.

    Falls back to the previous frame's pose (or identity) when fewer than 3 keypoints are
    valid, since Kabsch needs at least 3 non-collinear correspondences.
    """
    valid_mask = ~np.isnan(measured_points).any(axis=1)

    if valid_mask.sum() < 3:
        if fallback_pose is not None:
            return fallback_pose
        quat_ceres = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        translation = np.nanmean(measured_points, axis=0)
        if np.isnan(translation).any():
            translation = np.zeros(3, dtype=np.float64)
        return quat_ceres, translation

    measured_valid = measured_points[valid_mask]
    reference_valid = reference_points[valid_mask]

    measured_centroid = measured_valid.mean(axis=0)
    reference_centroid = reference_valid.mean(axis=0)

    measured_centered = measured_valid - measured_centroid
    reference_centered = reference_valid - reference_centroid

    H = reference_centered.T @ measured_centered
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

    translation = measured_centroid - R @ reference_centroid

    return quat_wxyz, translation


class OptimizationConfig(BaseModel):
    """Configuration for optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iter: int
    # Smoothness weights are relative to the (unit-weight) data term: the cost is
    #   1/2 [ sum ||m - (R r + p)||^2 + lambda_rot^2 sum ||dq||^2 + lambda_trans^2 sum ||dp||^2 ]
    # A former separate lambda_data was redundant (only the ratios matter); old configs
    # convert as lambda_smooth_new = lambda_smooth_old / lambda_data_old.
    lambda_rot_smooth: float
    lambda_trans_smooth: float
    function_tolerance: float = 1e-9
    # gradient_tolerance is absolute, so it scales with the cost: 1e-15 here is the
    # 1e-11 that was used when the data term carried lambda_data=100 (cost x 1e4).
    gradient_tolerance: float = 1e-15
    parameter_tolerance: float = 1e-10
    # Huber loss transition scale for measurement (data) residuals only, in mm (the
    # data residual is unweighted). None disables the robust loss (quadratic behavior).
    # NOTE: measurement residuals are batched per-frame (FrameMeasurementFactor), so
    # when set, Huber robustifies on the combined residual norm across all valid
    # keypoints in a frame, not per-keypoint as in a pre-batching single-keypoint
    # regime. Recalibrate this scale (expect needing a larger value, roughly
    # sqrt(n_valid_keypoints)x a single-keypoint-calibrated scale as a starting guess)
    # before relying on it.
    robust_loss_scale: float | None = None


class OptimizationResult(BaseModel):
    """Results from optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    quaternions_wxyz: NDArray[np.float64]  # (n_frames, 4) [w, x, y, z]
    translations: NDArray[np.float64]  # (n_frames, 3)
    keypoint_trajectories: NDArray[np.float64]  # (n_frames, n_keypoints, 3)
    reference_geometry: ReferenceGeometry  # Pydantic model
    initial_cost: float
    final_cost: float
    success: bool
    iterations: int
    time_seconds: float

    @field_validator("quaternions_wxyz", "translations", "keypoint_trajectories", mode="before")
    @classmethod
    def convert_to_numpy(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        return np.asarray(v, dtype=np.float64)

    @property
    def n_frames(self) -> int:
        return self.quaternions_wxyz.shape[0]

    @property
    def orientations(self) -> list[Quaternion]:
        """Convert quaternion array to list of Quaternion objects."""
        return [
            Quaternion(
                w=float(self.quaternions_wxyz[i, 0]),
                x=float(self.quaternions_wxyz[i, 1]),
                y=float(self.quaternions_wxyz[i, 2]),
                z=float(self.quaternions_wxyz[i, 3]),
            )
            for i in range(self.n_frames)
        ]

    @property
    def rotation_matricies(self) -> NDArray[np.float64]:
        """Get rotation matrices (n_frames, 3, 3)."""
        rotations = np.zeros((self.n_frames, 3, 3), dtype=np.float64)
        for i, q in enumerate(self.orientations):
            rotations[i] = q.to_rotation_matrix()
        return rotations


# =============================================================================
# COST FUNCTIONS
# =============================================================================


def _quaternion_point_jacobian(
    quat_wxyz: NDArray[np.float64], point: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Analytic Jacobian of R(q) @ point w.r.t. quat_wxyz = [w, x, y, z], evaluated at
    |q| = 1 (the quaternion parameter block stays on the unit sphere via
    QuaternionManifold, so this matches what Evaluate ever observes).

    Derived symbolically (sympy.diff on R(q/|q|) @ point w.r.t. the raw [w,x,y,z]
    components, so the derivative correctly accounts for the implicit renormalization,
    matching the finite-difference implementation it replaces) and validated against
    central differences to ~1e-7 before transcription; see
    `test_measurement_jacobian.py` for the ongoing regression check against that
    finite-difference reference.
    """
    w, x, y, z = quat_wxyz
    px, py, pz = point

    x0 = y**2
    x1 = z**2
    x2 = x0 + x1
    x3 = 2 * w**2
    x4 = x3 * z
    x5 = 2 * w
    x6 = x * x5
    x7 = x6 * y
    x8 = x7 + z
    x9 = x3 * y
    x10 = x6 * z
    x11 = x10 - y
    x12 = 2 * x
    x13 = x**2
    x14 = 2 * x13
    x15 = x14 * y
    x16 = x10 + y
    x17 = x14 * z
    x18 = x7 - z
    x19 = 2 * y
    x20 = x2 - 1
    x21 = 2 * x0
    x22 = x * x21
    x23 = y * z
    x24 = x23 * x5
    x25 = x + x24
    x26 = w * x21
    x27 = x12 * x23
    x28 = -w + x27
    x29 = 2 * x1
    x30 = w * x29
    x31 = w + x27
    x32 = x * x29
    x33 = -x + x24
    x34 = x1 + x13
    x35 = x * x3
    x36 = x34 - 1
    x37 = w * x14
    x38 = x21 * z
    x39 = 2 * z
    x40 = x29 * y
    x41 = x0 + x13
    x42 = x41 - 1

    jacobian = np.empty((3, 4), dtype=np.float64)
    jacobian[0, 0] = 4 * px * w * x2 - 2 * py * (-x4 + x8) - 2 * pz * (x11 + x9)
    jacobian[0, 1] = 2 * px * x12 * x2 + 2 * py * (-x15 + x16) - 2 * pz * (x17 + x18)
    jacobian[0, 2] = 2 * px * x19 * x20 + 2 * py * (-x22 + x25) - 2 * pz * (x26 + x28)
    jacobian[0, 3] = 4 * px * x20 * z - 2 * py * (-x30 + x31) - 2 * pz * (x32 + x33)
    jacobian[1, 0] = -2 * px * (x18 + x4) + 4 * py * w * x34 - 2 * pz * (x25 - x35)
    jacobian[1, 1] = -2 * px * (x11 + x15) + 4 * py * x * x36 - 2 * pz * (x31 - x37)
    jacobian[1, 2] = -2 * px * (x22 + x33) + 2 * py * x19 * x34 + 2 * pz * (-x38 + x8)
    jacobian[1, 3] = -2 * px * (x28 + x30) + 2 * py * x36 * x39 + 2 * pz * (x16 - x40)
    jacobian[2, 0] = -2 * px * (x16 - x9) - 2 * py * (x33 + x35) + 4 * pz * w * x41
    jacobian[2, 1] = 2 * px * (-x17 + x8) - 2 * py * (x28 + x37) + 2 * pz * x12 * x42
    jacobian[2, 2] = -2 * px * (-x26 + x31) - 2 * py * (x18 + x38) + 4 * pz * x42 * y
    jacobian[2, 3] = 2 * px * (x25 - x32) - 2 * py * (x11 + x40) + 2 * pz * x39 * x41
    return jacobian


class MeasurementFactor(pyceres.CostFunction):
    """Data fitting: measured point should match transformed reference point."""

    def __init__(
        self,
        *,
        measured_point: NDArray[np.float64],
        reference_point: NDArray[np.float64],
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.reference_point = reference_point.copy()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3])  # [quat, trans] only
        # Constant per-instance; hoisted out of Evaluate so it's not rebuilt every call.
        self._translation_jacobian = -np.eye(3, dtype=np.float64)

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64]] | None,
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        predicted = R @ self.reference_point + translation
        residuals[:] = self.measured_point - predicted

        if jacobians is not None:
            # Jacobian w.r.t. quaternion: residual = measured - R(q)@ref - trans,
            # so d(residual)/dq = -d(R(q)@ref)/dq.
            if jacobians[0] is not None:
                point_jacobian = _quaternion_point_jacobian(quat, self.reference_point)
                jacobians[0][:] = (-point_jacobian).ravel()

            # Jacobian w.r.t. translation
            if jacobians[1] is not None:
                jacobians[1][:] = self._translation_jacobian.ravel()

        return True


class FrameMeasurementFactor(pyceres.CostFunction):
    """Data fitting for one frame: all valid (non-NaN) measured keypoints in that
    frame vs. their transformed reference points, batched into a single residual
    block to avoid one Python/C++ Evaluate() boundary crossing per keypoint.

    Mathematically identical to adding `n_valid` separate MeasurementFactor
    residual blocks that share the same [quat, trans] parameter blocks — same
    residual values, same total Jacobian entries, just concatenated into one
    (3*n_valid,) residual vector instead of n_valid separate (3,) ones. See
    `test_frame_measurement_factor.py` for a bit-for-bit check of that equivalence.

    NOTE on robust loss: if a pyceres Loss is attached to this residual block, Ceres
    robustifies based on the combined norm of all n_valid stacked residuals, not each
    keypoint's own residual norm as happens with per-keypoint MeasurementFactor
    blocks. See OptimizationConfig.robust_loss_scale's docstring.
    """

    def __init__(
        self,
        *,
        measured_points: NDArray[np.float64],
        reference_points: NDArray[np.float64],
    ) -> None:
        super().__init__()
        self.measured_points = measured_points.copy()  # (n_valid, 3)
        self.reference_points = reference_points.copy()  # (n_valid, 3)
        self.n_valid = measured_points.shape[0]
        self.set_num_residuals(3 * self.n_valid)
        self.set_parameter_block_sizes([4, 3])  # [quat, trans] — one pose per frame

        # Translation Jacobian block is -I tiled n_valid times down the rows;
        # n_valid is fixed at construction, so precompute once.
        self._translation_jacobian = np.tile(
            -np.eye(3, dtype=np.float64), (self.n_valid, 1)
        )  # (3*n_valid, 3)
        self._unit_vectors = np.eye(3, dtype=np.float64)

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64]] | None,
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        predicted = (R @ self.reference_points.T).T + translation  # (n_valid, 3)
        residuals[:] = (self.measured_points - predicted).ravel()

        if jacobians is not None:
            if jacobians[0] is not None:
                # d(R(q) @ p)/dq is linear in p, so J(q, p) = sum_j p_j * J(q, e_j).
                # Evaluate the already-validated per-point formula at the 3 unit
                # vectors once, then combine for all n_valid points in one einsum —
                # cost no longer scales with a Python loop over keypoints, and no
                # second copy of the Jacobian formula is needed.
                basis = np.stack(
                    [_quaternion_point_jacobian(quat, e) for e in self._unit_vectors]
                )  # (3, 3, 4): basis[j] = J(q, e_j)
                point_jacobians = np.einsum("kj,jab->kab", self.reference_points, basis)
                jacobians[0][:] = (-point_jacobians).ravel()

            if jacobians[1] is not None:
                jacobians[1][:] = self._translation_jacobian.ravel()

        return True


class RotationSmoothnessFactor(pyceres.CostFunction):
    """Rotation smoothness between consecutive frames."""

    def __init__(self, *, weight: float = 5.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(4)
        self.set_parameter_block_sizes([4, 4])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64]] | None,
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

    def __init__(self, *, weight: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64]] | None,
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
    original_trajectories: NDArray[np.float64],
    rigid_edges: list[tuple[str, str]],
    config: OptimizationConfig,
    keypoint_names: list[str],
    display_edges: list[tuple[str, str]] | None = None,
    body_frame_origin_keypoints: list[str],
    body_frame_x_axis_keypoint: str,
    body_frame_y_axis_keypoint: str,
    units: str = "mm",
) -> OptimizationResult:
    """
    Optimize poses with FIXED reference geometry.

    Reference geometry is estimated once from the data (median distances + MDS)
    and then held constant. Only per-frame poses (rotation, translation) are optimized.

    Args:
        original_trajectories: (n_frames, n_keypoints, 3) measured positions (may contain NaN)
        rigid_edges: List of (name_i, name_j) pairs defining rigid body structure
        config: OptimizationConfig
        keypoint_names: Marker names
        display_edges: Edges to show in visualization (defaults to rigid_edges)
        body_frame_origin_keypoints: Marker names whose mean defines the origin
        body_frame_x_axis_keypoint: Marker name that defines X-axis direction
        body_frame_y_axis_keypoint: Marker name that defines Y-axis direction
        units: Units for the reference geometry ("mm" or "m")

    Returns:
        OptimizationResult with fixed reference geometry and optimized poses
    """
    n_frames, n_keypoints, _ = original_trajectories.shape

    if display_edges is None:
        display_edges = rigid_edges

    logger.info("=" * 80)
    logger.info("POSE OPTIMIZATION (FIXED REFERENCE GEOMETRY)")
    logger.info("=" * 80)
    logger.info(f"Frames: {n_frames}, Markers: {n_keypoints}")

    # =========================================================================
    # INITIALIZE REFERENCE GEOMETRY
    # =========================================================================
    reference_geometry_model, reference_geometry_array = estimate_reference_geometry(
        original_data=original_trajectories,
        keypoint_names=keypoint_names,
        origin_keypoints=body_frame_origin_keypoints,
        x_axis_keypoint=body_frame_x_axis_keypoint,
        y_axis_keypoint=body_frame_y_axis_keypoint,
        units=units,
        display_edges=display_edges,
        rigid_edges=rigid_edges,
    )

    logger.info("Reference geometry is FIXED (will not be optimized)")

    # =========================================================================
    # COUNT VALID MEASUREMENTS
    # =========================================================================
    valid_mask = ~np.isnan(original_trajectories).any(axis=2)
    n_valid_measurements = valid_mask.sum()
    n_total_measurements = n_frames * n_keypoints
    n_missing = n_total_measurements - n_valid_measurements

    logger.info("\nMeasurement statistics:")
    logger.info(f"  Total possible: {n_total_measurements}")
    logger.info(
        f"  Valid:          {n_valid_measurements} ({100 * n_valid_measurements / n_total_measurements:.1f}%)"
    )
    logger.info(
        f"  Missing (NaN):  {n_missing} ({100 * n_missing / n_total_measurements:.1f}%)"
    )

    # =========================================================================
    # INITIALIZE POSES
    # =========================================================================
    logger.info("\nInitializing poses...")
    poses: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    last_good_pose: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    for frame_idx in range(n_frames):
        pose = _estimate_initial_pose(
            measured_points=original_trajectories[frame_idx],
            reference_points=reference_geometry_array,
            fallback_pose=last_good_pose,
        )
        poses.append(pose)
        last_good_pose = pose

    # =========================================================================
    # BUILD OPTIMIZATION PROBLEM
    # =========================================================================
    logger.info("\nBuilding optimization problem...")
    problem = pyceres.Problem()

    for quat, trans in poses:
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(trans, 3)
        problem.set_manifold(quat, pyceres.QuaternionManifold())

    # DATA FITTING
    logger.info("  Adding measurement factors (skipping NaN)...")
    measurement_loss = (
        pyceres.HuberLoss(config.robust_loss_scale) if config.robust_loss_scale is not None else None
    )
    if config.robust_loss_scale is not None:
        logger.warning(
            "robust_loss_scale is set; measurement residuals are batched per-frame, so "
            "Huber now robustifies on each frame's combined residual norm across all "
            "valid keypoints, not per-keypoint as in a pre-batching regime — recalibrate "
            "before trusting this."
        )

    n_measurement_factors = 0
    n_skipped_empty_frames = 0
    for frame_idx in range(n_frames):
        quat, trans = poses[frame_idx]
        frame_valid_mask = valid_mask[frame_idx]
        n_valid = int(frame_valid_mask.sum())

        if n_valid == 0:
            # Matches prior per-keypoint behavior: a frame with zero valid keypoints
            # simply gets no measurement factor (it still gets smoothness factors to
            # its neighbors).
            n_skipped_empty_frames += 1
            continue

        cost = FrameMeasurementFactor(
            measured_points=original_trajectories[frame_idx, frame_valid_mask],
            reference_points=reference_geometry_array[frame_valid_mask],
        )
        problem.add_residual_block(cost, measurement_loss, [quat, trans])
        n_measurement_factors += n_valid

    logger.info(
        f"    Added {n_measurement_factors} measurement residuals across "
        f"{n_frames - n_skipped_empty_frames} frame factors "
        f"({n_skipped_empty_frames} frames skipped, 0 valid keypoints)"
    )

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
    translations = np.zeros((n_frames, 3), dtype=np.float64)
    reconstructed_keypoints = np.zeros((n_frames, n_keypoints, 3), dtype=np.float64)
    quaternions_wxyz = np.zeros((n_frames, 4), dtype=np.float64)

    for frame_idx in range(n_frames):
        quat_ceres = poses[frame_idx][0]
        trans = poses[frame_idx][1]

        # Ceres uses wxyz format
        quaternions_wxyz[frame_idx] = quat_ceres

        # Convert to scipy format (xyzw) for rotation matrix
        quat_scipy = np.array([quat_ceres[1], quat_ceres[2], quat_ceres[3], quat_ceres[0]])
        R = Rotation.from_quat(quat_scipy).as_matrix()

        translations[frame_idx] = trans
        reconstructed_keypoints[frame_idx] = (R @ reference_geometry_array.T).T + trans

    return OptimizationResult(
        quaternions_wxyz=quaternions_wxyz,
        translations=translations,
        keypoint_trajectories=reconstructed_keypoints,
        reference_geometry=reference_geometry_model,
        initial_cost=summary.initial_cost,
        final_cost=summary.final_cost,
        success=summary.termination_type == pyceres.TerminationType.CONVERGENCE,
        iterations=summary.num_successful_steps,
        time_seconds=summary.total_time_in_seconds,
    )
