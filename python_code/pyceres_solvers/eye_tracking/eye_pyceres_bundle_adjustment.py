"""Bundle adjustment for eye tracking with PyCeres."""
import os

import numpy as np
from scipy.spatial.transform import Rotation
from pydantic import BaseModel, Field
from numpydantic import NDArray, Shape
import pyceres

from python_code.pyceres_solvers.eye_tracking.camera_model import CameraIntrinsics, project_point


class OptimizationConfig(BaseModel):
    """Configuration for bundle adjustment."""

    max_iterations: int = 500
    function_tolerance: float = 1e-8
    gradient_tolerance: float = 1e-12
    parameter_tolerance: float = 1e-10

    use_huber_loss: bool = True
    huber_delta_px: float = 2.0

    pupil_weight: float = 1.0
    tear_duct_weight: float = 1.0
    rotation_smoothness_weight: float = 10.0
    scale_smoothness_weight: float = 5.0



class EyeModel(BaseModel):
    """Static eye model parameters."""

    model_config = {"arbitrary_types_allowed": True}

    eyeball_center_mm: NDArray[Shape["3"], float]
    base_semi_major_mm: float = Field(gt=0)
    base_semi_minor_mm: float = Field(gt=0)
    pupil_roundness: float = Field(gt=-4, lt=4)  # n in superellipse, 2=ellipse, <2=pointy ellipse, >2=rounded rectangle
    tear_duct_xyz_mm: NDArray[Shape["3 xyz"], float]

    @classmethod
    def create_initial_guess(
        cls,
        *,
        eyeball_distance_mm: float = 20.0,
        base_semi_major_mm: float = 2.0,
        base_semi_minor_mm: float = 1.5,
        pupil_roundness: float = 2.0,
        tear_duct_xyz_mm: NDArray[Shape["3 xyz"], float] | None = None
    ) -> "EyeModel":
        """Create reasonable initial parameter guess."""
        if tear_duct_xyz_mm is None:
            tear_duct_xyz_mm = np.array([2.0, 1.0, 0.0])

        return cls(
            eyeball_center_mm=np.array([0.0, 0.0, eyeball_distance_mm]),
            base_semi_major_mm=base_semi_major_mm,
            base_semi_minor_mm=base_semi_minor_mm,
            pupil_roundness=pupil_roundness,
            tear_duct_xyz_mm=tear_duct_xyz_mm
        )


class OptimizationResult(BaseModel):
    """Results from bundle adjustment."""

    model_config = {"arbitrary_types_allowed": True}

    # Optimized parameters
    eye_model: EyeModel
    eye_quaternions: NDArray[Shape["*, 4"], float]
    pupil_dialations: NDArray[Shape["* frame_number"], float]

    # Computed outputs
    gaze_directions: NDArray[Shape["*, 3"], float]
    pupil_centers_3d_mm: NDArray[Shape["*, 3"], float]
    tear_ducts_3d_mm: NDArray[Shape["*, 3"], float]

    # Reprojections
    projected_pupil_points_px: NDArray[Shape["*, 8, 2"], float]
    projected_pupil_centers_px: NDArray[Shape["*, 2"], float]
    projected_tear_ducts_px: NDArray[Shape["*, 2"], float]

    # Errors
    pupil_point_errors_px: NDArray[Shape["*, 8"], float]
    pupil_center_errors_px: NDArray[Shape["* frame_number"], float]
    tear_duct_errors_px: NDArray[Shape["*  frame_number"], float]

    # Optimization info
    success: bool
    num_iterations: int
    initial_cost: float
    final_cost: float

    @property
    def mean_pupil_error_px(self) -> float:
        """Mean pupil center reprojection error."""
        return float(np.mean(self.pupil_center_errors_px))

    @property
    def mean_tear_duct_error_px(self) -> float:
        """Mean tear duct reprojection error."""
        return float(np.mean(self.tear_duct_errors_px))


class PupilPointCost(pyceres.CostFunction):
    """Reprojection cost for one pupil boundary point."""

    def __init__(
        self,
        *,
        observed_px: np.ndarray,
        point_index: int,
        camera: CameraIntrinsics,
        weight: float
    ) -> None:
        super().__init__()
        self.observed_px = observed_px.copy()
        self.point_index = point_index
        self.camera = camera
        self.weight = weight
        self.angle = 2.0 * np.pi * point_index / 8.0

        self.set_num_residuals(2)
        # Parameters: [quaternion(4), eyeball_center(3), pupil_params(3), scale(1)]
        self.set_parameter_block_sizes([4, 3, 3, 1])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        eyeball_center = parameters[1]
        base_semi_major = parameters[2][0]
        base_semi_minor = parameters[2][1]
        # roundness = parameters[2][4]
        dilation = parameters[3][0]

        # Scale the axes
        semi_major = base_semi_major * dilation
        semi_minor = base_semi_minor * dilation

        #################
        ###############
        ### NOTE - I think this is wrong? We should be estimating the pupil ellipse as a thing on the eye sphere surface, scaled by dilation, and then rotated by pupil rotation before be projected on to the eye camera and compared with that frames's measured pupil point.
        ### THERE IS NO PUPIL ROTAATION TERM _- THe PUPIL ROTATION ARISES FROM THE EYE ROTATION QUATERNION
        ### THERE IS NO PUPIL DEPTH TERM - THE PUPIL IS ON THE EYE SPHERE SURFACE
        ############### NEEDS FIXING!!!

        # # Generate superellipse point in canonical 2D space
        # cos_angle = np.cos(self.angle)
        # sin_angle = np.sin(self.angle)
        # exponent = 2.0 / roundness
        #
        # x_canonical = semi_major * np.sign(cos_angle) * np.abs(cos_angle) ** exponent
        # y_canonical = semi_minor * np.sign(sin_angle) * np.abs(sin_angle) ** exponent
        #
        # # Apply in-plane rotation
        # cos_rot = np.cos(pupil_rotation)
        # sin_rot = np.sin(pupil_rotation)
        # x_local = cos_rot * x_canonical - sin_rot * y_canonical
        # y_local = sin_rot * x_canonical + cos_rot * y_canonical
        #
        # # Create 3D point on eyeball surface (pupil is at depth z from center)
        # point_local = np.array([x_local, y_local, pupil_depth])
        #
        # # Transform to camera frame using eye orientation
        # R = Rotation.from_quat(quat=quat)
        # point_cam = eyeball_center + R.applypoint_local)
        #
        # # Project to image
        # projected = project_point(point_3d=point_cam, camera=self.camera)
        #
        # # Residual (observed - predicted)
        # residuals[0] = self.weight * (self.observed_px[0] - projected[0])
        # residuals[1] = self.weight * (self.observed_px[1] - projected[1])

        return True


class TearDuctCost(pyceres.CostFunction):
    """Reprojection cost for tear duct."""

    def __init__(
        self,
        *,
        observed_px: np.ndarray,
        camera: CameraIntrinsics,
        weight: float
    ) -> None:
        super().__init__()
        self.observed_px = observed_px.copy()
        self.camera = camera
        self.weight = weight

        self.set_num_residuals(2)
        # Parameters: [quaternion(4), eyeball_center(3), tear_duct_local(3)]
        self.set_parameter_block_sizes([4, 3, 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        eyeball_center = parameters[1]
        tear_duct_local = parameters[2]

        # Transform to camera frame
        R = Rotation.from_quat(quat=quat)
        tear_duct_cam = eyeball_center + R.apply(tear_duct_local)

        # Project
        projected = project_point(point_3d=tear_duct_cam, camera=self.camera)

        # Residual
        residuals[0] = self.weight * (self.observed_px[0] - projected[0])
        residuals[1] = self.weight * (self.observed_px[1] - projected[1])

        return True


class RotationSmoothnessCost(pyceres.CostFunction):
    """Temporal smoothness for rotation."""

    def __init__(self, *, weight: float) -> None:
        super().__init__()
        self.weight = weight

        self.set_num_residuals(1)
        # Parameters: [quat_t(4), quat_t1(4)]
        self.set_parameter_block_sizes([4, 4])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat_t = parameters[0]
        quat_t1 = parameters[1]

        # Compute angular distance
        R_t = Rotation.from_quat(quat=quat_t)
        R_t1 = Rotation.from_quat(quat=quat_t1)
        R_diff = R_t.inv() * R_t1

        angle = R_diff.magnitude()

        residuals[0] = self.weight * angle

        return True


class ScaleSmoothnessCost(pyceres.CostFunction):
    """Temporal smoothness for pupil dilation."""

    def __init__(self, *, weight: float) -> None:
        super().__init__()
        self.weight = weight

        self.set_num_residuals(1)
        # Parameters: [scale_t(1), scale_t1(1)]
        self.set_parameter_block_sizes([1, 1])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        scale_t = parameters[0][0]
        scale_t1 = parameters[1][0]

        residuals[0] = self.weight * (scale_t1 - scale_t)

        return True


def optimize_eye_tracking_data(
    *,
    observed_pupil_points_px: NDArray[Shape["*, 8, 2"], float],
    observed_tear_ducts_px: NDArray[Shape["*, 2"], float],
    camera: CameraIntrinsics,
    initial_eye_model: EyeModel,
    config: OptimizationConfig
) -> OptimizationResult:
    """
    Optimize eye tracking parameters using bundle adjustment.

    Args:
        observed_pupil_points_px: (N, 8, 2) pupil boundary points
        observed_tear_ducts_px: (N, 2) tear duct positions
        camera: Camera intrinsics
        initial_eye_model: Initial parameter guess
        config: Optimization settings

    Returns:
        OptimizationResult with optimized parameters and diagnostics
    """
    n_frames = len(observed_pupil_points_px)

    print(f"\n{'='*80}")
    print("EYE TRACKING BUNDLE ADJUSTMENT")
    print(f"{'='*80}")
    print(f"Frames: {n_frames}")
    print(f"Observations: {n_frames * 18} (8×2 pupil + 1×2 tear duct per frame)")
    print(f"Parameters: {n_frames * 5} (per-frame: 4 quaternion + 1 scale)")
    print(f"            + 13 global (3 center + 5 pupil + 3 tear duct + 2 extra)")
    print(f"{'='*80}\n")

    # Initialize parameters
    quaternions = np.zeros(shape=(n_frames, 4))
    quaternions[:, 3] = 1.0  # Identity rotations (w=1 for scipy quaternions)

    pupil_dilations = np.ones(shape=(n_frames, 1))

    # Global parameters - make mutable copies
    eyeball_center = initial_eye_model.eyeball_center_mm.copy()
    pupil_params = np.array([
        initial_eye_model.base_semi_major_mm,
        initial_eye_model.base_semi_minor_mm,
        initial_eye_model.pupil_roundness
    ])
    tear_duct_local = initial_eye_model.tear_duct_xyz_mm.copy()

    # Build problem
    problem = pyceres.Problem()

    loss = pyceres.HuberLoss(config.huber_delta_px) if config.use_huber_loss else None

    # Add pupil point costs
    for i in range(n_frames):
        for j in range(8):
            cost = PupilPointCost(
                observed_px=observed_pupil_points_px[i, j],
                point_index=j,
                camera=camera,
                weight=config.pupil_weight
            )
            problem.add_residual_block(
                cost=cost,
                loss=loss,
                paramv=[quaternions[i], eyeball_center, pupil_params, pupil_dilations[i]]
            )

        # Tear duct cost
        cost_td = TearDuctCost(
            observed_px=observed_tear_ducts_px[i],
            camera=camera,
            weight=config.tear_duct_weight
        )
        problem.add_residual_block(
            cost=cost_td,
            loss=loss,
            paramv=[quaternions[i], eyeball_center, tear_duct_local]
        )

        # Set quaternion manifold
        problem.set_manifold(
            quaternions[i],
            pyceres.QuaternionManifold()
        )

    # Add smoothness costs
    if config.rotation_smoothness_weight > 0:
        for i in range(n_frames - 1):
            cost_smooth = RotationSmoothnessCost(weight=config.rotation_smoothness_weight)
            problem.add_residual_block(
                cost=cost_smooth,
                loss=None,
                paramv=[quaternions[i], quaternions[i + 1]]
            )

    if config.scale_smoothness_weight > 0:
        for i in range(n_frames - 1):
            cost_smooth = ScaleSmoothnessCost(weight=config.scale_smoothness_weight)
            problem.add_residual_block(
                cost=cost_smooth,
                loss=None,
                paramv=[pupil_dilations[i], pupil_dilations[i + 1]]
            )

    # Set bounds on scale
    for i in range(n_frames):
        problem.set_parameter_lower_bound(pupil_dilations[i], 0, 0.3)
        problem.set_parameter_upper_bound(pupil_dilations[i], 0, 3.0)

    # Configure solver
    options = pyceres.SolverOptions()
    options.max_num_iterations = config.max_iterations
    options.function_tolerance = config.function_tolerance
    options.gradient_tolerance = config.gradient_tolerance
    options.parameter_tolerance = config.parameter_tolerance
    options.num_threads = max(os.cpu_count()-1 if os.cpu_count() is not None else 1, 1)
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.trust_region_strategy_type = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    options.minimizer_progress_to_stdout = True

    # Solve
    summary = pyceres.SolverSummary()
    pyceres.solve(options,
                  problem,
                  summary)

    print("\n" + summary.BriefReport())
    print(f"\nOptimized Parameters:")
    print(f"  Eyeball center: {eyeball_center}")
    print(f"  Pupil scales: {pupil_dilations.mean():.3f} ± {pupil_dilations.std():.3f}")
    print(f"  Tear duct local: {tear_duct_local}")

    # Compute outputs
    optimized_eye_model = EyeModel(
        eyeball_center_mm=eyeball_center,
        base_semi_major_mm=float(pupil_params[0]),
        base_semi_minor_mm=float(pupil_params[1]),
        pupil_roundness=float(pupil_params[2]),
        tear_duct_xyz_mm=tear_duct_local
    )

    # Compute 3D positions and projections
    rotations = [Rotation.from_quat(quat=q) for q in quaternions]
    gaze_directions = np.array([R.apply(np.array([0, 0, 1])) for R in rotations])

    pupil_centers_3d = np.array([
        eyeball_center + R.apply(np.array([0, 0, pupil_params[0]]))
        for R in rotations
    ])

    tear_ducts_3d = np.array([
        eyeball_center + R.apply(tear_duct_local)
        for R in rotations
    ])

    # Compute reprojections and errors
    projected_pupil_points = np.zeros(shape=(n_frames, 8, 2))
    pupil_point_errors = np.zeros(shape=(n_frames, 8))

    for i in range(n_frames):
        for j in range(8):
            # Regenerate the point using optimized parameters
            angle = 2.0 * np.pi * j / 8.0
            scale = pupil_dilations[i, 0]
            semi_major = pupil_params[0] * scale
            semi_minor = pupil_params[1] * scale

            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            exponent = 2.0

            x_can = semi_major * np.sign(cos_angle) * np.abs(cos_angle) ** exponent
            y_can = semi_minor * np.sign(sin_angle) * np.abs(sin_angle) ** exponent

            cos_rot = np.cos(pupil_params[2])
            sin_rot = np.sin(pupil_params[2])
            x_local = cos_rot * x_can - sin_rot * y_can
            y_local = sin_rot * x_can + cos_rot * y_can
            x_local = x_can
            y_local = y_can

            point_local = np.array([x_local, y_local, pupil_params[0]])
            point_cam = eyeball_center + rotations[i].apply(point_local)

            proj = project_point(point_3d=point_cam, camera=camera)
            projected_pupil_points[i, j] = proj

            error = np.linalg.norm(observed_pupil_points_px[i, j] - proj)
            pupil_point_errors[i, j] = error

    projected_pupil_centers = project_point(point_3d=pupil_centers_3d, camera=camera)
    projected_tear_ducts = project_point(point_3d=tear_ducts_3d, camera=camera)

    # Compute centers from observed points
    observed_centers = observed_pupil_points_px.mean(axis=1)
    pupil_center_errors = np.linalg.norm(observed_centers - projected_pupil_centers, axis=1)
    tear_duct_errors = np.linalg.norm(observed_tear_ducts_px - projected_tear_ducts, axis=1)

    success = (
        summary.termination_type == pyceres.TerminationType.CONVERGENCE or
        summary.termination_type == pyceres.TerminationType.USER_SUCCESS
    )

    return OptimizationResult(
        eye_model=optimized_eye_model,
        eye_quaternions=quaternions,
        pupil_dialations=pupil_dilations.flatten(),
        gaze_directions=gaze_directions,
        pupil_centers_3d_mm=pupil_centers_3d,
        tear_ducts_3d_mm=tear_ducts_3d,
        projected_pupil_points_px=projected_pupil_points,
        projected_pupil_centers_px=projected_pupil_centers,
        projected_tear_ducts_px=projected_tear_ducts,
        pupil_point_errors_px=pupil_point_errors,
        pupil_center_errors_px=pupil_center_errors,
        tear_duct_errors_px=tear_duct_errors,
        success=success,
        num_iterations=summary.num_successful_steps + summary.num_unsuccessful_steps,
        initial_cost=summary.initial_cost,
        final_cost=summary.final_cost
    )