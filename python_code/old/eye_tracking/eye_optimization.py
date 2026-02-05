"""Optimizer to estimate 3D eye orientation from pupil center observations."""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from dataclasses import dataclass

from eye_camera_model import CameraIntrinsics, project_point_to_image


@dataclass
class OptimizationConfig:
    """Configuration for eye orientation optimization."""

    optimization_method: str = "Powell"
    """Scipy optimization method ('Powell', 'Nelder-Mead', 'L-BFGS-B', etc.)"""

    max_iterations: int = 200
    """Maximum number of optimization iterations"""

    max_rotation_degrees: float | None = 45.0
    """Maximum rotation angle constraint (None for unbounded)"""

    use_temporal_init: bool = True
    """Use previous frame's result as initialization for batch processing"""

    convergence_tolerance: float = 1e-6
    """Optimization convergence tolerance"""

    verbose: bool = False
    """Print optimization progress"""


@dataclass
class EyeModel:
    """3D eye model parameters."""

    eyeball_center_mm: np.ndarray
    """(3,) position of eyeball center in camera frame (mm)"""

    eyeball_radius_mm: float
    """Radius of eyeball sphere (mm)"""

    pupil_offset_mm: float
    """Distance from eyeball center to pupil along optical axis (mm)"""

    def __post_init__(self) -> None:
        """Validate parameters."""
        assert self.eyeball_center_mm.shape == (3,), "eyeball_center must be (3,)"
        assert self.pupil_offset_mm <= self.eyeball_radius_mm, \
            "pupil_offset must be <= eyeball_radius"


@dataclass
class GazeEstimate:
    """Result of gaze estimation."""

    eye_orientation: Rotation
    """3D rotation of the eye from neutral position"""

    pupil_center_3d_mm: np.ndarray
    """(3,) 3D position of pupil center in camera frame (mm)"""

    projected_pixel_px: np.ndarray
    """(2,) projected pixel coordinates"""

    reprojection_error_px: float
    """Reprojection error in pixels"""

    gaze_direction: np.ndarray
    """(3,) normalized gaze direction vector"""

    success: bool
    """Whether optimization converged successfully"""


def estimate_eye_orientation(
    *,
    observed_pupil_center_px: np.ndarray,
    eye_model: EyeModel,
    camera: CameraIntrinsics,
    config: OptimizationConfig,
    initial_orientation: Rotation | None = None
) -> GazeEstimate:
    """
    Estimate 3D eye orientation from observed pupil center.

    This function finds the eye rotation that minimizes the reprojection error
    between the observed pupil center and the projected pupil center.

    Eye coordinate system:
    - Origin at eyeball center
    - Neutral gaze (identity rotation) looks along +Z axis (into camera)
    - Pupil is at (0, 0, pupil_offset_mm) in eye frame

    Args:
        observed_pupil_center_px: (2,) observed pupil center in pixels
        eye_model: Eye model with geometry parameters
        camera: Camera intrinsics for projection
        config: Optimization configuration
        initial_orientation: Initial guess for eye orientation (default: identity)

    Returns:
        GazeEstimate with optimized orientation and diagnostics
    """
    assert observed_pupil_center_px.shape == (2,), "observed_pupil_center must be (2,)"

    if initial_orientation is None:
        initial_orientation = Rotation.identity()

    x0 = initial_orientation.as_rotvec()

    def objective(rotvec: np.ndarray) -> float:
        """Compute reprojection error for given eye orientation."""
        R = Rotation.from_rotvec(rotvec)

        # Pupil position in eye frame (looking along +Z when neutral)
        pupil_in_eye_frame = np.array([0.0, 0.0, eye_model.pupil_offset_mm])

        # Transform to camera frame
        pupil_in_camera_frame = (
            eye_model.eyeball_center_mm + R.apply(pupil_in_eye_frame)
        )

        # Project to image plane
        projected = project_point_to_image(
            point_3d=pupil_in_camera_frame,
            camera=camera
        )

        # Compute L2 error in pixels
        error = np.linalg.norm(projected - observed_pupil_center_px)

        return error

    # Set up bounds if requested
    bounds = None
    if (config.max_rotation_degrees is not None
        and config.optimization_method in ["L-BFGS-B", "TNC", "SLSQP"]):
        max_angle_rad = np.deg2rad(config.max_rotation_degrees)
        bounds = [(-max_angle_rad, max_angle_rad)] * 3

    # Optimization options
    options = {
        "maxiter": config.max_iterations,
        "disp": config.verbose
    }

    if config.optimization_method in ["Powell", "Nelder-Mead"]:
        options["ftol"] = config.convergence_tolerance
    elif config.optimization_method in ["L-BFGS-B", "TNC"]:
        options["ftol"] = config.convergence_tolerance
        options["gtol"] = config.convergence_tolerance

    # Run optimization
    result = minimize(
        fun=objective,
        x0=x0,
        method=config.optimization_method,
        bounds=bounds,
        options=options
    )

    # Extract optimal solution
    optimal_rotvec = result.x
    optimal_rotation = Rotation.from_rotvec(optimal_rotvec)

    # Compute final 3D pupil position
    pupil_in_eye_frame = np.array([0.0, 0.0, eye_model.pupil_offset_mm])
    optimal_pupil_3d = (
        eye_model.eyeball_center_mm + optimal_rotation.apply(pupil_in_eye_frame)
    )

    # Project to get final pixel coordinates
    projected_pixel = project_point_to_image(
        point_3d=optimal_pupil_3d,
        camera=camera
    )

    # Compute gaze direction (normalized)
    gaze_direction = optimal_rotation.apply(np.array([0.0, 0.0, 1.0]))

    return GazeEstimate(
        eye_orientation=optimal_rotation,
        pupil_center_3d_mm=optimal_pupil_3d,
        projected_pixel_px=projected_pixel,
        reprojection_error_px=float(result.fun),
        gaze_direction=gaze_direction,
        success=result.success
    )


def batch_estimate_eye_orientation(
    *,
    observed_pupil_centers_px: np.ndarray,
    eye_model: EyeModel,
    camera: CameraIntrinsics,
    config: OptimizationConfig
) -> list[GazeEstimate]:
    """
    Estimate eye orientation for multiple frames.

    Args:
        observed_pupil_centers_px: (N, 2) observed pupil centers
        eye_model: Eye model parameters
        camera: Camera intrinsics
        config: Optimization configuration

    Returns:
        List of GazeEstimate objects, one per frame
    """
    n_frames = len(observed_pupil_centers_px)
    results: list[GazeEstimate] = []

    for i, pupil_center in enumerate(observed_pupil_centers_px):
        if config.verbose:
            print(f"Processing frame {i+1}/{n_frames}")

        # Use previous result as initialization if requested
        init_rot = None
        if config.use_temporal_init and len(results) > 0 and results[-1].success:
            init_rot = results[-1].eye_orientation

        estimate = estimate_eye_orientation(
            observed_pupil_center_px=pupil_center,
            eye_model=eye_model,
            camera=camera,
            config=config,
            initial_orientation=init_rot
        )

        results.append(estimate)

    return results


def create_default_eye_model(
    *,
    eyeball_center_z_mm: float = 20.0,
    eyeball_radius_mm: float = 12.0
) -> EyeModel:
    """
    Create a default eye model with typical parameters.

    Args:
        eyeball_center_z_mm: Distance from camera to eyeball center (mm)
        eyeball_radius_mm: Eyeball radius (mm, typical: 11-13mm)

    Returns:
        EyeModel with default parameters
    """
    return EyeModel(
        eyeball_center_mm=np.array([0.0, 0.0, eyeball_center_z_mm]),
        eyeball_radius_mm=eyeball_radius_mm,
        pupil_offset_mm=eyeball_radius_mm  # Pupil on sphere surface
    )