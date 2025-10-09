import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import svd
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Any

# Configure logging with function name and line number
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s'
)
logger = logging.getLogger(__name__)


def generate_cube_vertices(size: float = 0.5) -> np.ndarray:
    """Generate 8 vertices of a cube centered at origin."""
    s = size
    vertices = np.array([
        [-s, -s, -s],
        [s, -s, -s],
        [s, s, -s],
        [-s, s, -s],
        [-s, -s, s],
        [s, -s, s],
        [s, s, s],
        [-s, s, s],
    ])
    return vertices


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix from axis-angle representation."""
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def rotation_error_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation error as angle in degrees between two rotation matrices."""
    R_error = R1.T @ R2
    trace = np.trace(R_error)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(angle_rad)


def apply_butterworth_filter(
        data: np.ndarray,
        cutoff_freq: float = 0.1,
        sampling_rate: float = 1.0,
        order: int = 4
) -> np.ndarray:
    """
    Apply zero-lag Butterworth low-pass filter to trajectory data.

    Args:
        data: (n_frames, n_dims) array to filter
        cutoff_freq: Cutoff frequency (normalized to Nyquist if < 1)
        sampling_rate: Sampling rate in Hz
        order: Filter order

    Returns:
        filtered: (n_frames, n_dims) filtered data
    """
    n_frames, n_dims = data.shape

    if n_frames < 2 * order:
        logger.warning(f"Not enough frames ({n_frames}) for filter order {order}, skipping filtering")
        return data

    # Design Butterworth filter
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist

    # Ensure cutoff is valid
    normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)

    b, a = butter(N=order, Wn=normalized_cutoff, btype='low', analog=False)

    # Apply zero-lag filter to each dimension
    filtered = np.zeros_like(data)
    for dim in range(n_dims):
        filtered[:, dim] = filtfilt(b=b, a=a, x=data[:, dim])

    return filtered


def generate_ground_truth_trajectory(
        n_frames: int,
        cube_size: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ground truth rigid body trajectory.

    Returns:
        rotations: (n_frames, 3, 3) rotation matrices
        translations: (n_frames, 3) translation vectors
        marker_positions: (n_frames, 9, 3) positions of 8 vertices + center
    """
    logger.info(f"Generating ground truth trajectory with {n_frames} frames")

    base_vertices = generate_cube_vertices(size=cube_size)
    n_markers = len(base_vertices) + 1

    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))
    marker_positions = np.zeros((n_frames, n_markers, 3))

    for i in range(n_frames):
        t = i / n_frames

        # Circular trajectory with vertical oscillation
        radius = 3.0
        translation = np.array([
            radius * np.cos(t * 2 * np.pi),
            radius * np.sin(t * 2 * np.pi),
            1.5 * np.sin(t * 4 * np.pi)
        ])

        # Rotation around an axis
        rot_axis = np.array([0.3, 1.0, 0.2])
        rot_angle = t * 4 * np.pi
        R = rotation_matrix_from_axis_angle(axis=rot_axis, angle=rot_angle)

        # Transform vertices
        transformed_vertices = (R @ base_vertices.T).T + translation

        rotations[i] = R
        translations[i] = translation
        marker_positions[i, :8, :] = transformed_vertices
        # Center is mean of the 8 vertices
        marker_positions[i, 8, :] = np.mean(transformed_vertices, axis=0)

    return rotations, translations, marker_positions


def add_noise_to_measurements(
        marker_positions: np.ndarray,
        noise_std: float = 0.3,
        seed: int | None = 42
) -> np.ndarray:
    """Add Gaussian noise to marker positions."""
    logger.info(f"Adding noise with std={noise_std * 1000:.1f} mm")

    if seed is not None:
        np.random.seed(seed=seed)

    n_frames, n_markers, _ = marker_positions.shape
    noisy_positions = marker_positions.copy()

    # Add noise only to the 8 vertices (not the center at index 8)
    noise = np.random.normal(loc=0, scale=noise_std, size=(n_frames, 8, 3))
    noisy_positions[:, :8, :] += noise

    # Recalculate center as mean of noisy vertices
    noisy_positions[:, 8, :] = np.mean(noisy_positions[:, :8, :], axis=1)

    return noisy_positions


def fit_rigid_transform_kabsch(
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
        noisy_measurements: np.ndarray,
        reference_geometry: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize trajectory using frame-by-frame Kabsch."""
    logger.info("Running Kabsch initialization...")

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

    return rotations, translations


def poses_to_vector(rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """
    Convert rotations and translations to optimization vector.

    Args:
        rotations: (n_frames, 3, 3) rotation matrices
        translations: (n_frames, 3) translation vectors

    Returns:
        x: (n_frames * 6,) vector [rotvec_0, trans_0, rotvec_1, trans_1, ...]
    """
    n_frames = rotations.shape[0]
    x = np.zeros(n_frames * 6)

    for i in range(n_frames):
        rotvec = Rotation.from_matrix(rotations[i]).as_rotvec()
        x[i * 6:i * 6 + 3] = rotvec
        x[i * 6 + 3:i * 6 + 6] = translations[i]

    return x


def vector_to_poses(x: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert optimization vector to rotations and translations.

    Args:
        x: (n_frames * 6,) vector
        n_frames: number of frames

    Returns:
        rotations: (n_frames, 3, 3) rotation matrices
        translations: (n_frames, 3) translation vectors
    """
    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))

    for i in range(n_frames):
        rotvec = x[i * 6:i * 6 + 3]
        rotations[i] = Rotation.from_rotvec(rotvec).as_matrix()
        translations[i] = x[i * 6 + 3:i * 6 + 6]

    return rotations, translations


def trajectory_objective_and_gradient(
        x: np.ndarray,
        noisy_measurements: np.ndarray,
        reference_geometry: np.ndarray,
        lambda_smooth_pos: float = 10.0,
        lambda_smooth_rot: float = 5.0,
        lambda_accel: float = 1.0,
        compute_gradient: bool = True
) -> tuple[float, np.ndarray | None]:
    """
    Objective function and analytical gradient for trajectory optimization.

    Args:
        x: optimization vector (n_frames * 6)
        noisy_measurements: (n_frames, 8, 3) noisy marker positions
        reference_geometry: (8, 3) reference cube vertices
        lambda_smooth_pos: position smoothness weight
        lambda_smooth_rot: rotation smoothness weight
        lambda_accel: acceleration smoothness weight
        compute_gradient: whether to compute gradient

    Returns:
        cost: scalar objective value
        gradient: (n_frames * 6,) gradient vector or None
    """
    n_frames = noisy_measurements.shape[0]
    rotations, translations = vector_to_poses(x=x, n_frames=n_frames)

    # Initialize gradient
    gradient = np.zeros_like(x) if compute_gradient else None

    total_cost = 0.0

    # DATA TERM
    for i in range(n_frames):
        reconstructed = (rotations[i] @ reference_geometry.T).T + translations[i]
        residuals = reconstructed - noisy_measurements[i, :8, :]
        total_cost += np.sum(residuals ** 2)

        if compute_gradient:
            grad_t = 2 * np.sum(residuals, axis=0)
            gradient[i * 6 + 3:i * 6 + 6] = grad_t

            # Finite diff for rotation gradient
            eps = 1e-7
            for j in range(3):
                x_plus = x.copy()
                x_plus[i * 6 + j] += eps
                R_plus = Rotation.from_rotvec(x_plus[i * 6:i * 6 + 3]).as_matrix()
                recon_plus = (R_plus @ reference_geometry.T).T + translations[i]
                cost_plus = np.sum((recon_plus - noisy_measurements[i, :8, :]) ** 2)

                x_minus = x.copy()
                x_minus[i * 6 + j] -= eps
                R_minus = Rotation.from_rotvec(x_minus[i * 6:i * 6 + 3]).as_matrix()
                recon_minus = (R_minus @ reference_geometry.T).T + translations[i]
                cost_minus = np.sum((recon_minus - noisy_measurements[i, :8, :]) ** 2)

                gradient[i * 6 + j] = (cost_plus - cost_minus) / (2 * eps)

    # VELOCITY SMOOTHNESS
    if n_frames > 2:
        pos_vel = translations[1:] - translations[:-1]
        pos_accel = pos_vel[1:] - pos_vel[:-1]
        total_cost += lambda_smooth_pos * np.sum(pos_accel ** 2)

        rotvecs = np.array([x[i * 6:i * 6 + 3] for i in range(n_frames)])
        rot_vel = rotvecs[1:] - rotvecs[:-1]
        rot_accel = rot_vel[1:] - rot_vel[:-1]
        total_cost += lambda_smooth_rot * np.sum(rot_accel ** 2)

        if compute_gradient:
            # Position smoothness gradient
            for i in range(n_frames):
                grad_contrib = np.zeros(3)

                if 1 <= i <= n_frames - 2:
                    accel_i = translations[i + 1] - 2 * translations[i] + translations[i - 1]
                    grad_contrib += 2 * lambda_smooth_pos * (-2) * accel_i

                if 2 <= i <= n_frames - 1:
                    accel_prev = translations[i] - 2 * translations[i - 1] + translations[i - 2]
                    grad_contrib += 2 * lambda_smooth_pos * accel_prev

                if 0 <= i <= n_frames - 3:
                    accel_next = translations[i + 2] - 2 * translations[i + 1] + translations[i]
                    grad_contrib += 2 * lambda_smooth_pos * accel_next

                gradient[i * 6 + 3:i * 6 + 6] += grad_contrib

            # Rotation smoothness gradient
            for i in range(n_frames):
                grad_contrib = np.zeros(3)

                if 1 <= i <= n_frames - 2:
                    accel_i = rotvecs[i + 1] - 2 * rotvecs[i] + rotvecs[i - 1]
                    grad_contrib += 2 * lambda_smooth_rot * (-2) * accel_i

                if 2 <= i <= n_frames - 1:
                    accel_prev = rotvecs[i] - 2 * rotvecs[i - 1] + rotvecs[i - 2]
                    grad_contrib += 2 * lambda_smooth_rot * accel_prev

                if 0 <= i <= n_frames - 3:
                    accel_next = rotvecs[i + 2] - 2 * rotvecs[i + 1] + rotvecs[i]
                    grad_contrib += 2 * lambda_smooth_rot * accel_next

                gradient[i * 6:i * 6 + 3] += grad_contrib

    # JERK TERM
    if n_frames > 3 and lambda_accel > 0:
        pos_vel = translations[1:] - translations[:-1]
        pos_acc = pos_vel[1:] - pos_vel[:-1]
        pos_jerk = pos_acc[1:] - pos_acc[:-1]
        accel_cost = lambda_accel * np.sum(pos_jerk ** 2)
        total_cost += accel_cost

        if compute_gradient:
            eps = 1e-7
            for i in range(n_frames):
                for j in range(3):
                    x_plus = x.copy()
                    x_plus[i * 6 + 3 + j] += eps
                    _, trans_plus = vector_to_poses(x_plus, n_frames)
                    vel_plus = trans_plus[1:] - trans_plus[:-1]
                    acc_plus = vel_plus[1:] - vel_plus[:-1]
                    jerk_plus = acc_plus[1:] - acc_plus[:-1]
                    cost_plus = lambda_accel * np.sum(jerk_plus ** 2)

                    gradient[i * 6 + 3 + j] += (cost_plus - accel_cost) / eps

    return total_cost, gradient


def trajectory_objective(
        x: np.ndarray,
        noisy_measurements: np.ndarray,
        reference_geometry: np.ndarray,
        lambda_smooth_pos: float = 10.0,
        lambda_smooth_rot: float = 5.0,
        lambda_accel: float = 1.0
) -> float:
    """Objective function only (for callback logging)."""
    cost, _ = trajectory_objective_and_gradient(
        x=x,
        noisy_measurements=noisy_measurements,
        reference_geometry=reference_geometry,
        lambda_smooth_pos=lambda_smooth_pos,
        lambda_smooth_rot=lambda_smooth_rot,
        lambda_accel=lambda_accel,
        compute_gradient=False
    )
    return cost


def optimize_trajectory_global(
        noisy_measurements: np.ndarray,
        reference_geometry: np.ndarray,
        lambda_smooth_pos: float = 10.0,
        lambda_smooth_rot: float = 5.0,
        lambda_accel: float = 1.0,
        apply_butterworth: bool = True,
        butterworth_cutoff: float = 0.15,
        max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize entire trajectory using global optimization.

    Returns:
        opt_rotations, opt_translations, opt_positions,
        opt_no_filter_rotations, opt_no_filter_translations, opt_no_filter_positions
    """
    n_frames = noisy_measurements.shape[0]

    logger.info("Initializing with Kabsch...")
    init_rotations, init_translations = kabsch_initialization(
        noisy_measurements=noisy_measurements,
        reference_geometry=reference_geometry
    )

    x0 = poses_to_vector(rotations=init_rotations, translations=init_translations)

    logger.info("Starting global trajectory optimization...")
    logger.info(f"  Frames: {n_frames}")
    logger.info(f"  Parameters: {len(x0)} (6 per frame)")
    logger.info(f"  Smoothness weights: pos={lambda_smooth_pos}, rot={lambda_smooth_rot}, accel={lambda_accel}")
    logger.info(f"  Max iterations: {max_iter}")

    iteration_count = [0]
    last_print = [0]

    def callback(xk):
        iteration_count[0] += 1
        if iteration_count[0] - last_print[0] >= 5:
            cost = trajectory_objective(
                x=xk,
                noisy_measurements=noisy_measurements,
                reference_geometry=reference_geometry,
                lambda_smooth_pos=lambda_smooth_pos,
                lambda_smooth_rot=lambda_smooth_rot,
                lambda_accel=lambda_accel
            )
            logger.info(f"  Iteration {iteration_count[0]:3d}: cost = {cost:.2f}")
            last_print[0] = iteration_count[0]

    logger.info("Running L-BFGS-B optimization with analytical gradients...")

    def objective_and_grad(x):
        cost, grad = trajectory_objective_and_gradient(
            x=x,
            noisy_measurements=noisy_measurements,
            reference_geometry=reference_geometry,
            lambda_smooth_pos=lambda_smooth_pos,
            lambda_smooth_rot=lambda_smooth_rot,
            lambda_accel=lambda_accel,
            compute_gradient=True
        )
        return cost, grad

    result = minimize(
        fun=objective_and_grad,
        x0=x0,
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': max_iter,
            'ftol': 1e-8,
            'gtol': 1e-6,
            'maxcor': 20,
            'maxls': 50
        },
        callback=callback
    )

    logger.info(f"Optimization {'converged' if result.success else 'completed'}")
    logger.info(f"  Final cost: {result.fun:.2f}")
    logger.info(f"  Total iterations: {iteration_count[0]}")

    # Extract optimized poses
    opt_rotations, opt_translations = vector_to_poses(x=result.x, n_frames=n_frames)

    # Save pre-filter version
    opt_rotations_no_filter = opt_rotations.copy()
    opt_translations_no_filter = opt_translations.copy()

    # Reconstruct without filter
    reconstructed_no_filter = np.zeros((n_frames, 9, 3))
    for i in range(n_frames):
        vertices = (opt_rotations_no_filter[i] @ reference_geometry.T).T + opt_translations_no_filter[i]
        reconstructed_no_filter[i, :8, :] = vertices
        reconstructed_no_filter[i, 8, :] = np.mean(vertices, axis=0)

    # Apply Butterworth filter if requested
    if apply_butterworth:
        logger.info(f"Applying Butterworth filter (cutoff={butterworth_cutoff})...")

        opt_translations = apply_butterworth_filter(
            data=opt_translations,
            cutoff_freq=butterworth_cutoff,
            sampling_rate=1.0,
            order=4
        )

        rotvecs = np.array([Rotation.from_matrix(R).as_rotvec() for R in opt_rotations])
        filtered_rotvecs = apply_butterworth_filter(
            data=rotvecs,
            cutoff_freq=butterworth_cutoff,
            sampling_rate=1.0,
            order=4
        )
        opt_rotations = np.array([Rotation.from_rotvec(rv).as_matrix() for rv in filtered_rotvecs])

    # Reconstruct with filter
    reconstructed = np.zeros((n_frames, 9, 3))
    for i in range(n_frames):
        vertices = (opt_rotations[i] @ reference_geometry.T).T + opt_translations[i]
        reconstructed[i, :8, :] = vertices
        reconstructed[i, 8, :] = np.mean(vertices, axis=0)

    return (opt_rotations, opt_translations, reconstructed,
            opt_rotations_no_filter, opt_translations_no_filter, reconstructed_no_filter)


def compute_detailed_errors(
        gt_positions: np.ndarray,
        noisy_positions: np.ndarray,
        kabsch_positions: np.ndarray,
        opt_no_filter_positions: np.ndarray,
        opt_positions: np.ndarray,
        gt_rotations: np.ndarray,
        kabsch_rotations: np.ndarray,
        opt_no_filter_rotations: np.ndarray,
        opt_rotations: np.ndarray,
        gt_translations: np.ndarray,
        kabsch_translations: np.ndarray,
        opt_no_filter_translations: np.ndarray,
        opt_translations: np.ndarray
) -> Dict[str, Any]:
    """Compute comprehensive error metrics."""
    logger.info("Computing error statistics...")

    n_frames = gt_positions.shape[0]

    # Per-frame position errors
    noisy_errors = np.linalg.norm(gt_positions - noisy_positions, axis=2)
    kabsch_errors = np.linalg.norm(gt_positions - kabsch_positions, axis=2)
    opt_no_filter_errors = np.linalg.norm(gt_positions - opt_no_filter_positions, axis=2)
    opt_errors = np.linalg.norm(gt_positions - opt_positions, axis=2)

    noisy_frame_errors = np.mean(noisy_errors, axis=1)
    kabsch_frame_errors = np.mean(kabsch_errors, axis=1)
    opt_no_filter_frame_errors = np.mean(opt_no_filter_errors, axis=1)
    opt_frame_errors = np.mean(opt_errors, axis=1)

    # Rotation errors
    kabsch_rotation_errors = np.array([
        rotation_error_angle(R1=gt_rotations[i], R2=kabsch_rotations[i])
        for i in range(n_frames)
    ])
    opt_no_filter_rotation_errors = np.array([
        rotation_error_angle(R1=gt_rotations[i], R2=opt_no_filter_rotations[i])
        for i in range(n_frames)
    ])
    opt_rotation_errors = np.array([
        rotation_error_angle(R1=gt_rotations[i], R2=opt_rotations[i])
        for i in range(n_frames)
    ])

    # Translation errors
    kabsch_translation_errors = np.linalg.norm(gt_translations - kabsch_translations, axis=1)
    opt_no_filter_translation_errors = np.linalg.norm(gt_translations - opt_no_filter_translations, axis=1)
    opt_translation_errors = np.linalg.norm(gt_translations - opt_translations, axis=1)

    # Center errors
    noisy_center_errors = np.linalg.norm(gt_positions[:, 8, :] - noisy_positions[:, 8, :], axis=1)
    kabsch_center_errors = np.linalg.norm(gt_positions[:, 8, :] - kabsch_positions[:, 8, :], axis=1)
    opt_no_filter_center_errors = np.linalg.norm(gt_positions[:, 8, :] - opt_no_filter_positions[:, 8, :], axis=1)
    opt_center_errors = np.linalg.norm(gt_positions[:, 8, :] - opt_positions[:, 8, :], axis=1)

    # Per-marker statistics
    noisy_marker_means = np.mean(noisy_errors, axis=0)
    kabsch_marker_means = np.mean(kabsch_errors, axis=0)
    opt_no_filter_marker_means = np.mean(opt_no_filter_errors, axis=0)
    opt_marker_means = np.mean(opt_errors, axis=0)

    stats = {
        'summary': {
            'noisy_mean_mm': float(np.mean(noisy_errors) * 1000),
            'noisy_std_mm': float(np.std(noisy_errors) * 1000),
            'noisy_rms_mm': float(np.sqrt(np.mean(noisy_errors ** 2)) * 1000),

            'kabsch_mean_mm': float(np.mean(kabsch_errors) * 1000),
            'kabsch_std_mm': float(np.std(kabsch_errors) * 1000),
            'kabsch_rms_mm': float(np.sqrt(np.mean(kabsch_errors ** 2)) * 1000),

            'opt_no_filter_mean_mm': float(np.mean(opt_no_filter_errors) * 1000),
            'opt_no_filter_std_mm': float(np.std(opt_no_filter_errors) * 1000),
            'opt_no_filter_rms_mm': float(np.sqrt(np.mean(opt_no_filter_errors ** 2)) * 1000),

            'opt_mean_mm': float(np.mean(opt_errors) * 1000),
            'opt_std_mm': float(np.std(opt_errors) * 1000),
            'opt_rms_mm': float(np.sqrt(np.mean(opt_errors ** 2)) * 1000),

            'kabsch_rotation_error_mean_deg': float(np.mean(kabsch_rotation_errors)),
            'kabsch_rotation_error_std_deg': float(np.std(kabsch_rotation_errors)),
            'opt_no_filter_rotation_error_mean_deg': float(np.mean(opt_no_filter_rotation_errors)),
            'opt_no_filter_rotation_error_std_deg': float(np.std(opt_no_filter_rotation_errors)),
            'opt_rotation_error_mean_deg': float(np.mean(opt_rotation_errors)),
            'opt_rotation_error_std_deg': float(np.std(opt_rotation_errors)),

            'kabsch_translation_error_mean_mm': float(np.mean(kabsch_translation_errors) * 1000),
            'kabsch_translation_error_std_mm': float(np.std(kabsch_translation_errors) * 1000),
            'opt_no_filter_translation_error_mean_mm': float(np.mean(opt_no_filter_translation_errors) * 1000),
            'opt_no_filter_translation_error_std_mm': float(np.std(opt_no_filter_translation_errors) * 1000),
            'opt_translation_error_mean_mm': float(np.mean(opt_translation_errors) * 1000),
            'opt_translation_error_std_mm': float(np.std(opt_translation_errors) * 1000),

            'noisy_center_error_mean_mm': float(np.mean(noisy_center_errors) * 1000),
            'kabsch_center_error_mean_mm': float(np.mean(kabsch_center_errors) * 1000),
            'opt_no_filter_center_error_mean_mm': float(np.mean(opt_no_filter_center_errors) * 1000),
            'opt_center_error_mean_mm': float(np.mean(opt_center_errors) * 1000),

            'kabsch_improvement_percent': float((1 - np.mean(kabsch_errors) / np.mean(noisy_errors)) * 100),
            'opt_no_filter_improvement_percent': float(
                (1 - np.mean(opt_no_filter_errors) / np.mean(noisy_errors)) * 100),
            'opt_improvement_percent': float((1 - np.mean(opt_errors) / np.mean(noisy_errors)) * 100),
            'opt_vs_kabsch_improvement_percent': float((1 - np.mean(opt_errors) / np.mean(kabsch_errors)) * 100),
            'butterworth_improvement_percent': float((1 - np.mean(opt_errors) / np.mean(opt_no_filter_errors)) * 100),
        },

        'per_frame': {
            'noisy_error_mm': (noisy_frame_errors * 1000).tolist(),
            'kabsch_error_mm': (kabsch_frame_errors * 1000).tolist(),
            'opt_no_filter_error_mm': (opt_no_filter_frame_errors * 1000).tolist(),
            'opt_error_mm': (opt_frame_errors * 1000).tolist(),
            'kabsch_rotation_error_deg': kabsch_rotation_errors.tolist(),
            'opt_no_filter_rotation_error_deg': opt_no_filter_rotation_errors.tolist(),
            'opt_rotation_error_deg': opt_rotation_errors.tolist(),
            'kabsch_translation_error_mm': (kabsch_translation_errors * 1000).tolist(),
            'opt_no_filter_translation_error_mm': (opt_no_filter_translation_errors * 1000).tolist(),
            'opt_translation_error_mm': (opt_translation_errors * 1000).tolist(),
            'noisy_center_error_mm': (noisy_center_errors * 1000).tolist(),
            'kabsch_center_error_mm': (kabsch_center_errors * 1000).tolist(),
            'opt_no_filter_center_error_mm': (opt_no_filter_center_errors * 1000).tolist(),
            'opt_center_error_mm': (opt_center_errors * 1000).tolist(),
        },

        'per_marker': {
            'noisy_mean_mm': (noisy_marker_means * 1000).tolist(),
            'kabsch_mean_mm': (kabsch_marker_means * 1000).tolist(),
            'opt_no_filter_mean_mm': (opt_no_filter_marker_means * 1000).tolist(),
            'opt_mean_mm': (opt_marker_means * 1000).tolist(),
        }
    }

    return stats


def save_combined_trajectory_to_csv(
        filepath: str | Path,
        gt_positions: np.ndarray,
        noisy_positions: np.ndarray,
        kabsch_positions: np.ndarray,
        opt_no_filter_positions: np.ndarray,
        opt_positions: np.ndarray
) -> None:
    """Save all trajectories to CSV."""
    logger.info(f"Saving trajectory data to {filepath}")

    n_frames, n_markers, _ = gt_positions.shape
    marker_names = [f"v{i}" for i in range(8)] + ["center"]

    data = {'frame': range(n_frames)}

    for dataset_name, positions in [('gt', gt_positions),
                                    ('noisy', noisy_positions),
                                    ('kabsch', kabsch_positions),
                                    ('opt_no_filter', opt_no_filter_positions),
                                    ('opt', opt_positions)]:
        for marker_idx, marker_name in enumerate(marker_names):
            for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
                col_name = f"{dataset_name}_{marker_name}_{coord_name}"
                data[col_name] = positions[:, marker_idx, coord_idx]

    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=filepath, index=False)

    logger.info(f"Saved trajectory data: shape={df.shape}")


def save_error_timeseries(filepath: str | Path, stats: Dict[str, Any]) -> None:
    """Save per-frame error timeseries to CSV."""
    logger.info(f"Saving error timeseries to {filepath}")
    df = pd.DataFrame(data=stats['per_frame'])
    df.to_csv(path_or_buf=filepath, index=False)


def save_statistics_json(filepath: str | Path, stats: Dict[str, Any]) -> None:
    """Save detailed statistics to JSON."""
    logger.info(f"Saving statistics to {filepath}")
    with open(filepath, 'w') as f:
        json.dump(obj=stats, fp=f, indent=2)


def print_statistics(stats: Dict[str, Any]) -> None:
    """Pretty print statistics."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPARISON")
    print("=" * 70)

    summary = stats['summary']

    print("\nNOISY MEASUREMENTS:")
    print(f"  Mean error:     {summary['noisy_mean_mm']:.2f} mm")
    print(f"  RMS error:      {summary['noisy_rms_mm']:.2f} mm")
    print(f"  Center error:   {summary['noisy_center_error_mean_mm']:.2f} mm")

    print("\nKABSCH (frame-by-frame):")
    print(f"  Mean error:     {summary['kabsch_mean_mm']:.2f} mm")
    print(f"  RMS error:      {summary['kabsch_rms_mm']:.2f} mm")
    print(f"  Center error:   {summary['kabsch_center_error_mean_mm']:.2f} mm")
    print(
        f"  Rotation error: {summary['kabsch_rotation_error_mean_deg']:.2f}° ± {summary['kabsch_rotation_error_std_deg']:.2f}°")
    print(f"  Translation err:{summary['kabsch_translation_error_mean_mm']:.2f} mm")
    print(f"  Improvement:    {summary['kabsch_improvement_percent']:.1f}% vs noisy")

    print("\nGLOBAL OPTIMIZATION (before Butterworth):")
    print(f"  Mean error:     {summary['opt_no_filter_mean_mm']:.2f} mm")
    print(f"  RMS error:      {summary['opt_no_filter_rms_mm']:.2f} mm")
    print(f"  Center error:   {summary['opt_no_filter_center_error_mean_mm']:.2f} mm")
    print(
        f"  Rotation error: {summary['opt_no_filter_rotation_error_mean_deg']:.2f}° ± {summary['opt_no_filter_rotation_error_std_deg']:.2f}°")
    print(f"  Translation err:{summary['opt_no_filter_translation_error_mean_mm']:.2f} mm")
    print(f"  Improvement:    {summary['opt_no_filter_improvement_percent']:.1f}% vs noisy")

    print("\nGLOBAL OPTIMIZATION (with Butterworth filter):")
    print(f"  Mean error:     {summary['opt_mean_mm']:.2f} mm")
    print(f"  RMS error:      {summary['opt_rms_mm']:.2f} mm")
    print(f"  Center error:   {summary['opt_center_error_mean_mm']:.2f} mm")
    print(
        f"  Rotation error: {summary['opt_rotation_error_mean_deg']:.2f}° ± {summary['opt_rotation_error_std_deg']:.2f}°")
    print(f"  Translation err:{summary['opt_translation_error_mean_mm']:.2f} mm")
    print(f"  Improvement:    {summary['opt_improvement_percent']:.1f}% vs noisy")
    print(f"  Improvement:    {summary['opt_vs_kabsch_improvement_percent']:.1f}% vs Kabsch")
    print(f"  Butterworth:    {summary['butterworth_improvement_percent']:.1f}% vs pre-filter")

    print("\nPER-MARKER COMPARISON (mm):")
    marker_names = [f"v{i}" for i in range(8)] + ["center"]
    print(f"  {'Marker':<8s}  {'Noisy':>8s}  {'Kabsch':>8s}  {'Opt(raw)':>9s}  {'Opt(filt)':>10s}")
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 9}  {'-' * 10}")
    for i, name in enumerate(marker_names):
        noisy = stats['per_marker']['noisy_mean_mm'][i]
        kabsch = stats['per_marker']['kabsch_mean_mm'][i]
        opt_no_filter = stats['per_marker']['opt_no_filter_mean_mm'][i]
        opt = stats['per_marker']['opt_mean_mm'][i]
        print(f"  {name:<8s}  {noisy:8.2f}  {kabsch:8.2f}  {opt_no_filter:9.2f}  {opt:10.2f}")


def main() -> None:
    """Generate synthetic data and run optimization."""
    print("=" * 70)
    print("RIGID BODY TRAJECTORY OPTIMIZATION")
    print("=" * 70)

    # Parameters
    n_frames = 200
    cube_size = 0.5
    noise_std = 0.3

    # Optimization parameters
    lambda_smooth_pos = 20.0
    lambda_smooth_rot = 10.0
    lambda_accel = 2.0
    apply_butterworth = True
    butterworth_cutoff = 0.15
    max_iter = 200

    logger.info("Starting rigid body trajectory optimization")
    logger.info(f"Frames: {n_frames}, Cube size: {cube_size}m, Noise std: {noise_std}m")

    # Generate ground truth
    logger.info("Step 1: Generating ground truth trajectory")
    gt_rotations, gt_translations, gt_positions = generate_ground_truth_trajectory(
        n_frames=n_frames,
        cube_size=cube_size
    )

    # Add noise
    logger.info("Step 2: Adding measurement noise")
    noisy_positions = add_noise_to_measurements(
        marker_positions=gt_positions,
        noise_std=noise_std,
        seed=42
    )

    # Kabsch baseline
    logger.info("Step 3: Running Kabsch optimization (baseline)")
    reference_geometry = generate_cube_vertices(size=cube_size)
    kabsch_rotations, kabsch_translations = kabsch_initialization(
        noisy_measurements=noisy_positions,
        reference_geometry=reference_geometry
    )
    kabsch_positions = np.zeros_like(gt_positions)
    for i in range(n_frames):
        vertices = (kabsch_rotations[i] @ reference_geometry.T).T + kabsch_translations[i]
        kabsch_positions[i, :8, :] = vertices
        kabsch_positions[i, 8, :] = np.mean(vertices, axis=0)

    # Global trajectory optimization
    logger.info("Step 4: Running global trajectory optimization")
    opt_rotations, opt_translations, opt_positions, \
        opt_no_filter_rotations, opt_no_filter_translations, opt_no_filter_positions = optimize_trajectory_global(
        noisy_measurements=noisy_positions,
        reference_geometry=reference_geometry,
        lambda_smooth_pos=lambda_smooth_pos,
        lambda_smooth_rot=lambda_smooth_rot,
        lambda_accel=lambda_accel,
        apply_butterworth=apply_butterworth,
        butterworth_cutoff=butterworth_cutoff,
        max_iter=max_iter
    )

    # Compute statistics
    logger.info("Step 5: Computing error statistics")
    stats = compute_detailed_errors(
        gt_positions=gt_positions,
        noisy_positions=noisy_positions,
        kabsch_positions=kabsch_positions,
        opt_no_filter_positions=opt_no_filter_positions,
        opt_positions=opt_positions,
        gt_rotations=gt_rotations,
        kabsch_rotations=kabsch_rotations,
        opt_no_filter_rotations=opt_no_filter_rotations,
        opt_rotations=opt_rotations,
        gt_translations=gt_translations,
        kabsch_translations=kabsch_translations,
        opt_no_filter_translations=opt_no_filter_translations,
        opt_translations=opt_translations
    )
    print_statistics(stats=stats)

    # Save outputs
    logger.info("Step 6: Saving output files")
    save_combined_trajectory_to_csv(
        filepath="trajectory_data.csv",
        gt_positions=gt_positions,
        noisy_positions=noisy_positions,
        kabsch_positions=kabsch_positions,
        opt_no_filter_positions=opt_no_filter_positions,
        opt_positions=opt_positions
    )

    save_error_timeseries(
        filepath="error_timeseries.csv",
        stats=stats
    )

    save_statistics_json(
        filepath="optimization_stats.json",
        stats=stats
    )

    logger.info("=" * 70)
    logger.info("COMPLETE - Results saved. Open HTML viewer to visualize.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()