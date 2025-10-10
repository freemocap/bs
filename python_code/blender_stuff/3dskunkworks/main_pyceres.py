"""
Rigid Body Motion Tracking with PyCeres
========================================

Estimates rigid body motion from noisy point measurements using:
- Robust reference geometry estimation from temporal data
- Pose graph optimization with Ceres Solver (via PyCeres)
- Rigid body distance constraints
- Temporal smoothness regularization

Uses asymmetric marker configurations to ensure unique orientation recovery.

Requirements:
    pip install pyceres numpy scipy pandas

Author: [Your Name]
License: MIT
"""

from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
from scipy.spatial.transform import Rotation
import pyceres

from geometry_module import rotation_matrix_from_axis_angle
from sba_utils import estimate_reference_geometry, compute_reference_distances

logger = logging.getLogger(__name__)


# =============================================================================
# PYCERES COST FUNCTIONS WITH ANALYTICAL JACOBIANS
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
            # Jacobian w.r.t. quaternion (3 residuals x 4 params = 12 elements)
            if jacobians[0] is not None:
                # Use finite differences for quaternion (complex derivatives)
                eps = 1e-8
                J_quat = np.zeros((3, 4))
                for i in range(4):
                    quat_plus = quat.copy()
                    quat_plus[i] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)  # Renormalize
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    predicted_plus = R_plus @ self.reference_point + translation
                    residual_plus = self.weight * (self.measured_point - predicted_plus)
                    J_quat[:, i] = (residual_plus - residuals) / eps
                # Flatten in row-major (C) order
                jacobians[0][:] = J_quat.ravel()

            # Jacobian w.r.t. translation (3 residuals x 3 params = 9 elements)
            if jacobians[1] is not None:
                J_trans = -self.weight * np.eye(3)
                # Flatten in row-major (C) order
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
            # Jacobian w.r.t. quaternion (1 residual x 4 params = 4 elements)
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

            # Jacobian w.r.t. translation (1 residual x 3 params = 3 elements)
            # d(||p_i - p_j||)/d(t) = 0 because translation cancels out
            if jacobians[1] is not None:
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
            # Jacobian w.r.t. first quaternion (4 residuals x 4 params = 16 elements)
            if jacobians[0] is not None:
                J0 = -self.weight * np.eye(4)
                jacobians[0][:] = J0.ravel()

            # Jacobian w.r.t. second quaternion (4 residuals x 4 params = 16 elements)
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
            # Jacobian w.r.t. first translation (3 residuals x 3 params = 9 elements)
            if jacobians[0] is not None:
                J0 = -self.weight * np.eye(3)
                jacobians[0][:] = J0.ravel()

            # Jacobian w.r.t. second translation (3 residuals x 3 params = 9 elements)
            if jacobians[1] is not None:
                J1 = self.weight * np.eye(3)
                jacobians[1][:] = J1.ravel()

        return True


# =============================================================================
# MARKER GEOMETRY
# =============================================================================

def generate_marker_configuration(
    *,
    size: float = 1.0,
    n_asymmetric_markers: int = 3
) -> np.ndarray:
    """
    Generate marker configuration with asymmetric points for unique orientation.

    Args:
        size: Characteristic size of the marker set
        n_asymmetric_markers: Number of additional markers to break symmetry

    Returns:
        (n_points, 3) marker positions
    """
    s = size

    # Base cube vertices
    cube_vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ])

    # Additional markers at strategic positions to break symmetry
    if n_asymmetric_markers >= 1:
        marker_1 = np.array([[0.0, -s * 1.5, 0.0]])
        cube_vertices = np.vstack([cube_vertices, marker_1])

    if n_asymmetric_markers >= 2:
        marker_2 = np.array([[s * 1.3, -s, -s * 0.7]])
        cube_vertices = np.vstack([cube_vertices, marker_2])

    if n_asymmetric_markers >= 3:
        marker_3 = np.array([[-s * 0.8, -s * 0.8, s * 1.4]])
        cube_vertices = np.vstack([cube_vertices, marker_3])

    return cube_vertices


# =============================================================================
# PYCERES OPTIMIZER
# =============================================================================

def optimize_rigid_body_pyceres(
    *,
    noisy_data: np.ndarray,
    reference_geometry: np.ndarray,
    reference_distances: np.ndarray,
    max_iter: int = 300,
    lambda_data: float = 100.0,
    lambda_rigid: float = 100.0,
    lambda_rot_smooth: float = 500.0,
    lambda_trans_smooth: float = 200.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize rigid body trajectory using PyCeres pose graph optimization.

    Args:
        noisy_data: (n_frames, n_points, 3) noisy measurements
        reference_geometry: (n_points, 3) reference shape
        reference_distances: (n_points, n_points) distance matrix
        max_iter: Maximum optimization iterations
        lambda_data: Weight for data fitting term
        lambda_rigid: Weight for rigid body constraints
        lambda_rot_smooth: Weight for rotation smoothness
        lambda_trans_smooth: Weight for translation smoothness

    Returns:
        - rotations: (n_frames, 3, 3)
        - translations: (n_frames, 3)
        - reconstructed: (n_frames, n_points, 3)
    """
    n_frames, n_points, _ = noisy_data.shape

    logger.info("=" * 80)
    logger.info("PYCERES POSE GRAPH OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"\nFrames: {n_frames}, Markers: {n_points}")
    logger.info(f"\nOptimization weights:")
    logger.info(f"  λ_data:        {lambda_data:8.1f}")
    logger.info(f"  λ_rigid:       {lambda_rigid:8.1f}")
    logger.info(f"  λ_rot_smooth:  {lambda_rot_smooth:8.1f}")
    logger.info(f"  λ_trans_smooth: {lambda_trans_smooth:8.1f}")

    # Initialize poses with Procrustes
    logger.info("\nInitializing poses with Procrustes alignment...")
    ref_centered = reference_geometry - np.mean(reference_geometry, axis=0)
    poses: list[tuple[np.ndarray, np.ndarray]] = []

    for frame_idx in range(n_frames):
        data_centered = noisy_data[frame_idx] - np.mean(noisy_data[frame_idx], axis=0)
        H = data_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        quat_scipy = Rotation.from_matrix(R).as_quat()
        quat_ceres = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        translation = np.mean(noisy_data[frame_idx], axis=0)
        poses.append((quat_ceres, translation))

    # Unwrap quaternions to ensure consistency
    init_quats = np.array([pose[0] for pose in poses])
    for i in range(1, n_frames):
        if np.dot(init_quats[i], init_quats[i - 1]) < 0:
            init_quats[i] = -init_quats[i]

    for i in range(n_frames):
        poses[i] = (init_quats[i], poses[i][1])

    # Build problem
    logger.info("\nBuilding pose graph...")
    problem = pyceres.Problem()
    pose_params: list[tuple[np.ndarray, np.ndarray]] = []

    for frame_idx in range(n_frames):
        quat, trans = poses[frame_idx]
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
                weight=lambda_data
            )
            problem.add_residual_block(cost, None, [quat, trans])

    # Add rigid body constraints (edges only to avoid over-constraining)
    logger.info("  Adding rigid body constraints...")
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for frame_idx in range(n_frames):
        quat, trans = pose_params[frame_idx]
        for i, j in edges:
            if i < n_points and j < n_points:
                cost = RigidBodyFactor(
                    ref_point_i=reference_geometry[i],
                    ref_point_j=reference_geometry[j],
                    reference_distance=reference_distances[i, j],
                    weight=lambda_rigid
                )
                problem.add_residual_block(cost, None, [quat, trans])

    # Add smoothness factors
    logger.info("  Adding smoothness factors...")
    for frame_idx in range(n_frames - 1):
        quat_t, trans_t = pose_params[frame_idx]
        quat_t1, trans_t1 = pose_params[frame_idx + 1]

        rot_cost = RotationSmoothnessFactor(weight=lambda_rot_smooth)
        problem.add_residual_block(rot_cost, None, [quat_t, quat_t1])

        trans_cost = TranslationSmoothnessFactor(weight=lambda_trans_smooth)
        problem.add_residual_block(trans_cost, None, [trans_t, trans_t1])

    logger.info(f"\n  Total residual blocks: {problem.num_residual_blocks()}")
    logger.info(f"  Total parameters: {problem.num_parameters()}")

    # Solve
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION")
    logger.info("=" * 80)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = max_iter
    options.function_tolerance = 1e-9
    options.gradient_tolerance = 1e-11
    options.parameter_tolerance = 1e-10

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Termination: {summary.termination_type}")
    logger.info(f"  Initial cost: {summary.initial_cost:.2f}")
    logger.info(f"  Final cost: {summary.final_cost:.2f}")
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

    return rotations, translations, reconstructed


# =============================================================================
# DATA GENERATION
# =============================================================================

@dataclass
class DataConfig:
    """Synthetic data generation parameters."""
    n_frames: int = 200
    marker_size: float = 1.0
    n_asymmetric_markers: int = 3
    noise_std: float = 0.1
    random_seed: int | None = 42


def generate_synthetic_trajectory(
    *,
    config: DataConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic rigid body trajectory with noise.

    Returns:
        - reference_geometry: (n_points, 3) marker configuration
        - gt_data: (n_frames, n_points, 3) ground truth trajectory
        - noisy_data: (n_frames, n_points, 3) noisy measurements
    """
    logger.info(f"Generating {config.n_frames} frames with {config.n_asymmetric_markers + 8} markers")

    reference_geometry = generate_marker_configuration(
        size=config.marker_size,
        n_asymmetric_markers=config.n_asymmetric_markers
    )

    n_points = len(reference_geometry)
    gt_data = np.zeros((config.n_frames, n_points, 3))

    for i in range(config.n_frames):
        t = i / config.n_frames

        # Circular trajectory with vertical oscillation
        radius = 3.0
        translation = np.array([
            radius * np.cos(t * 2 * np.pi),
            radius * np.sin(t * 2 * np.pi),
            1.5 * np.sin(t * 4 * np.pi)
        ])

        # Rotation around diagonal axis
        rot_axis = np.array([0.3, 1.0, 0.2])
        rot_angle = t * 4 * np.pi
        R = rotation_matrix_from_axis_angle(axis=rot_axis, angle=rot_angle)

        gt_data[i] = (R @ reference_geometry.T).T + translation

    # Add noise
    if config.random_seed is not None:
        np.random.seed(seed=config.random_seed)

    noise = np.random.normal(loc=0, scale=config.noise_std, size=gt_data.shape)
    noisy_data = gt_data + noise

    logger.info(f"  Noise level: σ={config.noise_std * 1000:.1f}mm")

    return reference_geometry, gt_data, noisy_data


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_reconstruction(
    *,
    gt_data: np.ndarray,
    noisy_data: np.ndarray,
    optimized_data: np.ndarray
) -> dict[str, float]:
    """
    Evaluate reconstruction quality.

    Returns:
        Dictionary of evaluation metrics
    """
    noisy_errors = np.linalg.norm(noisy_data - gt_data, axis=2)
    optimized_errors = np.linalg.norm(optimized_data - gt_data, axis=2)

    metrics = {
        'noisy_mean_mm': np.mean(noisy_errors) * 1000,
        'noisy_max_mm': np.max(noisy_errors) * 1000,
        'optimized_mean_mm': np.mean(optimized_errors) * 1000,
        'optimized_max_mm': np.max(optimized_errors) * 1000,
        'improvement_pct': (np.mean(noisy_errors) - np.mean(optimized_errors)) / np.mean(noisy_errors) * 100
    }

    logger.info("\nReconstruction quality:")
    logger.info(f"  Noisy:     mean={metrics['noisy_mean_mm']:.2f}mm, max={metrics['noisy_max_mm']:.2f}mm")
    logger.info(f"  Optimized: mean={metrics['optimized_mean_mm']:.2f}mm, max={metrics['optimized_max_mm']:.2f}mm")
    logger.info(f"  Improvement: {metrics['improvement_pct']:.1f}%")

    return metrics


# =============================================================================
# OUTPUT
# =============================================================================

def save_trajectory_csv(
    *,
    filepath: Path,
    gt_data: np.ndarray,
    noisy_data: np.ndarray,
    optimized_data: np.ndarray,
    n_base_markers: int = 8
) -> None:
    """
    Save trajectory data for visualization.

    Args:
        filepath: Output CSV path
        gt_data: Ground truth trajectory
        noisy_data: Noisy measurements
        optimized_data: Optimized trajectory
        n_base_markers: Number of base markers (before asymmetric markers)
    """
    n_frames, n_points, _ = gt_data.shape
    data: dict[str, np.ndarray | range] = {'frame': range(n_frames)}

    def add_dataset(*, name: str, positions: np.ndarray) -> None:
        for point_idx in range(n_points):
            if point_idx < n_base_markers:
                point_name = f"v{point_idx}"
            else:
                point_name = f"m{point_idx - n_base_markers}"

            for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
                col_name = f"{name}_{point_name}_{coord_name}"
                data[col_name] = positions[:, point_idx, coord_idx]

        # Add centroid
        center = np.mean(positions, axis=1)
        for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
            data[f"{name}_center_{coord_name}"] = center[:, coord_idx]

    add_dataset(name='gt', positions=gt_data)
    add_dataset(name='noisy', positions=noisy_data)
    add_dataset(name='optimized', positions=optimized_data)

    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=filepath, index=False)
    logger.info(f"\nSaved {len(df)} frames to {filepath}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    output_path: Path = Path("trajectory_data.csv")
    log_level: str = "INFO"

    # Optimizer parameters
    max_iter: int = 300
    lambda_data: float = 100.0
    lambda_rigid: float = 100.0
    lambda_rot_smooth: float = 500.0
    lambda_trans_smooth: float = 200.0


def run_pipeline(*, config: PipelineConfig) -> dict[str, float]:
    """
    Run complete rigid body tracking pipeline with PyCeres.

    Returns:
        Evaluation metrics
    """
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(levelname)s | %(message)s'
    )

    logger.info("=" * 80)
    logger.info("RIGID BODY TRACKING WITH PYCERES")
    logger.info("=" * 80)

    # Generate synthetic data
    logger.info("\n[1/4] Generating synthetic data")
    reference_geometry, gt_data, noisy_data = generate_synthetic_trajectory(config=config.data)

    # Estimate reference geometry from noisy data
    logger.info("\n[2/4] Estimating reference geometry")
    reference_geometry_estimated = estimate_reference_geometry(noisy_data=noisy_data)
    reference_distances = compute_reference_distances(reference=reference_geometry_estimated)

    # Optimize trajectory
    logger.info("\n[3/4] Running PyCeres optimization")
    _, _, optimized_data = optimize_rigid_body_pyceres(
        noisy_data=noisy_data,
        reference_geometry=reference_geometry_estimated,
        reference_distances=reference_distances,
        max_iter=config.max_iter,
        lambda_data=config.lambda_data,
        lambda_rigid=config.lambda_rigid,
        lambda_rot_smooth=config.lambda_rot_smooth,
        lambda_trans_smooth=config.lambda_trans_smooth
    )

    # Evaluate results
    logger.info("\n[4/4] Evaluating reconstruction")
    metrics = evaluate_reconstruction(
        gt_data=gt_data,
        noisy_data=noisy_data,
        optimized_data=optimized_data
    )

    # Save output
    logger.info("\nSaving results")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_trajectory_csv(
        filepath=config.output_path,
        gt_data=gt_data,
        noisy_data=noisy_data,
        optimized_data=optimized_data,
        n_base_markers=8
    )

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline complete")
    logger.info("=" * 80)
    logger.info("\nVisualize results by opening rigid-body-viewer.html")

    return metrics


def main() -> None:
    """Main entry point."""
    config = PipelineConfig(
        data=DataConfig(
            n_frames=200,
            marker_size=1.0,
            n_asymmetric_markers=3,
            noise_std=0.1,
            random_seed=42
        ),
        max_iter=300,
        lambda_data=100.0,
        lambda_rigid=100.0,
        lambda_rot_smooth=500.0,
        lambda_trans_smooth=200.0,
        output_path=Path("trajectory_data.csv")
    )
    
    run_pipeline(config=config)


if __name__ == "__main__":
    main()