"""Synthetic data demo with cube keypoints."""

from pathlib import Path
import numpy as np
import logging
from scipy.spatial.transform import Rotation

from python_code.kinematics_core.stick_figure_topology_model import StickFigureTopology
from python_code.rigid_body_solver.core.optimization import OptimizationConfig
from python_code.rigid_body_solver.core import TrackingConfig, process_tracking_data
from python_code.rigid_body_solver.data_io.data_savers import save_simple_csv

logger = logging.getLogger(__name__)


def rotation_matrix_from_axis_angle(*, axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix from axis-angle."""
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def generate_cube_keypoints(*, size: float = 1.0, n_extra: int = 3) -> np.ndarray:
    """
    Generate cube vertices plus asymmetric keypoints.

    Args:
        size: Cube half-width
        n_extra: Number of extra asymmetric keypoints

    Returns:
        (8 + n_extra, 3) keypoint positions
    """
    s = size

    # Base cube
    cube = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ])

    # Add asymmetric keypoints to break symmetry
    extra_keypoints = []
    if n_extra >= 1:
        extra_keypoints.append([0.0, -s * 1.5, 0.0])
    if n_extra >= 2:
        extra_keypoints.append([s * 1.3, -s, -s * 0.7])
    if n_extra >= 3:
        extra_keypoints.append([-s * 0.8, -s * 0.8, s * 1.4])

    if extra_keypoints:
        return np.vstack([cube, np.array(extra_keypoints)])
    return cube


def generate_synthetic_trajectory(
    *,
    reference_keypoints: np.ndarray,
    n_frames: int = 200,
    noise_std: float = 0.1,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic trajectory with circular motion.

    Args:
        reference_keypoints: (n_keypoints, 3) keypoint configuration
        n_frames: Number of frames to generate
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        - ground_truth: (n_frames, n_keypoints, 3)
        - original: (n_frames, n_keypoints, 3)
    """
    n_keypoints = len(reference_keypoints)
    ground_truth = np.zeros((n_frames, n_keypoints, 3))

    for i in range(n_frames):
        t = i / n_frames

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

        ground_truth[i] = (R @ reference_keypoints.T).T + translation

    # Add noise
    np.random.seed(seed=random_seed)
    noise = np.random.normal(loc=0, scale=noise_std, size=ground_truth.shape)
    original = ground_truth + noise

    return ground_truth, original


def create_cube_topology() -> StickFigureTopology:
    """Create topology for cube with asymmetric keypoints."""

    keypoint_names = [
        "v0", "v1", "v2", "v3",  # Bottom face
        "v4", "v5", "v6", "v7",  # Top face
        "m0", "m1", "m2"          # Asymmetric keypoints
    ]

    # Cube edges (12 edges)
    rigid_edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Diagonal connections to asymmetric keypoints
        (0, 8), (1, 8), (4, 8),
        (1, 9), (2, 9), (5, 9),
        (4, 10), (7, 10), (0, 10),
    ]

    return StickFigureTopology(
        keypoint_names=keypoint_names,
        rigid_edges=rigid_edges,
        name="cube_asymmetric"
    )


def run_synthetic_demo() -> None:
    """Run complete synthetic data demonstration."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    logger.info("="*80)
    logger.info("SYNTHETIC CUBE DEMO")
    logger.info("="*80)

    # Generate synthetic data
    logger.info("\nGenerating synthetic data...")
    reference_keypoints = generate_cube_keypoints(size=1.0, n_extra=3)
    ground_truth, original = generate_synthetic_trajectory(
        reference_keypoints=reference_keypoints,
        n_frames=200,
        noise_std=0.1,
        random_seed=42
    )

    logger.info(f"  Generated {len(original)} frames")
    logger.info(f"  Noise level: σ=100mm")

    # Save original data to CSV (simple format for input)
    output_dir = Path("output/synthetic_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = create_cube_topology()

    save_simple_csv(
        filepath=output_dir / "input_data.csv",
        data=original,
        keypoint_names=topology.keypoint_names
    )

    # Create tracking configuration
    config = TrackingConfig(
        input_csv=output_dir / "input_data.csv",
        topology=topology,
        output_dir=output_dir,
        optimization=OptimizationConfig(
            max_iter=300,
            lambda_data=100.0,
            lambda_rigid=500.0,
            lambda_rot_smooth=200.0,
            lambda_trans_smooth=200.0
        )
    )

    # Run pipeline
    result = process_tracking_data(
        config=config,
        ground_truth_data=ground_truth
    )

    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOpen {output_dir / 'rigid_body_viewer.html'} to visualize results")


if __name__ == "__main__":
    run_synthetic_demo()