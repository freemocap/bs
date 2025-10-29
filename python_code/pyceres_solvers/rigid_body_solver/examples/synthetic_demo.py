"""Synthetic data demo with cube markers."""

from pathlib import Path
import numpy as np
import logging
from scipy.spatial.transform import Rotation

from python_code.pyceres_solvers.rigid_body_solver.core.topology import RigidBodyTopology
from python_code.pyceres_solvers.rigid_body_solver.core.optimization import OptimizationConfig
from python_code.pyceres_solvers.rigid_body_solver.api import TrackingConfig, process_tracking_data
from python_code.pyceres_solvers.rigid_body_solver.io.savers import save_simple_csv

logger = logging.getLogger(__name__)


def rotation_matrix_from_axis_angle(*, axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix from axis-angle."""
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def generate_cube_markers(*, size: float = 1.0, n_extra: int = 3) -> np.ndarray:
    """
    Generate cube vertices plus asymmetric markers.

    Args:
        size: Cube half-width
        n_extra: Number of extra asymmetric markers

    Returns:
        (8 + n_extra, 3) marker positions
    """
    s = size

    # Base cube
    cube = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ])

    # Add asymmetric markers to break symmetry
    extra_markers = []
    if n_extra >= 1:
        extra_markers.append([0.0, -s * 1.5, 0.0])
    if n_extra >= 2:
        extra_markers.append([s * 1.3, -s, -s * 0.7])
    if n_extra >= 3:
        extra_markers.append([-s * 0.8, -s * 0.8, s * 1.4])

    if extra_markers:
        return np.vstack([cube, np.array(extra_markers)])
    return cube


def generate_synthetic_trajectory(
    *,
    reference_markers: np.ndarray,
    n_frames: int = 200,
    noise_std: float = 0.1,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic trajectory with circular motion.

    Args:
        reference_markers: (n_markers, 3) marker configuration
        n_frames: Number of frames to generate
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        - ground_truth: (n_frames, n_markers, 3)
        - noisy: (n_frames, n_markers, 3)
    """
    n_markers = len(reference_markers)
    ground_truth = np.zeros((n_frames, n_markers, 3))

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

        ground_truth[i] = (R @ reference_markers.T).T + translation

    # Add noise
    np.random.seed(seed=random_seed)
    noise = np.random.normal(loc=0, scale=noise_std, size=ground_truth.shape)
    noisy = ground_truth + noise

    return ground_truth, noisy


def create_cube_topology() -> RigidBodyTopology:
    """Create topology for cube with asymmetric markers."""

    marker_names = [
        "v0", "v1", "v2", "v3",  # Bottom face
        "v4", "v5", "v6", "v7",  # Top face
        "m0", "m1", "m2"          # Asymmetric markers
    ]

    # Cube edges (12 edges)
    rigid_edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Diagonal connections to asymmetric markers
        (0, 8), (1, 8), (4, 8),
        (1, 9), (2, 9), (5, 9),
        (4, 10), (7, 10), (0, 10),
    ]

    return RigidBodyTopology(
        marker_names=marker_names,
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
    reference_markers = generate_cube_markers(size=1.0, n_extra=3)
    ground_truth, noisy = generate_synthetic_trajectory(
        reference_markers=reference_markers,
        n_frames=200,
        noise_std=0.1,
        random_seed=42
    )

    logger.info(f"  Generated {len(noisy)} frames")
    logger.info(f"  Noise level: σ=100mm")

    # Save noisy data to CSV (simple format for input)
    output_dir = Path("output/synthetic_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = create_cube_topology()

    save_simple_csv(
        filepath=output_dir / "input_data.csv",
        data=noisy,
        marker_names=topology.marker_names
    )

    # Create tracking configuration
    config = TrackingConfig(
        input_csv=output_dir / "input_data.csv",
        topology=topology,
        output_dir=output_dir,
        scale_factor=1.0,
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