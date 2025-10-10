"""Ferret head tracking with automatic chunking and reference visualization."""
import multiprocessing
from pathlib import Path
import logging

from python_code.rigid_body_tracker.core.topology import RigidBodyTopology
from python_code.rigid_body_tracker.core.optimization import OptimizationConfig
from python_code.rigid_body_tracker.core.chunking import ChunkConfig
from python_code.rigid_body_tracker.api import TrackingConfig, process_tracking_data
from python_code.rigid_body_tracker.io.loaders import load_trajectories
from python_code.rigid_body_tracker.core.reference import estimate_reference_geometry

logger = logging.getLogger(__name__)


def create_ferret_head_topology() -> RigidBodyTopology:
    """Create topology for ferret head (8 markers)."""

    marker_names = [
        "nose",  # 0
        "left_eye",  # 1
        "right_eye",  # 2
        "left_ear",  # 3
        "right_ear",  # 4
        "base",  # 5
        "left_cam_tip",  # 6
        "right_cam_tip",  # 7
    ]

    # Define rigid connections
    rigid_edges = [
        # Skull structure - core rigidity
        (0, 1),  # nose to left_eye
        (0, 2),  # nose to right_eye
        (0, 3),  # nose to left_ear
        (0, 4),  # nose to right_ear
        (0, 5),  # nose to base
        (0, 6),  # nose to left_cam_tip
        (0, 7),  # nose to right_cam_tip
        (1, 2),  # left_eye to right_eye
        (1, 3),  # left_eye to left_ear
        (1, 4),  # left_eye to right_ear
        (1, 5),  # left_eye to base
        (1, 6),  # left_eye to left_cam_tip
        (1, 7),  # left_eye to right_cam_tip
        (2, 3),  # right_eye to left_ear
        (2, 4),  # right_eye to right_ear
        (2, 5),  # right_eye to base
        (2, 6),  # right_eye to left_cam_tip
        (2, 7),  # right_eye to right_cam_tip
        (3, 4),  # left_ear to right_ear
        (3, 5),  # left_ear to base
        (3, 6),  # left_ear to left_cam_tip
        (3, 7),  # left_ear to right_cam_tip
        (4, 5),  # right_ear to base
        (4, 6),  # right_ear to left_cam_tip
        (4, 7),  # right_ear to right_cam_tip
        (5, 6),  # base to left_cam_tip
        (5, 7),  # base to right_cam_tip
        (6, 7),  # left_cam_tip to right_cam_tip
    ]

    return RigidBodyTopology(
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        name="ferret_head_v1"
    )


def visualize_reference_alignment_only(
    *,
    input_csv: Path,
    topology: RigidBodyTopology,
    output_path: Path,
    show_n_frames: int = 200
) -> None:
    """
    Generate only the reference alignment visualization.

    Useful for debugging reference geometry issues.
    """
    logger.info("="*80)
    logger.info("REFERENCE ALIGNMENT VISUALIZATION")
    logger.info("="*80)

    # Load data
    trajectory_dict = load_trajectories(
        filepath=input_csv,
        scale_factor=1.0,
        z_value=0.0
    )

    noisy_data = topology.extract_trajectories(trajectory_dict=trajectory_dict)
    logger.info(f"Loaded data: {noisy_data.shape}")

    # Estimate reference with visualization
    reference_geometry = estimate_reference_geometry(
        noisy_data=noisy_data,
        show_n_frames=show_n_frames
    )


def run_ferret_tracking() -> None:
    """Run ferret head tracking with PARALLEL chunked optimization."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    # Create topology
    topology = create_ferret_head_topology()

    input_csv = Path(
        r"D:\bs\ferret_recordings\session_2025-07-01_ferret_757_EyeCameras_P33_EO5"
        r"\clips\1m_20s-2m_20s\mocap_data\output_data\processed_data"
        r"\head_spine_body_rigid_3d_xyz.csv"
    )

    output_dir = Path("output/ferret_head")
    n_cores = multiprocessing.cpu_count()
    logger.info(f"System has {n_cores} CPU cores available")

    chunk_config = ChunkConfig(
        chunk_size=500,
        overlap_size=50,
        blend_window=25,
        min_chunk_size=100
    )

    optimization_config = OptimizationConfig(
        max_iter=100,
        lambda_data=100.0,
        lambda_rigid=200.0,
        lambda_rot_smooth=100.0,
        lambda_trans_smooth=100.0
    )

    config = TrackingConfig(
        input_csv=input_csv,
        topology=topology,
        output_dir=output_dir,
        scale_factor=1.0,
        optimization=optimization_config,
        chunk_config=chunk_config,
        use_chunking=True,
        use_parallel=True,
        n_workers=None,
        chunking_threshold=1000
    )

    result = process_tracking_data(config=config)

    logger.info("\n" + "=" * 80)
    logger.info("PARALLEL OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✓ Processed {result.noisy_data.shape[0]} frames using parallel chunks")
    logger.info(f"✓ Results saved to: {config.output_dir}")
    logger.info(f"\nOpen {config.output_dir / 'rigid_body_viewer.html'} to visualize")

    if result.metrics:
        logger.info("\nKey Metrics:")
        logger.info(f"  Edge error (optimized): {result.metrics['optimized_edge_error_mean_mm']:.2f}mm")
        logger.info(f"  Edge consistency: {result.metrics['optimized_edge_std_mm']:.3f}mm std")
        logger.info(f"  Smoothness: {result.metrics['optimized_centroid_jitter_mm']:.3f}mm jitter")


if __name__ == "__main__":
    run_ferret_tracking()