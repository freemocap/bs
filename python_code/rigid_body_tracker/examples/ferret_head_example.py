"""Ferret head tracking with automatic chunking for long recordings."""
import multiprocessing
from pathlib import Path
import logging

from python_code.rigid_body_tracker.core.topology import RigidBodyTopology
from python_code.rigid_body_tracker.core.optimization import OptimizationConfig
from python_code.rigid_body_tracker.core.chunking import ChunkConfig
from python_code.rigid_body_tracker.api import TrackingConfig, process_tracking_data

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


def run_ferret_tracking_parallel() -> None:
    """Run ferret head tracking with PARALLEL chunked optimization (FASTEST!)."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    # Create topology
    topology = create_ferret_head_topology()

    # Check available CPU cores
    n_cores = multiprocessing.cpu_count()
    logger.info(f"System has {n_cores} CPU cores available")

    # Configure chunking for long recordings
    chunk_config = ChunkConfig(
        chunk_size=500,  # Process 500 frames at a time
        overlap_size=50,  # 50 frame overlap between chunks
        blend_window=25,  # Blend over 25 frames
        min_chunk_size=100  # Minimum viable chunk size
    )

    # Optimization config
    optimization_config = OptimizationConfig(
        max_iter=100,  # Can reduce since we're chunking
        lambda_data=100.0,
        lambda_rigid=500.0,
        lambda_rot_smooth=200.0,
        lambda_trans_smooth=200.0
    )

    # Pipeline config with PARALLEL processing enabled
    config = TrackingConfig(
        input_csv=Path(r"D:\bs\ferret_recordings\session_2025-07-01_ferret_757_EyeCameras_P33_EO5\clips\1m_20s-2m_20s\mocap_data\output_data\processed_data\head_spine_body_rigid_3d_xyz.csv"),
        topology=topology,
        output_dir=Path("output/ferret_head_parallel"),
        scale_factor=1.0,
        optimization=optimization_config,
        chunk_config=chunk_config,
        use_chunking=True,
        use_parallel=True,  # 🚀 ENABLE PARALLEL PROCESSING
        n_workers=None,  # Use all available cores (or set to specific number)
        chunking_threshold=1000
    )

    # Run pipeline
    result = process_tracking_data(config=config)

    logger.info("\n" + "=" * 80)
    logger.info("PARALLEL OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✓ Processed {result.noisy_data.shape[0]} frames using parallel chunks")
    logger.info(f"✓ Results saved to: {config.output_dir}")
    logger.info(f"\nOpen {config.output_dir / 'rigid_body_viewer.html'} to visualize")

    # Print key metrics
    if result.metrics:
        logger.info("\nKey Metrics:")
        logger.info(f"  Edge error (optimized): {result.metrics['optimized_edge_error_mean_mm']:.2f}mm")
        logger.info(f"  Edge consistency: {result.metrics['optimized_edge_std_mm']:.3f}mm std")
        logger.info(f"  Smoothness: {result.metrics['optimized_centroid_jitter_mm']:.3f}mm jitter")


def run_ferret_tracking_sequential() -> None:
    """Run with sequential chunking (slower but uses less memory)."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    topology = create_ferret_head_topology()

    config = TrackingConfig(
        input_csv=Path("data/head_spine_body_rigid_3d_xyz.csv"),
        topology=topology,
        output_dir=Path("output/ferret_head_sequential"),
        scale_factor=1.0,
        optimization=OptimizationConfig(
            max_iter=100,
            lambda_data=100.0,
            lambda_rigid=500.0,
            lambda_rot_smooth=200.0,
            lambda_trans_smooth=200.0
        ),
        chunk_config=ChunkConfig(
            chunk_size=500,
            overlap_size=50,
            blend_window=25
        ),
        use_chunking=True,
        use_parallel=False,  # Sequential processing
    )

    result = process_tracking_data(config=config)
    logger.info("\n✓ Sequential processing complete!")


def run_ferret_tracking_custom_workers() -> None:
    """Run with custom number of workers (e.g., leave cores free for other tasks)."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    topology = create_ferret_head_topology()
    n_cores = multiprocessing.cpu_count()

    # Use half the cores to leave system responsive
    n_workers = max(1, n_cores // 2)
    logger.info(f"Using {n_workers}/{n_cores} cores for optimization")

    config = TrackingConfig(
        input_csv=Path("data/head_spine_body_rigid_3d_xyz.csv"),
        topology=topology,
        output_dir=Path("output/ferret_head_custom"),
        scale_factor=1.0,
        optimization=OptimizationConfig(
            max_iter=100,
            lambda_data=100.0,
            lambda_rigid=500.0,
            lambda_rot_smooth=200.0,
            lambda_trans_smooth=200.0
        ),
        use_chunking=True,
        use_parallel=True,
        n_workers=n_workers,  # Use specific number of workers
    )

    result = process_tracking_data(config=config)
    logger.info("\n✓ Custom worker processing complete!")


if __name__ == "__main__":
    # RECOMMENDED: Run with parallel processing for maximum speed
    run_ferret_tracking_parallel()

    # Or run with sequential processing (slower but lower memory)
    # run_ferret_tracking_sequential()

    # Or run with custom number of workers
    # run_ferret_tracking_custom_workers()

    # Or run with automatic decision
    # run_ferret_tracking_auto()