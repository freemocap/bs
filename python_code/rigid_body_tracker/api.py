"""Main API with chunked optimization support."""

from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp

from python_code.rigid_body_tracker.core.topology import RigidBodyTopology
from python_code.rigid_body_tracker.core.optimization import (
    OptimizationConfig,
    optimize_rigid_body,
    OptimizationResult,
    align_reconstructed_to_noisy
)
from python_code.rigid_body_tracker.core.chunking import (
    ChunkConfig,
    optimize_chunked
)
from python_code.rigid_body_tracker.core.parallel_opt import (
    optimize_chunked_parallel,
    estimate_parallel_speedup
)
from python_code.rigid_body_tracker.core.reference import estimate_reference_geometry
from python_code.rigid_body_tracker.io.loaders import load_trajectories
from python_code.rigid_body_tracker.io.savers import (
    save_results,
    save_evaluation_report,
    print_summary
)
from python_code.rigid_body_tracker.core.metrics import evaluate_reconstruction

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Complete configuration for rigid body tracking pipeline."""

    input_csv: Path
    """Path to input CSV file"""

    topology: RigidBodyTopology
    """Rigid body topology defining structure"""

    output_dir: Path
    """Directory for output files"""

    scale_factor: float = 1.0
    """Scale factor for coordinates (e.g., 0.001 for mm to m)"""

    z_value: float = 0.0
    """Default z-coordinate for 2D data"""

    optimization: OptimizationConfig | None = None
    """Optimization configuration"""

    chunk_config: ChunkConfig | None = None
    """Chunking configuration (None = auto-decide based on length)"""

    use_chunking: bool | None = None
    """Force chunking on/off (None = auto-decide)"""

    chunking_threshold: int = 1000
    """Automatically use chunking if frames > threshold"""

    use_parallel: bool = True
    """Use parallel processing for chunks (recommended for speed)"""

    n_workers: int | None = None
    """Number of parallel workers (None = use all CPU cores)"""

    csv_format: str | None = None
    """Force CSV format ('tidy', 'wide', 'dlc'), or None for auto-detect"""

    reference_method: str = "median"
    """Method for reference estimation ('mean' or 'median')"""


class PipelineResult:
    """Results from complete tracking pipeline."""

    def __init__(
        self,
        *,
        noisy_data: np.ndarray,
        optimized_data: np.ndarray,
        reference_geometry: np.ndarray,
        rotations: np.ndarray,
        translations: np.ndarray,
        ground_truth_data: np.ndarray | None = None,
        metrics: dict[str, float] | None = None,
        optimization_result: OptimizationResult | None = None
    ) -> None:
        self.noisy_data = noisy_data
        self.optimized_data = optimized_data
        self.reference_geometry = reference_geometry
        self.rotations = rotations
        self.translations = translations
        self.ground_truth_data = ground_truth_data
        self.metrics = metrics
        self.optimization_result = optimization_result


def process_tracking_data(
    *,
    config: TrackingConfig,
    ground_truth_data: np.ndarray | None = None
) -> PipelineResult:
    """
    Complete rigid body tracking pipeline with automatic chunking.

    Args:
        config: Pipeline configuration
        ground_truth_data: Optional (n_frames, n_markers, 3) ground truth for validation

    Returns:
        PipelineResult with optimized trajectories and metrics
    """
    logger.info("="*80)
    logger.info("RIGID BODY TRACKING PIPELINE")
    logger.info("="*80)
    logger.info(f"\nInput:    {config.input_csv.name}")
    logger.info(f"Output:   {config.output_dir}")
    logger.info(f"Topology: {config.topology.name}")
    logger.info(f"Markers:  {len(config.topology.marker_names)}")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 1: LOAD DATA")
    logger.info("="*80)

    trajectory_dict = load_trajectories(
        filepath=config.input_csv,
        scale_factor=config.scale_factor,
        z_value=config.z_value,
        format=config.csv_format
    )

    # =========================================================================
    # STEP 2: EXTRACT MARKERS ACCORDING TO TOPOLOGY
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 2: EXTRACT MARKERS")
    logger.info("="*80)

    noisy_data = config.topology.extract_trajectories(
        trajectory_dict=trajectory_dict
    )
    logger.info(f"  Data shape: {noisy_data.shape}")

    n_frames = noisy_data.shape[0]

    # =========================================================================
    # STEP 3: ESTIMATE REFERENCE GEOMETRY
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 3: ESTIMATE REFERENCE GEOMETRY")
    logger.info("="*80)

    reference_geometry = estimate_reference_geometry(
        noisy_data=noisy_data,
    )

    reference_distances = config.topology.compute_reference_distances(
        reference_geometry=reference_geometry
    )

    # =========================================================================
    # STEP 4: DECIDE ON CHUNKING STRATEGY
    # =========================================================================
    if config.use_chunking is None:
        # Auto-decide based on length
        use_chunking = n_frames > config.chunking_threshold
    else:
        use_chunking = config.use_chunking

    if use_chunking:
        logger.info(f"\n{'='*80}")
        logger.info(f"USING CHUNKED OPTIMIZATION ({n_frames} frames)")
        logger.info("="*80)

    # =========================================================================
    # STEP 5: OPTIMIZE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 4: OPTIMIZE RIGID BODY POSES")
    logger.info("="*80)

    opt_config = config.optimization or OptimizationConfig()

    if use_chunking:
        chunk_config = config.chunk_config or ChunkConfig()

        # Estimate speedup
        if config.use_parallel and config.n_workers != 1:
            n_workers = config.n_workers or mp.cpu_count()
            estimate = estimate_parallel_speedup(
                n_frames=n_frames,
                chunk_size=chunk_config.chunk_size,
                n_workers=n_workers
            )
            logger.info(f"\nParallel processing estimate:")
            logger.info(f"  Chunks: {estimate['n_chunks']}")
            logger.info(f"  Workers: {n_workers}")
            logger.info(f"  Sequential time: ~{estimate['sequential_time_minutes']:.1f} minutes")
            logger.info(f"  Parallel time: ~{estimate['parallel_time_minutes']:.1f} minutes")
            logger.info(f"  Expected speedup: {estimate['speedup']:.1f}x")

        if config.use_parallel:
            rotations, translations, optimized_data = optimize_chunked_parallel(
                noisy_data=noisy_data,
                reference_geometry=reference_geometry,
                rigid_edges=config.topology.rigid_edges,
                reference_distances=reference_distances,
                optimization_config=opt_config,
                chunk_config=chunk_config,
                optimize_fn=optimize_rigid_body,
                n_workers=config.n_workers
            )
        else:
            rotations, translations, optimized_data = optimize_chunked(
                noisy_data=noisy_data,
                reference_geometry=reference_geometry,
                rigid_edges=config.topology.rigid_edges,
                reference_distances=reference_distances,
                optimization_config=opt_config,
                chunk_config=chunk_config,
                optimize_fn=optimize_rigid_body
            )

        optimization_result = None  # No single result object for chunked

    else:
        result = optimize_rigid_body(
            noisy_data=noisy_data,
            reference_geometry=reference_geometry,
            rigid_edges=config.topology.rigid_edges,
            reference_distances=reference_distances,
            config=opt_config
        )

        rotations = result.rotations
        translations = result.translations
        optimized_data = result.reconstructed
        optimization_result = result

    # =========================================================================
    # STEP 5.5: ALIGN TO INPUT COORDINATE FRAME
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 4.5: ALIGN TO INPUT COORDINATE FRAME")
    logger.info("="*80)

    optimized_data, rotations = align_reconstructed_to_noisy(
        noisy_data=noisy_data,
        reconstructed_data=optimized_data,
        rotations=rotations
    )

    # =========================================================================
    # STEP 6: EVALUATE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 5: EVALUATE RESULTS")
    logger.info("="*80)

    metrics = evaluate_reconstruction(
        noisy_data=noisy_data,
        optimized_data=optimized_data,
        reference_distances=reference_distances,
        topology=config.topology,
        ground_truth_data=ground_truth_data
    )

    # =========================================================================
    # STEP 7: SAVE RESULTS
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 6: SAVE RESULTS")
    logger.info("="*80)

    save_results(
        output_dir=config.output_dir,
        noisy_data=noisy_data,
        optimized_data=optimized_data,
        marker_names=config.topology.marker_names,
        topology_dict=config.topology.to_dict(),
        ground_truth_data=ground_truth_data
    )

    # Save metrics
    save_evaluation_report(
        filepath=config.output_dir / "metrics.json",
        metrics=metrics,
        config={
            "topology": config.topology.name,
            "n_frames": n_frames,
            "n_markers": len(config.topology.marker_names),
            "optimization": {
                "max_iter": opt_config.max_iter,
                "lambda_data": opt_config.lambda_data,
                "lambda_rigid": opt_config.lambda_rigid,
                "lambda_rot_smooth": opt_config.lambda_rot_smooth,
                "lambda_trans_smooth": opt_config.lambda_trans_smooth,
            },
            "chunked": use_chunking
        }
    )

    print_summary(
        noisy_data=noisy_data,
        optimized_data=optimized_data,
        ground_truth_data=ground_truth_data
    )

    return PipelineResult(
        noisy_data=noisy_data,
        optimized_data=optimized_data,
        reference_geometry=reference_geometry,
        rotations=rotations,
        translations=translations,
        ground_truth_data=ground_truth_data,
        metrics=metrics,
        optimization_result=optimization_result
    )