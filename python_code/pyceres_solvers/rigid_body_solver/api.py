"""Simplified rigid body tracking API with soft constraints support."""

from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp

from python_code.pyceres_solvers.rigid_body_solver.core.topology import RigidBodyTopology
from python_code.pyceres_solvers.rigid_body_solver.core.optimization import (
    OptimizationConfig,
    optimize_rigid_body,
    OptimizationResult
)
from python_code.pyceres_solvers.rigid_body_solver.core.parallel_opt import (
    optimize_chunked_parallel,
    estimate_parallel_speedup
)
from python_code.pyceres_solvers.rigid_body_solver.core.chunking import ChunkConfig
from python_code.pyceres_solvers.rigid_body_solver.io.loaders import load_trajectories
from python_code.pyceres_solvers.rigid_body_solver.io.savers import (
    save_results,
    save_evaluation_report
)
from python_code.pyceres_solvers.rigid_body_solver.core.metrics import evaluate_reconstruction

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Complete configuration for rigid body tracking."""

    input_csv: Path
    """Path to input CSV file"""

    timestamps: np.ndarray
    """Timestamps for each frame"""

    topology: RigidBodyTopology
    """Rigid body topology"""

    output_dir: Path
    """Output directory"""

    optimization: OptimizationConfig
    """Optimization configuration"""

    soft_edges: list[tuple[int, int]] | None = None
    """Optional soft (flexible) edges - e.g., spine segments"""

    lambda_soft: float = 10.0
    """Weight for soft constraints (lower = more flexible)"""

    use_parallel: bool = True
    """Use parallel processing for long recordings"""

    n_workers: int | None = None
    """Number of parallel workers (None = all cores)"""


def estimate_initial_distances(
    *,
    noisy_data: np.ndarray,
    edges: list[tuple[int, int]],
    edge_type: str = "rigid"
) -> np.ndarray:
    """
    Estimate initial edge distances from noisy data using median.

    Args:
        noisy_data: (n_frames, n_markers, 3)
        edges: List of (i, j) pairs
        edge_type: "rigid" or "soft" (for logging)

    Returns:
        (n_markers, n_markers) distance matrix
    """
    n_markers = noisy_data.shape[1]
    distances = np.zeros((n_markers, n_markers))

    logger.info(f"Estimating {edge_type} edge distances from data...")

    for i, j in edges:
        frame_distances = np.linalg.norm(
            noisy_data[:, i, :] - noisy_data[:, j, :],
            axis=1
        )
        median_dist = np.median(frame_distances)
        std_dist = np.std(frame_distances)
        distances[i, j] = median_dist
        distances[j, i] = median_dist

        logger.info(f"  Edge ({i},{j}): {median_dist:.4f}m ± {std_dist*1000:.1f}mm")

    return distances


def process_tracking_data(*, config: TrackingConfig) -> OptimizationResult:
    """
    Complete rigid body tracking pipeline with soft constraints.

    Pipeline:
    1. Load data
    2. Extract markers
    3. Estimate initial distances (rigid + soft)
    4. Optimize (with optional parallelization)
    5. Evaluate
    6. Save

    Args:
        config: TrackingConfig

    Returns:
        OptimizationResult
    """
    logger.info("="*80)
    logger.info("RIGID BODY TRACKING PIPELINE")
    logger.info("="*80)
    logger.info(f"Input:    {config.input_csv.name}")
    logger.info(f"Output:   {config.output_dir}")
    logger.info(f"Topology: {config.topology.name}")
    logger.info(f"Markers:  {len(config.topology.marker_names)}")

    if config.soft_edges:
        logger.info(f"Soft edges: {len(config.soft_edges)} (flexible constraints)")
        logger.info(f"Soft weight: {config.lambda_soft} (lower = more flexible)")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 1: LOAD DATA")
    logger.info("="*80)

    trajectory_dict = load_trajectories(
        filepath=config.input_csv,
        scale_factor=1.0,
        z_value=0.0
    )

    # =========================================================================
    # STEP 2: EXTRACT MARKERS
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 2: EXTRACT MARKERS")
    logger.info("="*80)

    noisy_data = config.topology.extract_trajectories(trajectory_dict=trajectory_dict)
    n_frames = noisy_data.shape[0]
    logger.info(f"  Data shape: {noisy_data.shape}")

    # =========================================================================
    # STEP 3: ESTIMATE INITIAL DISTANCES
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 3: ESTIMATE INITIAL DISTANCES")
    logger.info("="*80)

    # RIGID edges: should maintain exact distances
    reference_distances = estimate_initial_distances(
        noisy_data=noisy_data,
        edges=config.topology.rigid_edges,
        edge_type="rigid"
    )

    # SOFT edges: can vary, but prefer median distance
    soft_distances = None
    if config.soft_edges:
        soft_distances = estimate_initial_distances(
            noisy_data=noisy_data,
            edges=config.soft_edges,
            edge_type="soft"
        )

    # =========================================================================
    # STEP 4: DECIDE ON PARALLELIZATION
    # =========================================================================
    # use_chunking = n_frames > 1000
    use_chunking = False
    chunk_config = ChunkConfig(
        chunk_size=500,
        overlap_size=50,
        blend_window=25,
        min_chunk_size=100
    )

    if use_chunking and config.use_parallel:
        n_workers = config.n_workers or mp.cpu_count()
        logger.info(f"\n{'='*80}")
        logger.info(f"USING PARALLEL OPTIMIZATION ({n_frames} frames)")
        logger.info("="*80)
        logger.info(f"Workers: {n_workers}")

        estimate = estimate_parallel_speedup(
            n_frames=n_frames,
            chunk_size=chunk_config.chunk_size,
            n_workers=n_workers
        )
        logger.info(f"\nParallel processing estimate:")
        logger.info(f"  Chunks: {estimate['n_chunks']}")
        logger.info(f"  Sequential time: ~{estimate['sequential_time_minutes']:.1f} min")
        logger.info(f"  Parallel time: ~{estimate['parallel_time_minutes']:.1f} min")
        logger.info(f"  Expected speedup: {estimate['speedup']:.1f}x")

    # =========================================================================
    # STEP 5: OPTIMIZE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 4: OPTIMIZE")
    logger.info("="*80)

    if use_chunking and config.use_parallel:
        rotations, translations, reconstructed = optimize_chunked_parallel(
            noisy_data=noisy_data,
            rigid_edges=config.topology.rigid_edges,
            reference_distances=reference_distances,
            optimization_config=config.optimization,
            chunk_config=chunk_config,
            optimize_fn=optimize_rigid_body,
            n_workers=config.n_workers,
            soft_edges=config.soft_edges,
            soft_distances=soft_distances,
            lambda_soft=config.lambda_soft
        )

        # Create result object
        result = OptimizationResult(
            rotations=rotations,
            translations=translations,
            reconstructed=reconstructed,
            reference_geometry=np.zeros((len(config.topology.marker_names), 3)),
            initial_cost=0.0,
            final_cost=0.0,
            success=True,
            iterations=0,
            time_seconds=0.0
        )
    else:
        result = optimize_rigid_body(
            noisy_data=noisy_data,
            rigid_edges=config.topology.rigid_edges,
            reference_distances=reference_distances,
            config=config.optimization,
            soft_edges=config.soft_edges,
            soft_distances=soft_distances,
            lambda_soft=config.lambda_soft
        )

    # =========================================================================
    # STEP 6: EVALUATE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 5: EVALUATE")
    logger.info("="*80)

    metrics = evaluate_reconstruction(
        noisy_data=noisy_data,
        optimized_data=result.reconstructed,
        reference_distances=reference_distances,
        topology=config.topology,
        ground_truth_data=None
    )

    # =========================================================================
    # STEP 7: SAVE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 6: SAVE RESULTS")
    logger.info("="*80)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    save_results(
        output_dir=config.output_dir,
        noisy_data=noisy_data,
        optimized_data=result.reconstructed,
        marker_names=config.topology.marker_names,
        topology_dict=config.topology.to_dict(),
        ground_truth_data=None,
        rotations=result.rotations,
        translations=result.translations,
        timestamps=config.timestamps
    )

    save_evaluation_report(
        filepath=config.output_dir / "metrics.json",
        metrics=metrics,
        config={
            "topology": config.topology.name,
            "n_frames": n_frames,
            "n_markers": len(config.topology.marker_names),
            "optimization": {
                "max_iter": config.optimization.max_iter,
                "lambda_data": config.optimization.lambda_data,
                "lambda_rigid": config.optimization.lambda_rigid,
                "lambda_soft": config.lambda_soft,
                "lambda_rot_smooth": config.optimization.lambda_rot_smooth,
                "lambda_trans_smooth": config.optimization.lambda_trans_smooth,
            },
            "use_parallel": config.use_parallel,
            "n_workers": config.n_workers or mp.cpu_count(),
            "soft_edges_count": len(config.soft_edges) if config.soft_edges else 0
        }
    )

    logger.info(f"\n✓ Complete! Results saved to: {config.output_dir}")
    logger.info(f"  Open {config.output_dir / 'rigid_body_viewer.html'} to visualize")

    return result