"""Simplified rigid body tracking API with soft constraints support.

MODIFICATION: Adds 'head_origin' as a virtual marker in output data.
"""

from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp

from python_code.pyceres_solvers.rigid_body_solver.core.geometry import verify_trajectory_reconstruction, \
    print_reference_geometry_summary, save_reference_geometry_json
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

    body_frame_origin_markers: list[str] | None = None
    """Marker names whose mean defines the origin (e.g., ['left_eye', 'right_eye', 'left_ear', 'right_ear'])"""

    body_frame_x_axis_marker: str | None = None
    """Marker name that defines X-axis direction (e.g., 'nose')"""

    body_frame_y_axis_marker: str | None = None
    """Marker name that defines Y-axis direction (e.g., 'left_ear')"""



def estimate_initial_distances(
    *,
    original_data: np.ndarray,
    edges: list[tuple[int, int]],
    edge_type: str = "rigid"
) -> np.ndarray:
    """
    Estimate initial edge distances from original data using median.

    Args:
        original_data: (n_frames, n_markers, 3)
        edges: List of (i, j) pairs
        edge_type: "rigid" or "soft" (for logging)

    Returns:
        (n_markers, n_markers) distance matrix
    """
    n_markers = original_data.shape[1]
    distances = np.zeros((n_markers, n_markers))

    logger.info(f"Estimating {edge_type} edge distances from data...")

    for i, j in edges:
        frame_distances = np.linalg.norm(
            original_data[:, i, :] - original_data[:, j, :],
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
    3. Estimate initial distances
    4. Optimize (with optional parallelization)
    6. Evaluate
    7. Save

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

    original_data = config.topology.extract_trajectories(trajectory_dict=trajectory_dict)
    n_frames = original_data.shape[0]
    logger.info(f"  Data shape: {original_data.shape}")

    # =========================================================================
    # STEP 3: ESTIMATE INITIAL DISTANCES
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 3: ESTIMATE INITIAL DISTANCES")
    logger.info("="*80)

    # RIGID edges: should maintain exact distances
    reference_distances = estimate_initial_distances(
        original_data=original_data,
        edges=config.topology.rigid_edges,
        edge_type="rigid"
    )

    # SOFT edges: can vary, but prefer median distance
    soft_distances = None
    if config.soft_edges:
        soft_distances = estimate_initial_distances(
            original_data=original_data,
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

    result = optimize_rigid_body(
        original_data=original_data,
        rigid_edges=config.topology.rigid_edges,
        reference_distances=reference_distances,
        config=config.optimization,
        marker_names=config.topology.marker_names,
        display_edges=config.topology.display_edges,
        soft_edges=config.soft_edges,
        soft_distances=soft_distances,
        lambda_soft=config.lambda_soft,
        body_frame_origin_markers=config.body_frame_origin_markers,
        body_frame_x_axis_marker=config.body_frame_x_axis_marker,
        body_frame_y_axis_marker=config.body_frame_y_axis_marker
    )

    # =========================================================================
    # STEP 5.5: ADD HEAD ORIGIN AS VIRTUAL MARKER (NEW!)
    # =========================================================================
    marker_names_to_save = config.topology.marker_names.copy()
    optimized_data_to_save = result.reconstructed
    topology_dict_to_save = config.topology.to_dict()


    # =========================================================================
    # STEP 6: EVALUATE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 5: EVALUATE")
    logger.info("="*80)

    # Evaluate only the original markers (not the virtual head_origin)
    metrics = evaluate_reconstruction(
        original_data=original_data,
        optimized_data=result.reconstructed,
        reference_distances=reference_distances,
        topology=config.topology,
        ground_truth_data=None
    )

    # =========================================================================
    # STEP 6.5: VERIFY RECONSTRUCTION
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 5.5: VERIFY RECONSTRUCTION")
    logger.info("="*80)

    # Verify that reconstructed = R @ reference + t
    verification_passed = verify_trajectory_reconstruction(
        reference_geometry=result.reference_geometry,
        rotations=result.rotations,
        translations=result.translations,
        reconstructed=result.reconstructed,
        marker_names=config.topology.marker_names,
        n_frames_to_check=min(10, n_frames),
        tolerance=1e-6  # 1 micron tolerance
    )

    if not verification_passed:
        logger.warning("⚠ Reconstruction verification failed!")

    # Print reference geometry summary
    print_reference_geometry_summary(
        reference_geometry=result.reference_geometry,
        marker_names=config.topology.marker_names,
        units="mm"
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
        original_data=original_data,
        optimized_data=optimized_data_to_save,
        marker_names=marker_names_to_save,
        topology_dict=topology_dict_to_save,
        ground_truth_data=None,
        quaternions=result.quaternions,
        rotations=result.rotations,
        translations=result.translations,
        timestamps=config.timestamps
    )

    # Save reference geometry as JSON
    save_reference_geometry_json(
        filepath=config.output_dir / "reference_geometry.json",
        reference_geometry=result.reference_geometry,
        marker_names=config.topology.marker_names,
        units="mm"
    )

    save_evaluation_report(
        filepath=config.output_dir / "metrics.json",
        metrics=metrics,
        config={
            "topology": config.topology.name,
            "n_frames": n_frames,
            "n_markers": len(config.topology.marker_names),
            "n_markers_with_virtual": len(marker_names_to_save),
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