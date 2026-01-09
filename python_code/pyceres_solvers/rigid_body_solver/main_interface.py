"""Simplified rigid body tracking API with soft constraints support.

MODIFICATION: Adds 'head_origin' as a virtual marker in output data.
"""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from python_code.pyceres_solvers.rigid_body_solver.core.reference_geometry import verify_trajectory_reconstruction, \
    print_reference_geometry_summary, save_reference_geometry_json
from python_code.pyceres_solvers.rigid_body_solver.core.optimization import (
    OptimizationConfig,
    optimize_rigid_body,
    OptimizationResult
)
from python_code.pyceres_solvers.rigid_body_solver.core.topology import RigidBodyTopology
from python_code.pyceres_solvers.rigid_body_solver.io.loaders import load_trajectories
from python_code.pyceres_solvers.rigid_body_solver.io.savers import (
    save_results,
)

logger = logging.getLogger(__name__)


@dataclass
class RigidBodySolverConfig:
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

    rigid_body_name: str

    body_frame_origin_markers: list[str] | None = None
    """Marker names whose mean defines the origin (e.g., ['left_eye', 'right_eye', 'left_ear', 'right_ear'])"""

    body_frame_x_axis_marker: str | None = None
    """Marker name that defines X-axis direction (e.g., 'nose')"""

    body_frame_y_axis_marker: str | None = None
    """Marker name that defines Y-axis direction (e.g., 'left_ear')"""



def estimate_initial_distances(
    *,
    original_data: np.ndarray,
    edges: list[tuple[str, str]],
        marker_names: list[str]
) -> np.ndarray:
    """
    Estimate initial edge distances from original data using median.

    Args:
        original_data: (n_frames, n_markers, 3)
        edges: List of (i, j) pairs

    Returns:
        (n_markers, n_markers) distance matrix
    """
    def marker_name_to_index(name: str) -> int:
        try:
            return marker_names.index(name)
        except ValueError:
            raise ValueError(f"Marker name '{name}' not found in marker_names: {marker_names}")
    n_markers = original_data.shape[1]
    distances = np.zeros((n_markers, n_markers))

    logger.info(f"Estimating rigid edge distances from data...")

    for i, j in edges:
        i_idx = marker_name_to_index(i)
        j_idx = marker_name_to_index(j)
        frame_distances = np.linalg.norm(
            original_data[:, i_idx, :] - original_data[:, j_idx, :],
            axis=1
        )
        median_dist = np.median(frame_distances)
        std_dist = np.std(frame_distances)
        distances[i_idx, j_idx] = median_dist
        distances[j_idx, i_idx] = median_dist

        logger.info(f"  Edge ({i},{j}): {median_dist:.4f}m ± {std_dist:.1f}mm")

    return distances


def process_tracking_data(*, config: RigidBodySolverConfig) -> OptimizationResult:
    """
    Complete rigid body tracking pipeline with soft constraints.

    Pipeline:
    1. Load data
    2. Extract markers
    3. Estimate initial distances
    4. Optimize
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


    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 1: LOAD DATA")
    logger.info("="*80)

    trajectory_dict = load_trajectories(
        filepath=config.input_csv,
        scale_factor=1.0,
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
        marker_names=config.topology.marker_names
    )


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
        body_frame_origin_markers=config.body_frame_origin_markers,
        body_frame_x_axis_marker=config.body_frame_x_axis_marker,
        body_frame_y_axis_marker=config.body_frame_y_axis_marker
    )





    # =========================================================================
    # VERIFY RECONSTRUCTION
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("VERIFY RECONSTRUCTION")
    logger.info("="*80)

    # Verify that reconstructed = R @ reference + t
    verification_passed = verify_trajectory_reconstruction(
        reference_geometry=result.reference_geometry,
        rotations=result.rotations,
        translations=result.translations,
        reconstructed=result.reconstructed_keypoints,
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
        rigid_body_name=config.rigid_body_name,
        quaternions=result.quaternions,
        rotations=result.rotations,
        translations=result.translations,
        timestamps=config.timestamps
    )

    # Save reference geometry as JSON
    save_reference_geometry_json(
        filepath=config.output_dir / f"{config.rigid_body_name}_reference_geometry.json",
        reference_geometry=result.reference_geometry,
        marker_names=config.topology.marker_names,
        units="mm"
    )



    logger.info(f"\n✓ Complete! Results saved to: {config.output_dir}")
    logger.info(f"  Open {config.output_dir / 'rigid_body_viewer.html'} to visualize")

    return result