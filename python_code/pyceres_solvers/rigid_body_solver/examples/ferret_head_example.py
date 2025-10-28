"""Ferret tracking: RIGID skull only + attach raw spine markers."""

from pathlib import Path
import logging
import numpy as np

from python_code.pyceres_solvers.rigid_body_solver.core.topology import RigidBodyTopology
from python_code.pyceres_solvers.rigid_body_solver.core.optimization import OptimizationConfig
from python_code.pyceres_solvers.rigid_body_solver.api import TrackingConfig, process_tracking_data
from python_code.pyceres_solvers.rigid_body_solver.io.loaders import load_trajectories
from python_code.pyceres_solvers.rigid_body_solver.io.savers import save_results

logger = logging.getLogger(__name__)


def create_skull_only_topology() -> RigidBodyTopology:
    """
    Create topology for SKULL ONLY (no spine in optimization).

    Returns:
        RigidBodyTopology with only skull markers
    """

    marker_names = [
        # SKULL (0-7) - RIGID (bone doesn't bend!)
        "nose", # 0
        "left_eye", # 1
        "right_eye", # 2
        "left_ear", # 3
        "right_ear",    #    4
        "base", # 5
        "left_cam_tip", # 6
        "right_cam_tip", # 7
    ]

    # RIGID EDGES: all skull markers (maintain exact distances)
    rigid_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6), (5, 7),
        (6, 7)
    ]

    # DISPLAY EDGES: visualize skull structure
    display_edges = [
        (0, 1), (0, 2),
        (1, 2),  # Face triangle
        (1, 3), (2, 4),
        (3, 4),  # Ears
        (3, 5), (4, 5),  # Back of head
        (5, 6), (5, 7),   # Camera mount
    ]

    topology = RigidBodyTopology(
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        display_edges=display_edges,
        name="ferret_skull_only"
    )

    return topology


def attach_raw_spine_markers(
    *,
    optimized_skull: np.ndarray,
    raw_spine_data: np.ndarray,
    skull_marker_names: list[str],
    spine_marker_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """
    Attach raw (unoptimized) spine markers to optimized skull data.

    Args:
        optimized_skull: (n_frames, n_skull_markers, 3) optimized skull positions
        raw_spine_data: (n_frames, n_spine_markers, 3) raw spine measurements
        skull_marker_names: Names of skull markers
        spine_marker_names: Names of spine markers

    Returns:
        Tuple of:
        - combined_data: (n_frames, n_total_markers, 3)
        - combined_names: List of all marker names
    """
    n_frames = optimized_skull.shape[0]
    n_skull = optimized_skull.shape[1]
    n_spine = raw_spine_data.shape[1]

    # Concatenate skull + spine
    combined_data = np.concatenate([optimized_skull, raw_spine_data], axis=1)
    combined_names = skull_marker_names + spine_marker_names

    logger.info(f"\nAttached raw spine markers:")
    logger.info(f"  Skull (optimized): {n_skull} markers")
    logger.info(f"  Spine (raw):       {n_spine} markers")
    logger.info(f"  Total:             {n_skull + n_spine} markers")

    return combined_data, combined_names


def run_ferret_skull_solver(input_csv: Path, timestamps_path: Path, output_dir: Path) -> None:
    """Run ferret tracking: optimize SKULL only, attach raw spine."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    logger.info("="*80)
    logger.info("FERRET TRACKING: RIGID SKULL + RAW SPINE")
    logger.info("="*80)
    logger.info("Strategy: Optimize skull only, attach raw spine measurements")
    logger.info("="*80)

    # =========================================================================
    # STEP 1: LOAD ALL DATA
    # =========================================================================

    logger.info(f"\nLoading data from: {input_csv.name}")

    trajectory_dict = load_trajectories(
        filepath=input_csv,
        scale_factor=1.0,
        z_value=0.0
    )

    # Define which markers are skull vs spine
    skull_marker_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "base", "left_cam_tip", "right_cam_tip"
    ]

    spine_marker_names = ["spine_t1", "sacrum", "tail_tip"]

    # Extract spine data (keep as raw measurements)
    raw_spine_data = np.stack(
        arrays=[trajectory_dict[name] for name in spine_marker_names],
        axis=1
    )
    timestamps = np.load(timestamps_path) / 1e9

    logger.info(f"\nExtracted data:")
    logger.info(f"  Skull markers: {len(skull_marker_names)}")
    logger.info(f"  Spine markers: {len(spine_marker_names)} (will NOT be optimized)")
    logger.info(f"  Total frames:  {raw_spine_data.shape[0]}")

    # =========================================================================
    # STEP 2: OPTIMIZE SKULL ONLY
    # =========================================================================
    skull_topology = create_skull_only_topology()

    config = TrackingConfig(
        input_csv=input_csv,
        timestamps=timestamps,
        topology=skull_topology,
        output_dir=output_dir,
        optimization=OptimizationConfig(
            max_iter=100,
            lambda_data=100.0,       # Fit to measurements
            lambda_rigid=200.0,      # Skull MUST stay rigid
            lambda_rot_smooth=100.0,
            lambda_trans_smooth=100.0
        ),
        soft_edges=None,             # No soft constraints
        use_parallel=True,
        n_workers=None
    )

    logger.info(f"\nOptimizing skull markers only...")
    logger.info(f"  Rigid edges: {len(skull_topology.rigid_edges)}")
    logger.info(f"  Output: {config.output_dir}")

    # Run optimization on skull only
    result = process_tracking_data(config=config)

    # =========================================================================
    # STEP 3: ATTACH RAW SPINE TO OPTIMIZED SKULL
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("ATTACHING RAW SPINE MARKERS")
    logger.info("="*80)

    combined_data, combined_names = attach_raw_spine_markers(
        optimized_skull=result.reconstructed,
        raw_spine_data=raw_spine_data,
        skull_marker_names=skull_marker_names,
        spine_marker_names=spine_marker_names
    )

    # =========================================================================
    # STEP 4: SAVE COMBINED RESULTS
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("SAVING COMBINED RESULTS")
    logger.info("="*80)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Get noisy data for all markers (skull + spine)
    noisy_all = np.stack(
        arrays=[trajectory_dict[name] for name in combined_names],
        axis=1
    )

    # Create display edges that show skull + spine connections
    display_edges_combined = skull_topology.display_edges.copy()

    # Add spine chain visualization
    skull_base_idx = skull_marker_names.index("base")
    left_ear_idx = skull_marker_names.index("left_ear")
    right_ear_idx = skull_marker_names.index("right_ear")

    # Spine indices in combined array
    spine_t1_idx = len(skull_marker_names) + 0
    sacrum_idx = len(skull_marker_names) + 1
    tail_tip_idx = len(skull_marker_names) + 2

    display_edges_combined.extend([
        (left_ear_idx, spine_t1_idx),
        (right_ear_idx, spine_t1_idx),
        (spine_t1_idx, sacrum_idx),
        (sacrum_idx, tail_tip_idx),
    ])

    # Create combined topology for visualization
    combined_topology = {
        "name": "ferret_skull_plus_raw_spine",
        "marker_names": combined_names,
        "rigid_edges": skull_topology.rigid_edges,  # Only skull edges are rigid
        "display_edges": display_edges_combined,
    }

    # Save results
    save_results(
        output_dir=config.output_dir,
        noisy_data=noisy_all,
        optimized_data=combined_data,
        marker_names=combined_names,
        topology_dict=combined_topology,
        ground_truth_data=None,
        rotations=result.rotations,
        translations=result.translations, 
        timestamps=timestamps
    )

    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info(f"Open {config.output_dir / 'rigid_body_viewer.html'} to visualize")
    logger.info("\nNOTE:")
    logger.info("  - Skull markers (0-7):  OPTIMIZED (rigid body)")
    logger.info("  - Spine markers (8-10): RAW (unoptimized measurements)")
    logger.info("  - The spine will move freely, following raw measurements")


if __name__ == "__main__":
    data_3d_csv = Path("/Users/philipqueen/head_spine_freemocap_data_by_frame.csv")
    timestamps_npy = Path("/Users/philipqueen/session_2025-07-01_ferret_757_EyeCameras_P33EO5/clips/1m_20s-2m_20s/mocap_data/synchronized_videos/24676894_synchronized_corrected_synchronized_timestamps_utc_clipped_7200_12600.npy")
    output_dir = data_3d_csv.parent.parent / "solver_output"
    output_dir.mkdir(exist_ok=True, parents=True)
    run_ferret_skull_solver(input_csv=data_3d_csv, timestamps_path=timestamps_npy, output_dir=output_dir)
