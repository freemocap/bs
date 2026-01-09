"""Ferret tracking: RIGID skull only + attach raw spine markers.

MODIFICATION: Adds 'head_origin' virtual marker to output.
"""

import logging
from itertools import combinations
from pathlib import Path

import numpy as np

from python_code.pyceres_solvers.rigid_body_solver.core.geometry import save_reference_geometry_json, \
    print_reference_geometry_summary
from python_code.pyceres_solvers.rigid_body_solver.core.optimization import OptimizationConfig
from python_code.pyceres_solvers.rigid_body_solver.core.topology import RigidBodyTopology
from python_code.pyceres_solvers.rigid_body_solver.io.loaders import load_trajectories
from python_code.pyceres_solvers.rigid_body_solver.io.savers import save_results
from python_code.pyceres_solvers.rigid_body_solver.main_interface import RigidBodySolverConfig, process_tracking_data

logger = logging.getLogger(__name__)


def create_skull_only_topology() -> RigidBodyTopology:
    """
    Create topology for SKULL ONLY (no spine in optimization).

    Returns:
        RigidBodyTopology with only skull markers
    """

    marker_names = [
        # SKULL (0-7) - RIGID (bone doesn't bend!)
        "nose",  # 0
        "left_eye",  # 1
        "right_eye",  # 2
        "left_ear",  # 3
        "right_ear",  # 4
        "base",  # 5
        "left_cam_tip",  # 6
        "right_cam_tip",  # 7
    ]

    # RIGID EDGES: all skull markers (maintain exact distances)
    rigid_edges =list(combinations(marker_names, 2))

    # DISPLAY EDGES: visualize skull structure
    display_edges = [
        ('nose', 'left_eye'),
        ('nose', 'right_eye'),
        ('left_eye', 'right_eye'),
        ('left_eye', 'left_ear'),
        ('right_eye', 'right_ear'),
        ('left_ear', 'right_ear'),
        ('left_ear', 'base'),
        ('right_ear', 'base'),
        ('base', 'left_cam_tip'),
        ('base', 'right_cam_tip'),
    ]

    topology = RigidBodyTopology(
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        display_edges=display_edges,
        name="ferret_skull"
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
    Attach raw (unoptimized) spine markers to optimized skull data

    Args
        optimized_skull: (n_frames, n_skull_markers+1, 3) optimized skull
        raw_spine_data: (n_frames, n_spine_markers, 3) raw spine measurements
        skull_marker_names: Names of skull markers
        spine_marker_names: Names of spine markers

    Returns:
        Tuple of:
        - combined_data: (n_frames, n_total_markers, 3)
        - combined_names: List of all marker names
    """
    n_frames = optimized_skull.shape[0]
    n_skull_markers = optimized_skull.shape[1]
    n_spine = raw_spine_data.shape[1]

    # Concatenate skull+origin + spine
    combined_data = np.concatenate([optimized_skull, raw_spine_data], axis=1)
    combined_names = skull_marker_names + spine_marker_names

    logger.info(f"\nAttached raw spine markers:")
    logger.info(f"  Skull (optimized): {n_skull_markers - 1} markers")
    logger.info(f"  Spine (raw):       {n_spine} markers")
    logger.info(f"  Total:             {n_skull_markers + n_spine} markers")

    return combined_data, combined_names


def run_ferret_skull_solver(
        *,
        input_csv: Path,
        timestamps_path: Path,
        output_dir: Path
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    logger.info("=" * 80)
    logger.info("FERRET TRACKING: RIGID SKULL +  RAW SPINE")
    logger.info("=" * 80)
    logger.info("Strategy: Optimize skull, attach raw spine")
    logger.info("=" * 80)

    # =========================================================================
    # STEP 1: LOAD ALL DATA
    # =========================================================================

    logger.info(f"\nLoading data from: {input_csv.name}")

    trajectory_dict = load_trajectories(
        filepath=input_csv,
        scale_factor=1.0,
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

    config = RigidBodySolverConfig(
        input_csv=input_csv,
        timestamps=timestamps,
        topology=skull_topology,
        output_dir=output_dir,
        rigid_body_name="skull",
        optimization=OptimizationConfig(
            max_iter=100,
            lambda_data=100.0,  # Fit to measurements
            lambda_rigid=200.0,  # Skull MUST stay rigid
            lambda_rot_smooth=100.0, # Smooth rotations over time
            lambda_trans_smooth=100.0 # Smooth translations over time
        ),
        # Define ferret head coordinate system
        body_frame_origin_markers=["left_eye", "right_eye"],  # Head center
        body_frame_x_axis_marker="nose",  # X-axis points forward (towards nose)
        body_frame_y_axis_marker="left_eye",  # Y-axis points left
    )

    logger.info(f"\nOptimizing skull ...")
    logger.info(f"  Rigid edges: {len(skull_topology.rigid_edges)}")
    logger.info(f"  Output: {config.output_dir}")

    # Run optimization on skull only
    # NOTE: The returned result will have skull markers
    result = process_tracking_data(config=config)

    # =========================================================================
    # STEP 3: EXTRACT SKULL + HEAD_ORIGIN DATA
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("EXTRACTING OPTIMIZED SKULL + HEAD_ORIGIN")
    logger.info("=" * 80)

    # Get the optimized skull data with head_origin (last marker)
    # The process_tracking_data function already added head_origin
    optimized_skull = result.reconstructed_keypoints

    logger.info(f"  Skull markers:  {len(skull_marker_names)}")

    # =========================================================================
    # STEP 4: ATTACH RAW SPINE TO OPTIMIZED SKULL+ORIGIN
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("ATTACHING RAW SPINE MARKERS")
    logger.info("=" * 80)

    combined_data, combined_names = attach_raw_spine_markers(
        optimized_skull=optimized_skull,
        raw_spine_data=raw_spine_data,
        skull_marker_names=skull_marker_names,
        spine_marker_names=spine_marker_names
    )

    # =========================================================================
    # STEP 5: SAVE COMBINED RESULTS
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("SAVING COMBINED RESULTS")
    logger.info("=" * 80)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Get original data for all markers (skull + head_origin (NaN) + spine)
    original_skull = np.stack(
        arrays=[trajectory_dict[name] for name in skull_marker_names],
        axis=1
    )
    original_origin = np.full((original_skull.shape[0], 1, 3), np.nan)  # Virtual marker has no measurement
    original_all = np.concatenate([original_skull, original_origin, raw_spine_data], axis=1)


    # Spine indices in combined array (after skull)
    spine_t1_name = "spine_t1"
    sacrum_name =  "sacrum"
    tail_tip_name =  "tail_tip"



    skull_topology.display_edges.extend([
        ("left_ear", "spine_t1"),
        ("right_ear", "spine_t1"),
        ("spine_t1", "sacrum"),
        ("sacrum", "tail_tip"),
    ])

    # Create combined topology for visualization
    combined_topology = {
        "name": "ferret_skull_plus_origin_plus_raw_spine",
        "marker_names": combined_names,
        "rigid_edges": skull_topology.rigid_edges,  # Only skull edges are rigid
        "display_edges": skull_topology.display_edges,
    }

    # Save results
    save_results(
        output_dir=config.output_dir,
        original_data=original_all,
        rigid_body_name="skull_and_spine",
        optimized_data=combined_data,
        marker_names=combined_names,
        topology_dict=combined_topology,
        quaternions=result.quaternions,
        rotations=result.rotations,
        translations=result.translations,
        timestamps=timestamps
    )



    # Print skull reference geometry summary
    print_reference_geometry_summary(
        reference_geometry=result.reference_geometry,
        marker_names=skull_marker_names,
        units="mm"
    )

    logger.info("\n" + "=" * 80)
    logger.info("✓ COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info(f"Open {config.output_dir / 'rigid_body_viewer.html'} to visualize")
    logger.info("\nMARKER SUMMARY:")
    logger.info(f"  - Skull markers (0-{len(skull_marker_names) - 1}):  OPTIMIZED (rigid body)")
    logger.info(
        f"  - Spine markers ({len(skull_marker_names) + 1}-{len(combined_names) - 1}): RAW (unoptimized measurements)")
    logger.info("\nNOTE:")
    logger.info("  - The spine will move freely, following raw measurements")


if __name__ == "__main__":
    data_3d_csv = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\dlc\head_freemocap_data_by_frame.csv")
    timestamps_npy = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\synchronized_videos\24676894_synchronized_corrected_synchronized_timestamps_utc_clipped_3377_8754.npy")
    output_dir = data_3d_csv.parent.parent / "solver_output"
    output_dir.mkdir(exist_ok=True, parents=True)
    run_ferret_skull_solver(
        input_csv=data_3d_csv,
        timestamps_path=timestamps_npy,
        output_dir=output_dir
    )
