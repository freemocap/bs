"""Ferret tracking: RIGID skull + SOFT flexible spine."""

from pathlib import Path
import logging

from python_code.rigid_body_tracker.core.topology import RigidBodyTopology
from python_code.rigid_body_tracker.core.optimization import OptimizationConfig
from python_code.rigid_body_tracker.api import TrackingConfig, process_tracking_data

logger = logging.getLogger(__name__)


def create_ferret_topology() -> tuple[RigidBodyTopology, list[tuple[int, int]]]:
    """
    Create ferret topology: RIGID skull + SOFT spine.

    Returns:
        (topology, soft_edges) tuple
    """

    marker_names = [
        # SKULL (0-7) - RIGID (bone doesn't bend!)
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "base", "left_cam_tip", "right_cam_tip",
        # SPINE (8-10) - SOFT (flexible, can bend and twist!)
        "spine_t1", "sacrum", "tail_tip"
    ]

    # RIGID EDGES: only skull markers (these MUST maintain exact distances)
    rigid_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6), (5, 7),
        (6, 7)
    ]

    # SOFT EDGES: skull-spine connections (these can stretch/compress!)
    soft_edges = [
        # Primary skull-to-spine connections
        (3, 8),
        (4, 8),
        (8, 9),
        (9, 10),
        (5, 9)  # Base of skull to sacrum



    ]

    # DISPLAY EDGES: visualize both rigid skull and flexible spine
    display_edges = [
        (0, 1), (0, 2),
        (1, 2),  # Face triangle
        (1, 3), (2, 4),
        (3, 4),  # Ears
        (3, 5), (4, 5),  # Back of head
        (5, 6), (5, 7), (6, 7),  # Camera mount
        (3, 8), (4, 8), (8, 9), (9, 10),  # Spine chain
    ]

    topology = RigidBodyTopology(
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        display_edges=display_edges,
        name="ferret_rigid_skull_soft_spine"
    )

    return topology, soft_edges


def run_ferret_tracking() -> None:
    """Run ferret tracking with RIGID skull + SOFT spine."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    # Create topology
    topology, soft_edges = create_ferret_topology()

    logger.info("="*80)
    logger.info("FERRET TRACKING: RIGID SKULL + SOFT SPINE")
    logger.info("="*80)
    logger.info(f"Total markers: {len(topology.marker_names)}")
    logger.info(f"  Skull (RIGID): 8 markers - bone doesn't bend!")
    logger.info(f"  Spine (SOFT):  3 markers - flexible, can wiggle!")
    logger.info(f"Rigid edges: {len(topology.rigid_edges)} (exact distance constraints)")
    logger.info(f"Soft edges:  {len(soft_edges)} (flexible distance constraints)")

    # Configure
    config = TrackingConfig(
        input_csv=Path(
            r"D:\bs\ferret_recordings\session_2025-07-01_ferret_757_EyeCameras_P33_EO5"
            r"\clips\1m_20s-2m_20s\mocap_data\output_data\processed_data"
            r"\head_spine_body_rigid_3d_xyz.csv"
        ),
        topology=topology,
        output_dir=Path("output/ferret_rigid_skull_soft_spine"),
        optimization=OptimizationConfig(
            max_iter=100,
            lambda_data=100.0,       # Fit to measurements
            lambda_rigid=200.0,      # Skull MUST stay rigid
            lambda_rot_smooth=100.0,
            lambda_trans_smooth=100.0
        ),
        soft_edges=soft_edges,       # FLEXIBLE spine connections
        lambda_soft=10.0,            # Low weight = more flexible (can wiggle!)
        use_parallel=True,
        n_workers=None  # Use all cores
    )

    logger.info(f"\nλ_soft = {config.lambda_soft} (LOW = spine can bend and twist)")
    logger.info("The spine is WIGGLY! It can move independently from the skull!")

    # Run pipeline
    result = process_tracking_data(config=config)

    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE")
    logger.info("="*80)
    logger.info(f"Processed data successfully")
    logger.info(f"Output: {config.output_dir}")
    logger.info("\nThe skull should be RIGID (maintains shape)")
    logger.info("The spine should be SOFT (can bend and move)")


if __name__ == "__main__":
    run_ferret_tracking()