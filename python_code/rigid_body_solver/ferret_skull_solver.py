"""Ferret tracking: RIGID skull only + attach raw spine keypoints.

Updated to use Pydantic v2 models and the new kinematics_core.
"""

import logging
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.rigid_body_solver.core.main_solver_interface import RigidBodySolverConfig, process_tracking_data, \
    ProcessingResult, print_reference_geometry_summary
from python_code.rigid_body_solver.core.optimization import OptimizationConfig
from python_code.kinematics_core.stick_figure_topology_model import StickFigureTopology
from python_code.rigid_body_solver.data_io.load_measured_trajectories import  load_measured_trajectories_csv
from python_code.rigid_body_solver.viz.ferret_skull_rerun import run_ferret_skull_and_spine_visualization

logger = logging.getLogger(__name__)


def create_skull_topology() -> StickFigureTopology:
    """
    Create topology for SKULL ONLY (no spine in optimization).

    Returns:
        RigidBodyTopology with only skull keypoints
    """
    keypoint_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "base",
        "left_cam_tip",
        "right_cam_tip",
    ]

    # RIGID EDGES: all skull keypoints (maintain exact distances)
    rigid_edges = list(combinations(keypoint_names, 2))

    # DISPLAY EDGES: visualize skull structure
    display_edges = [
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
        ("left_ear", "right_ear"),
        ("left_ear", "base"),
        ("right_ear", "base"),
        ("base", "left_cam_tip"),
        ("base", "right_cam_tip"),
    ]

    topology = StickFigureTopology(
        keypoint_names=keypoint_names,
        rigid_edges=rigid_edges,
        display_edges=display_edges,
        name="ferret_skull",
    )

    return topology

def create_skull_and_spine_topology(
    skull_topology: StickFigureTopology,
    spine_keypoint_names: list[str],
    spine_display_edges: list[tuple[str, str]],
) -> StickFigureTopology:
    skull_and_spine_topology_dict = skull_topology.model_dump()
    skull_and_spine_topology_dict['name'] = "ferret_skull_and_spine"
    skull_and_spine_topology_dict['keypoint_names'] = skull_and_spine_topology_dict['keypoint_names'] + spine_keypoint_names
    skull_and_spine_topology_dict['display_edges'] = spine_display_edges
    return StickFigureTopology(**skull_and_spine_topology_dict)

def attach_raw_spine_keypoints(
    *,
    optimized_skull: NDArray[np.float64],
    raw_spine_data: NDArray[np.float64],
    skull_keypoint_names: list[str],
    spine_keypoint_names: list[str],
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Attach raw (unoptimized) spine keypoints to optimized skull data.

    Args:
        optimized_skull: (n_frames, n_skull_keypoints, 3) optimized skull
        raw_spine_data: (n_frames, n_spine_keypoints, 3) raw spine measurements
        skull_keypoint_names: Names of skull keypoints
        spine_keypoint_names: Names of spine keypoints

    Returns:
        Tuple of:
        - combined_data: (n_frames, n_total_keypoints, 3)
        - combined_names: List of all keypoint names
    """
    n_skull_keypoints = optimized_skull.shape[1]
    n_spine = raw_spine_data.shape[1]

    combined_data = np.concatenate([optimized_skull, raw_spine_data], axis=1)
    combined_names = skull_keypoint_names + spine_keypoint_names

    logger.info("\nAttached raw spine keypoints:")
    logger.info(f"  Skull (optimized): {n_skull_keypoints} keypoints")
    logger.info(f"  Spine (raw):       {n_spine} keypoints")
    logger.info(f"  Total:             {n_skull_keypoints + n_spine} keypoints")

    return combined_data, combined_names

def save_skull_and_spine_to_tidy_csv(
    trajectories: NDArray[np.float64],
    keypoint_names: list[str],
    timestamps: NDArray[np.float64],
    output_path: Path,
) -> None:
    """
    Save combined skull and spine trajectories to a tidy CSV file.

    Args:
        trajectories: (n_frames, n_keypoints, 3) array of keypoint trajectories
        keypoint_names: List of keypoint names corresponding to the second dimension of trajectories
        timestamps: (n_frames,) array of timestamps
        output_path: Path to save the CSV file
    """
    n_frames, n_keypoints, _ = trajectories.shape


    data_rows = []
    for frame_idx in range(n_frames):
        for keypoint_idx in range(n_keypoints):
            for component, comp_name in enumerate(["x", "y", "z"]):
                row = {
                    "frame": frame_idx,
                    "timestamp": timestamps[frame_idx],
                    "trajectory": keypoint_names[keypoint_idx],
                    "component": comp_name,
                    "value": trajectories[frame_idx, keypoint_idx, component],
                    "units": "mm",
                }
                data_rows.append(row)

    df = pl.DataFrame(data_rows)
    df.write_csv(output_path)
    logger.info(f"Saved combined skull and spine trajectories to: {output_path}")

def run_ferret_skull_solver(
    *,
    input_csv: Path,
    timestamps_path: Path,
    output_dir: Path,
) -> RigidBodyKinematics:
    """
    Run the ferret skull solver pipeline.

    Args:
        input_csv: Path to input CSV file with 3D trajectories
        timestamps_path: Path to timestamps numpy file
        output_dir: Output directory for results

    Returns:
        RigidBodyKinematics object for the skull
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    logger.info("=" * 80)
    logger.info("FERRET TRACKING: RIGID SKULL + RAW SPINE")
    logger.info("=" * 80)
    logger.info("Strategy: Optimize skull, attach raw spine")
    logger.info("=" * 80)

    # =========================================================================
    # LOAD ALL DATA
    # =========================================================================
    logger.info(f"\nLoading data from: {input_csv.name}")

    trajectory_dict = load_measured_trajectories_csv(
        filepath=input_csv,
        scale_factor=1.0,
    )

    skull_keypoint_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "base",
        "left_cam_tip",
        "right_cam_tip",
    ]

    spine_keypoint_names = ["spine_t1", "sacrum", "tail_tip"]

    original_trajectory_data = np.stack(
        arrays=[trajectory_dict[name] for name in trajectory_dict.keys()],
        axis=1,
    )
    all_keypoint_names = list(trajectory_dict.keys())
    timestamps = np.load(timestamps_path) / 1e9

    logger.info("\nExtracted data:")
    logger.info(f"  Skull keypoints: {len(skull_keypoint_names)}")
    logger.info(f"  Spine keypoints: {len(spine_keypoint_names)} (will NOT be optimized)")
    logger.info(f"  Total frames:  {original_trajectory_data.shape[0]}")

    # =========================================================================
    # OPTIMIZE SKULL ONLY
    # =========================================================================
    skull_topology = create_skull_topology()

    config = RigidBodySolverConfig(
        input_csv=input_csv,
        timestamps=timestamps,
        topology=skull_topology,
        output_dir=output_dir,
        rigid_body_name="skull",
        optimization=OptimizationConfig(
            max_iter=100,
            lambda_data=100.0, # How much should we weight the measured data?
            lambda_trans_smooth=300.0, # How much should we weight translational smoothness?
            lambda_rot_smooth=500.0,  # How much should we weight rotational smoothness?

        ),
        body_frame_origin_keypoints=["left_eye", "right_eye"],
        body_frame_x_axis_keypoint="nose",
        body_frame_y_axis_keypoint="left_eye",
    )

    logger.info("\nOptimizing skull...")
    logger.info(f"  Rigid edges: {len(skull_topology.rigid_edges)}")
    logger.info(f"  Output: {config.output_dir}")

    result: ProcessingResult = process_tracking_data(config=config)

    # =========================================================================
    # EXTRACT SKULL DATA
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("EXTRACTING OPTIMIZED SKULL")
    logger.info("=" * 80)

    optimized_skull = result.kinematics.keypoint_trajectories.trajectories_fr_id_xyz

    logger.info(f"  Skull keypoints: {len(skull_keypoint_names)}")


    # =========================================================================
    # SAVE COMBINED RESULTS
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("SAVING COMBINED RESULTS")
    logger.info("=" * 80)

    config.output_dir.mkdir(parents=True, exist_ok=True)


    skull_and_spine_display_edges = deepcopy(skull_topology.display_edges)
    skull_and_spine_display_edges.extend([
        ("left_ear", "spine_t1"),
        ("right_ear", "spine_t1"),
        ("spine_t1", "sacrum"),
        ("sacrum", "tail_tip"),
    ])

    # Create combined topology for visualization
    skull_and_spine_topology = create_skull_and_spine_topology(
        skull_topology=skull_topology,
        spine_keypoint_names=spine_keypoint_names,
        spine_display_edges=skull_and_spine_display_edges,
    )

    save_skull_and_spine_trajectories_csv_path = config.output_dir / "skull_and_spine_trajectories.csv"
    save_skull_and_spine_to_tidy_csv(
        trajectories=original_trajectory_data,
        keypoint_names=all_keypoint_names,
        timestamps=timestamps,
        output_path=save_skull_and_spine_trajectories_csv_path,
    )
    skull_and_spine_topology_path = config.output_dir / "skull_and_spine_topology.json"
    skull_and_spine_topology.save_json(filepath=skull_and_spine_topology_path)
    logger.info(f"Saved combined topology to: {skull_and_spine_topology_path}")

    # Print skull reference geometry summary
    print_reference_geometry_summary(
        reference_geometry=result.optimization_result.reference_geometry,
    )

    logger.info("\n" + "=" * 80)
    logger.info("✓ COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("\nMARKER SUMMARY:")
    logger.info(f"  - Skull keypoints (0-{len(skull_keypoint_names) - 1}): OPTIMIZED (rigid body)")
    logger.info(
        f"  - Spine keypoints ({len(skull_keypoint_names)}-{len(all_keypoint_names) - 1}): RAW (unoptimized)"
    )
    logger.info("\nNOTE:")
    logger.info("  - The spine will move freely, following raw measurements")

    return result.kinematics


if __name__ == "__main__":
    data_3d_csv = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\dlc\head_freemocap_data_by_frame.csv"
    )
    timestamps_npy = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\synchronized_videos\24676894_synchronized_corrected_synchronized_timestamps_utc_clipped_3377_8754.npy"
    )
    _output_dir = data_3d_csv.parent.parent / "solver_output"
    _output_dir.mkdir(exist_ok=True, parents=True)
    run_ferret_skull_solver(
        input_csv=data_3d_csv,
        timestamps_path=timestamps_npy,
        output_dir=_output_dir,
    )
    run_ferret_skull_and_spine_visualization(
        output_dir=_output_dir,
        spawn=True,
        time_window_seconds=5.0,
    )
