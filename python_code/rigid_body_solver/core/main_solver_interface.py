"""Main interface for rigid body tracking pipeline using Pydantic v2."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.rigid_body_solver.core.optimization import OptimizationConfig, OptimizationResult, optimize_rigid_body
from python_code.rigid_body_solver.core.topology import StickFigureTopology
from python_code.rigid_body_solver.data_io.load_measured_trajectories import load_measured_trajectories_csv

logger = logging.getLogger(__name__)


class RigidBodySolverConfig(BaseModel):
    """Complete configuration for rigid body tracking."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    input_csv: Path
    """Path to input CSV file"""

    timestamps: NDArray[np.float64]
    """Timestamps for each frame"""

    topology: StickFigureTopology
    """Rigid body topology"""

    output_dir: Path
    """Output directory"""

    optimization: OptimizationConfig
    """Optimization configuration"""

    rigid_body_name: str
    """Name of the rigid body"""

    body_frame_origin_markers: list[str]
    """Marker names whose mean defines the origin"""

    body_frame_x_axis_marker: str
    """Marker name that defines X-axis direction"""

    body_frame_y_axis_marker: str
    """Marker name that defines Y-axis direction"""

    units: Literal["mm", "m"] = "mm"
    """Units for coordinates"""

    @field_validator("timestamps", mode="before")
    @classmethod
    def convert_timestamps_to_numpy(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        return np.asarray(v, dtype=np.float64)

    @field_validator("body_frame_origin_markers")
    @classmethod
    def origin_markers_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("body_frame_origin_markers cannot be empty")
        return v


class ProcessingResult(BaseModel):
    """Result from processing tracking data."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    kinematics: RigidBodyKinematics
    """The computed kinematics"""

    optimization_result: OptimizationResult
    """Raw optimization result"""

    original_trajectories: NDArray[np.float64]
    """Original trajectory data (n_frames, n_markers, 3)"""

    @field_validator("original_trajectories", mode="before")
    @classmethod
    def convert_to_numpy(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        return np.asarray(v, dtype=np.float64)


def optimization_result_to_kinematics(
        result: OptimizationResult,
        timestamps: NDArray[np.float64],
) -> RigidBodyKinematics:
    """
    Convert an OptimizationResult to a RigidBodyKinematics object.

    Args:
        result: The optimization result
        timestamps: Timestamps for each frame

    Returns:
        RigidBodyKinematics with computed velocities and angular velocities
    """
    return RigidBodyKinematics.from_pose_arrays(
        reference_geometry=result.reference_geometry,
        timestamps=timestamps,
        position_xyz=result.translations,
        quaternions_wxyz=result.quaternions_wxyz,
    )


def verify_trajectory_reconstruction(
        *,
        reference_geometry: ReferenceGeometry,
        kinematics: RigidBodyKinematics,
        reconstructed: NDArray[np.float64],
        n_frames_to_check: int = 5,
        tolerance: float = 1e-6,
) -> bool:
    """
    Verify that reconstructed trajectories match: world = R @ reference + t

    Args:
        reference_geometry: Reference geometry model
        kinematics: RigidBodyKinematics object
        reconstructed: (n_frames, n_markers, 3) reconstructed positions
        n_frames_to_check: Number of frames to verify
        tolerance: Maximum allowed error

    Returns:
        True if verification passes, False otherwise
    """
    marker_names, ref_positions = reference_geometry.get_marker_array()
    n_frames = min(n_frames_to_check, kinematics.n_frames)
    n_markers = len(marker_names)

    logger.info("=" * 80)
    logger.info("VERIFYING TRAJECTORY RECONSTRUCTION")
    logger.info("=" * 80)
    logger.info("Formula: world = R @ reference + t")
    logger.info(f"Checking {n_frames} frames, {n_markers} markers")
    logger.info(f"Tolerance: {tolerance * 1000:.4f} mm")

    max_error = 0.0
    max_error_frame = -1
    max_error_marker = ""

    for frame_idx in range(n_frames):
        pose = kinematics[frame_idx]
        R = pose.basis_vectors
        t = pose.position

        expected = (R @ ref_positions.T).T + t
        actual = reconstructed[frame_idx]

        errors = np.linalg.norm(expected - actual, axis=1)

        frame_max_error = errors.max()
        if frame_max_error > max_error:
            max_error = frame_max_error
            max_error_frame = frame_idx
            max_error_marker = marker_names[errors.argmax()]

        if frame_idx < 3:
            logger.info(f"\nFrame {frame_idx}:")
            logger.info(
                f"  Max error: {frame_max_error * 1000:.6f} mm (marker: {marker_names[errors.argmax()]})"
            )
            logger.info(f"  Mean error: {errors.mean() * 1000:.6f} mm")

    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Frames checked: {n_frames}")
    logger.info(f"Max error: {max_error * 1000:.6f} mm")
    logger.info(f"  Frame: {max_error_frame}")
    logger.info(f"  Marker: {max_error_marker}")

    if max_error < tolerance:
        logger.info(f"✓ PASS: All errors < {tolerance * 1000:.4f} mm")
        return True
    else:
        logger.warning(
            f"✗ FAIL: Max error {max_error * 1000:.6f} mm exceeds tolerance {tolerance * 1000:.4f} mm"
        )
        return False


def print_reference_geometry_summary(
        reference_geometry: ReferenceGeometry,
) -> None:
    """Print a summary of the reference geometry."""
    logger.info("=" * 80)
    logger.info("REFERENCE GEOMETRY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Units: {reference_geometry.units}")
    logger.info("Coordinate system: Body-fixed frame")
    logger.info("\nMarker positions:")
    logger.info(f"{'Name':<20} {'X':>10} {'Y':>10} {'Z':>10}")
    logger.info("-" * 80)

    marker_positions = reference_geometry.get_marker_positions()
    for name, pos in marker_positions.items():
        x, y, z = pos
        logger.info(f"{name:<20} {x:>10.3f} {y:>10.3f} {z:>10.3f}")

    positions = np.array(list(marker_positions.values()))
    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)
    size = max_vals - min_vals

    logger.info("\nBounding box:")
    logger.info(
        f"  X: [{min_vals[0]:.3f}, {max_vals[0]:.3f}] {reference_geometry.units} (size: {size[0]:.3f})"
    )
    logger.info(
        f"  Y: [{min_vals[1]:.3f}, {max_vals[1]:.3f}] {reference_geometry.units} (size: {size[1]:.3f})"
    )
    logger.info(
        f"  Z: [{min_vals[2]:.3f}, {max_vals[2]:.3f}] {reference_geometry.units} (size: {size[2]:.3f})"
    )


def process_tracking_data(config: RigidBodySolverConfig) -> ProcessingResult:
    """
    Complete rigid body tracking pipeline with soft constraints.

    Pipeline:
    1. Load data
    2. Extract markers
    3. Estimate rigid body reference geometry
    4. Optimize
    5. Convert to RigidBodyKinematics
    6. Verify and save

    Args:
        config: RigidBodySolverConfig

    Returns:
        ProcessingResult containing kinematics and optimization result
    """
    logger.info("=" * 80)
    logger.info("RIGID BODY TRACKING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input:    {config.input_csv.name}")
    logger.info(f"Output:   {config.output_dir}")
    logger.info(f"Topology: {config.topology.name}")
    logger.info(f"Markers:  {len(config.topology.marker_names)}")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 1: LOAD DATA")
    logger.info("=" * 80)

    trajectory_dict = load_measured_trajectories_csv(
        filepath=config.input_csv,
        scale_factor=1.0,
    )

    # =========================================================================
    # STEP 2: EXTRACT MARKERS
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 2: EXTRACT MARKERS")
    logger.info("=" * 80)

    original_trajectories = config.topology.extract_trajectories(
        trajectory_dict=trajectory_dict
    )
    n_frames = original_trajectories.shape[0]
    logger.info(f"  Data shape: {original_trajectories.shape}")

    # =========================================================================
    # STEP 3: OPTIMIZE
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 3: OPTIMIZE")
    logger.info("=" * 80)

    optimization_result = optimize_rigid_body(
        original_trajectories=original_trajectories,
        rigid_edges=config.topology.rigid_edges,
        config=config.optimization,
        marker_names=config.topology.marker_names,
        display_edges=config.topology.display_edges,
        body_frame_origin_markers=config.body_frame_origin_markers,
        body_frame_x_axis_marker=config.body_frame_x_axis_marker,
        body_frame_y_axis_marker=config.body_frame_y_axis_marker,
        units=config.units,
    )

    # =========================================================================
    # STEP 4: CONVERT TO KINEMATICS
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 4: CONVERT TO KINEMATICS")
    logger.info("=" * 80)

    kinematics = RigidBodyKinematics.from_pose_arrays(
        name=config.rigid_body_name,
        reference_geometry=optimization_result.reference_geometry,
        timestamps=config.timestamps,
        position_xyz=optimization_result.translations,
        quaternions_wxyz=optimization_result.quaternions_wxyz,
    )

    logger.info(f"  Created RigidBodyKinematics with {kinematics.n_frames} frames")
    logger.info(f"  Duration: {kinematics.duration:.2f} seconds")

    # =========================================================================
    # STEP 5: VERIFY RECONSTRUCTION
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 5: VERIFY RECONSTRUCTION")
    logger.info("=" * 80)

    verification_passed = verify_trajectory_reconstruction(
        reference_geometry=optimization_result.reference_geometry,
        kinematics=kinematics,
        reconstructed=optimization_result.keypoint_trajectories,
        n_frames_to_check=min(10, n_frames),
        tolerance=1e-6,
    )

    if not verification_passed:
        logger.warning("⚠ Reconstruction verification failed!")

    print_reference_geometry_summary(optimization_result.reference_geometry)

    # =========================================================================
    # STEP 6: SAVE
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 6: SAVE RESULTS")
    logger.info("=" * 80)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    RigidBodyKinematics.from_pose_arrays(
        name=config.rigid_body_name,
        reference_geometry=optimization_result.reference_geometry,
        timestamps=config.timestamps,
        position_xyz=optimization_result.translations,
        quaternions_wxyz=optimization_result.quaternions_wxyz,
    ).save_to_disk(
        output_directory=config.output_dir,
    )

    # Save reference geometry as JSON
    ref_geom_path = config.output_dir / f"{config.rigid_body_name}_reference_geometry.json"
    optimization_result.reference_geometry.to_json_file(ref_geom_path)
    logger.info(f"Saved reference geometry to: {ref_geom_path}")

    logger.info(f"\n✓ Complete! Results saved to: {config.output_dir}")

    return ProcessingResult(
        kinematics=kinematics,
        optimization_result=optimization_result,
        original_trajectories=original_trajectories,
    )
