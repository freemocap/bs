"""
Rigid Body Tracking with PyCeres
=========================================================

"""

from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
from scipy.spatial.transform import Rotation

from geometry_module import rotation_matrix_from_axis_angle
from sba_utils import (
    optimize_rigid_body_sba,
    SBAConfig,
    compute_reference_distances
)

logger = logging.getLogger(__name__)



def generate_asymmetric_marker_set(*, size: float = 1.0) -> np.ndarray:
    """
    Generate an ASYMMETRIC marker configuration.

    Uses cube corners PLUS additional markers to break symmetry:
    - One marker extending from a face
    - One marker extending from an edge
    - One marker extending from a corner

    This ensures UNIQUE orientation recovery!
    """
    s = size

    # Original 8 cube vertices
    cube_vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Bottom face
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],      # Top face
    ])

    # Add asymmetric markers
    asymmetric_markers = np.array([
        # Marker 8: Extends from front face center (breaks front/back symmetry)
        [0.0, -s * 1.5, 0.0],

        # Marker 9: Extends from bottom-right edge (breaks left/right symmetry)
        [s * 1.3, -s, -s * 0.7],

        # Marker 10: Extends from top-front-left corner (breaks rotational symmetry)
        [-s * 0.8, -s * 0.8, s * 1.4],
    ])

    all_markers = np.vstack([cube_vertices, asymmetric_markers])

    logger.info(f"📐 Generated asymmetric marker set:")
    logger.info(f"   - 8 cube corners")
    logger.info(f"   - 3 asymmetric markers")
    logger.info(f"   - Total: {len(all_markers)} points")
    logger.info(f"   - This configuration has NO rotational symmetry!")

    return all_markers


def check_symmetry(*, geometry: np.ndarray, tolerance: float = 0.01) -> None:
    """
    Check if geometry has rotational symmetries.
    Helps verify that we broke the symmetry properly.
    """
    logger.info("\n🔍 Checking for symmetries...")

    # Test 90° rotations around each axis
    test_rotations = [
        ("X-axis 90°", Rotation.from_euler('x', 90, degrees=True)),
        ("Y-axis 90°", Rotation.from_euler('y', 90, degrees=True)),
        ("Z-axis 90°", Rotation.from_euler('z', 90, degrees=True)),
        ("Z-axis 180°", Rotation.from_euler('z', 180, degrees=True)),
    ]

    centered = geometry - np.mean(geometry, axis=0)

    symmetries_found = 0
    for name, rotation in test_rotations:
        rotated = rotation.apply(centered)

        # Check if rotated points match original (within tolerance)
        min_dists = []
        for point in rotated:
            dists = np.linalg.norm(centered - point, axis=1)
            min_dists.append(np.min(dists))

        max_deviation = np.max(min_dists)

        if max_deviation < tolerance:
            logger.info(f"   ⚠️  Found symmetry: {name} (deviation: {max_deviation:.4f}m)")
            symmetries_found += 1
        else:
            logger.info(f"   ✅ No symmetry: {name} (deviation: {max_deviation:.4f}m)")

    if symmetries_found == 0:
        logger.info("\n   🎉 SUCCESS! No symmetries detected - geometry is unique!")
    else:
        logger.warning(f"\n   ⚠️  WARNING: Found {symmetries_found} symmetries - may still spin!")


# =============================================================================
# DATA GENERATION WITH ASYMMETRIC MARKERS
# =============================================================================

@dataclass
class DataConfig:
    """Data generation configuration."""
    n_frames: int = 200
    marker_size: float = 1.0
    noise_std: float = 0.1
    random_seed: int | None = 42
    use_asymmetric_markers: bool = True


def generate_synthetic_trajectory(
    *,
    config: DataConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic trajectory with asymmetric markers.

    Returns:
        - reference_geometry: The marker configuration
        - gt_data: Ground truth trajectory
        - noisy_data: Noisy measurements
    """
    logger.info("=" * 80)
    logger.info("📦 DATA GENERATION")
    logger.info("=" * 80)

    reference_geometry = generate_asymmetric_marker_set(size=config.marker_size)
    check_symmetry(geometry=reference_geometry)


    n_points = len(reference_geometry)
    gt_data = np.zeros((config.n_frames, n_points, 3))

    logger.info(f"\n🎬 Generating {config.n_frames} frames...")
    for i in range(config.n_frames):
        t = i / config.n_frames

        # Circular trajectory with vertical oscillation
        radius = 3.0
        translation = np.array([
            radius * np.cos(t * 2 * np.pi),
            radius * np.sin(t * 2 * np.pi),
            1.5 * np.sin(t * 4 * np.pi)
        ])

        # Rotation around diagonal axis
        rot_axis = np.array([0.3, 1.0, 0.2])
        rot_angle = t * 4 * np.pi
        R = rotation_matrix_from_axis_angle(axis=rot_axis, angle=rot_angle)

        gt_data[i] = (R @ reference_geometry.T).T + translation

    # Add noise
    if config.random_seed is not None:
        np.random.seed(seed=config.random_seed)

    logger.info(f"   Adding Gaussian noise (σ={config.noise_std * 1000:.1f}mm)...")
    noise = np.random.normal(loc=0, scale=config.noise_std, size=gt_data.shape)
    noisy_data = gt_data + noise

    logger.info(f"   ✅ Generated {config.n_frames} frames × {n_points} markers")

    return reference_geometry, gt_data, noisy_data



def evaluate_with_spin_detection(
    *,
    reference_geometry: np.ndarray,
    gt_data: np.ndarray,
    noisy_data: np.ndarray,
    optimized_data: np.ndarray
) -> None:
    """
    Comprehensive evaluation including spinning detection.
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 EVALUATION")
    logger.info("=" * 80)

    # Position errors
    noisy_errors = np.linalg.norm(noisy_data - gt_data, axis=2)
    optimized_errors = np.linalg.norm(optimized_data - gt_data, axis=2)

    logger.info("\n📏 Position errors (mm):")
    logger.info(f"   Noisy:     mean={np.mean(noisy_errors) * 1000:.2f}, max={np.max(noisy_errors) * 1000:.2f}")
    logger.info(f"   Optimized: mean={np.mean(optimized_errors) * 1000:.2f}, max={np.max(optimized_errors) * 1000:.2f}")

    improvement = (np.mean(noisy_errors) - np.mean(optimized_errors)) / np.mean(noisy_errors) * 100
    logger.info(f"   Improvement: {improvement:.1f}%")

    # Rigid body constraint preservation
    n_frames = len(optimized_data)
    n_points = len(reference_geometry)

    logger.info("\n🔗 Rigid body constraint check:")
    ref_distances = compute_reference_distances(reference=reference_geometry)

    max_distance_error = 0.0
    mean_distance_error = 0.0
    n_pairs = 0

    for frame_idx in range(n_frames):
        for i in range(n_points):
            for j in range(i + 1, n_points):
                current_dist = np.linalg.norm(
                    optimized_data[frame_idx, i] - optimized_data[frame_idx, j]
                )
                ref_dist = ref_distances[i, j]
                error = abs(current_dist - ref_dist)
                max_distance_error = max(max_distance_error, error)
                mean_distance_error += error
                n_pairs += 1

    mean_distance_error /= n_pairs
    logger.info(f"   Mean distance error: {mean_distance_error * 1000:.3f}mm")
    logger.info(f"   Max distance error:  {max_distance_error * 1000:.3f}mm")

    logger.info("\n🔄 SPINNING DETECTION:")

    def estimate_rotation(source: np.ndarray, target: np.ndarray) -> Rotation:
        """Procrustes rotation estimation."""
        src_centered = source - np.mean(source, axis=0)
        tgt_centered = target - np.mean(target, axis=0)
        H = tgt_centered.T @ src_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        return Rotation.from_matrix(R)

    # Compute frame-to-frame rotations
    gt_rot_magnitudes: list[float] = []
    opt_rot_magnitudes: list[float] = []

    for i in range(1, n_frames):
        # Ground truth rotation
        R_gt_prev = estimate_rotation(reference_geometry, gt_data[i - 1])
        R_gt_curr = estimate_rotation(reference_geometry, gt_data[i])
        R_gt_delta = R_gt_prev.inv() * R_gt_curr
        gt_rot_magnitudes.append(np.linalg.norm(R_gt_delta.as_rotvec()))

        # Optimized rotation
        R_opt_prev = estimate_rotation(reference_geometry, optimized_data[i - 1])
        R_opt_curr = estimate_rotation(reference_geometry, optimized_data[i])
        R_opt_delta = R_opt_prev.inv() * R_opt_curr
        opt_rot_magnitudes.append(np.linalg.norm(R_opt_delta.as_rotvec()))

    gt_mean_rot = np.mean(gt_rot_magnitudes)
    opt_mean_rot = np.mean(opt_rot_magnitudes)
    rot_ratio = opt_mean_rot / gt_mean_rot

    logger.info(f"   Mean frame-to-frame rotation:")
    logger.info(f"      Ground truth: {np.degrees(gt_mean_rot):.3f}°/frame")
    logger.info(f"      Optimized:    {np.degrees(opt_mean_rot):.3f}°/frame")
    logger.info(f"      Ratio:        {rot_ratio:.2f}x")

    if rot_ratio > 1.5:
        logger.error(f"   🚨 SPINNING DETECTED! Optimized is rotating {rot_ratio:.1f}x faster!")
        logger.error(f"   💡 The geometry may still have symmetry issues.")
    elif rot_ratio < 0.6:
        logger.warning(f"   ⚠️  Under-rotation detected (ratio: {rot_ratio:.2f}x)")
    else:
        logger.info(f"   ✅ Rotation magnitude looks good!")

    # Check total cumulative rotation
    R_gt_total = estimate_rotation(reference_geometry, gt_data[0]).inv() * estimate_rotation(reference_geometry, gt_data[-1])
    R_opt_total = estimate_rotation(reference_geometry, optimized_data[0]).inv() * estimate_rotation(reference_geometry, optimized_data[-1])

    total_gt_rot = np.linalg.norm(R_gt_total.as_rotvec())
    total_opt_rot = np.linalg.norm(R_opt_total.as_rotvec())

    logger.info(f"\n   Total cumulative rotation:")
    logger.info(f"      Ground truth: {np.degrees(total_gt_rot):.1f}°")
    logger.info(f"      Optimized:    {np.degrees(total_opt_rot):.1f}°")
    logger.info(f"      Difference:   {np.degrees(abs(total_gt_rot - total_opt_rot)):.1f}°")


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(
    *,
    filepath: Path,
    gt_data: np.ndarray,
    noisy_data: np.ndarray,
    optimized_data: np.ndarray,
    n_cube_corners: int = 8
) -> None:
    """Save results to CSV for visualization."""
    logger.info(f"\n💾 Saving to {filepath}...")

    n_frames, n_points, _ = gt_data.shape
    data: dict[str, np.ndarray | range] = {'frame': range(n_frames)}

    def add_dataset(*, name: str, positions: np.ndarray) -> None:
        for point_idx in range(n_points):
            # Label cube corners as v0-v7, extra markers as m0-m2
            if point_idx < n_cube_corners:
                point_name = f"v{point_idx}"
            else:
                point_name = f"m{point_idx - n_cube_corners}"

            for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
                col_name = f"{name}_{point_name}_{coord_name}"
                data[col_name] = positions[:, point_idx, coord_idx]

        # Add center point
        center = np.mean(positions, axis=1)
        for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
            data[f"{name}_center_{coord_name}"] = center[:, coord_idx]

    add_dataset(name='gt', positions=gt_data)
    add_dataset(name='noisy', positions=noisy_data)
    add_dataset(name='optimized', positions=optimized_data)

    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=filepath, index=False)
    logger.info(f"   ✅ Saved {len(df)} frames")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    data: DataConfig = field(default_factory=lambda: DataConfig(use_asymmetric_markers=True))
    sba: SBAConfig = field(default_factory=lambda: SBAConfig(
        lambda_data=1.0,
        lambda_rigid=10000.0,
        lambda_smooth_pos=1.0,
        lambda_smooth_rot=0.5,
        max_iter=500
    ))
    output_path: Path = Path("trajectory_data.csv")
    log_level: str = "INFO"


def run_pipeline(*, config: PipelineConfig) -> None:
    """Run complete pipeline with asymmetric markers."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(levelname)s | %(message)s'
    )

    logger.info("\n" + "=" * 80)
    logger.info("🚀 RIGID BODY TRACKING WITH ASYMMETRIC MARKERS")
    logger.info("=" * 80)
    logger.info("\n💡 KEY INSIGHT: Symmetric geometry causes spinning!")
    logger.info("   Breaking symmetry with additional markers...")

    # Generate data
    reference_geometry, gt_data, noisy_data = generate_synthetic_trajectory(config=config.data)

    # Optimize using SBA
    logger.info("\n" + "=" * 80)
    logger.info("⚙️  SPARSE BUNDLE ADJUSTMENT")
    logger.info("=" * 80)

    _, _, optimized_data = optimize_rigid_body_sba(
        noisy_data=noisy_data,
        config=config.sba
    )

    # Evaluate
    evaluate_with_spin_detection(
        reference_geometry=reference_geometry,
        gt_data=gt_data,
        noisy_data=noisy_data,
        optimized_data=optimized_data
    )

    # Save
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(
        filepath=config.output_path,
        gt_data=gt_data,
        noisy_data=noisy_data,
        optimized_data=optimized_data,
        n_cube_corners=8
    )

    logger.info("\n" + "=" * 80)
    logger.info("🎉 COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\n✨ Open rigid-body-viewer.html to see results")
    logger.info(f"   If it STILL spins, the problem is deeper than symmetry...")


def main() -> None:
    """Entry point."""

    # Test both symmetric and asymmetric configurations
    logger = logging.getLogger(__name__)

    print("\n\n" + "=" * 80)
    print("=" * 80)
    config = PipelineConfig(
        data=DataConfig(use_asymmetric_markers=True),
        output_path=Path("trajectory_data.csv")
    )
    run_pipeline(config=config)

    print("\n\n" + "=" * 80)
    print("\nGenerated  trajectory_data.csv ")


if __name__ == "__main__":
    main()