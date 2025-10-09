"""
Main Execution Script for Rigid Body Trajectory Optimization
=============================================================

Complete pipeline with ANTI-SPIN FIXES:
- Unwrapped Kabsch initialization
- True geodesic distance regularization
- Explicit large rotation penalties

Author: AI Assistant
Date: 2025
"""

import numpy as np
import jax
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Optional

from geometry_module import (
    generate_cube_vertices,
    rotation_matrix_from_axis_angle
)
from trajectory_module import (
    kabsch_initialization,
    optimize_trajectory_jax
)
from analysis_module import print_comparison_report

# Configure JAX for high precision
jax.config.update("jax_enable_x64", True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(funcName)s() | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_ground_truth_trajectory(
    *,
    n_frames: int,
    cube_size: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic ground truth rigid body trajectory.

    The cube follows a circular path with oscillating height while rotating.

    Args:
        n_frames: Number of frames to generate
        cube_size: Half-width of the cube

    Returns:
        Tuple of (rotations, translations, marker_positions)
        - rotations: (n_frames, 3, 3)
        - translations: (n_frames, 3)
        - marker_positions: (n_frames, 9, 3) - 8 vertices + 1 center
    """
    logger.info(f"Generating ground truth trajectory ({n_frames} frames)...")

    base_vertices = generate_cube_vertices(size=cube_size)
    n_markers = len(base_vertices) + 1  # 8 vertices + center

    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))
    marker_positions = np.zeros((n_frames, n_markers, 3))

    for i in range(n_frames):
        t = i / n_frames  # Normalized time [0, 1]

        # Circular path in XY plane + oscillating Z
        radius = 3.0
        translation = np.array([
            radius * np.cos(t * 2 * np.pi),
            radius * np.sin(t * 2 * np.pi),
            1.5 * np.sin(t * 4 * np.pi)
        ])

        # Smooth rotation around a tilted axis
        rot_axis = np.array([0.3, 1.0, 0.2])
        rot_angle = t * 4 * np.pi  # 2 full rotations
        R = rotation_matrix_from_axis_angle(axis=rot_axis, angle=rot_angle)

        # Transform vertices
        transformed_vertices = (R @ base_vertices.T).T + translation

        # Store results
        rotations[i] = R
        translations[i] = translation
        marker_positions[i, :8, :] = transformed_vertices
        marker_positions[i, 8, :] = np.mean(transformed_vertices, axis=0)  # Center

    return rotations, translations, marker_positions


def add_noise_to_measurements(
    *,
    marker_positions: np.ndarray,
    noise_std: float = 0.3,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Add Gaussian noise to marker positions to simulate measurement error.

    Args:
        marker_positions: Clean marker positions (n_frames, n_markers, 3)
        noise_std: Standard deviation of Gaussian noise (in same units)
        seed: Random seed for reproducibility

    Returns:
        Noisy marker positions of same shape
    """
    logger.info(f"Adding noise (σ={noise_std * 1000:.1f} mm)...")

    if seed is not None:
        np.random.seed(seed=seed)

    n_frames, n_markers, _ = marker_positions.shape
    noisy_positions = marker_positions.copy()

    # Add noise only to the 8 vertices (not the center)
    noise = np.random.normal(loc=0, scale=noise_std, size=(n_frames, 8, 3))
    noisy_positions[:, :8, :] += noise

    # Recompute center as mean of noisy vertices
    noisy_positions[:, 8, :] = np.mean(noisy_positions[:, :8, :], axis=1)

    return noisy_positions


# =============================================================================
# FILE I/O
# =============================================================================

def save_trajectory_csv(
    *,
    filepath: str,
    gt_positions: np.ndarray,
    noisy_positions: np.ndarray,
    kabsch_positions: np.ndarray,
    opt_no_filter_positions: np.ndarray,
    opt_positions: np.ndarray
) -> None:
    """
    Save all trajectories to CSV.

    Args:
        filepath: Output CSV filepath
        gt_positions: Ground truth positions (n_frames, n_markers, 3)
        noisy_positions: Noisy measurements (n_frames, n_markers, 3)
        kabsch_positions: Kabsch results (n_frames, n_markers, 3)
        opt_no_filter_positions: Optimization before smoothing (n_frames, n_markers, 3)
        opt_positions: Final optimized positions (n_frames, n_markers, 3)
    """
    logger.info(f"Saving trajectory to {filepath}...")

    n_frames, n_markers, _ = gt_positions.shape
    marker_names = [f"v{i}" for i in range(8)] + ["center"]

    data: dict[str, np.ndarray | range] = {'frame': range(n_frames)}

    for dataset_name, positions in [
        ('gt', gt_positions),
        ('noisy', noisy_positions),
        ('kabsch', kabsch_positions),
        ('opt_no_filter', opt_no_filter_positions),
        ('opt', opt_positions)
    ]:
        for marker_idx, marker_name in enumerate(marker_names):
            for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
                col_name = f"{dataset_name}_{marker_name}_{coord_name}"
                data[col_name] = positions[:, marker_idx, coord_idx]

    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=filepath, index=False)
    logger.info(f"  Saved {df.shape[0]} frames × {df.shape[1]} columns")


def save_stats_json(*, filepath: str, stats: dict[str, object]) -> None:
    """
    Save statistics to JSON.

    Args:
        filepath: Output JSON filepath
        stats: Statistics dictionary
    """
    logger.info(f"Saving statistics to {filepath}...")
    with open(filepath, 'w') as f:
        json.dump(obj=stats, fp=f, indent=2)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Run complete rigid body trajectory optimization pipeline."""

    print("\n" + "=" * 80)
    print("RIGID BODY TRAJECTORY OPTIMIZATION - ANTI-SPIN FIXED 🔧")
    print("=" * 80)

    # Check JAX devices
    logger.info("\n🔧 JAX Configuration")
    devices = jax.devices()
    logger.info(f"  Devices:  {devices}")
    logger.info(f"  Platform: {devices[0].platform}")

    if devices[0].platform == 'gpu':
        logger.info("  ✅ GPU acceleration enabled")
    else:
        logger.info("  ℹ️  Using CPU (still fast with JIT!)")

    # Configuration
    n_frames = 200
    cube_size = 0.5
    noise_std = 0.3

    # 🔧 UPDATED OPTIMIZATION PARAMETERS
    lambda_data = 1.0
    lambda_smooth_pos = 30.0
    lambda_smooth_rot = 15.0
    lambda_accel = 3.0
    lambda_rot_geodesic = 200.0  # INCREASED from 150
    lambda_max_rotation = 1000.0  # NEW: Spin prevention
    max_rotation_per_frame_deg = 25.0  # NEW: Max rotation constraint
    apply_slerp_smoothing = True
    slerp_window = 5
    max_iter = 400

    logger.info("\n🎯 Key Features (ENHANCED):")
    logger.info("  ✓ Advanced rotation unwrapping (3-pass)")
    logger.info("  ✓ Unwrapped Kabsch initialization 🔧 NEW!")
    logger.info("  ✓ TRUE geodesic distance regularization 🔧 NEW!")
    logger.info("  ✓ Explicit large rotation penalty 🔧 NEW!")
    logger.info("  ✓ SLERP-smoothed initialization")
    logger.info("  ✓ Physical plausibility metrics")

    # Generate data
    logger.info("\n" + "=" * 80)
    logger.info("DATA GENERATION")
    logger.info("=" * 80)

    gt_rotations, gt_translations, gt_positions = generate_ground_truth_trajectory(
        n_frames=n_frames,
        cube_size=cube_size
    )

    noisy_positions = add_noise_to_measurements(
        marker_positions=gt_positions,
        noise_std=noise_std,
        seed=42
    )

    # Kabsch baseline
    logger.info("\n" + "=" * 80)
    logger.info("KABSCH BASELINE")
    logger.info("=" * 80)

    reference_geometry = generate_cube_vertices(size=cube_size)

    kabsch_rotations, kabsch_translations = kabsch_initialization(
        noisy_measurements=noisy_positions,
        reference_geometry=reference_geometry,
        apply_slerp=True,
        slerp_window=7
    )

    kabsch_positions = np.zeros_like(gt_positions)
    for i in range(n_frames):
        vertices = (kabsch_rotations[i] @ reference_geometry.T).T + \
                   kabsch_translations[i]
        kabsch_positions[i, :8, :] = vertices
        kabsch_positions[i, 8, :] = np.mean(vertices, axis=0)

    # Optimization with new parameters
    opt_rotations, opt_translations, opt_positions, \
        opt_no_filter_rotations, opt_no_filter_translations, opt_no_filter_positions = \
        optimize_trajectory_jax(
            noisy_measurements=noisy_positions,
            reference_geometry=reference_geometry,
            lambda_data=lambda_data,
            lambda_smooth_pos=lambda_smooth_pos,
            lambda_smooth_rot=lambda_smooth_rot,
            lambda_accel=lambda_accel,
            lambda_rot_geodesic=lambda_rot_geodesic,
            lambda_max_rotation=lambda_max_rotation,
            max_rotation_per_frame_deg=max_rotation_per_frame_deg,
            apply_slerp_smoothing=apply_slerp_smoothing,
            slerp_window=slerp_window,
            max_iter=max_iter
        )

    # Results
    print_comparison_report(
        gt_rotations=gt_rotations,
        gt_translations=gt_translations,
        gt_positions=gt_positions,
        kabsch_rotations=kabsch_rotations,
        kabsch_translations=kabsch_translations,
        kabsch_positions=kabsch_positions,
        opt_rotations=opt_rotations,
        opt_translations=opt_translations,
        opt_positions=opt_positions,
        reference_geometry=reference_geometry
    )

    # Save outputs
    logger.info("\n" + "=" * 80)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 80)

    save_trajectory_csv(
        filepath="trajectory_data.csv",
        gt_positions=gt_positions,
        noisy_positions=noisy_positions,
        kabsch_positions=kabsch_positions,
        opt_no_filter_positions=opt_no_filter_positions,
        opt_positions=opt_positions
    )

    logger.info("\n" + "=" * 80)
    logger.info("🎉 COMPLETE!")
    logger.info("=" * 80)
    logger.info("\n🔧 Anti-spin fixes applied:")
    logger.info("   1. Unwrapped Kabsch initialization")
    logger.info("   2. True geodesic distance (SO(3) manifold)")
    logger.info("   3. Explicit large rotation penalty")
    logger.info("\nOpen rigid-body-viewer-html.html to visualize results!")
    logger.info("Look for smooth motion WITHOUT spinning artifacts! 🎯")


if __name__ == "__main__":
    main()