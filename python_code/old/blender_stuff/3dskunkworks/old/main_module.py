"""
Main Execution Script - CONFIG-DRIVEN VERSION
==============================================

Now with centralized configuration management!

Author: AI Assistant
Date: 2025
"""

import numpy as np
import torch
import pandas as pd
import logging

from geometry_module import (
    generate_cube_vertices,
    rotation_matrix_from_axis_angle
)
from trajectory_module import (
    kabsch_initialization,
    optimize_trajectory_torch
)


from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataGenerationConfig:
    """Parameters for synthetic data generation."""

    n_frames: int = 200 # 2 seconds at 100 Hz
    cube_size: float = 1
    noise_std: float = 0.1  # in meters
    random_seed: int | None = 42


@dataclass
class KabschInitConfig:
    """Parameters for Kabsch initialization."""

    apply_slerp: bool = True
    slerp_window: int = 7


@dataclass
class OptimizationWeights:
    """Regularization weights for trajectory optimization."""

    # Data fitting
    lambda_data: float = 1.0

    # Position smoothness
    lambda_smooth_pos: float = 30.0
    lambda_accel: float = 3.0  # Position jerk penalty

    # Rotation smoothness
    lambda_smooth_rot: float = 15.0  # Tangent space smoothness
    lambda_rot_geodesic: float = 1000.0  # Geodesic smoothness
    lambda_rot_jerk: float = 50.0  # Rotation jerk penalty

    # Rotation constraints
    lambda_max_rotation: float = 5000.0  # Large rotation penalty
    lambda_quat_consistency: float = 2000.0  # Quaternion sign consistency
    lambda_orientation_anchor: float = 100.0  # Absolute orientation anchoring

    # Rotation limits
    max_rotation_per_frame_deg: float = 15.0


@dataclass
class OptimizerConfig:
    """PyTorch optimizer configuration."""

    max_iter: int = 500
    learning_rate: float = 0.1
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str = "strong_wolfe"

    # L-BFGS internal iterations per step
    lbfgs_max_iter_per_step: int = 20


@dataclass
class PostProcessingConfig:
    """Post-optimization smoothing configuration."""

    apply_slerp_smoothing: bool = True
    slerp_window: int = 5

    # Butterworth filter for translations
    butterworth_cutoff_freq: float = 0.15
    butterworth_sampling_rate: float = 1.0
    butterworth_order: int = 4


@dataclass
class OutputConfig:
    """Output file configuration."""

    trajectory_csv_path: Path = field(default_factory=lambda: Path("trajectory_data.csv"))
    stats_json_path: Path | None = None

    # Create output directory if it doesn't exist
    create_output_dir: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Sub-configurations
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    kabsch_init: KabschInitConfig = field(default_factory=KabschInitConfig)
    optimization_weights: OptimizationWeights = field(default_factory=OptimizationWeights)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Logging
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        # Data generation validation
        if self.data_generation.n_frames < 10:
            raise ValueError(f"n_frames must be >= 10, got {self.data_generation.n_frames}")

        if self.data_generation.cube_size <= 0:
            raise ValueError(f"cube_size must be > 0, got {self.data_generation.cube_size}")

        if self.data_generation.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.data_generation.noise_std}")

        # Optimization weights validation
        if self.optimization_weights.lambda_data < 0:
            raise ValueError("lambda_data must be >= 0")

        if self.optimization_weights.max_rotation_per_frame_deg <= 0:
            raise ValueError("max_rotation_per_frame_deg must be > 0")

        # Optimizer validation
        if self.optimizer.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.optimizer.max_iter}")

        if self.optimizer.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.optimizer.learning_rate}")

        # Post-processing validation
        if self.post_processing.slerp_window < 1:
            raise ValueError(f"slerp_window must be >= 1, got {self.post_processing.slerp_window}")

        if self.post_processing.slerp_window % 2 == 0:
            raise ValueError(f"slerp_window must be odd, got {self.post_processing.slerp_window}")

        if not (0 < self.post_processing.butterworth_cutoff_freq < 1):
            raise ValueError("butterworth_cutoff_freq must be in (0, 1)")


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_default_config() -> PipelineConfig:
    """Get default (anti-spin) configuration."""
    return PipelineConfig()



# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_ground_truth_trajectory(
    *,
    n_frames: int,
    cube_size: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic ground truth rigid body trajectory."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating ground truth trajectory ({n_frames} frames)...")

    base_vertices = generate_cube_vertices(size=cube_size)
    n_markers = len(base_vertices) + 1

    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))
    marker_positions = np.zeros((n_frames, n_markers, 3))

    for i in range(n_frames):
        t = i / n_frames

        radius = 3.0
        translation = np.array([
            radius * np.cos(t * 2 * np.pi),
            radius * np.sin(t * 2 * np.pi),
            1.5 * np.sin(t * 4 * np.pi)
        ])

        rot_axis = np.array([0.3, 1.0, 0.2])
        rot_angle = t * 4 * np.pi
        R = rotation_matrix_from_axis_angle(axis=rot_axis, angle=rot_angle)

        transformed_vertices = (R @ base_vertices.T).T + translation

        rotations[i] = R
        translations[i] = translation
        marker_positions[i, :8, :] = transformed_vertices
        marker_positions[i, 8, :] = np.mean(transformed_vertices, axis=0)

    return rotations, translations, marker_positions


def add_noise_to_measurements(
    *,
    marker_positions: np.ndarray,
    noise_std: float = 0.1,
    seed: int | None = 42
) -> np.ndarray:
    """Add Gaussian noise to marker positions."""
    logger = logging.getLogger(__name__)
    logger.info(f"Adding noise (σ={noise_std * 1000:.1f} mm)...")

    if seed is not None:
        np.random.seed(seed=seed)

    n_frames, n_markers, _ = marker_positions.shape
    original_positions = marker_positions.copy()

    noise = np.random.normal(loc=0, scale=noise_std, size=(n_frames, 8, 3))
    original_positions[:, :8, :] += noise
    original_positions[:, 8, :] = np.mean(original_positions[:, :8, :], axis=1)

    return original_positions


# =============================================================================
# FILE I/O
# =============================================================================

def save_trajectory_csv(
    *,
    filepath: Path,
    gt_positions: np.ndarray,
    original_positions: np.ndarray,
    kabsch_positions: np.ndarray,
    opt_no_filter_positions: np.ndarray,
    opt_positions: np.ndarray
) -> None:
    """Save all trajectories to CSV."""
    logger = logging.getLogger(__name__)
    logger.info(f"Saving trajectory to {filepath}...")

    n_frames, n_markers, _ = gt_positions.shape
    marker_names = [f"v{i}" for i in range(8)] + ["center"]

    data: dict[str, np.ndarray | range] = {'frame': range(n_frames)}

    for dataset_name, positions in [
        ('gt', gt_positions),
        ('original', original_positions),
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




# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_pipeline(*, config: PipelineConfig) -> None:
    """
    Run complete rigid body trajectory optimization pipeline.

    Args:
        config: Pipeline configuration
    """
    logger = logging.getLogger(__name__)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(levelname)s | %(funcName)s() | %(message)s'
    )

    print("\n" + "=" * 80)
    print("RIGID BODY TRAJECTORY OPTIMIZATION - CONFIG-DRIVEN VERSION 🔥")
    print("=" * 80)

    # Check PyTorch setup
    logger.info("\n🔧 PyTorch Configuration")
    logger.info(f"  Version:  {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU device: {torch.cuda.get_device_name(0)}")
        logger.info("  ✅ GPU acceleration enabled!")
    else:
        logger.info("  ℹ️  Using CPU (still fast!)")

    # Print configuration summary
    logger.info("\n📋 Configuration Summary")
    logger.info(f"  Frames: {config.data_generation.n_frames}")
    logger.info(f"  Noise: {config.data_generation.noise_std * 1000:.1f} mm")
    logger.info(f"  Max iterations: {config.optimizer.max_iter}")
    logger.info(f"  Learning rate: {config.optimizer.learning_rate}")

    # Generate data
    logger.info("\n" + "=" * 80)
    logger.info("DATA GENERATION")
    logger.info("=" * 80)

    gt_rotations, gt_translations, gt_positions = generate_ground_truth_trajectory(
        n_frames=config.data_generation.n_frames,
        cube_size=config.data_generation.cube_size
    )

    original_positions = add_noise_to_measurements(
        marker_positions=gt_positions,
        noise_std=config.data_generation.noise_std,
        seed=config.data_generation.random_seed
    )

    # Kabsch baseline
    logger.info("\n" + "=" * 80)
    logger.info("KABSCH BASELINE")
    logger.info("=" * 80)

    reference_geometry = generate_cube_vertices(size=config.data_generation.cube_size)

    kabsch_rotations, kabsch_translations = kabsch_initialization(
        original_measurements=original_positions,
        reference_geometry=reference_geometry,
        apply_slerp=config.kabsch_init.apply_slerp,
        slerp_window=config.kabsch_init.slerp_window
    )

    n_frames = config.data_generation.n_frames
    kabsch_positions = np.zeros_like(gt_positions)
    for i in range(n_frames):
        vertices = (kabsch_rotations[i] @ reference_geometry.T).T + \
                   kabsch_translations[i]
        kabsch_positions[i, :8, :] = vertices
        kabsch_positions[i, 8, :] = np.mean(vertices, axis=0)

    # PyTorch optimization
    w = config.optimization_weights
    opt_rotations, opt_translations, opt_positions, \
        opt_no_filter_rotations, opt_no_filter_translations, opt_no_filter_positions = \
        optimize_trajectory_torch(
            original_measurements=original_positions,
            reference_geometry=reference_geometry,
            lambda_data=w.lambda_data,
            lambda_smooth_pos=w.lambda_smooth_pos,
            lambda_smooth_rot=w.lambda_smooth_rot,
            lambda_accel=w.lambda_accel,
            lambda_rot_geodesic=w.lambda_rot_geodesic,
            lambda_max_rotation=w.lambda_max_rotation,
            lambda_quat_consistency=w.lambda_quat_consistency,
            lambda_orientation_anchor=w.lambda_orientation_anchor,
            lambda_rot_jerk=w.lambda_rot_jerk,
            max_rotation_per_frame_deg=w.max_rotation_per_frame_deg,
            apply_slerp_smoothing=config.post_processing.apply_slerp_smoothing,
            slerp_window=config.post_processing.slerp_window,
            max_iter=config.optimizer.max_iter,
            learning_rate=config.optimizer.learning_rate
        )



    # Save outputs
    logger.info("\n" + "=" * 80)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 80)

    # Create output directory if needed
    if config.output.create_output_dir:
        config.output.trajectory_csv_path.parent.mkdir(parents=True, exist_ok=True)

    save_trajectory_csv(
        filepath=config.output.trajectory_csv_path,
        gt_positions=gt_positions,
        original_positions=original_positions,
        kabsch_positions=kabsch_positions,
        opt_no_filter_positions=opt_no_filter_positions,
        opt_positions=opt_positions
    )



    logger.info("\n" + "=" * 80)
    logger.info("🎉 COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nOpen rigid-body-viewer-html.html to verify NO SPINNING! 🎯")



if __name__ == "__main__":
    run_pipeline(config=get_default_config())