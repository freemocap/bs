"""Complete eye tracking pipeline API."""

from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np
import time

from eye_camera_model import CameraIntrinsics
from eye_optimization import (
    OptimizationConfig,
    EyeModel,
    batch_estimate_eye_orientation,
    create_default_eye_model
)
from eye_loaders import (
    load_pupil_centers,
    filter_invalid_frames
)
from eye_savers import (
    save_eye_tracking_results,
    save_summary_stats,
    compute_reprojection_error_stats,
    print_summary
)
from viewer_generator import save_interactive_viewer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Complete configuration for eye tracking pipeline."""

    input_csv: Path
    """Path to pupil ellipse CSV (DLC format with p1-p8 points)"""

    output_dir: Path
    """Output directory"""

    camera: CameraIntrinsics
    """Camera intrinsics"""

    eye_model: EyeModel
    """Eye model parameters"""

    optimization: OptimizationConfig
    """Optimization configuration"""

    min_valid_points: int = 3
    """Minimum number of valid ellipse points required per frame"""

    video_path: Path | None = None
    """Optional video path for viewer"""


def process_eye_tracking(
    *,
    config: PipelineConfig
) -> dict[str, np.ndarray | float]:
    """
    Complete eye tracking pipeline.

    Pipeline:
    1. Load pupil ellipse data and compute centers
    2. Filter invalid frames
    3. Optimize 3D eye orientation for each frame
    4. Save results to CSV
    5. Generate interactive HTML viewer

    Args:
        config: PipelineConfig

    Returns:
        Dictionary with results arrays
    """
    logger.info("="*80)
    logger.info("EYE TRACKING PIPELINE")
    logger.info("="*80)
    logger.info(f"Input:  {config.input_csv.name}")
    logger.info(f"Output: {config.output_dir}")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 1: LOAD DATA")
    logger.info("="*80)

    data = load_pupil_centers(filepath=config.input_csv)
    pupil_centers = data['pupil_centers']
    frame_indices = data['frame_indices']
    n_valid_points = data['n_valid_points']

    logger.info(f"  Loaded {len(pupil_centers)} frames")

    # =========================================================================
    # STEP 2: FILTER INVALID FRAMES
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 2: FILTER INVALID FRAMES")
    logger.info("="*80)

    pupil_centers, frame_indices, valid_mask = filter_invalid_frames(
        pupil_centers=pupil_centers,
        frame_indices=frame_indices,
        n_valid_points=n_valid_points,
        min_valid_points=config.min_valid_points
    )

    # =========================================================================
    # STEP 3: OPTIMIZE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 3: OPTIMIZE 3D EYE ORIENTATION")
    logger.info("="*80)

    start_time = time.time()

    results = batch_estimate_eye_orientation(
        observed_pupil_centers_px=pupil_centers,
        eye_model=config.eye_model,
        camera=config.camera,
        config=config.optimization
    )

    optimization_time = time.time() - start_time

    # Extract results into arrays
    n_frames = len(results)
    pupil_centers_3d_mm = np.zeros(shape=(n_frames, 3))
    pupil_centers_reprojected_px = np.zeros(shape=(n_frames, 2))
    eyeball_centers_mm = np.zeros(shape=(n_frames, 3))
    gaze_directions = np.zeros(shape=(n_frames, 3))
    reprojection_errors_px = np.zeros(shape=n_frames)

    for i, result in enumerate(results):
        pupil_centers_3d_mm[i] = result.pupil_center_3d_mm
        pupil_centers_reprojected_px[i] = result.projected_pixel_px
        eyeball_centers_mm[i] = config.eye_model.eyeball_center_mm
        gaze_directions[i] = result.gaze_direction
        reprojection_errors_px[i] = result.reprojection_error_px

    logger.info(f"  Optimization completed in {optimization_time:.2f}s")
    logger.info(f"  Time per frame: {optimization_time/n_frames*1000:.1f}ms")

    # =========================================================================
    # STEP 4: EVALUATE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 4: EVALUATE RESULTS")
    logger.info("="*80)

    error_stats = compute_reprojection_error_stats(
        reprojection_errors_px=reprojection_errors_px
    )

    print_summary(
        reprojection_errors=error_stats,
        eyeball_centers_mm=eyeball_centers_mm,
        gaze_directions=gaze_directions
    )

    # =========================================================================
    # STEP 5: SAVE RESULTS
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 5: SAVE RESULTS")
    logger.info("="*80)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save main results CSV
    results_csv = config.output_dir / "eye_tracking_results.csv"
    save_eye_tracking_results(
        filepath=results_csv,
        frame_indices=frame_indices,
        pupil_centers_observed_px=pupil_centers,
        pupil_centers_reprojected_px=pupil_centers_reprojected_px,
        pupil_centers_3d_mm=pupil_centers_3d_mm,
        eyeball_centers_mm=eyeball_centers_mm,
        gaze_directions=gaze_directions,
        reprojection_errors_px=reprojection_errors_px
    )

    # Save summary statistics
    save_summary_stats(
        filepath=config.output_dir / "summary_stats.csv",
        reprojection_errors=error_stats,
        eyeball_centers_mm=eyeball_centers_mm,
        gaze_directions=gaze_directions,
        optimization_time_sec=optimization_time,
        n_frames=n_frames
    )

    # =========================================================================
    # STEP 6: GENERATE INTERACTIVE VIEWER
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 6: GENERATE INTERACTIVE VIEWER")
    logger.info("="*80)

    save_interactive_viewer(
        output_dir=config.output_dir,
        csv_filepath=results_csv,
        video_path=config.video_path
    )

    logger.info(f"\n✓ Complete! Results saved to: {config.output_dir}")
    logger.info(f"\n📊 Output files:")
    logger.info(f"  • eye_tracking_results.csv - Main results with gaze angles")
    logger.info(f"  • summary_stats.csv - Summary statistics")
    logger.info(f"  • eye_tracking_viewer.html - Interactive 3D viewer ⭐")

    return {
        'pupil_centers_observed_px': pupil_centers,
        'pupil_centers_reprojected_px': pupil_centers_reprojected_px,
        'pupil_centers_3d_mm': pupil_centers_3d_mm,
        'eyeball_centers_mm': eyeball_centers_mm,
        'gaze_directions': gaze_directions,
        'reprojection_errors_px': reprojection_errors_px,
        'optimization_time_sec': optimization_time
    }