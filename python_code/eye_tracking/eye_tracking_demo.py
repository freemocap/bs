"""Demo script for 3D eye tracking from pupil centers."""

from pathlib import Path
import logging

from eye_camera_model import CameraIntrinsics
from eye_optimization import OptimizationConfig, create_default_eye_model
from eye_tracking_api import PipelineConfig, process_eye_tracking

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


def run_eye_tracking_demo() -> None:
    """Run complete eye tracking demo with interactive viewer generation."""

    logger.info("="*80)
    logger.info("3D EYE TRACKING FROM PUPIL CENTERS")
    logger.info("="*80)
    logger.info("\nThis pipeline:")
    logger.info("1. Loads 2D pupil ellipse tracking data (p1-p8 points)")
    logger.info("2. Computes pupil centers as mean of valid points")
    logger.info("3. Estimates 3D eyeball orientation via optimization")
    logger.info("4. Computes 3D gaze direction from eyeball rotation")
    logger.info("5. Generates interactive HTML viewer with 3D model")
    logger.info("="*80)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Input data
    input_csv = Path(
        "/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/eye_data/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1_0m_37s-1m_37s_eye_data.csv"
    )

    # Optional: video file for viewer
    video_path = Path(
        "/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/eye_data/eye_videos/eye0_clipped_4340_11510.mp4"
    )

    # Check if video exists
    if not video_path.exists():
        logger.warning(f"Video not found: {video_path}")
        logger.info("Viewer will be generated without video")
        video_path = None
    else:
        logger.info(f"Video found: {video_path.name}")

    output_dir = Path("output/eye_tracking_demo")

    # Camera parameters
    camera = CameraIntrinsics.from_eye_camera_spec()

    logger.info(f"\nCamera Configuration:")
    logger.info(f"  Focal length: {camera.focal_length_mm:.2f} mm")
    logger.info(f"  Sensor: {camera.sensor_width_mm:.2f}mm x {camera.sensor_height_mm:.2f}mm")
    logger.info(f"  Resolution: {camera.image_width_px}x{camera.image_height_px} px")
    logger.info(f"  Pixel size: {camera.pixel_size_x*1000:.2f}µm x {camera.pixel_size_y*1000:.2f}µm")
    logger.info(f"  Focal length (px): fx={camera.fx:.1f}, fy={camera.fy:.1f}")

    # Eye model (typical parameters)
    eye_model = create_default_eye_model(
        eyeball_center_z_mm=20.0,  # Distance from camera to eye # TODO: Made up
        eyeball_radius_mm=3.5     # Typical eyeball radius 
    )

    logger.info(f"\nEye Model:")
    logger.info(f"  Eyeball radius: {eye_model.eyeball_radius_mm} mm")
    logger.info(f"  Pupil offset: {eye_model.pupil_offset_mm} mm")
    logger.info(f"  Eyeball center: {eye_model.eyeball_center_mm}")

    # Optimization configuration
    optimization = OptimizationConfig(
        optimization_method="Powell",
        max_iterations=200,
        max_rotation_degrees=45.0,
        use_temporal_init=True,
        convergence_tolerance=1e-6,
        verbose=True
    )

    logger.info(f"\nOptimization Configuration:")
    logger.info(f"  Method: {optimization.optimization_method}")
    logger.info(f"  Max iterations: {optimization.max_iterations}")
    logger.info(f"  Max rotation: {optimization.max_rotation_degrees}°")
    logger.info(f"  Temporal initialization: {optimization.use_temporal_init}")

    # =========================================================================
    # RUN PIPELINE
    # =========================================================================

    config = PipelineConfig(
        input_csv=input_csv,
        output_dir=output_dir,
        camera=camera,
        eye_model=eye_model,
        optimization=optimization,
        min_valid_points=3,
        video_path=video_path
    )

    result = process_eye_tracking(config=config)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80)
    logger.info(f"\n📁 Results saved to: {output_dir}")
    logger.info(f"\n📊 Output files:")
    logger.info(f"  • eye_tracking_results.csv - Full results with gaze angles")
    logger.info(f"  • summary_stats.csv - Summary statistics")
    logger.info(f"  • eye_tracking_viewer.html - Interactive 3D viewer ⭐")

    logger.info(f"\n🚀 Next steps:")
    logger.info(f"  1. Open {output_dir / 'eye_tracking_viewer.html'} in your web browser")
    logger.info(f"  2. The viewer has all data pre-loaded - just open and play!")
    logger.info(f"  3. Use the CSV files for downstream analysis")

    viewer_path = output_dir / "eye_tracking_viewer.html"
    logger.info(f"\n💡 To view results, run:")
    logger.info(f"     start {viewer_path}  (Windows)")
    logger.info(f"     open {viewer_path}   (Mac)")
    logger.info(f"     xdg-open {viewer_path}  (Linux)")

    logger.info("\n" + "="*80)
    logger.info("KEY OUTPUTS")
    logger.info("="*80)
    logger.info(f"  🎯 Eyeball center: 3D position in camera frame (mm)")
    logger.info(f"  🔄 Eye rotation: Computed from pupil center displacement")
    logger.info(f"  👁️ Gaze direction: 3D unit vector from eye rotation")
    logger.info(f"  📐 Gaze angles: Azimuth/Elevation in degrees")

    mean_error = result['reprojection_errors_px'].mean()
    logger.info(f"  📏 Mean reprojection error: {mean_error:.2f} px")


if __name__ == "__main__":
    run_eye_tracking_demo()