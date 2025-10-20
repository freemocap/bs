from pathlib import Path

from python_code.eye_analysis.eye_video_viewers.eye_viewer import (
    SVGEyeTrackingViewer,
    ViewMode
)
from python_code.eye_analysis.eye_video_dataset import EyeVideoData
from python_code.eye_analysis.data_processing.active_contour_fit import SnakeParams

if __name__ == "__main__":
    # Setup paths
    base_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37"
    )
    video_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4"
    )
    timestamps_npy_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"
    )
    csv_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )

    # Configure snake parameters
    # STRATEGY: Use high rigidity + displacement constraints to keep snake near initial ellipse
    # The snake can refine the pupil boundary but won't collapse or wander far
    #
    # Parameters:
    # - n_points: Number of points in the snake (20 is a good default for pupils)
    # - alpha: Elasticity - HIGH values resist length changes (prevents collapse)
    # - beta: Smoothness - HIGH values keep the curve smooth like the initial ellipse
    # - w_line: Attraction to dark/light regions (0 = ignore intensity)
    # - w_edge: Attraction to edges
    # - gamma: Step size for convergence
    # - max_iterations: FEWER iterations = stays closer to initial ellipse (500 not 2500!)
    # - sigma: Gaussian blur for edge detection
    # - max_displacement: Maximum allowed movement from initial position (10 pixels)
    snake_params: SnakeParams = SnakeParams(
        n_points=20,
        alpha=0.5,  # HIGH elasticity - resists shrinking/expanding
        beta=1.0,  # HIGH smoothness - stays ellipse-like
        w_line=0.0,  # Ignore intensity (prevents collapse toward dark center)
        w_edge=1.0,  # Use edge gradients to refine boundary
        gamma=0.01,
        max_iterations=500,  # Only 500 iterations = less time to collapse
        sigma=2.0,
        max_displacement=10.0  # Hard constraint: can only move 10px from initial ellipse
    )

    # Create dataset - both raw and cleaned data are automatically loaded
    eye_dataset: EyeVideoData = EyeVideoData.create(
        data_name="ferret_757_eye_tracking",
        recording_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,  # Hz
        butterworth_sampling_rate=90.0,  # Hz (video framerate)
        snake_params=snake_params
    )

    # Now you can access data via clean dot notation:
    # eye_dataset.pixel_trajectories.raw['p1']  # Raw Trajectory2D for p1
    # eye_dataset.pixel_trajectories.cleaned['p1']  # Cleaned Trajectory2D for p1
    # eye_dataset.pixel_trajectories.pairs['p1'].raw  # Also raw
    # eye_dataset.pixel_trajectories.pairs['p1'].cleaned  # Also cleaned

    # Create viewer with snake enabled
    viewer: SVGEyeTrackingViewer = SVGEyeTrackingViewer(
        dataset=eye_dataset,
        window_name="SVG Pupil Tracking with Snake",
        initial_view_mode=ViewMode.CLEANED,
        enable_snake=True  # Enable snake contours
    )

    # Run viewer
    # Keyboard shortcuts during playback:
    # - 'r' = raw data only (red/orange dots, orange ellipse, yellow snake)
    # - 'c' = cleaned data only (cyan dots + lines, magenta ellipse, green snake)
    # - 'b' = both overlaid
    # - 'n' = toggle snake contours on/off
    # The green snake should form-fit to the dark pupil edge!
    viewer.run(start_frame=0)