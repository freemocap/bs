from pathlib import Path

from python_code.eye_data_cleanup.eye_viewer import (
    EyeVideoDataset,
    SVGEyeTrackingViewer,
    ViewMode
)

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

    # Create dataset - both raw and cleaned data are automatically loaded
    eye_dataset: EyeVideoDataset = EyeVideoDataset.create(
        data_name="ferret_757_eye_tracking",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,  # Hz
        butterworth_sampling_rate=30.0  # Hz (video framerate)
    )

    # Now you can access data via clean dot notation:
    # eye_dataset.pixel_trajectories.raw['p1']  # Raw Trajectory2D for p1
    # eye_dataset.pixel_trajectories.cleaned['p1']  # Cleaned Trajectory2D for p1
    # eye_dataset.pixel_trajectories.pairs['p1'].raw  # Also raw
    # eye_dataset.pixel_trajectories.pairs['p1'].cleaned  # Also cleaned

    # Create viewer
    viewer: SVGEyeTrackingViewer = SVGEyeTrackingViewer(
        dataset=eye_dataset,
        window_name="SVG Pupil Tracking",
        initial_view_mode=ViewMode.CLEANED
    )

    # Run viewer
    # Keyboard shortcuts during playback:
    # - 'r' = raw data only (red/orange dots)
    # - 'c' = cleaned data only (cyan dots + lines)
    # - 'b' = both overlaid
    viewer.run(start_frame=0)