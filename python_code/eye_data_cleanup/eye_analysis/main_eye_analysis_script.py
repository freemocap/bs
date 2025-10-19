"""Main script to run eye tracking analysis - simplified API."""

from pathlib import Path

from python_code.eye_data_cleanup.eye_analysis.eye_analyzer import EyeDataPlotter
from python_code.eye_data_cleanup.eye_viewer import EyeVideoDataset


def main() -> None:
    """Run complete eye tracking analysis pipeline."""
    # Setup paths
    base_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37"
    )
    video_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4"
    )
    timestamps_npy_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"
    )
    csv_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )

    # Create dataset - raw and cleaned data are automatically loaded and filtered
    print("Loading eye tracking dataset...")
    eye_dataset = EyeVideoDataset.create(
        data_name="ferret_757_eye_tracking",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,  # Hz - applied to create cleaned data
        butterworth_sampling_rate=90.0  # Hz (video framerate)
    )

    # Create analyzer
    print("Initializing analyzer...")
    analyzer = EyeDataPlotter(dataset=eye_dataset)

    # Show integrated dashboard with video frame
    # You can specify which frame to display (default is frame 0)
    print("\nDisplaying integrated dashboard with video frame...")
    analyzer.plot_integrated_dashboard(
        nbins=100,
        show=True,
        frame_index=50  # Show frame 50 from the video
    )

    # # Generate complete analysis report with frame
    # print("\nGenerating complete analysis report...")
    # output_dir = base_path / "analysis_output"
    #
    # analyzer.create_analysis_report(
    #     output_dir=output_dir,
    #     nbins=50,
    #     frame_index=50  # Include frame 50 in the dashboard
    # )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()