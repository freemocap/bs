"""Main script to run eye tracking analysis."""

from pathlib import Path

from python_code.eye_data_cleanup.eye_analysis.analyzer_main import EyeTrackingAnalyzer
from python_code.eye_data_cleanup.eye_viewer import EyeVideoDataset


def main() -> None:
    """Run complete eye tracking analysis pipeline."""
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

    # Create dataset
    print("Loading eye tracking dataset...")
    eye_dataset: EyeVideoDataset = EyeVideoDataset.create(
        data_name="ferret_757_eye_tracking",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
    )

    # Create analyzer
    print("Initializing analyzer...")
    analyzer: EyeTrackingAnalyzer = EyeTrackingAnalyzer(dataset=eye_dataset)

    #  Show integrated dashboard (all plots in one view)
    print("\nDisplaying integrated dashboard...")
    analyzer.plot_integrated_dashboard(
        cutoff=5.0,
        fs=30.0,
        order=4,
        nbins=100,
        show=True
    )

    # # Generate complete analysis report
    # print("\nGenerating complete analysis report...")
    # output_dir: Path = base_path / "analysis_output"
    #
    # analyzer.create_analysis_report(
    #     output_dir=output_dir,
    #     cutoff=5.0,
    #     fs=30.0,
    #     order=4,
    #     nbins=50
    # )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
