from pathlib import Path


from python_code.eye_analysis.data_models.eye_video_dataset import EyeVideoData
from python_code.eye_analysis.video_viewers.eye_viewer import EyeVideoDataViewer

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
    # # Define snake parameters (Active Contour Model)
    # # TODO - I suspect we can make this work, but for now the simple ellipse works best
    # snake_params: SnakeParams = SnakeParams(
    #     n_points=20,
    #     alpha=0.5,  # HIGH elasticity - resists shrinking/expanding
    #     beta=1.0,  # HIGH smoothness - stays ellipse-like
    #     w_line=0.0,  # Ignore intensity (prevents collapse toward dark center)
    #     w_edge=1.0,  # Use edge gradients to refine boundary
    #     gamma=0.01,
    #     max_iterations=500,  # Only 500 iterations = less time to collapse
    #     sigma=2.0,
    #     max_displacement=10.0  # Hard constraint: can only move 10px from initial ellipse
    # )

    # Create dataset - both raw and cleaned data are automatically loaded
    eye_dataset: EyeVideoData = EyeVideoData.create(
        data_name="ferret_757_eye_tracking",
        recording_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,  # Hz
    )

    viewer: EyeVideoDataViewer = EyeVideoDataViewer.create(
        dataset=eye_dataset,
    )

    viewer.run(start_frame=0)