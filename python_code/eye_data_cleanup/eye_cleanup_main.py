from pathlib import Path

from python_code.eye_data_cleanup.eye_viewer import EyeVideoDataset, create_full_eye_topology, SVGEyeTrackingViewer

if __name__ == "__main__":
    # Setup paths (same as your original)
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
    eye_dataset: EyeVideoDataset = EyeVideoDataset.create(
        data_name="ferret_757_eye_tracking",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
    )


    # Option 2: Full topology (all features)
    topology = create_full_eye_topology(
        width=eye_dataset.video.width,
        height=eye_dataset.video.height
    )

    # Create viewer with SVG topology
    viewer: SVGEyeTrackingViewer = SVGEyeTrackingViewer(
        dataset=eye_dataset,
        topology=topology,
        window_name="SVG Pupil Tracking"
    )

    # Run viewer
    viewer.run(start_frame=0)
