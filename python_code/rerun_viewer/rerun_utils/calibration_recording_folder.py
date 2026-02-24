from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel


class CalibrationRecordingFolder(BaseModel):
    base_recordings_folder: Path
    recording_name: str
    
    # Derived paths
    recording_folder: Path
    mocap_annotated_videos_folder: Path
    mocap_synchronized_videos_folder: Path
    mocap_timestamps_folder: Path
    mocap_output_data_folder: Path

    topdown_video_name: str
    topdown_annotated_video_path: Path
    topdown_video_path: Path
    topdown_timestamps_npy_path: Path

    data_3d_npy_path: Path

    @classmethod
    def create_from_recording_name(cls, 
        recording_name: str, 
        base_recordings_folder: Path = Path("/Users/philipqueen")
    ) -> 'RecordingFolder':
        """Create a RecordiongFolder instance from recording and clip names."""
        recording_folder = Path(base_recordings_folder) / recording_name / "calibration"
        print(f"Parsing recording folder: {recording_folder}")

        mocap_annotated_videos_folder = recording_folder / "charuco_annotated_videos"
        mocap_synchronized_videos_folder = recording_folder / "synchronized_videos"
        if not mocap_synchronized_videos_folder.exists():
            mocap_synchronized_videos_folder = recording_folder / "synchronized_corrected_videos"
        mocap_timestamps_folder = mocap_synchronized_videos_folder
        mocap_output_data_folder = recording_folder / "output_data"
        for path in [recording_folder, mocap_annotated_videos_folder, mocap_synchronized_videos_folder, mocap_timestamps_folder, mocap_output_data_folder]:
            if not path.exists():
                raise ValueError(f"Path does not exist: {path}")

        topdown_video_name = "24676894"
        topdown_annotated_video_path = list(mocap_annotated_videos_folder.glob(f"{topdown_video_name}*.mp4"))[0]
        topdown_video_path = list(mocap_synchronized_videos_folder.glob(f"{topdown_video_name}*.mp4"))[0]
        topdown_timestamps_npy_path = mocap_output_data_folder / f"spoofed_timestamps.npy"

        cap = cv2.VideoCapture(str(topdown_video_path))
        recording_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamps = np.array(range(0, recording_length)) * 1e9 / cv2.CAP_PROP_FPS
        print(f"Saving spoofed timestamps to {topdown_timestamps_npy_path}")
        np.save(topdown_timestamps_npy_path, timestamps)

        data_3d_npy_path = mocap_output_data_folder / "charuco_3d_xyz.npy"
        if not data_3d_npy_path.exists():
            raise ValueError(f"Path does not exist: {data_3d_npy_path}")

        return cls(
            base_recordings_folder=base_recordings_folder,
            recording_name=recording_name,
            recording_folder=recording_folder,
            mocap_annotated_videos_folder=mocap_annotated_videos_folder,
            mocap_synchronized_videos_folder=mocap_synchronized_videos_folder,
            mocap_timestamps_folder=mocap_timestamps_folder,
            mocap_output_data_folder=mocap_output_data_folder,
            topdown_video_name=topdown_video_name,
            topdown_annotated_video_path=topdown_annotated_video_path,
            topdown_video_path=topdown_video_path,
            topdown_timestamps_npy_path=topdown_timestamps_npy_path,
            data_3d_npy_path=data_3d_npy_path
        )