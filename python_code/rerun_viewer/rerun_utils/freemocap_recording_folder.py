from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel


class FreemocapRecordingFolder(BaseModel):
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

    @classmethod
    def create_from_clip(cls, 
        recording_name: str, 
        base_recordings_folder: Path = Path("/Users/philipqueen/freemocap_data/recording_sessions")
    ) -> 'RecordingFolder':
        """Create a RecordiongFolder instance from recording and clip names."""
        recording_folder = base_recordings_folder / recording_name
        print(f"Parsing recording folder: {recording_folder}")

        mocap_annotated_videos_folder = recording_folder / "annotated_videos"
        mocap_synchronized_videos_folder = recording_folder / "synchronized_videos"
        mocap_timestamps_folder = mocap_synchronized_videos_folder
        mocap_output_data_folder = recording_folder / "output_data"
        for path in [recording_folder, mocap_annotated_videos_folder, mocap_synchronized_videos_folder, mocap_timestamps_folder, mocap_output_data_folder]:
            if not path.exists():
                raise ValueError(f"Path does not exist: {path}")

        topdown_video_name = "sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam1"
        topdown_annotated_video_path = list(mocap_annotated_videos_folder.glob(f"{topdown_video_name}*.mp4"))[0]
        topdown_video_path = list(mocap_synchronized_videos_folder.glob(f"{topdown_video_name}*.mp4"))[0]
        topdown_timestamps_npy_path = mocap_output_data_folder / f"spoofed_timestamps.npy"

        cap = cv2.VideoCapture(str(topdown_video_path))
        recording_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamps = np.array(range(0, recording_length)) * 1e9 / 30
        print(f"Saving spoofed timestamps to {topdown_timestamps_npy_path}")
        print(timestamps)
        np.save(topdown_timestamps_npy_path, timestamps)

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
            topdown_timestamps_npy_path=topdown_timestamps_npy_path
        )