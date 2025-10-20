import logging
from pathlib import Path

import numpy as np

from python_code.eye_analysis.csv_io import ABaseModel, load_trajectory_csv
from python_code.eye_analysis.eye_video_viewers.eye_viewer import DEFAULT_MIN_CONFIDENCE
from python_code.eye_analysis.eye_video_viewers.video_helper import VideoHelper
from python_code.eye_analysis.trajectory_dataset import TrajectoryDataset

logger = logging.getLogger(__name__)

class EyeType(str):
    LEFT = "left"
    RIGHT = "right"

class EyeVideoData(ABaseModel):
    """Dataset for eye tracking video with pupil landmarks."""

    data_name: str
    base_path: Path
    video: VideoHelper
    dataset: TrajectoryDataset
    eye_type: EyeType

    @classmethod
    def create(
        cls,
        *,
        data_name: str,
        recording_path: Path,
        raw_video_path: Path,
        timestamps_npy_path: Path,
        data_csv_path: Path,
        eye_type: EyeType | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        butterworth_cutoff: float = 6.0,

    ) -> "EyeVideoData":
        """Create an EyeVideoDataset instance."""
        timestamps = np.load(str(timestamps_npy_path))
        framerate = 1.0 / float(np.median(np.diff(timestamps)))

        if "eye1" in str(raw_video_path).lower() or "left" in str(raw_video_path).lower():
            if eye_type and eye_type != EyeType.LEFT:
                raise ValueError(f"Conflicting eye type information: {eye_type} vs LEFT inferred from filename.")
            eye_type = EyeType.LEFT
        elif 'eye0' in str(raw_video_path).lower() or "right" in str(raw_video_path).lower():
            if eye_type and eye_type != EyeType.RIGHT:
                raise ValueError(f"Conflicting eye type information: {eye_type} vs RIGHT inferred from filename.")
            eye_type = EyeType.RIGHT
        else:
            if eye_type is None:
                raise ValueError("Eye type could not be inferred from filename; please specify explicitly.")
        return cls(
            data_name=data_name,
            base_path=recording_path,
            eye_type=eye_type,
            video=VideoHelper.create(
                video_path=raw_video_path,
                timestamps_npy_path=timestamps_npy_path,
            ),
            dataset=load_trajectory_csv(
                filepath=data_csv_path,
                min_confidence=min_confidence,
                butterworth_cutoff=butterworth_cutoff,
                framerate=framerate
            ),
        )


