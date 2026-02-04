import enum
import logging
from pathlib import Path

import numpy as np

from python_code.eye_analysis.data_models.abase_model import ABaseModel, FrozenABaseModel
from python_code.eye_analysis.data_models.csv_io import load_trajectory_dataset
from python_code.eye_analysis.data_models.video_helper import VideoHelper
from python_code.eye_analysis.data_models.trajectory_dataset import TrajectoryDataset, DEFAULT_MIN_CONFIDENCE, \
    DEFAULT_BUTTERWORTH_CUTOFF, DEFAULT_BUTTERWORTH_ORDER

logger = logging.getLogger(__name__)

class EyeType(enum.Enum):
    LEFT = "left"
    RIGHT = "right"

class EyeVideoData(FrozenABaseModel):
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
        butterworth_cutoff: float = DEFAULT_BUTTERWORTH_CUTOFF,
            butterworth_order: int = DEFAULT_BUTTERWORTH_ORDER,

    ) -> "EyeVideoData":
        """Create an EyeVideoDataset instance."""
        timestamps = np.load(str(timestamps_npy_path)) /1e9 # Convert ns to s

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
            dataset=load_trajectory_dataset(
                filepath=data_csv_path,
                min_confidence=min_confidence,
                butterworth_cutoff=butterworth_cutoff,
                butterworth_order=butterworth_order,
                timestamps=timestamps,
            ),
        )


