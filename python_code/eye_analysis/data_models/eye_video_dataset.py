import enum
import logging
from pathlib import Path

import numpy as np

from python_code.eye_analysis.data_models.abase_model import (
    ABaseModel,
    FrozenABaseModel,
)
from python_code.eye_analysis.data_models.csv_io import load_trajectory_dataset
from python_code.eye_analysis.data_models.video_helper import VideoHelper
from python_code.eye_analysis.data_models.trajectory_dataset import (
    TrajectoryDataset,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_EYE_FILTER_SPEC,
)
from python_code.utilities.temporal_filters import FilterSpec

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
        filter_spec: FilterSpec = DEFAULT_EYE_FILTER_SPEC,
    ) -> "EyeVideoData":
        """Create an EyeVideoDataset instance."""
        timestamps = np.load(str(timestamps_npy_path)) / 1e9  # Convert ns to s

        # Eye videos are named left_eye*/right_eye*; legacy eye0/eye1 names are not supported.
        video_name = raw_video_path.name.lower()
        inferred: EyeType | None = None
        if "left" in video_name and "right" not in video_name:
            inferred = EyeType.LEFT
        elif "right" in video_name and "left" not in video_name:
            inferred = EyeType.RIGHT

        if eye_type is None:
            if inferred is None:
                raise ValueError(
                    f"Could not infer eye type from {raw_video_path.name} (expected 'left' or 'right' "
                    f"in the filename); please specify eye_type explicitly."
                )
            eye_type = inferred
        elif inferred is not None and inferred != eye_type:
            raise ValueError(
                f"Conflicting eye type information: {eye_type} given, but filename "
                f"{raw_video_path.name} indicates {inferred}."
            )
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
                filter_spec=filter_spec,
                timestamps=timestamps,
            ),
        )
