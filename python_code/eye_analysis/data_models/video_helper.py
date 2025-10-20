from pathlib import Path

import cv2
import numpy as np

from python_code.eye_analysis.data_models.abase_model import FrozenABaseModel


class VideoHelper(FrozenABaseModel):
    """Helper class for managing video capture and metadata."""

    video_path: Path
    video_capture: cv2.VideoCapture | None = None
    timestamps: list[float] | None = None

    @property
    def video_name(self) -> str:
        """Get the video filename without extension."""
        return self.video_path.stem

    @property
    def width(self) -> int:
        """Get the video width after applying resize factor."""
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")
        return int(self.video_capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH) )

    @property
    def height(self) -> int:
        """Get the video height after applying resize factor."""
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")
        return int(self.video_capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT) )

    @classmethod
    def create(
        cls,
        *,
        video_path: Path,
        timestamps_npy_path: Path | None = None,
    ) -> "VideoHelper":
        """Create a VideoHelper instance from a video file."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vid_cap: cv2.VideoCapture = cv2.VideoCapture(filename=str(video_path))
        if not vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        timestamps_array: np.ndarray | None = None
        if timestamps_npy_path is not None:
            if not Path(timestamps_npy_path).exists():
                raise FileNotFoundError(f"Timestamps file not found: {timestamps_npy_path}")
            timestamps_array = np.load(file=timestamps_npy_path).astype(np.float64)
            if timestamps_array.ndim != 1:
                raise ValueError(f"Timestamps array must be 1D, got shape: {timestamps_array.shape}")
            if len(timestamps_array) != int(vid_cap.get(propId=cv2.CAP_PROP_FRAME_COUNT)):
                raise ValueError(
                    f"Number of timestamps ({len(timestamps_array)}) does not match "
                    f"number of video frames ({int(vid_cap.get(propId=cv2.CAP_PROP_FRAME_COUNT))})"
                )

        return cls(
            video_path=video_path,
            video_capture=vid_cap,
            timestamps=list(timestamps_array) if timestamps_array is not None else None,
        )
