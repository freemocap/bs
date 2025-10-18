from pathlib import Path

import cv2
import numpy as np

from python_code.data_loaders.trajectory_loader.trajectory_csv_io import load_trajectory_csv
from python_code.data_loaders.trajectory_loader.trajectory_dataset import (
    ABaseModel,
    TrajectoryND,
    TrajectoryDataset,
    TrajectoryType,
)

DEFAULT_RESIZE_FACTOR: float = 1.0
DEFAULT_MIN_CONFIDENCE: float = 0.3


class VideoHelper(ABaseModel):
    """Helper class for managing video capture and metadata."""

    video_path: Path
    video_capture: cv2.VideoCapture | None = None
    timestamps: list[float] | None = None
    resize_factor: float = DEFAULT_RESIZE_FACTOR

    @property
    def video_name(self) -> str:
        """Get the video filename without extension."""
        return self.video_path.stem

    @property
    def width(self) -> int:
        """Get the video width after applying resize factor."""
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")
        return int(self.video_capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH) * self.resize_factor)

    @property
    def height(self) -> int:
        """Get the video height after applying resize factor."""
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")
        return int(self.video_capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT) * self.resize_factor)

    @classmethod
    def create(
            cls,
            *,
            video_path: Path,
            timestamps_npy_path: Path | None = None,
            resize_factor: float = DEFAULT_RESIZE_FACTOR,
    ) -> "VideoHelper":
        """
        Create a VideoHelper instance from a video file.

        Args:
            video_path: Path to the video file
            timestamps_npy_path: Optional path to numpy array of timestamps
            resize_factor: Factor to resize video frames

        Returns:
            VideoHelper instance

        Raises:
            FileNotFoundError: If video file doesn't exist
            IOError: If video cannot be opened
        """
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

        return cls(
            video_path=video_path,
            video_capture=vid_cap,
            timestamps=list(timestamps_array) if timestamps_array is not None else None,
            resize_factor=resize_factor,
        )


class RerunVideoDataset(ABaseModel):
    """Base model representing video data and associated tracking."""

    data_name: str
    base_path: Path
    video: VideoHelper
    pixel_trajectories: TrajectoryDataset

    @classmethod
    def create(
            cls,
            *,
            data_name: str,
            base_path: Path,
            raw_video_path: Path,
            timestamps_npy_path: Path,
            data_csv_path: Path,
            resize_factor: float = DEFAULT_RESIZE_FACTOR,
            min_confidence: float = DEFAULT_MIN_CONFIDENCE
    ) -> "RerunVideoDataset":
        """
        Create a RerunVideoDataset instance.

        Args:
            data_name: Descriptive name for this dataset
            base_path: Base directory path
            raw_video_path: Path to video file
            timestamps_npy_path: Path to timestamps numpy array
            data_csv_path: Path to trajectory CSV data
            resize_factor: Video resize factor
            min_confidence: Minimum confidence threshold for trajectories

        Returns:
            RerunVideoDataset instance
        """
        return cls(
            data_name=data_name,
            base_path=base_path,
            video=VideoHelper.create(
                video_path=raw_video_path,
                timestamps_npy_path=timestamps_npy_path,
                resize_factor=resize_factor
            ),
            pixel_trajectories=load_trajectory_csv(
                filepath=data_csv_path,
                min_confidence=min_confidence,
                trajectory_type=TrajectoryType.POSITION_2D,
            )
        )


class RerunEyeVideoDataset(RerunVideoDataset):
    """Dataset for eye tracking video with pupil landmarks."""

    # Define the landmark indices for pupil points
    landmarks: dict[str, int] = {
        "p1": 0,
        "p2": 1,
        "p3": 2,
        "p4": 3,
        "p5": 4,
        "p6": 5,
        "p7": 6,
        "p8": 7,
        "tear_duct": 8,
        "outer_eye": 9,
    }

    # Define connections between landmarks (for visualization)
    connections: tuple[tuple[int, int], ...] = (
        # Pupil outline (closed loop)
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7), (7, 0),
        # Additional connections
        (8, 9),  # tear duct to outer eye
        (8, 0),  # tear duct to p1
    )

    @property
    def pupil_mean_x(self) -> TrajectoryND:
        """
        Calculate mean x-coordinate of pupil points (p1-p8).

        Returns:
            1D array of mean x coordinates over time
        """
        pupil_points_x: list[np.ndarray] = []
        for landmark_name in self.landmarks:
            if "p" in landmark_name and landmark_name != "tear_duct":
                point_idx: int = self.landmarks[landmark_name]
                pupil_points_x.append(self.pixel_trajectories.to_array()[:, point_idx, 0])
        return np.nanmean(a=np.array(pupil_points_x), axis=0)

    @property
    def pupil_mean_y(self) -> TrajectoryND:
        """
        Calculate mean y-coordinate of pupil points (p1-p8).

        Returns:
            1D array of mean y coordinates over time
        """
        pupil_points_y: list[np.ndarray] = []
        for landmark_name in self.landmarks:
            if "p" in landmark_name and landmark_name != "tear_duct":
                point_idx: int = self.landmarks[landmark_name]
                pupil_points_y.append(self.pixel_trajectories.to_array()[:, point_idx, 1])
        return np.nanmean(a=np.array(pupil_points_y), axis=0)