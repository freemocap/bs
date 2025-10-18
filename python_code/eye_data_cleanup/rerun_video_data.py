from pathlib import Path

import cv2
import numpy as np

from python_code.data_loaders.trajectory_loader.trajectory_csv_io import load_trajectory_csv
from python_code.data_loaders.trajectory_loader.trajectory_dataset import ABaseModel, TrajectoryND, TrajectoryDataset, \
    TrajectoryType

DEFAULT_RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
DEFAULT_MIN_CONFIDENCE = 0.3  # Minimum confidence for trajectory points
DEFAULT_COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)n


class VideoHelper(ABaseModel):
    video_path: Path
    video_capture: cv2.VideoCapture | None = None
    timestamps: list[float] | None = None

    resize_factor: float = DEFAULT_RESIZE_FACTOR

    @property
    def video_name(self) -> str:
        return self.video_path.stem
    @property
    def width(self) -> int:
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)*self.resize_factor)
    @property
    def height(self) -> int:
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)*self.resize_factor)

    @classmethod
    def create(cls,
               video_path: Path,
               timestamps_npy_path: Path | None = None,
               resize_factor: float = DEFAULT_RESIZE_FACTOR,
               ):
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vid_cap = cv2.VideoCapture(str(video_path))
        if not vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        timestamps_array = None
        if timestamps_npy_path is not None:
            if not Path(timestamps_npy_path).exists():
                raise FileNotFoundError(f"NPY file not found: {timestamps_npy_path}")
            timestamps_array = np.load(timestamps_npy_path).astype(np.float64)

        return cls(
            video_path=video_path,
            video_capture=vid_cap,
            timestamps=list(timestamps_array),
            resize_factor=resize_factor,
        )


class RerunVideoDataset(ABaseModel):
    """Base model representing video data and associated tracking."""
    data_name: str  # E.g. "Left Eye", "Right Eye", "TopDown Mocap"
    base_path: Path
    video: VideoHelper
    pixel_trajectories: TrajectoryDataset

    @classmethod
    def create(cls,
               data_name: str,
               base_path: Path,
               raw_video_path: Path,
               timestamps_npy_path: Path,
               data_csv_path: Path,
               resize_factor: float = DEFAULT_RESIZE_FACTOR,
               min_confidence: float = DEFAULT_MIN_CONFIDENCE
               ):
       return cls(data_name=data_name,
            base_path=base_path,
            video=VideoHelper.create(
                video_path=raw_video_path,
                timestamps_npy_path=timestamps_npy_path,
                resize_factor=resize_factor
            ),
            pixel_trajectories=load_trajectory_csv(filepath=data_csv_path,
                                                   min_confidence=min_confidence)
            )

class RerunEyeVideoDataset(RerunVideoDataset):

    landmarks:dict[str,int] = {
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

    connections:tuple[int,int] = (
        #pupil outline
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 0),
        # additional connections
        (8, 9), # tear duct to outer eye
        (8, 0), # tear duct to p1
    )

    @property
    def pupil_mean_x(self) -> TrajectoryND:
        pupil_points_x = []
        for landmark_name in self.landmarks:
            if "p" in landmark_name:
                point_idx = self.landmarks[landmark_name]
                pupil_points_x.append(self.pixel_trajectories.to_array()[:, point_idx, 0])
        return np.nanmean(np.array(pupil_points_x), axis=0)

    @property
    def pupil_mean_y(self) -> TrajectoryND:
        pupil_points_y = []
        for landmark_name in self.landmarks:
            if "p" in landmark_name:
                point_idx = self.landmarks[landmark_name]
                pupil_points_y.append(self.pixel_trajectories.to_array()[:, point_idx, 1])
        return np.nanmean(np.array(pupil_points_y), axis=0)
