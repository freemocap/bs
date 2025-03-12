from typing import Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict


class VideoScalingParameters(BaseModel):
    """Parameters for scaling and positioning video frames in the grid."""
    scale: float
    x_offset: int
    y_offset: int
    scaled_width: int
    scaled_height: int

class VideoMetadata(BaseModel):
    """Metadata about a video file."""
    path: str
    name: str
    width: int
    height: int
    frame_count: int

class VideoPlaybackState(BaseModel):
    """Current state of video playback."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: VideoMetadata
    cap: cv2.VideoCapture
    current_frame: Optional[np.ndarray] = None
    processed_frame: Optional[np.ndarray] = None
    scaling_params: Optional[VideoScalingParameters] = None

    @property
    def name(self) -> str:
        return self.metadata.name

class ClickData(BaseModel):
    """Data associated with a mouse click."""
    window_x: int
    window_y: int
    video_x: int
    video_y: int
    frame_number: int
    video_index: int

    @property
    def x(self) -> int:
        return int(self.video_x)

    @property
    def y(self) -> int:
        return int(self.video_y)

class GridParameters(BaseModel):
    """Parameters defining the video grid layout."""
    rows: int
    columns: int
    cell_width: int
    cell_height: int
    total_width: int
    total_height: int

    @property
    def cell_size(self) -> Tuple[int, int]:
        return self.cell_width, self.cell_height

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self.rows, self.columns

    @classmethod
    def calculate(cls, video_count: int, max_window_size: Tuple[int, int]) -> "GridParameters":
        """Calculate grid parameters based on video count and window constraints."""
        max_width, max_height = max_window_size

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(video_count)))

        # Calculate cell size
        cell_width = max_width // grid_size
        cell_height = max_height // grid_size

        return cls(
            rows=grid_size,
            columns=grid_size,
            cell_width=cell_width,
            cell_height=cell_height,
            total_width=cell_width * grid_size,
            total_height=cell_height * grid_size
        )