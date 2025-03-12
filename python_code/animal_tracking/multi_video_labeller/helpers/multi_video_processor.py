import mimetypes
import os
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from .click_handler import ClickHandler
from .models import VideoMetadata, VideoPlaybackState, VideoScalingParameters, GridParameters


class MultiVideoHandler(BaseModel):
    """Handles video loading and frame processing."""

    video_folder: str
    videos: list[VideoPlaybackState] = []
    click_handler: ClickHandler

    @classmethod
    def from_folder(cls, video_folder: str, max_window_size: tuple[int, int] ):
        videos = cls._load_videos(video_folder)
        return cls(
            video_folder=video_folder,
            videos=videos,
            click_handler=ClickHandler(output_path=str(Path(video_folder).parent / "clicks.csv", ),
                                       grid_parameters=GridParameters.calculate(video_count=len(videos),
                                                                                max_window_size=max_window_size
                                                                                ),
                                       videos=videos
                                       )
        )

    @classmethod
    def _load_videos(cls, video_folder: str) -> list[VideoPlaybackState]:
        """Load all videos from the folder and calculate their scaling parameters."""
        video_files = [f for f in os.listdir(video_folder) if mimetypes.guess_type(f)[0].startswith('video')]

        if not video_files:
            raise ValueError(f"No videos found in {video_folder}")

        videos = []
        frame_counts = set()

        for idx, video_path in enumerate(video_files):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            metadata = VideoMetadata(
                path=video_path,
                name=Path(video_path).name,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                index=idx
            )

            frame_counts.add(metadata.frame_count)

            scaling_params = cls._calculate_scaling_parameters(
                metadata.width,
                metadata.height,
                cls.grid_params.cell_size
            )

            videos.append(VideoPlaybackState(
                metadata=metadata,
                cap=cap,
                scaling_params=scaling_params
            ))

        if len(frame_counts) > 1:
            raise ValueError("All videos must have the same number of frames")

        return videos

    @staticmethod
    def _calculate_scaling_parameters(
            orig_width: int,
            orig_height: int,
            cell_size: tuple[int, int]
    ) -> VideoScalingParameters:
        """Calculate scaling parameters for a video to fit in a grid cell."""
        cell_width, cell_height = cell_size

        # Calculate scale factor preserving aspect ratio
        scale = min(cell_width / orig_width, cell_height / orig_height)

        scaled_width = int(orig_width * scale)
        scaled_height = int(orig_height * scale)

        # Calculate offsets to center the video
        x_offset = (cell_width - scaled_width) // 2
        y_offset = (cell_height - scaled_height) // 2

        return VideoScalingParameters(
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset,
            scaled_width=scaled_width,
            scaled_height=scaled_height
        )

    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process a video frame - resize and add overlays."""
        if frame is None:
            return np.zeros(self.grid_params.cell_size + (3,), dtype=np.uint8)

        # Resize frame
        resized = cv2.resize(frame, (self.scaled_width, self.scaled_height))

        # Create padded frame
        padded = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8)
        padded[self.y_offset:self.y_offset + self.scaled_height,
        self.x_offset:self.x_offset + self.scaled_width] = resized

        # Add frame number
        cv2.putText(padded, f"Frame {frame_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return padded
