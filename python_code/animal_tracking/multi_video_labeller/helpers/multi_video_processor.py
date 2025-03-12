import logging
import mimetypes
import os
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from .click_handler import ClickHandler
from .image_annotator import ImageAnnotator
from .video_models import VideoMetadata, VideoPlaybackState, VideoScalingParameters, GridParameters

logger = logging.getLogger(__name__)

class MultiVideoProcessor(BaseModel):
    """Handles video loading and image processing."""

    video_folder: str
    videos: list[VideoPlaybackState] = []
    click_handler: ClickHandler
    grid_parameters: GridParameters
    image_annotator: ImageAnnotator = ImageAnnotator()
    frame_count: int

    @classmethod
    def from_folder(cls, video_folder: str, max_window_size: tuple[int, int]):

        videos, grid_parameters, frame_count= cls._load_videos(video_folder, max_window_size)

        return cls(
            video_folder=video_folder,
            videos=videos,
            click_handler=ClickHandler(output_path=str(Path(video_folder).parent / "clicks.csv", ),
                                       grid_parameters=grid_parameters,
                                       videos=videos
                                       ),
            grid_parameters=grid_parameters,
            frame_count=frame_count
        )

    @classmethod
    def _load_videos(cls, video_folder: str,max_window_size:tuple[int,int]) -> tuple[list[VideoPlaybackState], GridParameters, int]:
        """Load all videos from the folder and calculate their scaling parameters."""
        video_files = [f for f in os.listdir(video_folder) if mimetypes.guess_type(f)[0].startswith('video')]
        video_paths = [str(Path(video_folder) / filename) for filename in video_files]

        grid_parameters = GridParameters.calculate(video_count=len(video_paths),
                                                   max_window_size=max_window_size
                                                   )
        if not video_files:
            raise ValueError(f"No videos found in {video_folder}")

        videos = []
        image_counts = set()

        for video_name, video_path in zip(video_files, video_paths):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            metadata = VideoMetadata(
                path=video_path,
                name=video_name,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )

            image_counts.add(metadata.frame_count)

            scaling_params = cls._calculate_scaling_parameters(
                metadata.width,
                metadata.height,
                grid_parameters.cell_size
            )

            videos.append(VideoPlaybackState(
                metadata=metadata,
                cap=cap,
                scaling_params=scaling_params
            ))

        if len(image_counts) > 1:
            raise ValueError("All videos must have the same number of images")

        return videos, grid_parameters, image_counts.pop()

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

    def prepare_single_image(self, image: np.ndarray, frame_number: int) -> np.ndarray:
        """Process a video image - resize and add overlays."""
        if image is None:
            return np.zeros(self.grid_params.cell_size + (3,), dtype=np.uint8)

        # Resize image
        resized = cv2.resize(image, (self.scaled_width, self.scaled_height))

        # Create padded image
        padded = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8)
        padded[self.y_offset:self.y_offset + self.scaled_height,
        self.x_offset:self.x_offset + self.scaled_width] = resized

        # Add frame number
        cv2.putText(padded, f"Frame {frame_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return padded

    def create_grid_image(self, frame_number: int, annotate_images:bool) -> np.ndarray:
        """Create a grid of video images."""
        # Create empty grid
        grid_image = np.zeros((self.grid_parameters.total_height,
                               self.grid_parameters.total_width,
                               3), dtype=np.uint8)

        for video_index, video in enumerate(self.videos):
            # Calculate grid position
            row = video_index // self.grid_parameters.rows
            col = video_index % self.grid_parameters.columns

            # Read image
            video.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = video.cap.read()

            if success:
                if annotate_images:
                    image = self.image_annotator.annotate_image(image,
                                                                click_data=self.click_handler.get_clicks_by_video_name(video.name),
                                                                camera_index=video_index,
                                                                frame_number=frame_number
                                                                )
                # Resize image
                scaled_image = cv2.resize(image,
                                          (video.scaling_params.scaled_width,
                                           video.scaling_params.scaled_height))

                # Calculate position in grid
                y_start = row * self.grid_parameters.cell_height + video.scaling_params.y_offset
                x_start = col * self.grid_parameters.cell_width + video.scaling_params.x_offset

                # Place image in grid
                grid_image[y_start:y_start + video.scaling_params.scaled_height,
                x_start:x_start + video.scaling_params.scaled_width] = scaled_image

        return grid_image

    def close(self):
        logger.info("VideoHandler closing")