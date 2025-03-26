import logging
import mimetypes
import os
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from python_code.animal_tracking.multi_video_labeller.helpers.data_handler import DataHandler, DataHandlerConfig

from .click_handler import ClickHandler
from .image_annotator import ImageAnnotator
from .video_models import VideoMetadata, VideoPlaybackState, VideoScalingParameters, GridParameters

logger = logging.getLogger(__name__)
from copy import deepcopy


class MultiVideoProcessor(BaseModel):
    video_folder: str
    videos: list[VideoPlaybackState] = []
    click_handler: ClickHandler
    data_handler: DataHandler
    grid_parameters: GridParameters
    image_annotator: ImageAnnotator = ImageAnnotator()
    frame_count: int

    @classmethod
    def from_folder(cls, video_folder: str, max_window_size: tuple[int, int], data_handler_config: str | Path):

        videos, grid_parameters, frame_count = cls._load_videos(video_folder, max_window_size)

        data_handler = DataHandler.from_config(
            DataHandlerConfig.from_config_file(videos=videos, config_path=data_handler_config)
        )

        return cls(
            video_folder=video_folder,
            videos=videos,
            click_handler=ClickHandler(output_path=str(Path(video_folder).parent / "clicks.csv", ),
                                       grid_parameters=grid_parameters,
                                       videos=videos
                                       ),
            data_handler=data_handler,
            grid_parameters=grid_parameters,
            frame_count=frame_count
        )

    @classmethod
    def _load_videos(cls, video_folder: str, max_window_size: tuple[int, int]) -> tuple[
        list[VideoPlaybackState], GridParameters, int]:
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

    def prepare_single_image(self, image: np.ndarray, frame_number: int, scaling_params: VideoScalingParameters) -> np.ndarray:
        """Process a video image - resize and add overlays."""
        if image is None:
            return np.zeros(self.grid_parameters.cell_size + (3,), dtype=np.uint8)
        
        # Resize image
        resized = cv2.resize(image, (scaling_params.scaled_width, scaling_params.scaled_height))

        # Create padded image
        padded = np.zeros((self.grid_parameters.cell_height, self.grid_parameters.cell_width, 3), dtype=np.uint8)
        padded[scaling_params.y_offset:scaling_params.y_offset + scaling_params.scaled_height,
        scaling_params.x_offset:scaling_params.x_offset + scaling_params.scaled_width] = resized

        # Add frame number
        cv2.putText(padded, f"Frame {frame_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return padded
    
    def handle_clicks(self, x: int, y: int, frame_number: int, auto_next_point: bool = False):
        click_data = self.click_handler.process_click(x, y, frame_number)
        if click_data is None:
            return
        self.data_handler.update_dataframe(click_data)

        if auto_next_point:
            self.data_handler.move_active_point_by_index(index_change=1)

    def move_active_point_by_index(self, index_change: int):
        self.data_handler.move_active_point_by_index(index_change=index_change)


    def create_grid_image(self, frame_number: int, annotate_images: bool) -> np.ndarray:
        """Create a grid of video images."""
        # Create a deep copy of video states to prevent race conditions
        video_states = [deepcopy(video.zoom_state) for video in self.videos]

        grid_image = np.zeros((self.grid_parameters.total_height,
                               self.grid_parameters.total_width,
                               3), dtype=np.uint8)

        for video_index, (video, zoom_state) in enumerate(zip(self.videos, video_states)):
            # Calculate grid position
            row = video_index // self.grid_parameters.columns
            col = video_index % self.grid_parameters.columns

            # Read image
            video.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = video.cap.read()

            if success:
                if annotate_images:
                    image = self.image_annotator.annotate_image(
                        image,
                        click_data=self.data_handler.get_data_by_video_frame(video_index=video_index, frame_number=frame_number),
                        camera_index=video_index,
                        frame_number=frame_number
                    )

                if zoom_state.scale > 1.0:
                    # Calculate zoomed dimensions
                    zoomed_width = int(video.scaling_params.scaled_width * zoom_state.scale)
                    zoomed_height = int(video.scaling_params.scaled_height * zoom_state.scale)

                    # Resize image to zoomed size
                    zoomed = cv2.resize(image, (zoomed_width, zoomed_height))

                    # Calculate the relative position within the actual image area
                    relative_x = (zoom_state.center_x - video.scaling_params.x_offset) / video.scaling_params.scaled_width
                    relative_y = (zoom_state.center_y - video.scaling_params.y_offset) / video.scaling_params.scaled_height
                    # Calculate the center point in the zoomed image
                    center_x = int(relative_x * zoomed_width)
                    center_y = int(relative_y * zoomed_height)

                    # Calculate extraction region centered on this point
                    x1 = max(0, center_x - video.scaling_params.scaled_width // 2)
                    y1 = max(0, center_y - video.scaling_params.scaled_height // 2)
                    x2 = min(zoomed_width, x1 + video.scaling_params.scaled_width)
                    y2 = min(zoomed_height, y1 + video.scaling_params.scaled_height)

                    # Adjust x1,y1 if x2,y2 are at their bounds
                    if x2 == zoomed_width:
                        x1 = zoomed_width - video.scaling_params.scaled_width
                    if y2 == zoomed_height:
                        y1 = zoomed_height - video.scaling_params.scaled_height

                    # Extract visible region
                    scaled_image = zoomed[y1:y2, x1:x2]

                else:
                    # Normal scaling without zoom
                    scaled_image = cv2.resize(image,
                                              (video.scaling_params.scaled_width,
                                               video.scaling_params.scaled_height))

                # Calculate position in grid
                y_start = row * self.grid_parameters.cell_height + video.scaling_params.y_offset
                x_start = col * self.grid_parameters.cell_width + video.scaling_params.x_offset

                # Place image in grid
                try:
                    grid_image[y_start:y_start + scaled_image.shape[0],
                    x_start:x_start + scaled_image.shape[1]] = scaled_image
                except ValueError as e:
                    logger.error(f"Error placing image in grid: {e}")

        return self.image_annotator.annotate_grid(image=grid_image, active_point=self.data_handler.active_point)

    def close(self):
        """Clean up resources."""
        logger.info("VideoHandler closing")
        for video in self.videos:
            video.cap.release()

        while True:
            save_data = input("Save data? (yes/no): ")
            if save_data == "yes" or save_data == "y":
                save_path = Path(self.video_folder).parent / "output.csv"
                self.data_handler.save_csv(output_path=str(save_path))
                break   
            else:
                confirmation = input("Are you sure? Type 'yes' to confirm data will be discarded: ")
                if confirmation == "yes":
                    logger.info("Data not saved.")
                    break

