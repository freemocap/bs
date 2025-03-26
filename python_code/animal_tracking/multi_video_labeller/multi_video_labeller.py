import logging
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from python_code.animal_tracking.multi_video_labeller.helpers.data_handler import DataHandler, DataHandlerConfig
from python_code.animal_tracking.multi_video_labeller.helpers.multi_video_processor import MultiVideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
MAX_WINDOW_SIZE = (1920, 1080)
ZOOM_STEP = 1.1
ZOOM_MIN = 1.0
ZOOM_MAX = 10.0
POSITION_EPSILON = 1e-6  # Small threshold for position changes


class MultiVideoLabeller(BaseModel):
    video_folder: str
    max_window_size: tuple[int, int]
    video_processor: MultiVideoProcessor
    frame_number: int = 0
    is_playing: bool = True
    step_size: int = 1
    zoom_scale: float = 1.0
    zoom_center: tuple[int, int] = (0, 0)
    active_cell: tuple[int, int] | None = None  # Track which cell the mouse is in

    @classmethod
    def create(cls, video_folder: str, max_window_size: tuple[int, int] = MAX_WINDOW_SIZE, data_handler_config: str | Path = "python_code/animal_tracking/multi_video_labeller/helpers/face_points.json"):
        video_processor = MultiVideoProcessor.from_folder(video_folder, max_window_size, data_handler_config)
        return cls(video_processor=video_processor,
                   video_folder=video_folder,
                   max_window_size=max_window_size)

    @property
    def frame_count(self):
        return self.video_processor.frame_count

    def _handle_keypress(self, key: int):
        if key == 27:  # ESC
            return False
        elif key == 32:  # spacebar
            self.is_playing = not self.is_playing
        elif key == ord('r'):  # reset zoom
            for video in self.video_processor.videos:
                video.zoom_state.reset()
        elif key == ord('a'):  # Left arrow
            self._jump_n_frames(-1)
        elif key == ord('d'):  # Right arrow
            self._jump_n_frames(1)
        elif key == ord('w'):  # Up arrow
            self.video_processor.move_active_point_by_index(index_change=-1)
        elif key == ord('s'):  # Down arrow
            self.video_processor.move_active_point_by_index(index_change=1)
        elif key == ord('q'):  # q
            print("q pressed")
        elif key == ord('e'):  # e
            print("e pressed")
        return True
    
    def _jump_n_frames(self, num_frames: int = 1):
        self.is_playing = False
        self.frame_number += num_frames
        self.frame_number = max(0, self.frame_number)
        self.frame_number = min(self.frame_count, self.frame_number)

    def _mouse_callback(self, event, x, y, flags, param):
        # Calculate which grid cell contains the mouse
        cell_x = x // self.video_processor.grid_parameters.cell_width
        cell_y = y // self.video_processor.grid_parameters.cell_height

        # Store current cell
        self.active_cell = (cell_x, cell_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.video_processor.handle_clicks(x, y, self.frame_number, auto_next_point=False)
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Only zoom if mouse is within a valid video cell
            video_idx = cell_y * self.video_processor.grid_parameters.columns + cell_x
            if video_idx < len(self.video_processor.videos):
                video = self.video_processor.videos[video_idx]
                scaling = video.scaling_params
                zoom_state = video.zoom_state

                # Get relative position within cell
                cell_relative_x = x % self.video_processor.grid_parameters.cell_width
                cell_relative_y = y % self.video_processor.grid_parameters.cell_height

                if zoom_state.scale > 1.0:
                    # Calculate current visible region
                    relative_x = (zoom_state.center_x - scaling.x_offset) / scaling.scaled_width
                    relative_y = (zoom_state.center_y - scaling.y_offset) / scaling.scaled_height

                    # Calculate center point in zoomed coordinates
                    zoomed_width = int(scaling.scaled_width * zoom_state.scale)
                    zoomed_height = int(scaling.scaled_height * zoom_state.scale)

                    center_x = int(relative_x * zoomed_width)
                    center_y = int(relative_y * zoomed_height)

                    # Calculate visible region bounds
                    x1 = max(0, center_x - scaling.scaled_width // 2)
                    y1 = max(0, center_y - scaling.scaled_height // 2)

                    # Convert mouse position to be relative to current visible region
                    new_center_x = x1 + (cell_relative_x - scaling.x_offset)
                    new_center_y = y1 + (cell_relative_y - scaling.y_offset)

                    if abs(new_center_x - video.zoom_state.center_x) > POSITION_EPSILON:
                        video.zoom_state.center_x = scaling.x_offset + int(new_center_x / zoom_state.scale)
                    if abs(new_center_y - video.zoom_state.center_y) > POSITION_EPSILON:
                        video.zoom_state.center_y = scaling.y_offset + int(new_center_y / zoom_state.scale)
                else:
                    # Initial zoom, use raw cell coordinates
                    video.zoom_state.center_x = cell_relative_x
                    video.zoom_state.center_y = cell_relative_y

                # Update zoom scale
                if flags > 0:  # Scroll up to zoom in
                    video.zoom_state.scale *= 1.1
                else:  # Scroll down to zoom out
                    video.zoom_state.scale /= 1.1

                # Keep zoom scale within reasonable limits
                video.zoom_state.scale = np.clip(video.zoom_state.scale, 1.0, 10.0)

    def run(self):
        """Run the video grid viewer."""
        cv2.namedWindow(self.video_folder, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.video_folder, *self.max_window_size)
        cv2.setMouseCallback(self.video_folder,
                             lambda event, x, y, flags, param: self._mouse_callback(event, x, y, flags, param))

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keypress(key):
                    break
                grid_image = self.video_processor.create_grid_image(self.frame_number, annotate_images=True)
                cv2.imshow(str(self.video_folder), grid_image)
                if self.is_playing:
                    self.frame_number = (self.frame_number + self.step_size) % self.frame_count
        finally:
            self.video_processor.close()


if __name__ == '__main__':
    DEMO_VIDEO_PATH = Path.home() / "freemocap_data/recording_sessions/freemocap_test_data/synchronized_videos"
    if not DEMO_VIDEO_PATH.exists():
        logger.error(f"Demo video path not found: {DEMO_VIDEO_PATH}")
        exit(1)
    try:
        viewer = MultiVideoLabeller.create(video_folder=str(DEMO_VIDEO_PATH))
        viewer.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
