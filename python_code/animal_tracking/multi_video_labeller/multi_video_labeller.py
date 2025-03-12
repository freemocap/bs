import logging
from pathlib import Path

import cv2
from pydantic import BaseModel

from python_code.animal_tracking.multi_video_labeller.helpers.multi_video_processor import MultiVideoHandler

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
COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 0)
]


class MultiVideoLabeller(BaseModel):
    video_folder: str
    max_window_size: tuple[int, int] = MAX_WINDOW_SIZE

    def setup(self):
        self.video_processor = MultiVideoHandler.from_folder(self.video_folder, self.max_window_size)
        self.videos = self.video_processor.load_videos()

        # State
        self.frame_number = 0
        self.is_playing = False
        self.current_video_idx = -1

    def run(self):
        """Run the video grid viewer."""
        cv2.namedWindow(str(self.video_folder), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(str(self.video_folder), *self.max_window_size)

        try:
            while True:
                grid_image = self._create_grid_image()
                cv2.imshow(str(self.video_folder), grid_image)

                key = cv2.waitKey(25) & 0xFF
                if not self._handle_key(key):
                    break

        finally:
            self._cleanup()


if __name__ == '__main__':
    DEMO_VIDEO_PATH = Path.home() / "freemocap_data/recording_sessions/freemocap_test_data/synchronized_videos"
    if not DEMO_VIDEO_PATH.exists():
        logger.error(f"Demo video path not found: {DEMO_VIDEO_PATH}")
        exit(1)
    try:
        viewer = MultiVideoLabeller(video_folder=str(DEMO_VIDEO_PATH))
        viewer.setup()
        viewer.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
