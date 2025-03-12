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
    max_window_size: tuple[int, int]
    video_handler: MultiVideoHandler
    frame_number: int = 0
    is_playing: bool = True
    step_size: int = 1

    @classmethod
    def create(cls, video_folder: str, max_window_size: tuple[int, int] = MAX_WINDOW_SIZE):
        return cls(video_handler=MultiVideoHandler.from_folder(video_folder, max_window_size),
                   video_folder=video_folder,
                   max_window_size=max_window_size)

    @property
    def frame_count(self):
        return self.video_handler.frame_count

    def _handle_keypress(self, key: int):
        if key == ord('q') or key == 27:  # q or ESC
            return False
        elif key == 32:  # spacebar
            self.is_playing = not self.is_playing
        return True

    def run(self):
        """Run the video grid viewer."""
        cv2.namedWindow(str(self.video_folder), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(str(self.video_folder), *self.max_window_size)

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keypress(key):
                    break

                if self.is_playing:
                    grid_image = self.video_handler.create_grid_image(self.frame_number)
                    cv2.imshow(str(self.video_folder), grid_image)
                    self.frame_number = (self.frame_number + self.step_size) % self.frame_count
        finally:
            self.video_handler.close()


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
