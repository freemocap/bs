import csv
import logging
import mimetypes
import os
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 0)
]


class ClickData(BaseModel):
    window_x: int
    window_y: int
    video_x: int
    video_y: int
    frame_number: int

class VideoScalingParameters(BaseModel):
    scale: float
    x_offset: int
    y_offset: int

class VideoPlaybackObject(BaseModel):
    width: int
    height: int
    frame_count: int
    path: str
    name: str
    index: int
    cap: cv2.VideoCapture
    current_frame: np.ndarray = None
    processed_frame: np.ndarray = None
    scaling_parameters: VideoScalingParameters | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    @property
    def original_size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def scaled_width(self) -> int:
        return int(self.width * self.scaling_parameters.scale)

    @property
    def scaled_height(self) -> int:
        return int(self.height * self.scaling_parameters.scale)

    @property
    def scaled_size(self) -> tuple[int, int]:
        return self.scaled_width, self.scaled_height


class VideoGridViewer(BaseModel):
    video_folder: str
    output_csv: str
    max_window_size: tuple[int, int] = (1920, 1080)
    videos: list[VideoPlaybackObject] = []
    grid_size: tuple[int, int] = (0, 0)
    cell_size: tuple[int, int] = (0, 0)
    clicks: list[ClickData] = []

    current_video_idx: int = -1
    mouse_pos: tuple[int, int] = (-1, -1)
    frame_number: int = 0
    is_playing: bool = False

    @classmethod
    def create(cls, video_folder: str):
        video_folder = str(Path(video_folder).resolve())
        output_csv = str(Path(video_folder).parent / 'clicks.csv')
        instance = cls(video_folder=video_folder, output_csv=output_csv)
        instance.load_videos()
        instance.calculate_grid()
        instance.calculate_cell_size()
        instance.setup_csv()
        return instance

    def load_videos(self):
        logger.info(f"Loading videos from {self.video_folder}")
        if not os.path.exists(self.video_folder):
            raise FileNotFoundError(f"Folder '{self.video_folder}' not found")

        # video_files = [f for f in os.listdir(self.video_folder) if f.endswith('.mp4')]
        video_files = [f for f in os.listdir(self.video_folder) if mimetypes.guess_type(f)[0].startswith('video')]

        if not video_files:
            raise ValueError("No MP4 videos found in directory")

        self.videos = []
        for filename in video_files:
            path = os.path.join(self.video_folder, filename)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {path}")
                continue

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.videos.append(VideoPlaybackObject(
                width=width,
                height=height,
                frame_count=frame_count,
                path=path,
                name=filename,
                index=len(self.videos),
                cap=cap
            ))

        if len(self.videos) == 0:
            raise ValueError("No valid videos loaded")

        # Verify frame counts
        frame_counts = [info.frame_count for info in self.videos]
        if len(set(frame_counts)) > 1:
            logger.error("Videos have different frame counts:")
            for info in self.videos:
                logger.error(f"{info.name}: {info.frame_count} frames")
            raise ValueError("Videos have different frame counts")

        logger.info(f"Successfully loaded {len(self.videos)} videos")

    def calculate_grid(self):
        num_videos = len(self.videos)
        grid_size = int(np.ceil(np.sqrt(num_videos)))
        self.grid_size = (grid_size, grid_size)
        logger.info(f"Calculated grid size: {self.grid_size}")

    def calculate_cell_size(self):
        max_width, max_height = self.max_window_size
        cols, rows = self.grid_size
        self.cell_size = (
            int(max_width / cols),
            int(max_height / rows)
        )
        logger.info(f"Cell size calculated: {self.cell_size} (width, height)")

    def setup_csv(self):
        logger.info(f"Initializing CSV output: {self.output_csv}")
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_number', 'video_index', 'window_x', 'window_y', 'video_x', 'video_y'])

    def process_frame(self,
                      frame: np.ndarray,
                      original_size: tuple[int, int],
                      frame_index: int) -> tuple[np.ndarray, VideoScalingParameters]:
        """Resize frame with aspect ratio preservation and padding"""
        cell_width, cell_height = self.cell_size
        original_width, original_height = original_size

        # Calculate scaling factor
        scale = min(cell_width / original_width, cell_height / original_height)
        scaled_width = int(original_width * scale)
        scaled_height = int(original_height * scale)

        # Resize frame
        resized = cv2.resize(frame, (scaled_width, scaled_height))

        # Create padded frame
        padded = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        x_offset = (cell_width - scaled_width) // 2
        y_offset = (cell_height - scaled_height) // 2
        padded[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized

        # Add frame number text
        cv2.putText(padded, f"Frame {frame_index}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return padded, VideoScalingParameters(
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset
        )

    def create_grid_image(self, frame_index: int) -> np.ndarray:
        grid_rows = []
        video_idx = 0

        for row in range(self.grid_size[0]):
            row_images = []
            for col in range(self.grid_size[1]):
                if video_idx < len(self.videos):
                    video = self.videos[video_idx]
                    cap = video.cap
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, video.current_frame = cap.read()
                    if ret:
                        video.processed_frame, video.scaling_parameters = self.process_frame(video.current_frame,
                                                                                             video.size,
                                                                                             frame_index)
                        row_images.append(video.processed_frame)
                    else:
                        logger.warning(f"Failed to read frame {frame_index} from {Path(video.path).name}")
                        row_images.append(np.zeros((self.cell_size[1], self.cell_size[0], 3), dtype=np.uint8))
                    video_idx += 1
                else:
                    row_images.append(np.zeros((self.cell_size[1], self.cell_size[0], 3), dtype=np.uint8))

            grid_row = np.hstack(row_images)
            grid_rows.append(grid_row)

        return np.vstack(grid_rows)

    def draw_overlays(self,
                      image: np.ndarray,
                      frame_number: int) -> np.ndarray:
        cell_w, cell_h = self.cell_size

        # Draw grid lines
        for i in range(1, self.grid_size[0]):
            cv2.line(image, (0, i * cell_h), (image.shape[1], i * cell_h), (255, 255, 255), 1)
        for i in range(1, self.grid_size[1]):
            cv2.line(image, (i * cell_w, 0), (i * cell_w, image.shape[0]), (255, 255, 255), 1)

        # Draw mouse position
        if self.current_video_idx != -1:
            color = self.colors[self.current_video_idx % len(self.colors)]
            cv2.putText(image,
                        f"Video {self.current_video_idx + 1} | Pos: {self.mouse_pos}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        #
        # # Draw clicks
        # for click in self.clicks:
        #     if click.frame == frame_number:
        #         color = self.colors[click['video_idx'] % len(self.colors)]
        #         cv2.circle(image, click['pos'], 8, color, -1)
        #         if click['original_pos'][0] != -1:
        #             cv2.putText(image, f"{click['original_pos']}",
        #                        (click['pos'][0]+10, click['pos'][1]-10),
        #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    def mouse_callback(self,
                       event: int,
                       x: int,
                       y: int,
                       flags: int,
                       param: dict):
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Mouse click detected at ({x}, {y}) in video {self.current_video_idx} (param: {param})")
            cols, rows = self.grid_size
            cell_w, cell_h = self.cell_size

            grid_x = x // cell_w
            grid_y = y // cell_h
            video_idx = grid_y * cols + grid_x

            if video_idx < len(self.videos):
                video = self.videos[video_idx]

                # Get scaling parameters
                scale, x_offset, y_offset, scaled_width, scaled_height = video.scaling_parameters
                # Calculate original coordinates
                cell_x = x % cell_w
                cell_y = y % cell_h

                if (x_offset <= cell_x < x_offset + scaled_width and
                        y_offset <= cell_y < y_offset + scaled_height):
                    orig_x = int((cell_x - x_offset) / scale)
                    orig_y = int((cell_y - y_offset) / scale)
                else:
                    orig_x, orig_y = -1, -1

                self.clicks.append(ClickData(x=x, y=y, frame=param['frame_number']))

                # Write to CSV
                with open(self.output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        param['frame_number'],
                        video_idx,
                        orig_x,
                        orig_y,
                        x,
                        y,
                    ])
                logger.info(
                    f"Click recorded: Video {video_idx} - Window Coordinates ({orig_x}, {orig_y}) - Video Coordinates ({x}, {y})")

    def run(self):
        logger.info("Starting video grid viewer")
        cv2.namedWindow(self.video_folder, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.video_folder, *self.max_window_size)
        frame_count = self.videos[0].frame_count

        try:
            while True:
                grid_image = self.create_grid_image(self.frame_number)
                annotated_image = self.draw_overlays(grid_image.copy(), self.frame_number)
                cv2.imshow(self.video_folder, annotated_image)
                cv2.setMouseCallback(self.video_folder, self.mouse_callback, {'frame_number': self.frame_number})

                key = cv2.waitKey(25) & 0xFF
                if key in [27, ord('q')]:  # ESC or Q to quit
                    logger.info("User requested exit")
                    break
                elif key == ord(' '):  # Space to toggle play/pause
                    self.is_playing = not self.is_playing
                elif key == ord('r'):  # Reset to frame 0
                    self.is_playing = False
                    self.frame_number = 0
                elif key == 83:  # Right arrow
                    self.is_playing = False
                    self.frame_number = min(self.frame_number + 1, frame_count - 1)
                elif key == 81:  # Left arrow
                    self.is_playing = False
                    self.frame_number = max(self.frame_number - 1, 0)

                if self.is_playing:
                    self.frame_number += 1
                    if self.frame_number >= frame_count:
                        self.frame_number = frame_count - 1
                        self.is_playing = False

        finally:
            logger.info("Cleaning up resources")
            for video in self.videos:
                video.cap.release()
            cv2.destroyAllWindows()
            logger.info("Processing complete")


if __name__ == '__main__':
    DEMO_VIDEO_PATH = Path.home() / "freemocap_data/recording_sessions/freemocap_test_data/synchronized_videos"
    if not DEMO_VIDEO_PATH.exists():
        logger.error(f"Demo video path not found: {DEMO_VIDEO_PATH}")
        exit(1)
    try:
        viewer = VideoGridViewer.create(video_folder=str(DEMO_VIDEO_PATH))
        viewer.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
