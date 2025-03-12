import csv
from typing import List

from pydantic import BaseModel

from .video_models import ClickData, VideoPlaybackState, GridParameters


class ClickHandler(BaseModel):
    """Handles mouse click processing and data recording."""

    output_path: str
    videos: List[VideoPlaybackState]
    grid_parameters: GridParameters
    clicks: dict[str, list[ClickData]] = {}
    csv_ready: bool = False

    def _setup_csv(self):
        """Initialize the CSV output file."""
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_number',
                'video_index',
                'window_x',
                'window_y',
                'video_x',
                'video_y'
            ])

    def process_click(self, x: int, y: int, frame_number: int) -> ClickData | None:
        """Process a mouse click and return click data if valid."""
        # Calculate which grid cell was clicked
        cell_x = x // self.grid_parameters.cell_width
        cell_y = y // self.grid_parameters.cell_height

        video_idx = cell_y * self.grid_parameters.columns + cell_x
        if video_idx >= len(self.videos):
            return None

        video = self.videos[video_idx]

        # Get position within cell
        cell_x = x % self.grid_parameters.cell_width
        cell_y = y % self.grid_parameters.cell_height

        # Convert to video coordinates
        scaling = video.scaling_params
        if (scaling.x_offset <= cell_x < scaling.x_offset + scaling.scaled_width and
                scaling.y_offset <= cell_y < scaling.y_offset + scaling.scaled_height):
            video_x = int((cell_x - scaling.x_offset) / scaling.scale)
            video_y = int((cell_y - scaling.y_offset) / scaling.scale)
        else:
            video_x = video_y = -1

        click_data = ClickData(
            window_x=x,
            window_y=y,
            video_x=video_x,
            video_y=video_y,
            frame_number=frame_number,
            video_index=video_idx
        )

        self._record_click(click_data , video.name)
        return click_data

    def _record_click(self, click: ClickData, video_name: str):
        """Record click data to CSV."""
        if not self.csv_ready:
            self._setup_csv()
            self.csv_ready = True
        if not video_name in self.clicks:
            self.clicks[video_name] = []
        self.clicks[video_name].append(click)
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                click.frame_number,
                click.video_index,
                click.window_x,
                click.window_y,
                click.video_x,
                click.video_y
            ])

    def get_clicks_by_video_name(self, video_name: str) -> List[ClickData]:
        """Get all clicks for a specific video."""
        return self.clicks.get(video_name, [])
