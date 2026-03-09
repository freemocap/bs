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
        scaling = video.scaling_params
        zoom_state = video.zoom_state

        # Get position within cell
        relative_cell_x = x % self.grid_parameters.cell_width
        relative_cell_y = y % self.grid_parameters.cell_height

        # Check if click is within the actual video area
        if (scaling.x_offset <= relative_cell_x < scaling.x_offset + scaling.scaled_width and
                scaling.y_offset <= relative_cell_y < scaling.y_offset + scaling.scaled_height):

            # Get position relative to video area (not cell)
            relative_x = relative_cell_x - scaling.x_offset
            relative_y = relative_cell_y - scaling.y_offset

            if zoom_state.scale > 1.0:
                # Calculate the center point in the zoomed image
                center_relative_x = (zoom_state.center_x - scaling.x_offset) / scaling.scaled_width
                center_relative_y = (zoom_state.center_y - scaling.y_offset) / scaling.scaled_height

                # Calculate the offset in the zoomed space
                zoomed_width = int(scaling.scaled_width * zoom_state.scale)
                zoomed_height = int(scaling.scaled_height * zoom_state.scale)

                center_zoomed_x = int(center_relative_x * zoomed_width)
                center_zoomed_y = int(center_relative_y * zoomed_height)

                # Calculate top-left corner of visible area
                x1 = max(0, center_zoomed_x - scaling.scaled_width // 2)
                y1 = max(0, center_zoomed_y - scaling.scaled_height // 2)

                # Adjust if at bounds
                if x1 + scaling.scaled_width > zoomed_width:
                    x1 = zoomed_width - scaling.scaled_width
                if y1 + scaling.scaled_height > zoomed_height:
                    y1 = zoomed_height - scaling.scaled_height

                # Convert click position to zoomed space
                zoomed_click_x = x1 + relative_x
                zoomed_click_y = y1 + relative_y

                # Convert back to original image coordinates
                video_x = int(zoomed_click_x / zoom_state.scale * video.metadata.width / scaling.scaled_width)
                video_y = int(zoomed_click_y / zoom_state.scale * video.metadata.height / scaling.scaled_height)
            else:
                # For non-zoomed image, convert directly to original coordinates
                video_x = int(relative_x * video.metadata.width / scaling.scaled_width)
                video_y = int(relative_y * video.metadata.height / scaling.scaled_height)
        else:
            video_x = video_y = -1

        click_data = ClickData(
            window_x=x,
            window_y=y,
            video_x=video_x,
            video_y=video_y,
            frame_number=frame_number,
            video_index=video_idx,
        )

        # self._record_click(click_data, video.name)
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
