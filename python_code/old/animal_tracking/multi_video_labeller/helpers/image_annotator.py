import cv2
import numpy as np
from pydantic import BaseModel

from python_code.animal_tracking.multi_video_labeller.helpers.video_models import ClickData


class ImageAnnotatorConfig(BaseModel):
    keypoint_type: int = cv2.MARKER_DIAMOND
    keypoint_size: int = 20
    keypoint_thickness: int = 2

    text_color: tuple[int, int, int] = (215, 115, 40)
    text_size: float = 1.25
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX

    show_help: bool = False
    tracked_points: list[str] = []


class ImageAnnotator(BaseModel):
    config: ImageAnnotatorConfig = ImageAnnotatorConfig()

    @staticmethod
    def draw_doubled_text(image: np.ndarray,
                          text: str,
                          x: int,
                          y: int,
                          font_scale: float,
                          color: tuple[int, int, int],
                          thickness: int):
        for line in text.split("\n"):
            if line:
                cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness * 3)
                cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            y += 40

    @property
    def short_help_text(self) -> str:
        return "H for Help, \nEsc to Quit"
    
    @property
    def show_help_text(self) -> str:
        return (
            "Click on the video to add a point.\n"
            "Use 'a' and 'd' to navigate through frames.\n"
            "Use 'w' and 's' to change the active point.\n"
            "Press 'u' to clear the data for active point\n" \
            "for the current frame.\n"
            "Press 'H' to toggle help text.\n"
            "Press 'Esc' to quit.\n"
            "You will be prompted to save the data in the terminal."
        )

    @property
    def colors(self) -> dict[str, tuple[int, int, int]]:
        np.random.seed(42)
    
        hues = np.linspace(0, 1, len(self.config.tracked_points), endpoint=False)
        
        # Convert HSV to RGB
        rgb_values = []
        for hue in hues:
            # Using saturation=0.7 and value=0.95 for vibrant but not overwhelming colors
            hsv = np.array([hue, 0.7, 0.95])
            rgb = self._hsv_to_rgb(hsv)
            rgb_values.append(tuple(map(int, rgb * 255)))

        colors = {}
        for tracked_point, color in zip(self.config.tracked_points, rgb_values):
            colors[tracked_point] = color
        
        return colors

    def annotate_image(
            self,
            image: np.ndarray,
            frame_number: int,
            click_data: dict[str, ClickData] | None = None,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        text_offset = int(image_height * 0.05)

        if click_data is None:
            click_data = {}
        # Copy the original image for annotation
        annotated_image = image.copy()

        # Draw a keypoint for each click
        for active_point, click in click_data.items():
            cv2.drawMarker(
                annotated_image,
                position=(click.x, click.y),
                color=self.colors[active_point],
                keypointType=self.config.keypoint_type,
                keypointSize=self.config.keypoint_size,
                thickness=self.config.keypoint_thickness,
            )
        self.draw_doubled_text(image=annotated_image,
                               text=f"Frame Number: {frame_number}, \nClick Count: {len(click_data)}",
                               x=text_offset,
                               y=text_offset,
                               font_scale=self.config.text_size,
                               color=self.config.text_color,
                               thickness=self.config.text_thickness)

        return annotated_image
    
    def annotate_grid(self, image: np.ndarray, active_point: str) -> np.ndarray:
        if self.config.show_help:
            help_text = self.show_help_text
        else:
            help_text = self.short_help_text
        self.draw_doubled_text(image=image,
                               text=f"Active Point:\n{active_point}",
                               x=10,
                               y=(image.shape[0] // 10) * 1,
                               font_scale=self.config.text_size,
                               color=self.config.text_color,       
                               thickness=self.config.text_thickness)
        self.draw_doubled_text(image=image,
                               text=help_text,
                               x=10,
                               y=(image.shape[0] // 10) * 5,
                               font_scale=self.config.text_size,
                               color=self.config.text_color,
                               thickness=self.config.text_thickness)
        return image
    
    @staticmethod
    def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        """Convert HSV color to RGB."""
        h, s, v = hsv
        hi = int(h * 6.) % 6
        f = h * 6. - int(h * 6.)
        p = v * (1. - s)
        q = v * (1. - f * s)
        t = v * (1. - (1. - f) * s)
        
        if hi == 0:
            return np.array([v, t, p])
        elif hi == 1:
            return np.array([q, v, p])
        elif hi == 2:
            return np.array([p, v, t])
        elif hi == 3:
            return np.array([p, q, v])
        elif hi == 4:
            return np.array([t, p, v])
        else:
            return np.array([v, p, q])
