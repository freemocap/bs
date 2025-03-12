import cv2
import numpy as np

from pydantic import BaseModel

from python_code.animal_tracking.multi_video_labeller.helpers.video_models import ClickData

class ImageAnnotatorConfig(BaseModel):

    marker_type: int = cv2.MARKER_DIAMOND
    marker_size: int = 10
    marker_thickness: int = 2
    marker_color: tuple[int, int, int] = (255, 0, 255)

    text_color: tuple[int, int, int] = (215, 115, 40)
    text_size: float = .5
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


class ImageAnnotator(BaseModel):
    config: ImageAnnotatorConfig = ImageAnnotatorConfig()
    @staticmethod
    def draw_doubled_text(image: np.ndarray,
                          text: str,
                          x: int,
                          y: int,
                          font_scale: float,
                          color: tuple[int, int, int],
                          thickness:int):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness * 3)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def annotate_image(
            self,
            image: np.ndarray,
            click_data: list[ClickData] | None = None,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        text_offset = int(image_height * 0.01)

        if click_data is None:
            click_data = []
        # Copy the original image for annotation
        annotated_image = image.copy()

        # Draw a marker for each click
        for click in click_data:
                cv2.drawMarker(
                    annotated_image,
                    position= (click.x, click.y),
                    color=self.config.marker_color,
                    markerType=self.config.corner_marker_type,
                    markerSize=self.config.marker_size,
                    thickness=self.config.marker_thickness,
                )
        self.draw_doubled_text(image=annotated_image,
                               text=f"Click Count: {len(click_data)}",
                               x=50,
                               y=50,
                               font_scale=self.config.text_size,
                               color=self.config.text_color,
                               thickness=self.config.text_thickness)

        return annotated_image
