# import logging
# from abc import ABC, abstractmethod
# from typing import List
#
# import cv2
# import numpy as np
# from pydantic import BaseModel, ConfigDict
#
# from skellytracker.trackers.demo_viewers.webcam_demo_viewer import (
#     WebcamDemoViewer,
# )
#
# logger = logging.getLogger(__name__)

# class BaseImageAnnotatorConfig(BaseModel, ABC):
#     pass
#
#
# class BaseImageAnnotator(BaseModel, ABC):
#     config: BaseImageAnnotatorConfig
#     observations: BaseObservations  #make it a list to allow plotting trails, etc.
#
#     @classmethod
#     def create(cls, config: BaseImageAnnotatorConfig):
#         raise NotImplementedError("Must implement a method to create an image annotator from a config.")
#
#     @abstractmethod
#     def annotate_image(self, image: np.ndarray, latest_observation: BaseObservation) -> np.ndarray:
#         pass
#
#     @staticmethod
#     def draw_doubled_text(image: np.ndarray,
#                           text: str,
#                           x: int,
#                           y: int,
#                           font_scale: float,
#                           color: tuple[int, int, int],
#                           thickness:int):
#         cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness * 3)
#         cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
#