import cv2
import json
import numpy as np

from dataclasses import dataclass
from pathlib import Path

INTRINSICS_JSON_PATH = Path(__file__).parent / "intrinsics.json"

@dataclass
class IntrinsicsCorrector:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        include_all_pixels = 0  # 1: all pixels are retained with some extra black in margins, 0: only valid pixels are shown
        # using 0 preserves image size

        self.new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.distortion_coefficients,
            (self.width, self.height),
            include_all_pixels,
            (self.width, self.height),
        )


    @classmethod
    def from_dict(cls, dict: dict[str, np.ndarray], width: int, height: int) -> "IntrinsicsCorrector":
        return cls(
            camera_matrix=np.array(dict["camera_matrix"]),
            distortion_coefficients=np.array(dict["distortion_coefficients"]),
            width=width,
            height=height,
        )

    def correct_frame(self, frame: np.ndarray) -> np.ndarray:
        corrected_image = cv2.undistort(frame, self.camera_matrix, self.distortion_coefficients, None, self.new_camera_matrix)

        cropped_image = corrected_image

        if cropped_image.shape != frame.shape:
            raise ValueError(f"Shape changed! original shape was {frame.shape} but corrected image is {cropped_image.shape}")

        return cropped_image
    
def get_calibrations_from_json(json_path: Path = INTRINSICS_JSON_PATH) -> dict[str, dict[str, np.ndarray]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    for index, dictionary in data.items():
        data[index] = {
            "camera_matrix": np.array(dictionary["camera_matrix"]),
            "distortion_coefficients": np.array(dictionary["distortion_coefficients"])
        }

    return data
