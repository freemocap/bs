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
        include_all_pixels = 1  # 1: all pixels are retained with some extra black in margins, 0: only valid pixels are shown
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

if __name__ == "__main__":
    video = Path("/home/scholl-lab/recordings/session_2025-06-28/ferret_757_EyeCameras_P30_EO2/raw_videos/24908832.mp4")

    intrinsic_data = get_calibrations_from_json()

    intrinsics = intrinsic_data[video.stem]

    cap = cv2.VideoCapture(str(video))

    corrector = IntrinsicsCorrector.from_dict(dict=intrinsics, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    ret, frame = cap.read()

    if not ret:
        raise ValueError("Unable to read frame")
    
    corrected_frame = corrector.correct_frame(frame)

    cv2.imshow("corrected_frame", corrected_frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()