from typing import Optional, Tuple, Sequence
import cv2
import numpy as np


def find_crops(image: np.ndarray) -> Tuple[int, int, int, int]:
    r = cv2.selectROI("Select Region", image)
    cv2.destroyAllWindows()
    
    x, y, w, h = r
    print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h


if __name__ == "__main__":
    pupil_video_path = "/Users/philipqueen/ferret_0776_P35_EO5/basler_pupil_synchronized/eye0.mp4"
    cap = cv2.VideoCapture(pupil_video_path)
    ret, frame = cap.read()
    find_crops(frame)
    cap.release()
