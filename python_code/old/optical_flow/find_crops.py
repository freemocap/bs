from json import JSONDecodeError
from typing import Optional, Tuple, Sequence
import cv2
import numpy as np
from pathlib import Path

from pupil_info import JSON_PATH, load_json, PupilInfo, save_models_to_json, recording_id_from_path


def find_crops(image: np.ndarray) -> Tuple[int, int, int, int]:
    r = cv2.selectROI("Select Region", image)
    cv2.destroyAllWindows()
    
    x, y, w, h = r
    print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h

def find_and_save_crops(pupil_video_path: Path | str) -> Tuple[int, int, int, int]:
    cap = cv2.VideoCapture(str(pupil_video_path))
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("Failed to read video")
    crops = find_crops(frame)
    cap.release()
    try:
        pupil_info_dict = load_json(Path(JSON_PATH))
    except JSONDecodeError:
        pupil_info_dict = {}
    pupil_info_dict[recording_id_from_path(pupil_video_path)] = PupilInfo.from_path_and_crop(path=pupil_video_path, crop=crops)
    save_models_to_json(pupil_info_dict, Path(JSON_PATH))

    return crops


if __name__ == "__main__":
    pupil_video_path = Path("/Users/philipqueen/session_2024-12-18/ferret_0776_P44_E14/eye1.mp4")
    find_and_save_crops(pupil_video_path)