from pathlib import Path
import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
from pupil_info import JSON_PATH, load_json, recording_id_from_path
from python_code.optical_flow.dense_optical_flow import dense_optical_flow
from python_code.optical_flow.find_crops import find_and_save_crops


def main(pupil_video_path: Path | str, force_recalculate_crops: bool = False, display: bool = False, record: bool = True, full_plot: bool = False) -> None:
    pupil_video_path = Path(pupil_video_path)

    pupil_info_dict = load_json(Path(JSON_PATH))
    if recording_id_from_path(pupil_video_path) in pupil_info_dict and not force_recalculate_crops:
        crop = pupil_info_dict[recording_id_from_path(pupil_video_path)].crop
    else:
        crop = find_and_save_crops(pupil_video_path)

    cap = cv2.VideoCapture(str(pupil_video_path))
    dense_optical_flow(
        cap,
        crop=crop,
        display=False,
        record=True,
        full_plot=False,
        output_path=pupil_video_path.parent / (pupil_video_path.stem + "optical_flow.mp4"),
    )


if __name__ == "__main__":
    pupil_video_path = Path(
        r"C:\Users\jonma\Downloads\eye0.mp4"
    )

    main(pupil_video_path=pupil_video_path, force_recalculate_crops=False, display=False, record=True, full_plot=True)
