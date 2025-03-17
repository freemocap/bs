from pathlib import Path

import cv2
from pupil_info import JSON_PATH, load_json, recording_id_from_path
from python_code.optical_flow.dense_optical_flow import dense_optical_flow
from python_code.optical_flow.find_crops import find_and_save_crops


def main(pupil_video_path: Path | str, force_recalculate_crops: bool = False):
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
        output_path=pupil_video_path.parent / (pupil_video_path.stem + "optical_flow.mp4"),
    )


if __name__ == "__main__":
    pupil_video_path = Path(
        "/Users/philipqueen/session_2024-12-18/ferret_0776_P44_E14/eye1.mp4"
    )

    main(pupil_video_path=pupil_video_path, force_recalculate_crops=False)
