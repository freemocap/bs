from typing import Optional, Tuple
import numpy as np
import cv2 
from matplotlib import pyplot as plt

from pathlib import Path

from python_code.optical_flow.plot_optical_flow_histograms import plot_optical_flow_histograms


def dense_optical_flow(
    cap: cv2.VideoCapture,
    crop: Optional[Tuple[int, int, int, int]] = None,
    display: bool = True,
    record: bool = False,
    output_path: Optional[str | Path] = None,
):
    if crop is not None:
        x, y, w, h = crop
    else:
        x, y, w, h = (
            0,
            0,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    if record:
        if output_path is None:
            raise Exception("Must specify output path if recording is enabled")
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(2 * w), int(h))
        )

    ret, frame1 = cap.read()
    if not ret:
        raise Exception("Failed to read video")
    frame1 = frame1[y : y + h, x : x + w]
    previous = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while True:
        ret, frame2 = cap.read()
        if not ret:
            print("No frames grabbed!")
            break
        frame2 = frame2[y : y + h, x : x + w]
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            next,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )  # magic values are OpenCV defaults  
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        combined = np.concatenate((frame2, bgr), axis=1)
        plot = plot_optical_flow_histograms(raw_image=frame2, flow_image=bgr, flow=flow)
        if display:
            cv2.imshow("Dense Optical Flow", plot)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            previous = next
        if record:
            writer.write(combined)
    if display:
        cv2.destroyAllWindows()
    if record:
        print(f"saving video to {output_path}")
        writer.release()

    cap.release()


if __name__ == "__main__":
    from pathlib import Path
    from pupil_info import JSON_PATH, load_json, recording_id_from_path

    pupil_video_path = Path(
        "/Users/philipqueen/session_2024-12-18/ferret_0776_P44_E14/eye1.mp4"
    )

    pupil_info_dict = load_json(Path(JSON_PATH))
    if recording_id_from_path(pupil_video_path) in pupil_info_dict:
        crop = pupil_info_dict[recording_id_from_path(pupil_video_path)].crop
    else:
        raise Exception(
            "Model not found in JSON file, please run find_crops.py for this recording"
        )

    cap = cv2.VideoCapture(str(pupil_video_path))
    dense_optical_flow(cap, crop=crop, display=True, record=False, output_path=pupil_video_path.stem + "optical_flow.mp4")
