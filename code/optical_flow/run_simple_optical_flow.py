from typing import Optional, Tuple
import numpy as np
import cv2 as cv

def run_simple_optical_flow(cap: cv.VideoCapture, crop: Optional[Tuple[int, int, int, int]] = None, display: bool = True, record: bool = False):
    if crop is not None:
        x, y, w, h = crop
    else:
        x, y, w, h = 0, 0, int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if record:
        output_pathstring = "output.mp4"
        fourcc = cv.VideoWriter.fourcc(*"mp4v")
        writer = cv.VideoWriter(output_pathstring, fourcc, cap.get(cv.CAP_PROP_FPS), (int(2*w), int(h)))


    ret, frame1 = cap.read()
    if not ret:
        raise Exception("Failed to read video")
    frame1 = frame1[y:y+h, x:x+w]
    previous = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while True:
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame2 = frame2[y:y+h, x:x+w]
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # magic values come from OpenCV docs
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle*180/np.pi/2
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        combined = np.concatenate((frame2, bgr), axis=1)
        if display:
            cv.imshow('frame2', combined)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            previous = next
        if record:
            writer.write(combined)
    if display:
        cv.destroyAllWindows()
    if record:
        print(f"saving video to {output_pathstring}")
        writer.release()

    cap.release()

if __name__ == "__main__":
    from pathlib import Path
    from pupil_info import JSON_PATH, load_json, recording_id_from_path

    pupil_video_path = Path("/Users/philipqueen/session_2024-12-18/ferret_0776_P44_E14/eye1.mp4")

    pupil_info_dict = load_json(Path(JSON_PATH))
    if recording_id_from_path(pupil_video_path) in pupil_info_dict:
        crop = pupil_info_dict[recording_id_from_path(pupil_video_path)].crop
    else:
        raise Exception("Model not found in JSON file, please run find_crops.py for this recording")

    cap = cv.VideoCapture(str(pupil_video_path))
    run_simple_optical_flow(cap, crop=crop, display=False, record=True)