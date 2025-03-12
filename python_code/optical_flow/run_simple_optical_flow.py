from typing import Optional, Tuple
import numpy as np
import cv2 as cv

def run_simple_optical_flow(cap: cv.VideoCapture, crop: Optional[Tuple[int, int, int, int]] = None, display: bool = False):
    if crop is not None:
        x, y, w, h = crop
    else:
        x, y, w, h = 0, 0, int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    ret, frame1 = cap.read()
    if not ret:
        raise Exception("Failed to read video")
    previous = cv.cvtColor(frame1[x:x+w, y:y+h], cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1[x:x+w, y:y+h])
    hsv[..., 1] = 255
    while True:
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2[x:x+w, y:y+h], cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        combined = np.concatenate((frame2[x:x+w, y:y+h], bgr), axis=1)
        if display:
            cv.imshow('frame2', combined)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            previous = next
    if display:
        cv.destroyAllWindows()

    cap.release()

if __name__ == "__main__":
    pupil_video_path = "/Users/philipqueen/ferret_0776_P35_EO5/basler_pupil_synchronized/eye0.mp4"
    crop = (224, 194, 85, 87)
    cap = cv.VideoCapture(pupil_video_path)
    run_simple_optical_flow(cap, crop=crop, display=True)