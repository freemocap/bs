import cv2
from pathlib import Path
from enum import Enum

class FlipMethod(Enum):
    HORIZONTAL = 1
    VERTICAL = 0
    BOTH = -1


def flip_video(video: Path, flip_method: FlipMethod):
    cap = cv2.VideoCapture(str(video))

    writer = cv2.VideoWriter(
        str(video.parent / f"{video.stem}_flipped.mp4"),
        fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished reading video")
            break

        flipped_frame = cv2.flip(frame, flip_method.value)

        writer.write(flipped_frame)

    cap.release()
    writer.release()

if __name__=="__main__":
    video=Path('/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5/clips/1m_20s-2m_20s/eye_data/eye_videos/eye1_clipped_9033_16234.mp4')

    flip_video(video=video, flip_method=FlipMethod.HORIZONTAL)