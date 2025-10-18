import cv2
import numpy as np

from python_code.eye_data_cleanup.rerun_video_data import VideoHelper
import rerun as rr





def process_video_frame(frame: np.ndarray,
                        resize_factor: float, 
                        resize_width: int,
                        resize_height: int,
                        flip_horizontal: bool = False,
                        jpeg_quality: int = 80) -> bytes:
    """Process a single video frame."""
    if flip_horizontal:
        frame = cv2.flip(frame, 1)

    # Resize if needed
    if resize_factor != 1.0:
        frame = cv2.resize(frame, (resize_width, resize_height))

    # Encode to JPEG
    return cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes()

def process_video_frames_for_rerun(video_cap: cv2.VideoCapture,
                                   resize_factor: float,
                                   resize_width: int,
                                   resize_height: int,
                                   flip_horizontal: bool = False) -> list[bytes]:
    """Process a batch of video frames."""
    encoded_frames = []
    success = True
    while success:
        success, frame = video_cap.read()
        if not success:
            continue
        encoded_frames.append(process_video_frame(frame=frame,
                                                    resize_factor=resize_factor,
                                                    resize_width=resize_width,
                                                    resize_height=resize_height,
                                                    flip_horizontal=flip_horizontal))

    return encoded_frames

def process_video_for_rerun(video: VideoHelper,
                            entity_path: str,
                            flip_horizontal: bool = False):
    """Process a video and send it to Rerun."""
    print(f"Processing {video.video_name} video...")

    # Log video stream
    encoded_frames = process_video_frames_for_rerun(
        video_cap=video.video_capture,
        resize_factor=video.resize_factor,
        resize_width=video.width,
        resize_height=video.height,
        flip_horizontal=flip_horizontal
    )
    t0 = video.timestamps[0]
    ts = []
    for t in video.timestamps:
        ts.append((t - t0)/1e9)  # Convert to seconds
    print(f"Sending time series data to {entity_path} with {len(ts)} timestamps...")

    rr.send_columns(
        entity_path=f"{entity_path}/video",
        indexes=[rr.TimeColumn("time", duration=ts)],
        columns=rr.EncodedImage.columns(
            blob=encoded_frames,
            media_type=['image/jpeg'] * len(encoded_frames))
    )
