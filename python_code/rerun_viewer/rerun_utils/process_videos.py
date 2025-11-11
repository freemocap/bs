import cv2
import numpy as np
from python_code.rerun_viewer.rerun_utils.video_data import VideoData
import rerun as rr

GOOD_PUPIL_POINT = "p2"
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)n




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

def process_video_frames(video_cap: cv2.VideoCapture,
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

def process_video(video_data: VideoData, entity_path: str, flip_horizontal: bool = False, include_annotated: bool = True):
    """Process a video and send it to Rerun."""
    print(f"Processing {video_data.data_name} video...")
    video_types = ["raw", "annotated"] if include_annotated else ["raw"]
    # Log video stream
    for video_type in video_types:
        encoded_frames = process_video_frames(
            video_cap=video_data.raw_vid_cap if video_type == "raw" else video_data.annotated_vid_cap,
            resize_factor=video_data.resize_factor,
            resize_width=video_data.resized_width,
            resize_height=video_data.resized_height,
            flip_horizontal=flip_horizontal
        )

        rr.send_columns(
            entity_path=f"{entity_path}/{video_type}",
            indexes=[rr.TimeColumn("time", duration=video_data.timestamps)],
            columns=rr.EncodedImage.columns(
                blob=encoded_frames,
                media_type=['image/jpeg'] * len(encoded_frames))
        )