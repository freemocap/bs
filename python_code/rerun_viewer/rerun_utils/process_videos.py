import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
                        flip_vertical: bool = False,
                        jpeg_quality: int = 80) -> bytes:
    """Process a single video frame."""
    if flip_horizontal:
        frame = cv2.flip(frame, 1)

    if flip_vertical:
        frame = cv2.flip(frame, 0)

    # Resize if needed
    if resize_factor != 1.0:
        frame = cv2.resize(frame, (resize_width, resize_height))

    # Encode to JPEG
    return cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes()

def process_video_frames(video_cap: cv2.VideoCapture,
                            resize_factor: float,
                            resize_width: int,
                            resize_height: int,
                            flip_horizontal: bool = False,
                            flip_vertical: bool = False,
                            jpeg_quality: int = 80) -> list[bytes]:
    """Process video frames in parallel using threads.

    cv2.imencode releases the GIL so threads encode concurrently. frame.copy()
    is called before submission to ensure each thread owns its frame data
    independently of OpenCV's internal VideoCapture buffer.
    """
    futures = []
    with ThreadPoolExecutor() as executor:
        success = True
        while success:
            success, frame = video_cap.read()
            if success:
                futures.append(executor.submit(
                    process_video_frame,
                    frame=frame.copy(),
                    resize_factor=resize_factor,
                    resize_width=resize_width,
                    resize_height=resize_height,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    jpeg_quality=jpeg_quality,
                ))
    return [f.result() for f in futures]


_MAX_CHUNK_BYTES = 1 * 1024 ** 3  # 1 GB — well under Arrow's 2 GB 32-bit offset limit


def _send_encoded_frames_chunked(
    entity_path: str,
    timestamps: np.ndarray,
    encoded_frames: list[bytes],
) -> None:
    start = 0
    while start < len(encoded_frames):
        end = start
        chunk_bytes = 0
        while end < len(encoded_frames):
            chunk_bytes += len(encoded_frames[end])
            if chunk_bytes > _MAX_CHUNK_BYTES:
                break
            end += 1
        if end == start:
            end = start + 1  # always send at least one frame
        rr.send_columns(
            entity_path=entity_path,
            indexes=[rr.TimeColumn("time", duration=timestamps[start:end])],
            columns=rr.EncodedImage.columns(
                blob=encoded_frames[start:end],
                media_type=["image/jpeg"] * (end - start),
            ),
        )
        start = end


def process_video(video_data: VideoData, entity_path: str, flip_horizontal: bool = False, flip_vertical: bool = False, include_annotated: bool = True):
    """Process a video and send it to Rerun."""
    print(f"Processing {video_data.data_name} video...")
    video_types = ["raw", "annotated"] if include_annotated else ["raw"]
    timestamps = video_data.timestamps - video_data.timestamps[0]
    # Log video stream
    for video_type in video_types:
        encoded_frames = process_video_frames(
            video_cap=video_data.raw_vid_cap if video_type == "raw" else video_data.annotated_vid_cap,
            resize_factor=video_data.resize_factor,
            resize_width=video_data.resized_width,
            resize_height=video_data.resized_height,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )

        # Arrow binary arrays use 32-bit offsets, capping total blob size at ~2GB.
        # Send in chunks to stay safely under that limit.
        _send_encoded_frames_chunked(
            entity_path=f"{entity_path}/{video_type}",
            timestamps=timestamps,
            encoded_frames=encoded_frames,
        )