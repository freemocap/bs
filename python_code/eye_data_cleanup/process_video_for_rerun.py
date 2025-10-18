import cv2
import numpy as np
import rerun as rr

from python_code.eye_data_cleanup.rerun_video_data import VideoHelper


def process_video_frame(
        *,
        frame: np.ndarray,
        resize_factor: float,
        resize_width: int,
        resize_height: int,
        flip_horizontal: bool = False,
        jpeg_quality: int = 80
) -> bytes:
    """Process a single video frame into JPEG bytes."""
    if flip_horizontal:
        frame = cv2.flip(src=frame, flipCode=1)

    # Resize if needed
    if resize_factor != 1.0:
        frame = cv2.resize(src=frame, dsize=(resize_width, resize_height))

    # Encode to JPEG
    return cv2.imencode(ext='.jpg', img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes()


def process_video_frames_for_rerun(
        *,
        video_cap: cv2.VideoCapture,
        resize_factor: float,
        resize_width: int,
        resize_height: int,
        flip_horizontal: bool = False
) -> list[bytes]:
    """Process all video frames into JPEG byte arrays."""
    encoded_frames: list[bytes] = []
    success: bool = True

    while success:
        success, frame = video_cap.read()
        if not success:
            continue
        encoded_frames.append(
            process_video_frame(
                frame=frame,
                resize_factor=resize_factor,
                resize_width=resize_width,
                resize_height=resize_height,
                flip_horizontal=flip_horizontal
            )
        )

    return encoded_frames


def process_video_for_rerun(
        *,
        video: VideoHelper,
        entity_path: str,
        flip_horizontal: bool = False
) -> None:
    """
    Process a video and send it to Rerun using send_columns for efficiency.

    Args:
        video: VideoHelper instance containing video data
        entity_path: The entity path where the video should be logged
        flip_horizontal: Whether to flip the video horizontally
    """
    print(f"Processing {video.video_name} video...")

    # Process all frames into encoded JPEG blobs
    encoded_frames: list[bytes] = process_video_frames_for_rerun(
        video_cap=video.video_capture,
        resize_factor=video.resize_factor,
        resize_width=video.width,
        resize_height=video.height,
        flip_horizontal=flip_horizontal
    )

    # Convert timestamps to seconds (duration) from nanoseconds
    t0: float = video.timestamps[0]
    timestamps_seconds: np.ndarray = np.array([(t - t0) / 1e9 for t in video.timestamps])

    print(f"Sending video data to {entity_path} with {len(timestamps_seconds)} timestamps and {len(encoded_frames)} frames...")

    # Ensure lengths match
    n_frames: int = min(len(timestamps_seconds), len(encoded_frames))
    if len(timestamps_seconds) != len(encoded_frames):
        print(f"WARNING: Timestamp count ({len(timestamps_seconds)}) doesn't match frame count ({len(encoded_frames)})")
        timestamps_seconds = timestamps_seconds[:n_frames]
        encoded_frames = encoded_frames[:n_frames]

    # Log video using send_columns for efficient bulk upload
    # Following pattern from Rerun examples for encoded images
    rr.send_columns(
        entity_path=entity_path,
        indexes=[rr.TimeColumn("time", duration=timestamps_seconds)],
        columns=rr.EncodedImage.columns(
            blob=encoded_frames,
            media_type=['image/jpeg'] * n_frames
        )
    )

    print(f"Successfully logged {n_frames} video frames to {entity_path}")