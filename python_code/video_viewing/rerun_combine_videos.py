from enum import Enum
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rerun as rr
import av
import numpy.typing as np

from python_code.video_viewing.combine_videos import VideoInfo, VideoType, create_video_info, get_first_timestamp

def combine_videos(
    videos: List[VideoInfo], 
    output_path: Path, 
    session_name: str, 
    recording_name: str, 
    dataframe: pd.DataFrame | None = None
) -> Path:
    active_pupil_point = "pupil_outer"

    basler_videos = [video for video in videos if video.video_type == VideoType.BASLER]
    basler_videos.sort(key=lambda video: video.key)
    pupil_videos = [video for video in videos if video.video_type == VideoType.PUPIL]
    pupil_videos.sort(key=lambda video: video.name)

    active_pupil_video = pupil_videos[0]
    frame_count = active_pupil_video.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    top_video = None
    top_videos = [
        video for video in videos if video.video_type == VideoType.TOP_DOWN
    ]
    if len(top_videos) > 0:
        top_video = top_videos[0]
    else:
        raise ValueError(
            "No top down video found"
        )

    if f"{active_pupil_point}_x" not in dataframe.columns or f"{active_pupil_point}_y" not in dataframe.columns:
        raise ValueError(f"Expected column '{active_pupil_point}' not found in CSV data.")
    active_pupil_x = dataframe.loc[dataframe["video"] == active_pupil_video.name][f"{active_pupil_point}_x"]
    active_pupil_y = dataframe.loc[dataframe["video"] == active_pupil_video.name][f"{active_pupil_point}_y"]

    if len(active_pupil_x) != int(frame_count):
        raise ValueError(f"Expected {frame_count} pupil points, but found {len(active_pupil_x)} in CSV data.")


    rr.init(f"{session_name}: {recording_name}", spawn=True)

    # rr.log("pupil_x_line", rr.SeriesLines(colors=[125, 0, 0], names="Horizonal Pupil Position", widths=2), static=True)
    # rr.log("pupil_y_line", rr.SeriesLines(colors=[0, 62, 125], names="Vertical Pupil Position ", widths=2), static=True)
    # rr.log("pupil_x_dots", rr.SeriesPoints(colors=[255, 0, 0], names="Horizonal Pupil Position", keypoints="circle", keypoint_sizes=2), static=True)
    # rr.log("pupil_y_dots", rr.SeriesPoints(colors=[0, 125, 255], names="Vertical Pupil Position", keypoints="circle", keypoint_sizes=2), static=True)
            
    # rr.log("pupil_x_line", rr.Scalars(active_pupil_x[frame_number]))
    # rr.log("pupil_y_line", rr.Scalars(active_pupil_y[frame_number]))
    # rr.log("pupil_x_dots", rr.Scalars(active_pupil_x[frame_number]))
    # rr.log("pupil_y_dots", rr.Scalars(active_pupil_y[frame_number]))

    for video in videos:
        frame_duration = 1 / video.fps
        video_frame_count = video.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Setup encoding pipeline.
        av.logging.set_level(av.logging.VERBOSE)
        container = av.open("/dev/null", "w", format="h264")  # Use AnnexB H.264 stream.
        stream = container.add_stream("libx264", rate=int(video.fps))
        stream.width = video.width
        stream.height = video.height
        # TODO(#10090): Rerun Video Streams don't support b-frames yet.
        # Note that b-frames are generally not recommended for low-latency streaming and may make logging more complex.
        stream.max_b_frames = 0

        # Log codec only once as static data (it naturally never changes). This isn't strictly necessary, but good practice.
        rr.log(video.name, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

        # Generate frames and stream them directly to Rerun.
        for frame_number in range(video_frame_count):
        
            ret, img = video.cap.read()
            if not ret:
                print(f"Failed to read frame {frame_number}, stopping.")
                break
            
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                if packet.pts is None:
                    continue
                rr.set_time("time", duration=float(packet.pts * packet.time_base))
                rr.log(video.name, rr.VideoStream.from_fields(sample=bytes(packet)))
                rr.set_time("time", duration= float(frame_number * frame_duration))


        # Flush stream.
        for packet in stream.encode():
            if packet.pts is None:
                continue
            rr.set_time("time", duration=float(packet.pts * packet.time_base))
            rr.log(video.name, rr.VideoStream.from_fields(sample=bytes(packet)))

if __name__ == "__main__":
    video_folder = Path(
        "/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/dlc_annotated_videos/current_best"
    )
    # video_folder = Path("/home/scholl-lab/recordings/session_2024-12-11/ferret_0776_P37_EO7/basler_pupil_synchronized")

    data_path = "/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/dlc_pupil_tracking/new_model_iteration_2/skellyclicker_machine_labels_iteration_2.csv"
    dataframe = pd.read_csv(data_path)

    videos = create_video_info(folder_path=video_folder)

    session_name = video_folder.parent.parent.stem
    recording_name = video_folder.parent.stem

    combine_videos(
        videos=videos,
        output_path=video_folder / "combined.mp4",
        session_name=session_name,
        recording_name=recording_name,
        dataframe=dataframe
    )