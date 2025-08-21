from enum import Enum
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python_code.video_viewing.combine_videos import VideoInfo, VideoType, create_video_info, ROTATIONS

def get_pupil_videos(pupil_path: Path) -> list[VideoInfo]:
    pupil_videos = [pupil_path / "eye0.mp4", pupil_path / "eye1.mp4"]

    with open(Path(__file__).parent / "video_rotations.json", "r") as f:
        video_rotations = json.load(f)
    print(video_rotations)
    videos: List[VideoInfo] = []
    for video_path in pupil_videos:
        try:
            rotation_info = video_rotations[video_path.stem]
        except KeyError:
            raise ValueError(f"{video_path.name} not found in rotation info")
        rotation = ROTATIONS[rotation_info["rotation"]]
        position = rotation_info.get("position", "")
        key = rotation_info.get("key", None)
        flip = rotation_info.get("flip", None)
        if "eye0" in video_path.stem:
            timestamps = np.load(pupil_path / "eye0_timestamps_utc.npy")
        elif "eye1" in video_path.stem:
            timestamps = np.load(pupil_path / "eye1_timestamps_utc.npy")
        else:
            raise ValueError()
        video_info = VideoInfo.from_path_and_timestamp(
            path=video_path,
            timestamps=timestamps,
            rotation=rotation,
            flip=flip,
            position=position,
            key=key,
        )
        print(video_info)
        videos.append(video_info)

    return videos

def rotate_frame(video: VideoInfo, frame: np.ndarray) -> np.ndarray:
    if (
        video.cv2_rotation is not None
    ):
        frame = cv2.rotate(frame, video.cv2_rotation)
    return frame

def flip_frame(video: VideoInfo, frame: np.ndarray) -> np.ndarray:
    if video.flip is not None:
        frame = cv2.flip(frame, video.flip)
    return frame

def trim_video(video: VideoInfo, writer: cv2.VideoWriter, frame_range: tuple[int, int]):
    frame_number = 0
    valid_frames = set(range(frame_range[0], frame_range[1] + 1))
    while True:
        ret, frame = video.cap.read()
        if not ret:
            print("fan out of frames")
            break

        if frame_number in valid_frames:
            frame = rotate_frame(video, frame)
            frame = flip_frame(video, frame)
            writer.write(frame)
        elif frame_number > frame_range[1]:
            break

        frame_number += 1

def create_writer_from_cap(cap: cv2.VideoCapture, output_path: Path, name: str) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        str(output_path / name),
        cv2.VideoWriter.fourcc(*"mp4v"),
        int(cap.get(cv2.CAP_PROP_FPS)),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )

def clip_videos_basler(videos: list[VideoInfo], output_folder: Path, frame_range: tuple[int, int], include_timestamps: bool = True):
    basler_videos = [video for video in videos if video.video_type == VideoType.BASLER or video.video_type == VideoType.TOP_DOWN]
    basler_videos.sort(key=lambda video: video.key)
    # pupil_videos = [video for video in videos if video.video_type == VideoType.PUPIL]
    # pupil_videos.sort(key=lambda video: video.name) 

    for video in basler_videos:
        name=f"{video.name}_clipped_{frame_range[0]}_{frame_range[1]}.mp4"
        writer = create_writer_from_cap(video.cap, output_path=output_folder, name=name)
        trim_video(video=video, writer=writer, frame_range=frame_range)
        writer.release()
        timestamps_name = f"{video.name}_clipped_{frame_range[0]}_{frame_range[1]}_timestamps.npy"
        np.save(str(output_folder / timestamps_name), video.timestamps[frame_range[0]:frame_range[1]+1], allow_pickle=False)

def clip_video_pupil(video: VideoInfo, output_folder: Path, frame_range: tuple[int, int], include_timestamps: bool = True):
    name=f"{video.name}_clipped_{frame_range[0]}_{frame_range[1]}.mp4"
    writer = create_writer_from_cap(video.cap, output_path=output_folder, name=name)
    trim_video(video=video, writer=writer, frame_range=frame_range)
    writer.release()
    timestamps_name = f"{video.name}_clipped_{frame_range[0]}_{frame_range[1]}_timestamps.npy"
    np.save(str(output_folder / timestamps_name), video.timestamps[frame_range[0]:frame_range[1]+1], allow_pickle=False)



    

if __name__ == "__main__":
    # video_folder = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/basler_pupil_synchronized")
    # output_folder = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/clips/0_37-1_37/mocap_data")


    # output_folder.mkdir(exist_ok=True, parents=True)
    # videos = create_video_info(video_folder)

    # clip_videos_basler(videos=videos, output_folder=output_folder, frame_range=(3377, 8754))

    pupil_folder = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/pupil_output/")
    output_folder = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/clips/0_37-1_37/eye_data")

    output_folder.mkdir(exist_ok=True, parents=True)

    pupil_videos = get_pupil_videos(pupil_path=pupil_folder)

    clip_video_pupil(video=pupil_videos[0], output_folder=output_folder, frame_range=(4451, 11621))
    clip_video_pupil(video=pupil_videos[1], output_folder=output_folder, frame_range=[4469, 11638])