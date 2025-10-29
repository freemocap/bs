
import json
from typing import List
import cv2
from pathlib import Path
import numpy as np

from python_code.video_viewing.closest_pupil_frame_to_basler_frame import closest_pupil_frame_to_basler_frame
from python_code.video_viewing.combine_videos import VideoInfo, VideoType, ROTATIONS

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
        # print(video_info)
        videos.append(video_info)

    return videos

def get_basler_videos(basler_path: Path) -> list[VideoInfo]:
    basler_videos = list(basler_path.glob("*.mp4"))
    print(len(basler_videos))

    with open(Path(__file__).parent / "video_rotations.json", "r") as f:
        video_rotations = json.load(f)
    print(video_rotations)
    videos: List[VideoInfo] = []
    for video_path in basler_videos:
        video_name = video_path.stem.split("_")[0]
        try:
            rotation_info = video_rotations[video_name]
        except KeyError:
            raise ValueError(f"{video_path.name} not found in rotation info")
        rotation = ROTATIONS[rotation_info["rotation"]]
        position = rotation_info.get("position", "")
        key = rotation_info.get("key", None)
        flip = rotation_info.get("flip", None)
        timestamps = np.load(basler_path / f"{video_name}_synchronized_timestamps_utc.npy")
        video_info = VideoInfo.from_path_and_timestamp(
            path=video_path,
            timestamps=timestamps,
            rotation=rotation,
            flip=flip,
            position=position,
            key=key,
        )
        # print(video_info)
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

def trim_video(video: VideoInfo, writer: cv2.VideoWriter, frame_range: tuple[int, int], apply_transforms: bool = False):
    frame_number = 0
    valid_frames = set(range(frame_range[0], frame_range[1] + 1))
    while True:
        ret, frame = video.cap.read()
        if not ret:
            print("ran out of frames")
            break

        if frame_number in valid_frames:
            if apply_transforms:
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

def clip_videos_basler(
        videos: list[VideoInfo], 
        output_folder: Path, 
        frame_range: tuple[int, int], 
        include_timestamps: bool = True,
        apply_transforms: bool = False):
    basler_videos = [video for video in videos if video.video_type == VideoType.BASLER or video.video_type == VideoType.TOP_DOWN]
    basler_videos.sort(key=lambda video: video.key)
    # pupil_videos = [video for video in videos if video.video_type == VideoType.PUPIL]
    # pupil_videos.sort(key=lambda video: video.name) 

    for video in basler_videos:
        name=f"{video.name}_clipped_{frame_range[0]}_{frame_range[1]}.mp4"
        writer = create_writer_from_cap(video.cap, output_path=output_folder, name=name)
        trim_video(video=video, writer=writer, frame_range=frame_range, apply_transforms=apply_transforms)
        writer.release()
        if include_timestamps:
            timestamps_name = f"{video.name}_synchronized_timestamps_utc_clipped_{frame_range[0]}_{frame_range[1]}.npy"
            np.save(str(output_folder / timestamps_name), video.timestamps[frame_range[0]:frame_range[1]+1], allow_pickle=False)

def clip_video_pupil(
        video: VideoInfo, 
        output_folder: Path, 
        frame_range: tuple[int, int], 
        include_timestamps: bool = True,
        apply_transforms: bool = False):
    name=f"{video.name}_clipped_{frame_range[0]}_{frame_range[1]}.mp4"
    writer = create_writer_from_cap(video.cap, output_path=output_folder, name=name)
    trim_video(video=video, writer=writer, frame_range=frame_range, apply_transforms=True)
    writer.release()
    if include_timestamps:
        timestamps_name = f"{video.name}_timestamps_utc_clipped_{frame_range[0]}_{frame_range[1]}.npy"
        np.save(str(output_folder / timestamps_name), video.timestamps[frame_range[0]:frame_range[1]+1], allow_pickle=False)
    

if __name__ == "__main__":
    recording_path = Path("/home/scholl-lab/ferret_recordings/session_2025-07-07_ferret_410_P39_E09")
    clip_folder = recording_path / "clips/1m_20s-2m_20s"

    basler_folder = recording_path / "full_recording/mocap_data/synchronized_corrected_videos"
    output_folder = clip_folder / "mocap_data/synchronized_videos"
    basler_start_frame = 50
    basler_end_frame = 5050

    output_folder.mkdir(exist_ok=True, parents=True)
    videos = get_basler_videos(basler_folder)

    clip_videos_basler(videos=videos, output_folder=output_folder, frame_range=(basler_start_frame, basler_end_frame))

    # Comment out below this if you don't want to clip eye videos

    #pupil_folder = recording_path / "full_recording/eye_data/eye_videos"
    #output_folder = clip_folder / "eye_data/eye_videos"

    #output_folder.mkdir(exist_ok=True, parents=True)

    #pupil_videos = get_pupil_videos(pupil_path=pupil_folder)

    
    #clip_info = closest_pupil_frame_to_basler_frame(
        #session_folder=recording_path,
        #starting_basler_frame=basler_start_frame,
        #ending_basler_frame=basler_end_frame
    #)

    #for video in pupil_videos:
        #video_name = video.path.stem
        #eye_start_frame = clip_info[video_name]["start_frame"]
        #eye_end_frame = clip_info[video_name]["end_frame"]
        #clip_video_pupil(video=video, output_folder=output_folder, frame_range=[eye_start_frame, eye_end_frame])

    #clip_info_json_path = clip_folder / "clip_info.json"
   # with open(clip_info_json_path, "w") as file:
       # json.dump(clip_info, file, indent=4)