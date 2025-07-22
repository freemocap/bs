import json
from dataclasses import dataclass
from typing import List, Tuple
import cv2
from pathlib import Path

import numpy as np


@dataclass
class VideoInfo:
    name: str
    is_pupil: bool
    path: Path
    width: int
    height: int
    fps: float
    timestamps: np.ndarray
    cap: cv2.VideoCapture

    @classmethod
    def from_path_and_timestamp(cls, path: Path, timestamps: np.ndarray):
        cap = cv2.VideoCapture(str(path))
        return cls(
            path.stem,
            "eye" in path.stem,
            path,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
            timestamps,
            cap,
        )
    
def create_video_info(folder_path: Path) -> Tuple[List[VideoInfo], List[VideoInfo]]:
    """
    Makes video information for basler and pupil videos from folder path containing synchronized videos and timestamps

    Args:
        folder_path (Path): Path to folder containing synchronized videos and timestamps

    Returns:  
        Tuple[List[VideoInfo], List[VideoInfo]]: Tuple containing basler and pupil video information
    """
    all_videos = list(video_path for video_path in folder_path.glob("*.mp4") if ("combined" not in video_path.stem and "clip" not in video_path.stem))
    print(all_videos)

    basler_videos = []
    pupil_videos = []
    cam_number = 0
    for video_path in all_videos:
        if "eye0" in video_path.stem:
            timestamps = load_synchronized_timestamps(folder_path, cam_name="eye0")
            pupil_videos.append(VideoInfo.from_path_and_timestamp(video_path, timestamps))
        elif "eye1" in video_path.stem:
            timestamps = load_synchronized_timestamps(folder_path, cam_name="eye1")
            pupil_videos.append(VideoInfo.from_path_and_timestamp(video_path, timestamps))
        else:
            timestamps = load_synchronized_timestamps(folder_path, cam_name=str(cam_number))
            print(f"loading video {video_path.stem} for cam {cam_number}")
            basler_videos.append(VideoInfo.from_path_and_timestamp(video_path, timestamps))
            cam_number += 1
        
    if len(pupil_videos) not in {0, 2}:
        raise NotImplementedError(f"Expected 0 or 2 pupil videos, got {len(pupil_videos)}")
    print(f"Found {len(basler_videos)} basler videos and {len(pupil_videos)} pupil videos")
    
    print("Basler Videos:")
    for video in basler_videos:
        print(f"\t{video.name}")
        print(f"\t\tfps: {video.fps}")
        print(f"\t\twidth: {video.width}")
        print(f"\t\theight: {video.height}")
        print(f"\t\tframe count: {video.cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        print(f"\t\tlength of timestamps: {len(video.timestamps)}")
    
    print("Pupil Videos:")
    for video in pupil_videos:
        print(f"\t{video.name}")
        print(f"\t\tfps: {video.fps}")
        print(f"\t\twidth: {video.width}")
        print(f"\t\theight: {video.height}")
        print(f"\t\tframe count: {video.cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        print(f"\t\tlength of timestamps: {len(video.timestamps)}")

    return basler_videos, pupil_videos

def load_synchronized_timestamps(folder_path: Path, cam_name: str) -> np.ndarray:
    timestamp_path = folder_path / f"cam_{cam_name}_synchronized_timestamps.npy"
    if not timestamp_path.exists():
        timestamp_path = folder_path / f"cam_{cam_name}_corrected_timestamps.npy"
        if not timestamp_path.exists():
            raise FileNotFoundError("Unable to find synchronized (or corrected) timestamps in path")
    timestamps = np.load(timestamp_path, allow_pickle=True)
    return timestamps

def find_closest_frame(target_timestamp: int | float, search_timestamps: np.ndarray) -> int:
    # Filter out None values
    valid_indices = [i for i, ts in enumerate(search_timestamps) if ts is not None]
    valid_timestamps = [search_timestamps[i] for i in valid_indices]

    # Find the index of the closest timestamp
    closest_index = int(np.argmin(np.abs(np.array(valid_timestamps) - target_timestamp)))

    # Return the original index
    return valid_indices[closest_index]


def get_first_timestamp(basler_videos: List[VideoInfo], pupil_videos: List[VideoInfo]) -> int:
    basler_timestamps = [video.timestamps[0] for video in basler_videos if video.timestamps[0] is not None]
    pupil_timestamps = [video.timestamps[0] for video in pupil_videos if video.timestamps[0] is not None]

    starting_timestamps = basler_timestamps + pupil_timestamps

    return min(starting_timestamps)

def convert_utc_timestamp_to_seconds_since_start(utc_timestamp: int) -> float:
    return utc_timestamp / 1e9

def annotate(
        frame: np.ndarray,
        video_name: str,
        frame_number: int,
        timestamp: float,       
) -> np.ndarray:
    text_1_offset = (10, 50)
    text_2_offset = (10, 120)
    text_3_offset = (10, 190)
    font_size = 2
    font_thickness = 2
    black = (0, 0, 0)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    annotated_frame = cv2.putText(
        frame,
        f"video: {video_name}",
        text_1_offset,
        font,
        font_size,
        black,
        font_thickness + 1,
    )
    annotated_frame = cv2.putText(
        annotated_frame,
        f"video: {video_name}",
        text_1_offset,
        font,
        font_size,
        white,
        font_thickness,
    )
    annotated_frame = cv2.putText(
        annotated_frame,
        f"frame: {frame_number}",
        text_2_offset,
        font,
        font_size,
        black,
        font_thickness + 1,
    )
    annotated_frame = cv2.putText(
        annotated_frame,
        f"frame: {frame_number}",
        text_2_offset,
        font,
        font_size,
        white,
        font_thickness,
    )
    annotated_frame = cv2.putText(
        annotated_frame,
        f"timestamp: {timestamp:.3f}",
        text_3_offset,
        font,
        font_size / 2,
        black,
        font_thickness + 1,
    ) 
    annotated_frame = cv2.putText(
        annotated_frame,
        f"timestamp: {timestamp:.3f}",
        text_3_offset,
        font,
        font_size / 2,
        white,
        font_thickness,
    ) 

    return annotated_frame


def combine_videos(basler_videos: List[VideoInfo], pupil_videos: List[VideoInfo]) -> Path:
    """
    Combine videos into a single video file.

    Args:
        basler_videos: List of VideoInfo objects for the basler videos.
        pupil_videos: List of VideoInfo objects for the pupil videos.

    Returns:
        Path to the combined video file.
    """
    earliest_timestamp = get_first_timestamp(basler_videos=basler_videos, pupil_videos=pupil_videos)
    print(f"earliest timestamp: {earliest_timestamp}")

    top_video = None
    for i, video in enumerate(basler_videos):
        if "24676894" in video.name:
            print(f"Found top video: {video.name}")
            top_video = video
            basler_videos.pop(i)

    if top_video is None:
        num_rows = 2
    else:
        num_rows = 3

    # get widths and heights of each video pair
    basler_widths = [int(video.width) for video in basler_videos]
    basler_heights = [int(video.height) for video in basler_videos]

    if len(set(basler_widths)) != 1 or len(set(basler_heights)) != 1:
        raise ValueError("Videos must have the same resolution.")

    output_width = basler_widths[0] * 3
    output_height = basler_heights[0] * num_rows

    print(f"widths: {basler_widths}")
    print(f"heights: {basler_heights}")
    print(f"output_width: {output_width} output_height: {output_height}")

    output_path = basler_videos[0].path.parent / "combined.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        basler_videos[0].fps,
        (output_width, output_height),
    )

    frame_number = 0
    eye0_frame_number = 0
    eye1_frame_number = 0
    while True:
        new_frame_list = []
        current_timestamps = []
        # get frames from basler videos
        for video in basler_videos:
            ret, frame = video.cap.read()
            if not ret:
                print(f"Failed to read frame {frame_number} from {video.name}")
                break
            try:
                timestamp = video.timestamps[frame_number]
            except IndexError:
                print(f"exceeded Basler timestamps for camera {video.name}")
                break
            current_timestamps.append(timestamp)
            timstamp_seconds = convert_utc_timestamp_to_seconds_since_start(timestamp - earliest_timestamp)
            annotated_frame = annotate(frame, video.name, frame_number, timstamp_seconds)
            new_frame_list.append(annotated_frame)

        if len(new_frame_list) != len(basler_videos):
            break

        average_timestamp = sum(current_timestamps) / len(current_timestamps)
        pupil_frames = {}
        # get pupil frames with closest timestamps to average of basler timestamps
        for video in pupil_videos:
            pupil_crop_size = 250
            if video.name == "eye0":
                active_pupil_frame_number = eye0_frame_number
                x, y = 114, 115
            elif video.name == "eye1":
                active_pupil_frame_number = eye1_frame_number
                x, y = 130, 80
            else:
                raise ValueError(f"Unknown video name: {video.name}")
            closest_frame = find_closest_frame(average_timestamp, video.timestamps)
            
            if active_pupil_frame_number >= len(video.timestamps):
                print(f"passed end of pupil video: {active_pupil_frame_number} >= {len(video.timestamps)}")
                print("using previous frame")
                # TODO: may need to handle this better
                frame = previous_frame
                timestamp = video.timestamps[-1]
            elif active_pupil_frame_number > closest_frame:
                print(f"passed closest frame: {active_pupil_frame_number} > {closest_frame}")
                print("using previous frame")
                frame = previous_frame
            elif active_pupil_frame_number == closest_frame:
                ret, frame = video.cap.read()
                if not ret:
                    print(f"Failed to read frame {active_pupil_frame_number} from {video.name}")
                    break
                timestamp = video.timestamps[active_pupil_frame_number]
            else:
                while active_pupil_frame_number < closest_frame:
                    ret, frame = video.cap.read()
                    if not ret:
                        print(f"Failed to read frame {active_pupil_frame_number} from {video.name}")
                        break
                    timestamp = video.timestamps[active_pupil_frame_number]
                    active_pupil_frame_number += 1
                
                ret, frame = video.cap.read()
                if not ret:
                    print(f"Failed to read frame {active_pupil_frame_number} from {video.name}")
                    break
                timestamp = video.timestamps[active_pupil_frame_number]

            previous_frame = frame
            # frame = frame[y:y + pupil_crop_size, x:x + pupil_crop_size]
            # frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            if video.name == "eye0":
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.resize(frame, (basler_widths[0], basler_heights[0]))


            timestamp_seconds = convert_utc_timestamp_to_seconds_since_start(timestamp - earliest_timestamp)
            annotated_frame = annotate(frame, video.name, frame_number, timestamp_seconds)
            if video.name == "eye0":
                eye0_frame_number = active_pupil_frame_number + 1
                pupil_frames["eye0"] = annotated_frame
            elif video.name == "eye1":
                eye1_frame_number = active_pupil_frame_number + 1
                pupil_frames["eye1"] = annotated_frame
        new_frame_list.append(pupil_frames["eye0"])
        new_frame_list.append(pupil_frames["eye1"])

        if top_video is not None:
            ret, frame = top_video.cap.read()
            if not ret:
                print(f"Failed to read frame {frame_number} from {top_video.name}")
                break
            try:
                timestamp = top_video.timestamps[frame_number]
            except IndexError:
                print(f"exceeded Basler timestamps for camera {top_video.name}")
                break
            current_timestamps.append(timestamp)
            timstamp_seconds = convert_utc_timestamp_to_seconds_since_start(timestamp - earliest_timestamp)
            annotated_frame = annotate(frame, top_video.name, frame_number, timstamp_seconds)

            dummy_width = (output_width - annotated_frame.shape[1]) // 2

            dummy_frame_left = np.zeros(
                (basler_heights[0], dummy_width, 3), dtype=np.uint8
            )
            dummy_frame_right = np.zeros(
                (basler_heights[0], (output_width - dummy_width - annotated_frame.shape[1]), 3), dtype=np.uint8
            )
            new_frame_list.append(dummy_frame_left)
            new_frame_list.append(annotated_frame)
            new_frame_list.append(dummy_frame_right)

        # combine basler and pupil videos into single frame

        top_row = np.concatenate(
            [new_frame_list[4], new_frame_list[5], new_frame_list[2]],
            axis=1
        )
        middle_row = np.concatenate(
            [new_frame_list[0], new_frame_list[1], new_frame_list[3]],
            axis=1
        )
        bottom_row = np.concatenate(
            [new_frame_list[6], new_frame_list[7], new_frame_list[8]],
            axis=1
        )

        for row in [top_row, middle_row, bottom_row]:
            if row.shape[1] != output_width:
                raise ValueError(f"Row width {row.shape[1]} does not match output width {output_width}")
            if row.shape[0] != basler_heights[0]:
                raise ValueError(f"Row height {row.shape[0]} does not match output height {basler_heights[0]}")
        
        new_frame = np.concatenate([top_row, middle_row, bottom_row], axis=0)
        # cv2.imshow("frame", new_frame)
        writer.write(new_frame)
        frame_number += 1

    writer.release()

    return output_path


if __name__ == "__main__":
    # video_folder = Path("/Users/philipqueen/ferret_NoImplant_P35_EO5/synchronized_videos")
    # video_folder = Path("/home/scholl-lab/recordings/session_2024-12-11/ferret_0776_P37_EO7/basler_pupil_synchronized")

    basler_videos, pupil_videos = create_video_info(folder_path=video_folder)

    combine_videos(basler_videos=basler_videos, pupil_videos=pupil_videos)
