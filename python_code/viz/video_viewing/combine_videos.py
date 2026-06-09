from enum import Enum
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class VideoType(Enum):
    TOP_DOWN = "top_down"
    BASLER = "basler"
    PUPIL = "pupil"


ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class VideoInfo:
    name: str
    video_type: VideoType
    path: Path
    width: int
    height: int
    fps: float
    timestamps: np.ndarray
    cap: cv2.VideoCapture
    position: str = ""
    cv2_rotation: int | None = None
    flip: int | None = None # 0 for vertical, >0 for horizontal, <0 for both
    key: int | None = None

    @classmethod
    def from_path_and_timestamp(
        cls,
        path: Path,
        timestamps: np.ndarray,
        position: str = "",
        rotation: int | None = None,
        flip: int | None = None,
        key: int | None = None,
    ):
        cap = cv2.VideoCapture(str(path))
        if "eye" in path.stem:
            video_type = VideoType.PUPIL
        elif "24676894" in path.stem:
            video_type = VideoType.TOP_DOWN
        else:
            video_type = VideoType.BASLER
        return cls(
            path.stem,
            video_type,
            path,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
            timestamps,
            cap,
            position=position,
            cv2_rotation=rotation,
            flip=flip,
            key=key,
        )


def create_video_info(folder_path: Path) -> List[VideoInfo]:
    """
    Makes video information for basler and pupil videos from folder path containing synchronized videos and timestamps

    Args:
        folder_path (Path): Path to folder containing synchronized videos and timestamps

    Returns:
        Tuple[List[VideoInfo], List[VideoInfo]]: Tuple containing basler and pupil video information
    """
    all_videos = list(
        video_path
        for video_path in folder_path.glob("*.mp4")
        if ("combined" not in video_path.stem and "clip" not in video_path.stem and "rotated" not in video_path.stem)
    )
    print(all_videos)

    with open(Path(__file__).parent / "video_rotations.json", "r") as f:
        video_rotations = json.load(f)
    videos: List[VideoInfo] = []
    cam_number = 0
    for video_path in all_videos:
        try:
            rotation_info = video_rotations[video_path.stem]
        except KeyError:
            raise ValueError(f"{video_path.name} not found in rotation info")
        rotation = ROTATIONS[rotation_info["rotation"]]
        position = rotation_info.get("position", "")
        key = rotation_info.get("key", None)
        flip = rotation_info.get("flip", None)
        if "eye0" in video_path.stem:
            timestamps = load_synchronized_timestamps(folder_path, cam_name="eye0")
        elif "eye1" in video_path.stem:
            timestamps = load_synchronized_timestamps(folder_path, cam_name="eye1")
        else:
            # TODO: fix implicit sorting of timestamps by cam number
            timestamps = load_synchronized_timestamps(
                folder_path, cam_name=str(cam_number)
            )
            print(f"loading video {video_path.stem} for cam {cam_number}")
            cam_number += 1
        video_info = VideoInfo.from_path_and_timestamp(
            path=video_path,
            timestamps=timestamps,
            rotation=rotation,
            flip=flip,
            position=position,
            key=key,
        )
        videos.append(video_info)

    basler_videos = [video for video in videos if video.video_type == VideoType.BASLER]
    pupil_videos = [video for video in videos if video.video_type == VideoType.PUPIL]

    if len(pupil_videos) not in {0, 2}:
        raise NotImplementedError(
            f"Expected 0 or 2 pupil videos, got {len(pupil_videos)}"
        )
    print(
        f"Found {len(basler_videos)} basler videos and {len(pupil_videos)} pupil videos"
    )

    basler_videos.sort(key=lambda video: video.key)
    pupil_videos.sort(key=lambda video: video.key)

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

    return videos


def load_synchronized_timestamps(folder_path: Path, cam_name: str) -> np.ndarray:
    timestamp_path = folder_path / f"cam_{cam_name}_synchronized_timestamps.npy"
    if not timestamp_path.exists():
        timestamp_path = folder_path / f"cam_{cam_name}_corrected_timestamps.npy"
        if not timestamp_path.exists():
            raise FileNotFoundError(
                "Unable to find synchronized (or corrected) timestamps in path"
            )
    timestamps = np.load(timestamp_path, allow_pickle=True)
    return timestamps


def find_closest_frame(
    target_timestamp: int | float, search_timestamps: np.ndarray
) -> int:
    # Filter out None values
    valid_indices = [i for i, ts in enumerate(search_timestamps) if ts is not None]
    valid_timestamps = [search_timestamps[i] for i in valid_indices]

    # Find the index of the closest timestamp
    closest_index = int(
        np.argmin(np.abs(np.array(valid_timestamps) - target_timestamp))
    )

    # Return the original index
    return valid_indices[closest_index]


def get_first_timestamp(
    basler_videos: List[VideoInfo], pupil_videos: List[VideoInfo]
) -> int:
    basler_timestamps = [
        video.timestamps[0]
        for video in basler_videos
        if video.timestamps[0] is not None
    ]
    pupil_timestamps = [
        video.timestamps[0] for video in pupil_videos if video.timestamps[0] is not None
    ]

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
    text_1_offset = (10, 70)
    text_2_offset = (10, 150)
    text_3_offset = (10, 230)
    font_size = 3
    font_thickness = 3
    black = (0, 0, 0)
    white = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    annotated_frame = cv2.putText(
        frame,
        f"video: {video_name}",
        text_1_offset,
        font,
        font_size,
        black,
        font_thickness + 4,
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
        frame,
        f"frame: {frame_number}",
        text_2_offset,
        font,
        font_size,
        black,
        font_thickness + 4,
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
        frame,
        f"time: {timestamp}",
        text_3_offset,
        font,
        font_size,
        black,
        font_thickness + 4,
    )
    annotated_frame = cv2.putText(
        annotated_frame,
        f"time: {timestamp}",
        text_3_offset,
        font,
        font_size,
        white,
        font_thickness,
    )

    return annotated_frame


def add_title(frame: np.ndarray, session_name: str | None, recording_name: str | None) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    if session_name and recording_name:
        title = f"{session_name}: {recording_name}"
    elif session_name:
        title = f"{session_name}"
    elif recording_name:
        title = f"{recording_name}"
    else:
        return frame
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = 50
    cv2.putText(
        frame,
        title,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )
    return frame

def draw_plot_on_frame(frame, size, dataframe, timestamps, frame_number, video_name):
    # Convert frame to RGB for plotting
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    dpi=100
    fig, ax = plt.subplots(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)

    pupil_tracked_point = "pupil_outer_x"

    frame_range = 50
    frame_start = min(0, frame_number - frame_range)
    frame_end = max(frame_number + frame_range, max(dataframe['frame']))

    timestamp_start = timestamps[frame_start]
    timestamp_end = timestamps[frame_end]

    ax.set_xlim(frame_number - frame_range, frame_number+frame_range)
    ax.set_ylim(250, 350)

    # Plot data
    ax.plot(dataframe['frame'][frame_start:frame_end], dataframe[pupil_tracked_point][frame_start:frame_end])
    ax.scatter(dataframe['frame'][frame_start:frame_end], dataframe[pupil_tracked_point][frame_start:frame_end])
    ax.vlines(frame_number, ymin=0, ymax=350) 

    ax.set_title(f"{video_name}: {pupil_tracked_point}")
    
    # Convert plot to OpenCV format
    fig.canvas.draw()
    plot_img = np.array(fig.canvas.renderer._renderer)[:, :, :-1]
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
    
    # Resize plot to fit corner of frame
    plot_img = cv2.resize(plot_img, size)
    
    # Place plot in top-right corner
    y_offset = frame.shape[0] - size[1]
    x_offset = frame.shape[1] - size[0]
    frame[y_offset:y_offset+size[1], x_offset:x_offset+size[0]] = plot_img
    
    plt.close(fig)
    return frame


def combine_videos(
    videos: List[VideoInfo], 
    output_path: Path, 
    session_name: str, 
    recording_name: str, 
    add_plot: bool = False,
    dataframe: pd.DataFrame | None = None
) -> Path:
    """
    Combine videos into a single video file.

    Args:
        videos: List of VideoInfo objects for all the videos

    Returns:
        Path to the combined video file.
    """
    basler_videos = [video for video in videos if video.video_type == VideoType.BASLER]
    basler_videos.sort(key=lambda video: video.key)
    pupil_videos = [video for video in videos if video.video_type == VideoType.PUPIL]
    pupil_videos.sort(key=lambda video: video.name)
    earliest_timestamp = get_first_timestamp(
        basler_videos=basler_videos, pupil_videos=pupil_videos
    )
    print(f"earliest timestamp: {earliest_timestamp}")

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

    # get widths and heights of each video pair
    basler_widths = [int(video.width) for video in basler_videos]
    basler_heights = [int(video.height) for video in basler_videos]

    if len(set(basler_widths)) != 1 or len(set(basler_heights)) != 1:
        raise ValueError("Basler side view videos must have the same resolution.")
    if basler_widths[0] != basler_heights[0]:
        raise ValueError("Basler side view videos must be square.")

    print(f"widths: {basler_widths}")
    print(f"heights: {basler_heights}")

    output_width = 2560
    output_height = 1440
    plot_height = 480

    if add_plot:
        if dataframe is None:
            raise RuntimeError("Cannot add plot if dataframe isn't provided")
        output_height += plot_height 
        for video in pupil_videos:
            if video.name == "eye0":
                plot_video=video

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        basler_videos[0].fps,
        (output_width, output_height),
    )

    print(f"output_width: {output_width} output_height: {output_height}")

    frame_number = 0
    current_pupil_frame_numbers = {
        "eye0": 0,
        "eye1": 0
    }

    dummy_frame = np.zeros((int(pupil_videos[0].cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(pupil_videos[0].cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 3))
    previous_frames = {
        "eye0": dummy_frame,
        "eye1": dummy_frame
    }
    while True:
    # while frame_number < 3000:
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
            frame = rotate_frame(video, frame)
            timstamp_seconds = convert_utc_timestamp_to_seconds_since_start(
                timestamp - earliest_timestamp
            )
            annotated_frame = annotate(
                frame, video.name, frame_number, timstamp_seconds
            )
            new_frame_list.append(annotated_frame)

        if len(new_frame_list) != len(basler_videos):
            print("did not find a frame for every basler side video")
            break

        average_timestamp = sum(current_timestamps) / len(current_timestamps)
        pupil_frames = {}

        # get pupil frames with closest timestamps to average of basler timestamps
        for video in pupil_videos:
            # pupil_crop_size = 250
            active_pupil_frame_number = current_pupil_frame_numbers[video.name]
            closest_frame = find_closest_frame(average_timestamp, video.timestamps)

            if active_pupil_frame_number >= len(video.timestamps):
                print(
                    f"passed end of pupil video: {active_pupil_frame_number} >= {len(video.timestamps)}"
                )
                print("using previous frame")
                # TODO: may need to handle this better
                frame = previous_frames[video.name]
                timestamp = video.timestamps[-1]
            elif active_pupil_frame_number > closest_frame:
                print(
                    f"passed closest frame: {active_pupil_frame_number} > {closest_frame}"
                )
                if not np.any(previous_frames[video.name]):
                    print("using dummy frame")
                    print(f"previous frames _ {previous_frames}")
                print("using previous frame")
                frame = previous_frames[video.name]
            elif active_pupil_frame_number == closest_frame:
                ret, frame = video.cap.read()
                if not ret:
                    print(
                        f"Failed to read frame {active_pupil_frame_number} from {video.name}"
                    )
                    break
                timestamp = video.timestamps[active_pupil_frame_number]
            else:
                while active_pupil_frame_number < closest_frame:
                    ret, frame = video.cap.read()
                    if not ret:
                        print(
                            f"Failed to read frame {active_pupil_frame_number} from {video.name}"
                        )
                        break
                    timestamp = video.timestamps[active_pupil_frame_number]
                    active_pupil_frame_number += 1

                ret, frame = video.cap.read()
                if not ret:
                    print(
                        f"Failed to read frame {active_pupil_frame_number} from {video.name}"
                    )
                    break
                timestamp = video.timestamps[active_pupil_frame_number]
            previous_frames[video.name] = frame
            frame = rotate_frame(video, frame)
            frame = flip_frame(video, frame)
            
            # frame = frame[y:y + pupil_crop_size, x:x + pupil_crop_size]
            # frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            frame = cv2.resize(frame, (basler_widths[0], basler_heights[0]))

            timestamp_seconds = convert_utc_timestamp_to_seconds_since_start(
                timestamp - earliest_timestamp
            )
            current_pupil_frame_numbers[video.name] = active_pupil_frame_number + 1
            annotated_frame = annotate(
                frame, video.name, current_pupil_frame_numbers[video.name], timestamp_seconds
            )
            
            pupil_frames[video.name] = annotated_frame


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
            frame = rotate_frame(top_video, frame)
            current_timestamps.append(timestamp)
            timestamp_seconds = convert_utc_timestamp_to_seconds_since_start(
                timestamp - earliest_timestamp
            )
            top_video_frame = annotate(
                frame, top_video.name, frame_number, timestamp_seconds
            )

        # combine basler and pupil videos into single frame
        new_frame = construct_video_frame(
            basler_videos=new_frame_list,
            pupil_frames=pupil_frames,
            top_video=top_video_frame,
            session_name=session_name,
            recording_name=recording_name,
            output_width=output_width,
            output_height=output_height,
            frame_number=frame_number,
            timestamp=timestamp_seconds
        )

        if add_plot:
            new_frame = draw_plot_on_frame(
                frame=new_frame, 
                size=(output_width, plot_height), 
                dataframe=dataframe.loc[dataframe["video"]==plot_video.name], 
                timestamps=plot_video.timestamps, 
                frame_number=current_pupil_frame_numbers[plot_video.name],
                video_name=plot_video.name)


        # cv2.imshow("frame", new_frame)
        writer.write(new_frame)
        frame_number += 1

    writer.release()

    return output_path


def construct_video_frame(
    basler_videos: List[np.ndarray],
    pupil_frames: Dict[str, np.ndarray],
    top_video: np.ndarray,
    session_name: str | None,
    recording_name: str | None,
    output_width: int,
    output_height: int,
    frame_number: int,
    timestamp: float
):
    background = np.zeros((output_height, output_width, 3), np.uint8)
    new_frame = add_title(background, session_name, recording_name)
    # if output_height != 1440 or output_width != 2560:
    #     raise ValueError(
    #         f"output height and width must be 1440x2560, but are {output_height}x{output_width}"
    #     )
    
    with open(Path(__file__).parent / "layout_locations.json", 'r') as f:
        layout_locations = json.load(f)

    layout_info = layout_locations[f"{output_width}x{output_height}"]

    new_frame = add_image_to_frame(
        new_frame, top_video, layout_info["top_video"]
    )
    new_frame = add_image_to_frame(
        new_frame, pupil_frames["eye0"], layout_info["pupil_videos"][0]
    )
    new_frame = add_image_to_frame(
        new_frame, pupil_frames["eye1"], layout_info["pupil_videos"][1]
    )
    for i, basler_video in enumerate(basler_videos):
        new_frame = add_image_to_frame(
            new_frame, basler_video, layout_info["basler_videos"][i]
        )

    text_2_offset = (10, 120)
    text_3_offset = (10, 190)
    font_size = 2
    font_thickness = 2
    black = (0, 0, 0)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    annotated_frame = cv2.putText(
        new_frame,
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

def add_image_to_frame(frame: np.ndarray, image: np.ndarray, layout_info: dict) -> np.ndarray:
    x = layout_info["x"]
    y = layout_info["y"]
    width = layout_info["width"]
    height = layout_info["height"]

    image = cv2.resize(image, (width, height))

    frame[y:y + height, x:x + width] = image

    return frame


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


if __name__ == "__main__":
    # video_folder = Path(
    #     "/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/dlc_annotated_videos/current_best"
    # )
    video_folder = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/basler_pupil_synchronized")

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
        add_plot=False,
        dataframe=dataframe
    )
