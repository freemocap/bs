import cv2
from pathlib import Path

import numpy as np


def combine_videos(video_paths: list[Path], timestamps: np.ndarray) -> Path:
    """
    Combine videos into a single video file.

    Args:
        video_paths: List of paths to video files.

    Returns:
        Path to the combined video file.
    """
    name_to_path = {video_path.stem: video_path for video_path in video_paths}
    name_to_cap = {name: cv2.VideoCapture(str(path)) for name, path in name_to_path.items()}

    if len(name_to_cap) % 2 != 0:
        raise NotImplementedError("Odd number of videos not supported yet!.")

    # get widths and heights of each video pair
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in name_to_cap.values()]
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in name_to_cap.values()]

    if len(set(widths)) != 1 or len(set(heights)) != 1:
        raise ValueError("Videos must have the same resolution.")

    output_width = widths[0] * 2
    output_height = heights[0] * (len(name_to_cap) // 2)

    print(f"widths: {widths}")
    print(f"heights: {heights}")
    print(f"output_width: {output_width} output_height: {output_height}")

    output_path = video_paths[0].parent / "combined.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        name_to_cap[list(name_to_cap.keys())[0]].get(cv2.CAP_PROP_FPS),
        (output_width, output_height),
    )

    if output_width > 400:
        text_1_offset = (10, 50)
        text_2_offset = (10, 120)
        text_3_offset = (10, 190)
        font_size = 2
        font_thickness = 2
    else:
        text_1_offset = (5, 5)
        text_2_offset = (5, 25)
        text_3_offset = (5, 45)
        font_size = 0.5
        font_thickness = 1

    frame_number = 0
    while True:
        new_frame_list = []
        for camera_number, (name, cap) in enumerate(name_to_cap.items()):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_number} from {name}")
                break
            annotated_frame = cv2.putText(frame, f"video: {name}", text_1_offset, cv2.FONT_HERSHEY_SIMPLEX, font_size,
                                          (255, 255, 255), font_thickness)
            annotated_frame = cv2.putText(annotated_frame, f"frame: {frame_number}", text_2_offset,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          font_size, (255, 255, 255), font_thickness)
            annotated_frame = cv2.putText(annotated_frame, f"timestamp: {timestamps[camera_number, frame_number]}",
                                          text_3_offset, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_size, (255, 255, 255), font_thickness)
            new_frame_list.append(annotated_frame)
        if len(new_frame_list) != len(name_to_cap):
            break
        left_column = np.concatenate([frame for index, frame in enumerate(new_frame_list) if index % 2 == 0], axis=0)
        right_column = np.concatenate([frame for index, frame in enumerate(new_frame_list) if index % 2 == 1], axis=0)
        new_frame = np.concatenate((left_column, right_column), axis=1)
        # cv2.imshow("frame", new_frame)
        # cv2.waitKey(1)
        writer.write(new_frame)
        frame_number += 1

    writer.release()

    return output_path


if __name__ == "__main__":
    video_folder = Path("/Users/philipqueen/ferret_NoImplant_P35_EO5/synchronized_videos")
    # video_folder = Path("/Users/philipqueen/basler_pupil_synch_test/pupil_output")

    timestamps_path = video_folder / "timestamps.npy"
    timestamps = np.load(timestamps_path)

    # print(timestamps.shape)
    # print(timestamps)

    video_paths = list(video_folder.glob("*.mp4"))
    video_paths = [video_path for video_path in video_paths if
                   "combined" not in video_path.stem]  # someday save video in a better place
    print(video_paths)
    combine_videos(video_paths=video_paths, timestamps=timestamps)
