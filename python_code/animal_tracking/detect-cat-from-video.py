import os
from pathlib import Path
from typing import Dict
from typing import List

import cv2
import numpy as np
from PIL import Image

from transformers import pipeline


def detect_and_crop_objects_from_videos(
    object_detector: pipeline,
    video_path: str,
    object_list: List[str],
    output_folder: str,
    crop_scale: float = 1.2,
) -> None:
    # Initialize video capture
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (30*2) != 0:
            print(f"Skipping frame {frame_count} of {total_frames}")
            frame_count += 1
            continue

        # Convert the frame to RGB for PIL compatibility
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        detections = object_detector(pil_image)
        if not detections:
            print(f"No objects detected in frame {frame_count} of {total_frames}")
            frame_count += 1
            continue

        # zero pad the frame count to ensure the images are sorted correctly
        frame_count_str = str(frame_count).zfill(6)
        total_frames_str = str(total_frames).zfill(6)

        for detection in detections:
            if detection["label"] in object_list:
                crop_object_from_frame(
                    frame=rgb_frame,
                    box=detection["box"],
                    scale=crop_scale,
                    output_path=f"{output_folder}/{video_name}_{detection['label']}_frame_{frame_count_str}_of_{total_frames_str}.png",
                )
                print(f"Saved {detection['label']} from video {video_name} at frame {frame_count} of {total_frames_str}")

        frame_count += 1
    cap.release()


def crop_object_from_frame(
    frame: np.array, box: Dict[str, int], scale: float, output_path: str
) -> None:
    # Create the output folder if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract the box coordinates
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

    # Calculate the center of the box and the size to ensure a square
    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
    box_size = max(xmax - xmin, ymax - ymin) * scale

    # Calculate the new box coordinates, ensuring the crop is centered and square
    new_xmin = int(max(center_x - box_size // 2, 0))
    new_ymin = int(max(center_y - box_size // 2, 0))
    new_xmax = int(min(center_x + box_size // 2, frame.shape[1]))
    new_ymax = int(min(center_y + box_size // 2, frame.shape[0]))

    # Perform the crop
    pil_image = Image.fromarray(frame)
    cropped_image = pil_image.crop((new_xmin, new_ymin, new_xmax, new_ymax))
    cropped_image.save(output_path)

    if not Path(output_path).exists():
        raise ValueError(f"Error: {output_path} was not saved!")
    



if __name__ == "__main__":
    folder_path = (
        r"C:\Users\jonma\Sync\videos\2024-03-03-pooh-5GoPro-5k@30fps-Linear-22deg"
    )
    object_list = ["cat"]
    output_folder = r"C:\Users\jonma\Sync\videos\2024-03-03-pooh-5GoPro-5k@30fps-Linear-22deg\cat-crops"

    object_detector = pipeline("object-detection")

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            detect_and_crop_objects_from_videos(
                object_detector, video_path, object_list, output_folder
            )

    print("Processing complete!")
