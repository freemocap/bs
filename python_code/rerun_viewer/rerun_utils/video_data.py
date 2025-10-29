
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel

GOOD_PUPIL_POINT = "p2"
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)n




class VideoData(BaseModel):
    """Base model representing video data and associated tracking."""
    data_name: str  # E.g. "Left Eye", "Right Eye", "TopDown Mocap"
    annotated_video_path: Path
    raw_video_path: Path
    timestamps_npy_path: Path
    framerate: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    frame_duration: float
    resize_factor: float
    resized_width: int
    resized_height: int
    timestamps: np.ndarray
    data_csv_path: Optional[Path] = None
    annotated_vid_cap: Optional[cv2.VideoCapture] = None
    raw_vid_cap: Optional[cv2.VideoCapture] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def annotated_video_name(self) -> str:
        return self.annotated_video_path.stem

    @property
    def raw_video_name(self) -> str:
        return self.raw_video_path.stem

    @classmethod
    def create(cls,
               annotated_video_path: Path,
               raw_video_path: Path,
               timestamps_npy_path: Path,
               data_name: str,
               data_csv_path: Path | None = None,
               resize_factor: float = 1.0
               ):
        """Load video information from the video file."""

        if not Path(annotated_video_path).exists():
            raise FileNotFoundError(f"Video file not found: {annotated_video_path}")

        raw_vid_cap = cv2.VideoCapture(str(raw_video_path))
        if not raw_vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {raw_video_path}")

        framerate = raw_vid_cap.get(cv2.CAP_PROP_FPS)
        width = int(raw_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(raw_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(raw_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        annotated_vid_cap = cv2.VideoCapture(str(annotated_video_path))
        if not annotated_vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {annotated_video_path}")

        # Validate that annotated and raw videos have matching properties
        if not all([
            framerate == annotated_vid_cap.get(cv2.CAP_PROP_FPS),
            width == int(annotated_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height == int(annotated_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count == int(annotated_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        ]):
            raise ValueError(
                f"Video properties do not match between annotated and raw videos for {data_name}.\n"
                f"Raw properties: \n"
                f"Framerate: {framerate}, Width: {width}, Height: {height}, Frame Count: {frame_count}\n"
                f"Annotated properties: \n"
                f"Framerate: {annotated_vid_cap.get(cv2.CAP_PROP_FPS)}, "
                f"Width: {int(annotated_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, "
                f"Height: {int(annotated_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                f"Frame Count: {int(annotated_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

        duration_seconds = frame_count / framerate
        frame_duration = 1.0 / framerate

        # Calculate new dimensions if resizing
        resized_width = int(width * resize_factor)
        resized_height = int(height * resize_factor)

        print(f"{data_name} video info: {width}x{height} @ {framerate} FPS, "
              f"{frame_count} frames, {duration_seconds:.1f}s duration")
        if resize_factor != 1.0:
            print(f"Resizing to: {resized_width}x{resized_height}")

        # Load timestamps
        timestamps_array = np.load(timestamps_npy_path).astype(np.float64) / 1e9  # Convert from nanoseconds to seconds
        if len(timestamps_array) != frame_count:
            raise ValueError(
                f"Expected {frame_count} timestamps, but found {len(timestamps_array)} in NPY data.")
        
        print(f"video {data_name} first timestamp: {timestamps_array[0]} last timestamp: {timestamps_array[-1]}")
        print(f"timestamps duration: {timestamps_array[-1] - timestamps_array[0]}")

        return cls(
            annotated_video_path=annotated_video_path,
            raw_video_path=raw_video_path,
            timestamps_npy_path=timestamps_npy_path,
            framerate=framerate,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
            frame_duration=frame_duration,
            resize_factor=resize_factor,
            resized_width=resized_width,
            resized_height=resized_height,
            timestamps=timestamps_array,
            annotated_vid_cap=annotated_vid_cap,
            raw_vid_cap=raw_vid_cap,
            data_csv_path=data_csv_path,
            data_name=data_name
        )


class MocapVideoData(VideoData):
    pass


class EyeVideoData(VideoData):
    """Model representing eye video data and associated pupil tracking."""
    pupil_x: Optional[np.ndarray] = None
    pupil_y: Optional[np.ndarray] = None

    def load_pupil_data(self, pupil_point_name: str = GOOD_PUPIL_POINT) -> None:
        """Load pupil tracking data from CSV."""
        if not Path(self.data_csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.data_csv_path}")

        # Skip the first row (scorer) and use the second and third rows as the header
        pupil_df = pd.read_csv(self.data_csv_path)
        pupil_df = pupil_df[pupil_df['video'].str.contains(self.raw_video_name)]

        # Check if the pupil point exists in the bodyparts level
        if f"{pupil_point_name}_x" not in pupil_df.columns.get_level_values(0) or f"{pupil_point_name}_y" not in pupil_df.columns.get_level_values(0):
            raise ValueError(f"Expected bodypart '{pupil_point_name}' not found in CSV data.")

        # Extract x and y coordinates for the specified pupil point
        pupil_x = pupil_df[f'{pupil_point_name}_x']
        pupil_y = pupil_df[f'{pupil_point_name}_y']

        if len(pupil_x) != self.frame_count:
            print(f"Warning: Expected {self.frame_count} pupil points, but found {len(pupil_x)} in CSV data.")

        # Convert to numpy arrays for faster processing
        self.pupil_x = pupil_x.to_numpy()
        self.pupil_y = pupil_y.to_numpy()

        print(f"Loaded pupil data for {self.data_name} eye: {len(self.pupil_x)} points")
