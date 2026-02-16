
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
            # framerate == annotated_vid_cap.get(cv2.CAP_PROP_FPS),
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
    good_pupil_point_x: Optional[np.ndarray] = None
    good_pupil_point_y: Optional[np.ndarray] = None
    pupil_mean_x: Optional[np.ndarray] = None
    pupil_mean_y: Optional[np.ndarray] = None
    pupil_point_names: Optional[list[str]] = None
    dataframe: Optional[pd.DataFrame] = None

    def load_good_pupil_point(self, pupil_point_name: str = GOOD_PUPIL_POINT) -> None:
        """Load pupil tracking data from CSV."""
        if self.data_csv_path is None:
            raise ValueError("data_csv_path is not set.")
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
        self.good_pupil_point_x = pupil_x.to_numpy()
        self.good_pupil_point_y = pupil_y.to_numpy()

        print(f"Loaded pupil data for {self.data_name} eye: {len(self.good_pupil_point_x)} points")

    def set_point_names(self) -> list[str]:
        if self.data_csv_path is None or not self.data_csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.data_csv_path}")
        
        data = pd.read_csv(self.data_csv_path)
        all_columns = list(data.columns)
        self.pupil_point_names = []
        for column in all_columns:
            if column == "video" or column == "frame":
                continue
            column = column.removesuffix("_x").removesuffix("_y")
            if column not in self.pupil_point_names:
                self.pupil_point_names.append(column)

        return self.pupil_point_names

    def get_point_names(self) -> list[str]:
        if self.pupil_point_names is None:
            return self.set_point_names()
        return self.pupil_point_names
    
    def pupil_video_name(self) -> str:
        if "eye0" in self.raw_video_name:
            return "eye0"
        elif "eye1" in self.raw_video_name:
            return "eye1"
        else:
            raise ValueError(f"Neither 'eye0' or 'eye1' found in raw video name: {self.raw_video_name}")
    
    def get_dataframe(self) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe
        if self.data_csv_path is None or not self.data_csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.data_csv_path}")
        df = pd.read_csv(self.data_csv_path)
        df["video"] = df["video"].apply(lambda x: "eye0" if "eye0" in x else "eye1")
        df = df[(df["video"] == self.pupil_video_name())]
        self.dataframe = df
        print(self.dataframe.head(5))
        return self.dataframe
    
    def calculate_pupil_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        selected_columns_x = [column for column in df.columns if "p" in column and "_x" in column]
        selected_columns_y = [column for column in df.columns if "p" in column and "_y" in column]
        df["eye_mean_x"] = df[selected_columns_x].mean(axis=1)
        df["eye_mean_y"] = df[selected_columns_y].mean(axis=1)
        return df
    
    def load_pupil_means(self) -> None:
        df = self.get_dataframe()
        df = self.calculate_pupil_mean(df)
        self.pupil_mean_x = df["eye_mean_x"].to_numpy()
        self.pupil_mean_y = df["eye_mean_y"].to_numpy()
    
    def data_array(self) -> np.ndarray:
        data_frame = self.get_dataframe()
        data_array = np.zeros((self.frame_count, len(self.get_point_names()), 2))
        for i, point_name in enumerate(self.get_point_names()):
            data_array[:, i, 0] = data_frame[f"{point_name}_x"]
            data_array[:, i, 1] = data_frame[f"{point_name}_y"]
        return data_array
    
    def flip_data_horizontal(self, array: np.ndarray, image_width: int) -> np.ndarray:
        flipped_array = np.copy(array)
        flipped_array[:, :, 0] = image_width - array[:, :, 0]
        return flipped_array
    
    def flip_data_vertical(self, array: np.ndarray, image_height: int) -> np.ndarray:
        flipped_array = np.copy(array)
        flipped_array[:, :, 1] = image_height - array[:, :, 1]
        return flipped_array
    

class AlignedEyeVideoData(VideoData):
    """Model representing eye video data and associated pupil tracking."""
    pupil_point_names: Optional[list[str]] = None
    dataframe: Optional[pd.DataFrame] = None

    def set_point_names(self) -> list[str]:
        if self.data_csv_path is None or not self.data_csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.data_csv_path}")
        
        data = pd.read_csv(self.data_csv_path)
        try:
            self.pupil_point_names = data["marker"].unique().tolist()
        except KeyError:
            self.pupil_point_names = data["keypoint"].unique().tolist()
        return self.pupil_point_names

    def get_point_names(self) -> list[str]:
        if self.pupil_point_names is None:
            return self.set_point_names()
        return self.pupil_point_names
    
    def pupil_video_name(self) -> str:
        if "eye0" in self.raw_video_name:
            return "eye0"
        elif "eye1" in self.raw_video_name:
            return "eye1"
        else:
            raise ValueError(f"Neither 'eye0' or 'eye1' found in raw video name: {self.raw_video_name}")
    
    def get_dataframe(self) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe
        if self.data_csv_path is None or not self.data_csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.data_csv_path}")
        self.dataframe = pd.read_csv(self.data_csv_path)
        print(self.dataframe.head(5))
        return self.dataframe
    
    def data_array(self) -> np.ndarray:
        data_frame = self.get_dataframe()
        data_array = np.zeros((self.frame_count, len(self.get_point_names()), 2))
        for i, point_name in enumerate(self.get_point_names()):
            try:
                mask = (data_frame['marker'] == point_name) & (data_frame['processing_level'] == "cleaned")
            except KeyError:
                mask = (data_frame['keypoint'] == point_name) & (data_frame['processing_level'] == "cleaned")
            try:
                data_array[:, i, :] = data_frame[mask][['x', 'y']].to_numpy()
            except ValueError:
                print(data_frame[mask])
                raise ValueError("incorrect indexing")
        return data_array
    
    def flip_data_horizontal(self, array: np.ndarray, image_width: int) -> np.ndarray:
        flipped_array = np.copy(array)
        flipped_array[:, :, 0] = image_width - array[:, :, 0]
        return flipped_array
    
    def flip_data_vertical(self, array: np.ndarray, image_height: int) -> np.ndarray:
        flipped_array = np.copy(array)
        flipped_array[:, :, 1] = image_height - array[:, :, 1]
        return flipped_array

if __name__ == "__main__":
    from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-18_ferret_420_E09/full_recording"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)
    recording_folder.check_triangulation(enforce_toy=False, enforce_annotated=False)

    left_eye = EyeVideoData.create(
        annotated_video_path=recording_folder.left_eye_annotated_video,
        raw_video_path=recording_folder.left_eye_video,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy,
        data_csv_path=recording_folder.left_eye_plot_points_csv,
        data_name="Left Eye"
    )

    print(left_eye.get_point_names())
    left_eye.get_dataframe()

    right_eye = EyeVideoData.create(
        annotated_video_path=recording_folder.right_eye_annotated_video,
        raw_video_path=recording_folder.right_eye_video,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy,
        data_csv_path=recording_folder.right_eye_plot_points_csv,
        data_name="Right Eye"
    )

    print(right_eye.get_point_names())
    right_eye.get_dataframe()