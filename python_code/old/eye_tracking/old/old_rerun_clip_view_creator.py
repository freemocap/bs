"""Video encode images using av and stream them to Rerun with optimized performance."""

from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from pydantic import BaseModel
from rerun.blueprint import VisualBounds2D
from rerun.blueprint.archetypes import TimeAxis
from rerun.blueprint.components import LinkAxis
from rerun.datatypes import Range2D

# Configuration
GOOD_PUPIL_POINT = "pupil_outer"
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)

# Define paths
RECORDING_NAME = "2025-07-11_ferret_757_EyeCameras_P43_E15__1"
CLIP_NAME = "0m_37s-1m_37s"

BASE_RECORDINGS_FOLDER = Path(r"D:\bs\ferret_recordings")
RECORDING_FOLDER = BASE_RECORDINGS_FOLDER / RECORDING_NAME
CLIP_FOLDER = RECORDING_FOLDER / "clips" / CLIP_NAME

# Eye data paths
EYE_DATA_FOLDER = CLIP_FOLDER / "eye_data"
EYE_ANNOTATED_VIDEOS_FOLDER = EYE_DATA_FOLDER / "annotated_videos"
EYE_SYNCHRONIZED_VIDEOS_FOLDER = EYE_DATA_FOLDER / "synchronized_videos"
EYE_TIMESTAMPS_FOLDER = EYE_SYNCHRONIZED_VIDEOS_FOLDER / "timestamps"
EYE_OUTPUT_DATA_FOLDER = EYE_DATA_FOLDER / "output_data" / "dlc_output"

RIGHT_EYE_ANNOTATED_VIDEO_PATH = EYE_ANNOTATED_VIDEOS_FOLDER / "eye0_clipped_4451_11621.mp4"
RIGHT_EYE_RAW_VIDEO_PATH = EYE_SYNCHRONIZED_VIDEOS_FOLDER / "eye0_clipped_4451_11621.mp4"
RIGHT_EYE_TIMESTAMPS_NPY_PATH = EYE_TIMESTAMPS_FOLDER / "eye0_clipped_4451_11621_timestamps.npy"
RIGHT_EYE_DATA_CSV_PATH = EYE_OUTPUT_DATA_FOLDER / "eye0_clipped_4451_11621DLC_Resnet50_pupil_tracking_ferret_757_EyeCameras_P43_E15__1_shuffle1_snapshot_030.csv"

LEFT_EYE_ANNOTATED_VIDEO_PATH = EYE_ANNOTATED_VIDEOS_FOLDER / "eye1_clipped_4469_11638.mp4"
LEFT_EYE_RAW_VIDEO_PATH = EYE_SYNCHRONIZED_VIDEOS_FOLDER / "eye1_clipped_4469_11638.mp4"
LEFT_EYE_TIMESTAMPS_NPY_PATH = EYE_TIMESTAMPS_FOLDER / "eye1_clipped_4469_11638_timestamps.npy"
LEFT_EYE_DATA_CSV_PATH = EYE_OUTPUT_DATA_FOLDER / "eye1_clipped_4469_11638DLC_Resnet50_pupil_tracking_ferret_757_EyeCameras_P43_E15__1_shuffle1_snapshot_030.csv"

# Mocap data paths
MOCAP_DATA_FOLDER = CLIP_FOLDER / "mocap_data"
MOCAP_ANNOTATED_VIDEOS_FOLDER = MOCAP_DATA_FOLDER / "annotated_videos"
MOCAP_SYNCHRONIZED_VIDEOS_FOLDER = MOCAP_DATA_FOLDER / "synchronized_videos"
MOCAP_TIMESTAMPS_FOLDER = MOCAP_SYNCHRONIZED_VIDEOS_FOLDER / "timestamps"
MOCAP_OUTPUT_DATA_FOLDER = MOCAP_DATA_FOLDER / "output_data" / "dlc_output"

TOPDOWN_ANNOTATED_VIDEO_PATH = MOCAP_ANNOTATED_VIDEOS_FOLDER / "24676894_clipped_3377_8754.mp4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        "
TOPDOWN_RAW_VIDEO_PATH = MOCAP_SYNCHRONIZED_VIDEOS_FOLDER / "24676894_clipped_3377_8754.mp4"
TOPDOWN_TIMESTAMPS_NPY_PATH = MOCAP_TIMESTAMPS_FOLDER / "24676894_clipped_3377_8754.npy"


class MocapVideoData(BaseModel):
    """Model representing mocap video data and associated tracking."""
    annotated_video_path: Path
    raw_video_path: Path
    timestamps_npy_path: Path
    mocap_name: str
    raw_video_name: str = ""
    annotated_video_name: str = ""
    framerate: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration_seconds: float = 0.0
    frame_duration: float = 0.0
    resized_width: int = 0
    resized_height: int = 0
    timestamps_array: Optional[np.ndarray] = None
    annotated_vid_cap: Optional[cv2.VideoCapture] = None
    raw_vid_cap: Optional[cv2.VideoCapture] = None

    class Config:
        arbitrary_types_allowed = True

    def load_video_info(self) -> None:
        """Load video information from the video file."""
        if not Path(self.annotated_video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.annotated_video_path}")

        self.annotated_video_name = str(Path(self.annotated_video_path).stem)
        self.raw_video_name = str(Path(self.raw_video_path).stem)

        self.raw_vid_cap = cv2.VideoCapture(str(self.raw_video_path))
        if not self.raw_vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {self.raw_video_path}")

        self.framerate = self.raw_vid_cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.raw_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.raw_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.raw_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.annotated_vid_cap = cv2.VideoCapture(str(self.annotated_video_path))
        if not self.annotated_vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {self.annotated_video_path}")
        if not all([
            self.framerate == self.annotated_vid_cap.get(cv2.CAP_PROP_FPS),
            self.width == int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            self.height == int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.frame_count == int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        ]):
            raise ValueError(
                f"Video properties do not match between annotated and raw videos for {self.mocap_name} mocap: "
                f"Raw properties: \n"
                f"Framerate: {self.framerate}, Width: {self.width}, Height: {self.height}, Frame Count: {self.frame_count}\n"
                f"Annotated properties: \n"
                f"Framerate: {self.annotated_vid_cap.get(cv2.CAP_PROP_FPS)}, "
                f"Width: {int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, "
                f"Height: {int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                f"Frame Count: {int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        self.duration_seconds = self.frame_count / self.framerate
        self.frame_duration = 1.0 / self.framerate
        # Calculate new dimensions if resizing
        self.resized_width = int(self.width * RESIZE_FACTOR)
        self.resized_height = int(self.height * RESIZE_FACTOR)
        print(f"{self.mocap_name} video info: {self.width}x{self.height} @ {self.framerate} FPS, "
              f"{self.frame_count} frames, {self.duration_seconds:.1f}s duration")
        if RESIZE_FACTOR != 1.0:
            print(f"Resizing to: {self.resized_width}x{self.resized_height}")
        # Create time array
        # Create time array
        self.timestamps_array = np.load(self.timestamps_npy_path)
        if len(self.timestamps_array) != self.frame_count:
            raise ValueError(
                f"Expected {self.frame_count} timestamps, but found {len(self.timestamps_array)} in NPY data.")


class EyeVideoData(BaseModel):
    """Model representing eye video data and associated pupil tracking."""
    annotated_video_path: Path
    raw_video_path: Path
    eye_data_csv_path: Path
    timestamps_npy_path: Path
    eye_name: str
    raw_video_name: str = ""
    annotated_video_name: str = ""
    framerate: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration_seconds: float = 0.0
    frame_duration: float = 0.0
    resized_width: int = 0
    resized_height: int = 0
    pupil_x: Optional[np.ndarray] = None
    pupil_y: Optional[np.ndarray] = None
    timestamps_array: Optional[np.ndarray] = None
    annotated_vid_cap: Optional[cv2.VideoCapture] = None
    raw_vid_cap: Optional[cv2.VideoCapture] = None

    class Config:
        arbitrary_types_allowed = True

    def load_video_info(self) -> None:
        """Load video information from the video file."""
        if not Path(self.annotated_video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.annotated_video_path}")

        self.annotated_video_name = str(Path(self.annotated_video_path).stem)
        self.raw_video_name = str(Path(self.raw_video_path).stem)

        self.raw_vid_cap = cv2.VideoCapture(str(self.raw_video_path))
        if not self.raw_vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {self.raw_video_path}")

        self.framerate = self.raw_vid_cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.raw_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.raw_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.raw_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.annotated_vid_cap = cv2.VideoCapture(str(self.annotated_video_path))
        if not self.annotated_vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {self.annotated_video_path}")
        if not all([
            self.framerate == self.annotated_vid_cap.get(cv2.CAP_PROP_FPS),
            self.width == int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            self.height == int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.frame_count == int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        ]):
            raise ValueError(f"Video properties do not match between annotated and raw videos for {self.eye_name} eye: "
                             f"Raw properties: \n"
                             f"Framerate: {self.framerate}, Width: {self.width}, Height: {self.height}, Frame Count: {self.frame_count}\n"
                             f"Annotated properties: \n"
                             f"Framerate: {self.annotated_vid_cap.get(cv2.CAP_PROP_FPS)}, "
                             f"Width: {int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, "
                             f"Height: {int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                             f"Frame Count: {int(self.annotated_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

        self.duration_seconds = self.frame_count / self.framerate
        self.frame_duration = 1.0 / self.framerate

        # Calculate new dimensions if resizing
        self.resized_width = int(self.width * RESIZE_FACTOR)
        self.resized_height = int(self.height * RESIZE_FACTOR)

        print(f"{self.eye_name} video info: {self.width}x{self.height} @ {self.framerate} FPS, "
              f"{self.frame_count} frames, {self.duration_seconds:.1f}s duration")
        if RESIZE_FACTOR != 1.0:
            print(f"Resizing to: {self.resized_width}x{self.resized_height}")

    def load_pupil_data(self, pupil_point_name: str = GOOD_PUPIL_POINT) -> None:
        """Load pupil tracking data from CSV."""
        if not Path(self.eye_data_csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.eye_data_csv_path}")

        # Skip the first row (scorer) and use the second and third rows as the header
        # The second row contains bodyparts, third row contains coords (x, y, likelihood)
        pupil_df = pd.read_csv(self.eye_data_csv_path, header=[0,
                                                               1])  # , skiprows=[0]) # Uncomment skiprows if dealing with data with the dumb scorer row still present

        # Check if the pupil point exists in the bodyparts level
        if pupil_point_name not in pupil_df.columns.get_level_values(0):
            raise ValueError(f"Expected bodypart '{pupil_point_name}' not found in CSV data.")

        # Extract x and y coordinates for the specified pupil point
        pupil_x = pupil_df[pupil_point_name, 'x']
        pupil_y = pupil_df[pupil_point_name, 'y']

        if len(pupil_x) != self.frame_count:
            print(f"Warning: Expected {self.frame_count} pupil points, but found {len(pupil_x)} in CSV data.")

        # Convert to numpy arrays for faster processing
        self.pupil_x = pupil_x.to_numpy()
        self.pupil_y = pupil_y.to_numpy()

        print(f"Loaded pupil data for {self.eye_name} eye: {len(self.pupil_x)} points")

        # Create time array
        self.timestamps_array = np.load(self.timestamps_npy_path).astype(np.float64)
        if len(self.timestamps_array) != self.frame_count:
            raise ValueError(
                f"Expected {self.frame_count} timestamps, but found {len(self.timestamps_array)} in NPY data.")


def process_eye_data(recording_name: str,
                     left_eye_video_data: EyeVideoData,
                     right_eye_video_data: EyeVideoData,
                     topdown_mocap_video: MocapVideoData,
                     start_time: float | None = None,
                     end_time: float | None = None
                     ) -> None:
    """Process both eye videos and visualize them with Rerun."""
    # Initialize Rerun
    rr.init(recording_name, spawn=True)

    # Define a blueprint with separate views for both eyes
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Vertical(
                # rrb.Spatial2DView(name="TopDown Mocap Video(Annotated)",
                #                   origin=f"/mocap_video/top_down/annotated",
                #                   visual_bounds=VisualBounds2D.from_fields(
                #                       range=Range2D(
                #                           x_range=(0, topdown_mocap_video.resized_width),
                #                           y_range=(0, topdown_mocap_video.resized_height)
                #                       )
                #                   ),
                #                   visible=False
                #                   ),
                rrb.Spatial2DView(name="TopDown Mocap Video(Raw)",
                                  origin=f"/mocap_video/top_down/raw",
                                  visual_bounds=VisualBounds2D.from_fields(
                                      range=Range2D(
                                          x_range=(0, topdown_mocap_video.resized_width),
                                          y_range=(0, topdown_mocap_video.resized_height)
                                      )
                                  ),
                                  ),
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial2DView(name="Right Eye Video (Annotated)",
                                          origin=f"/right_eye_video/annotated",
                                          visual_bounds=VisualBounds2D.from_fields(
                                              range=Range2D(
                                                  x_range=(0, right_eye_video_data.resized_width),
                                                  y_range=(0, right_eye_video_data.resized_height)
                                              )
                                          ),
                                          visible=False
                                          ),
                        rrb.Spatial2DView(name="Right Eye Video (Raw)",
                                          origin=f"/right_eye_video/raw",
                                          visual_bounds=VisualBounds2D.from_fields(
                                              range=Range2D(
                                                  x_range=(0, right_eye_video_data.resized_width),
                                                  y_range=(0, right_eye_video_data.resized_height)
                                              )
                                          ),
                                          ),
                        rrb.TimeSeriesView(name="Right Eye Horizontal Position",
                                           contents=[f"+ /right_eye/pupil_x_line",
                                                     f"+ /right_eye/pupil_x_dots"],
                                           axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal)),
                        rrb.TimeSeriesView(name="Right Eye Vertical Position",
                                           contents=[f"+ /right_eye/pupil_y_line",
                                                     f"+ /right_eye/pupil_y_dots"],
                                           axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal)),
                    ),
                    rrb.Vertical(
                        rrb.Spatial2DView(name="Left Eye Video (Annotated)",
                                          origin=f"/left_eye_video/annotated",
                                          visual_bounds=VisualBounds2D.from_fields(
                                              range=Range2D(
                                                  x_range=(0, left_eye_video_data.resized_width),
                                                  y_range=(0, left_eye_video_data.resized_height)
                                              )
                                          ),
                                          visible=False
                                          ),
                        rrb.Spatial2DView(name="Left Eye Video (Raw)",
                                          origin=f"/left_eye_video/raw",
                                          visual_bounds=VisualBounds2D.from_fields(
                                              range=Range2D(
                                                  x_range=(0, left_eye_video_data.resized_width),
                                                  y_range=(0, left_eye_video_data.resized_height)
                                              )
                                          ),
                                          ),
                        rrb.TimeSeriesView(name="Left Eye Horizontal Position",
                                           contents=[f"+ /left_eye/pupil_x_line",
                                                     f"+ /left_eye/pupil_x_dots"],
                                           axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal),
                                           ),

                        rrb.TimeSeriesView(name="Left Eye Vertical Position",
                                           contents=[f"+ /left_eye/pupil_y_line",
                                                     f"+ /left_eye/pupil_y_dots"],
                                           axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal)),
                    ),

                ),
            ),
        ),
        rrb.BlueprintPanel(state="expanded"),
    )
    rr.send_blueprint(blueprint)

    # Setup optimized encoding pipeline for both eyes
    av.logging.set_level(av.logging.ERROR)  # Reduce logging noise

    right_eye_horizontal_color = [255, 0, 0]
    right_eye_vertical_color = [255, 0, 255]
    left_eye_horizontal_color = [0, 255, 255]
    left_eye_vertical_color = [65, 85, 255]
    # Log static data for both eyes
    for eye, prefix in [(right_eye_video_data, "right_eye"), (left_eye_video_data, "left_eye")]:
        # Video stream
        # Filter timestamps and pupil data based on time range

        if start_time is not None or end_time is not None:
            # Create mask for timestamps within range
            eye_timestamp_mask = np.ones(len(eye.timestamps_array), dtype=bool)
            if start_time is not None:
                eye_timestamp_mask = eye_timestamp_mask & (eye.timestamps_array >= start_time)
            if end_time is not None:
                eye_timestamp_mask = eye_timestamp_mask & (eye.timestamps_array <= end_time)
        else:
            eye_timestamp_mask = np.ones(len(eye.timestamps_array), dtype=bool)

        # Apply the mask consistently to all data
        eye_filtered_timestamps = eye.timestamps_array[eye_timestamp_mask]
        filtered_pupil_x = eye.pupil_x[eye_timestamp_mask]
        filtered_pupil_y = eye.pupil_y[eye_timestamp_mask]

        # Time series components
        rr.log(f"{prefix}/pupil_x_line",
               rr.SeriesLines(colors=right_eye_horizontal_color if prefix == "right_eye" else left_eye_horizontal_color,
                              names=f"{eye.eye_name} Horizontal Position",
                              widths=2),
               static=True)
        rr.log(f"{prefix}/pupil_y_line",
               rr.SeriesLines(colors=right_eye_vertical_color if prefix == "right_eye" else left_eye_vertical_color,
                              names=f"{eye.eye_name} Vertical Position",
                              widths=2),
               static=True)
        rr.log(f"{prefix}/pupil_x_dots",
               rr.SeriesPoints(
                   colors=right_eye_horizontal_color if prefix == "right_eye" else left_eye_horizontal_color,
                   names=f"{eye.eye_name} Horizontal Position",
                   markers="circle",
                   marker_sizes=2),
               static=True)
        rr.log(f"{prefix}/pupil_y_dots",
               rr.SeriesPoints(colors=right_eye_vertical_color if prefix == "right_eye" else left_eye_vertical_color,
                               names=f"{eye.eye_name} Vertical Position",
                               markers="circle",
                               marker_sizes=2),
               static=True)

        rr.send_columns(
            entity_path=f"{prefix}/pupil_x_line",
            indexes=[rr.TimeColumn("time", duration=eye_filtered_timestamps)],
            columns=rr.Scalars.columns(scalars=filtered_pupil_x),
        )
        rr.send_columns(
            entity_path=f"{prefix}/pupil_y_line",
            indexes=[rr.TimeColumn("time", duration=eye_filtered_timestamps)],
            columns=rr.Scalars.columns(scalars=filtered_pupil_y),
        )
        rr.send_columns(
            entity_path=f"{prefix}/pupil_x_dots",
            indexes=[rr.TimeColumn("time", duration=eye_filtered_timestamps)],
            columns=rr.Scalars.columns(scalars=filtered_pupil_x),
        )
        rr.send_columns(
            entity_path=f"{prefix}/pupil_y_dots",
            indexes=[rr.TimeColumn("time", duration=eye_filtered_timestamps)],
            columns=rr.Scalars.columns(scalars=filtered_pupil_y),

        )

        # Process video frames

        for video_type, cap in [("raw", eye.raw_vid_cap), ("annotated", eye.annotated_vid_cap)]:
            video_entity_path = f"{prefix}_video/{video_type}"
            rr.log(video_entity_path, rr.VideoStream(codec=rr.VideoCodec.H264))
            print(f"Processing {eye.eye_name} eye {video_type} video frames...")
            encoded_image_blobs = []
            for frame_number in range(0, eye.frame_count):
                # Process left eye
                success, image = cap.read()
                if not success:
                    print(f"Failed to read  eye frame {frame_number}, stopping.")
                    break
                if start_time is not None and eye.timestamps_array[frame_number] < start_time:
                    continue
                if end_time is not None and eye.timestamps_array[frame_number] > end_time:
                    continue
                if prefix == "left_eye":
                    image = cv2.flip(image, 1)  # Mirror left so tear ducts face inwards
                # Resize if needed
                if RESIZE_FACTOR != 1.0:
                    resized_image = cv2.resize(image, (eye.resized_width, eye.resized_height))
                else:
                    resized_image = image

                encoded_image_blobs.append(cv2.imencode('.jpg', resized_image,
                                                        [cv2.IMWRITE_JPEG_QUALITY,
                                                         80])[1].tobytes())

                # Print progress every 100 frames
                if frame_number % 1000 == 0:
                    print(
                        f"Processed {frame_number}/{eye.frame_count} frames ({frame_number / eye.frame_count * 100:.1f}%)")
            rr.send_columns(
                entity_path=video_entity_path,
                indexes=[rr.TimeColumn("time", duration=eye_filtered_timestamps)],
                columns=rr.EncodedImage.columns(blob=encoded_image_blobs,
                                                media_type=['image/jpeg'] * len(encoded_image_blobs))
            )

    if start_time is not None or end_time is not None:
        # Create mask for timestamps within range
        mocap_timestamp_mask = np.ones(len(topdown_mocap_video.timestamps_array), dtype=bool)
        if start_time is not None:
            mocap_timestamp_mask = mocap_timestamp_mask & (topdown_mocap_video.timestamps_array >= start_time)
        if end_time is not None:
            mocap_timestamp_mask = mocap_timestamp_mask & (topdown_mocap_video.timestamps_array <= end_time)
    else:
        mocap_timestamp_mask = np.ones(len(topdown_mocap_video.timestamps_array), dtype=bool)

    # Apply the mask consistently to all data
    topdown_mocap_video_timestamps = topdown_mocap_video.timestamps_array[mocap_timestamp_mask]
    for video_type, cap in [
        ("raw", topdown_mocap_video.raw_vid_cap)]:  # , ("annotated", topdown_mocap_video.annotated_vid_cap)]:
        video_entity_path = f"mocap_video/top_down/{video_type}"
        rr.log(video_entity_path, rr.VideoStream(codec=rr.VideoCodec.H264))
        print(f"Processing TopDown Mocap {video_type} video frames...")
        encoded_image_blobs = []
        for frame_number in range(0, eye.frame_count):
            # Process left eye
            success, image = cap.read()
            if not success:
                print(f"Failed to read topdown mocap video frame {frame_number}, stopping.")
                break
            if start_time is not None and topdown_mocap_video.timestamps_array[frame_number] < start_time:
                continue
            if end_time is not None and topdown_mocap_video.timestamps_array[frame_number] > end_time:
                continue
            # Resize if needed
            if RESIZE_FACTOR != 1.0:
                resized_image = cv2.resize(image,
                                           (topdown_mocap_video.resized_width, topdown_mocap_video.resized_height))
            else:
                resized_image = image

            encoded_image_blobs.append(cv2.imencode('.jpg', resized_image,
                                                    [cv2.IMWRITE_JPEG_QUALITY,
                                                     80])[1].tobytes())

            # Print progress every 100 frames
            if frame_number % 1000 == 0:
                print(
                    f"Processed {frame_number}/{topdown_mocap_video.frame_count} frames ({frame_number / topdown_mocap_video.frame_count * 100:.1f}%)")
        rr.send_columns(
            entity_path=video_entity_path,
            indexes=[rr.TimeColumn("time", duration=topdown_mocap_video_timestamps)],
            columns=rr.EncodedImage.columns(blob=encoded_image_blobs,
                                            media_type=['image/jpeg'] * len(encoded_image_blobs))
        )
    print(f"Processing complete!")


def main_rerun_viewer_maker(start_time: float | None = None, end_time: float | None = None):
    """Main function to run the eye tracking visualization."""

    # Create eye data models
    left_eye = EyeVideoData(
        annotated_video_path=LEFT_EYE_ANNOTATED_VIDEO_PATH,
        raw_video_path=LEFT_EYE_RAW_VIDEO_PATH,
        timestamps_npy_path=LEFT_EYE_TIMESTAMPS_NPY_PATH,
        eye_data_csv_path=LEFT_EYE_DATA_CSV_PATH,
        eye_name="Left"
    )
    left_eye.load_video_info()
    left_eye.load_pupil_data()

    right_eye = EyeVideoData(
        annotated_video_path=RIGHT_EYE_ANNOTATED_VIDEO_PATH,
        raw_video_path=RIGHT_EYE_RAW_VIDEO_PATH,
        timestamps_npy_path=RIGHT_EYE_TIMESTAMPS_NPY_PATH,
        eye_data_csv_path=RIGHT_EYE_DATA_CSV_PATH,
        eye_name="Right"
    )
    right_eye.load_video_info()
    right_eye.load_pupil_data()

    topdown_mocap_video = MocapVideoData(
        annotated_video_path=TOPDOWN_RAW_VIDEO_PATH,
        raw_video_path=TOPDOWN_RAW_VIDEO_PATH,
        timestamps_npy_path=TOPDOWN_TIMESTAMPS_NPY_PATH,
        mocap_name="TopDown Mocap"
    )
    topdown_mocap_video.load_video_info()
    recording_start_time = np.min([
        float(left_eye.timestamps_array[0]),
        float(right_eye.timestamps_array[0]),
        float(topdown_mocap_video.timestamps_array[0]),
    ])

    left_eye.timestamps_array -= recording_start_time
    right_eye.timestamps_array -= recording_start_time
    topdown_mocap_video.timestamps_array -= recording_start_time

    # Process and visualize the eye videos
    process_eye_data(left_eye_video_data=left_eye,
                     right_eye_video_data=right_eye,
                     topdown_mocap_video=topdown_mocap_video,
                     recording_name=RECORDING_NAME,
                     start_time=start_time,
                     end_time=end_time)


if __name__ == "__main__":
    main_rerun_viewer_maker()
