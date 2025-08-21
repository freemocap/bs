"""Video encode images using av and stream them to Rerun with optimized performance."""

import time
from pathlib import Path
from typing import Optional, Tuple

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
RECORDING_NAME = "ferret_757_EyeCameras_P43_E15__1"
RIGHT_EYE_VIDEO_PATH = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\basler_pupil_synchronized\synchronized_eye_videos\eye0.mp4")
RIGHT_EYE_TIMESTAMPS_NPY_PATH = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\basler_pupil_synchronized\synchronized_eye_videos\cam_eye1_synchronized_timestamps.npy")
RIGHT_EYE_DATA_CSV_PATH = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\dlc_pupil_tracking\new_model_iteration_2\eye0DLC_Resnet50_pupil_tracking_ferret_757_EyeCameras_P43_E15__1_shuffle1_snapshot_030.csv")

LEFT_EYE_VIDEO_PATH = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\basler_pupil_synchronized\synchronized_eye_videos\eye1.mp4")
LEFT_EYE_TIMESTAMPS_NPY_PATH = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\basler_pupil_synchronized\synchronized_eye_videos\cam_eye1_synchronized_timestamps.npy")
LEFT_EYE_DATA_CSV_PATH = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\dlc_pupil_tracking\new_model_iteration_2\eye0DLC_Resnet50_pupil_tracking_ferret_757_EyeCameras_P43_E15__1_shuffle1_snapshot_030.csv")


class EyeVideoData(BaseModel):
    """Model representing eye video data and associated pupil tracking."""
    video_path: Path
    eye_data_csv_path: Path
    timestamps_npy_path: Path
    eye_name: str
    video_name: str = ""
    framerate: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration_seconds: float = 0.0
    frame_duration: float = 0.0
    new_width: int = 0
    new_height: int = 0
    pupil_x: Optional[np.ndarray] = None
    pupil_y: Optional[np.ndarray] = None
    timestamps_array: Optional[np.ndarray] = None
    vid_cap: Optional[cv2.VideoCapture] = None

    class Config:
        arbitrary_types_allowed = True

    def load_video_info(self) -> None:
        """Load video information from the video file."""
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.video_name = str(Path(self.video_path).stem)
        self.vid_cap = cv2.VideoCapture(str(self.video_path))
        if not self.vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        self.framerate = self.vid_cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.frame_count / self.framerate
        self.frame_duration = 1.0 / self.framerate

        # Calculate new dimensions if resizing
        self.new_width = int(self.width * RESIZE_FACTOR)
        self.new_height = int(self.height * RESIZE_FACTOR)

        print(f"{self.eye_name} video info: {self.width}x{self.height} @ {self.framerate} FPS, "
              f"{self.frame_count} frames, {self.duration_seconds:.1f}s duration")
        if RESIZE_FACTOR != 1.0:
            print(f"Resizing to: {self.new_width}x{self.new_height}")

    def load_pupil_data(self, pupil_point_name: str) -> None:
        """Load pupil tracking data from CSV."""
        if not Path(self.eye_data_csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.eye_data_csv_path}")

        # Skip the first row (scorer) and use the second and third rows as the header
        # The second row contains bodyparts, third row contains coords (x, y, likelihood)
        pupil_df = pd.read_csv(self.eye_data_csv_path, header=[0, 1], skiprows=[0])

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
        self.timestamps_array = np.arange(self.frame_count) * self.frame_duration  # FAKE TIMESTAMPS WILL BE REPLACED
        if len(self.timestamps_array) != self.frame_count:
            raise ValueError(
                f"Expected {self.frame_count} timestamps, but found {len(self.timestamps_array)} in NPY data.")

    def setup_video_encoder(self) -> Tuple[av.container.OutputContainer, av.stream.Stream]:
        """Set up the video encoder for this eye."""
        container = av.open("/dev/null", "w", format="h264")
        stream = container.add_stream("libx264", rate=int(self.framerate))
        stream.width = self.new_width
        stream.height = self.new_height
        stream.max_b_frames = 0  # Keep this for compatibility with Rerun
        stream.pix_fmt = "yuv420p"  # More efficient pixel format
        stream.options = {
            "crf": str(COMPRESSION_LEVEL),  # Higher = more compression
            "preset": "slow",  # Slower = better compression
            "tune": "zerolatency",  # Good for streaming
            "profile": "baseline",  # More compatible profile
        }
        return container, stream


def process_eye_data(recording_name: str,
                     left_eye: EyeVideoData,
                     right_eye: EyeVideoData,
                     start_time: float | None = None,
                     end_time: float | None = None
                     ) -> None:
    """Process both eye videos and visualize them with Rerun."""
    # Initialize Rerun
    rr.init(recording_name, spawn=True)

    # Define a blueprint with separate views for both eyes
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Left Eye Video",
                                  origin=f"/{left_eye.video_name}",
                                  visual_bounds=VisualBounds2D.from_fields(
                                      range=Range2D(
                                          x_range=(0, left_eye.new_width),
                                          y_range=(0, left_eye.new_height)
                                      )
                                  )),
                rrb.Spatial2DView(name="Right Eye Video",
                                  origin=f"/{right_eye.video_name}",
                                  visual_bounds=VisualBounds2D.from_fields(
                                      range=Range2D(
                                          x_range=(0, right_eye.new_width),
                                          y_range=(0, right_eye.new_height)
                                      )
                                  )),
            ),
            rrb.Horizontal(
                rrb.Vertical(
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
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Right Eye Horizontal Position",
                                       contents=[f"+ /right_eye/pupil_x_line",
                                                 f"+ /right_eye/pupil_x_dots"],
                                       axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal)),
                    rrb.TimeSeriesView(name="Right Eye Vertical Position",
                                       contents=[f"+ /right_eye/pupil_y_line",
                                                 f"+ /right_eye/pupil_y_dots"],
                                        axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal)),
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
    left_eye_horizontal_color = [0, 0, 255]
    left_eye_vertical_color = [0, 255, 255]
    # Log static data for both eyes
    for eye, prefix in [(left_eye, "left_eye"), (right_eye, "right_eye")]:
        # Video stream
        rr.log(eye.video_name, rr.VideoStream(codec=rr.VideoCodec.H264))

        # Filter timestamps and pupil data based on time range

        if start_time is not None or end_time is not None:
            # Create mask for timestamps within range
            timestamp_mask = np.ones(len(eye.timestamps_array), dtype=bool)
            if start_time is not None:
                timestamp_mask = timestamp_mask & (eye.timestamps_array >= start_time)
            if end_time is not None:
                timestamp_mask = timestamp_mask & (eye.timestamps_array <= end_time)
        else:
            timestamp_mask = np.ones(len(eye.timestamps_array), dtype=bool)

        # Apply the mask consistently to all data
        filtered_timestamps = eye.timestamps_array[timestamp_mask]
        filtered_pupil_x = eye.pupil_x[timestamp_mask]
        filtered_pupil_y = eye.pupil_y[timestamp_mask]

        rr.set_time("time", timestamp=float(filtered_timestamps[0]))  # Set initial time to first timestamp

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
            indexes=[rr.TimeColumn("time", timestamp=filtered_timestamps)],
            columns=rr.Scalars.columns(scalars=filtered_pupil_x),
        )
        rr.send_columns(
            entity_path=f"{prefix}/pupil_y_line",
            indexes=[rr.TimeColumn("time", timestamp=filtered_timestamps)],
            columns=rr.Scalars.columns(scalars=filtered_pupil_y),
        )
        rr.send_columns(
            entity_path=f"{prefix}/pupil_x_dots",
            indexes=[rr.TimeColumn("time", timestamp=filtered_timestamps)],
            columns=rr.Scalars.columns(scalars=filtered_pupil_x),
        )
        rr.send_columns(
            entity_path=f"{prefix}/pupil_y_dots",
            indexes=[rr.TimeColumn("time", timestamp=filtered_timestamps)],
            columns=rr.Scalars.columns(scalars=filtered_pupil_y),

        )

        # Process video frames

        # Determine the number of frames to process (minimum of both eyes)

        for frame_number in range(0, eye.frame_count):
            # Process left eye
            success, image = eye.vid_cap.read()
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
                resized_image = cv2.resize(image, (eye.new_width, eye.new_height))
            else:
                resized_image = image

            rr.set_time("time", timestamp=float(eye.timestamps_array[frame_number]))
            rr.log(eye.video_name, rr.EncodedImage.from_fields(media_type='image/jpeg',
                                                               blob=cv2.imencode('.jpg', resized_image,
                                                                                 [cv2.IMWRITE_JPEG_QUALITY, 90])[
                                                                   1].tobytes()))

            # Print progress every 100 frames
            if frame_number % 100 == 0:
                print(
                    f"Processed {frame_number}/{eye.frame_count} frames ({frame_number / eye.frame_count * 100:.1f}%)")

    print(f"Processing complete in {time.time() - start_time:.1f}s")


def main_rerun_eye_viewer(start_time: float | None = None, end_time: float | None = None):
    """Main function to run the eye tracking visualization."""

    # Create eye data models
    left_eye = EyeVideoData(
        video_path=LEFT_EYE_VIDEO_PATH,
        timestamps_npy_path=LEFT_EYE_TIMESTAMPS_NPY_PATH,
        eye_data_csv_path=LEFT_EYE_DATA_CSV_PATH,
        eye_name="Left"
    )

    right_eye = EyeVideoData(
        video_path=RIGHT_EYE_VIDEO_PATH,
        timestamps_npy_path=RIGHT_EYE_TIMESTAMPS_NPY_PATH,
        eye_data_csv_path=RIGHT_EYE_DATA_CSV_PATH,
        eye_name="Right"
    )

    # Load video info and pupil data
    for eye in [left_eye, right_eye]:
        eye.load_video_info()
        eye.load_pupil_data(GOOD_PUPIL_POINT)

    # Process and visualize the eye videos
    process_eye_data(left_eye=left_eye,
                     right_eye=right_eye,
                     recording_name=RECORDING_NAME,
                     start_time=start_time,
                     end_time=end_time)


if __name__ == "__main__":
    main_rerun_eye_viewer(start_time=10.0, end_time=70.0)
