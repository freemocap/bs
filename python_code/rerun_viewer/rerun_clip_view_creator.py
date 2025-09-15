"""Video encode images using av and stream them to Rerun with optimized performance."""

from pathlib import Path
from typing import Optional

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
GOOD_PUPIL_POINT = "p2"
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)

# TODO: keep fixing paths

# Define paths
RECORDING_NAME = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
CLIP_NAME = "0_37-1_37"

BASE_RECORDINGS_FOLDER = Path("/home/scholl-lab/ferret_recordings")
RECORDING_FOLDER = BASE_RECORDINGS_FOLDER / RECORDING_NAME
CLIP_FOLDER = RECORDING_FOLDER / "clips" / CLIP_NAME

# Eye data paths
EYE_DATA_FOLDER = CLIP_FOLDER / "eye_data"
EYE_ANNOTATED_VIDEOS_FOLDER = EYE_DATA_FOLDER / "annotated_videos"
EYE_SYNCHRONIZED_VIDEOS_FOLDER = EYE_DATA_FOLDER / "eye_videos"
EYE_TIMESTAMPS_FOLDER = EYE_SYNCHRONIZED_VIDEOS_FOLDER
EYE_OUTPUT_DATA_FOLDER = EYE_DATA_FOLDER / "dlc_output"

EYE_DATA_CSV_PATH = list(EYE_OUTPUT_DATA_FOLDER.glob("skellyclicker_machine_labels*.csv"))[0]

RIGHT_EYE_ANNOTATED_VIDEO_PATH = list(EYE_ANNOTATED_VIDEOS_FOLDER.glob("eye0*.mp4"))[0]
RIGHT_EYE_RAW_VIDEO_PATH = list(EYE_SYNCHRONIZED_VIDEOS_FOLDER.glob("eye0*.mp4"))[0]
RIGHT_EYE_TIMESTAMPS_NPY_PATH = list(EYE_TIMESTAMPS_FOLDER.glob("eye0_timestamps*.npy"))[0]


LEFT_EYE_ANNOTATED_VIDEO_PATH = list(EYE_ANNOTATED_VIDEOS_FOLDER.glob("eye1*.mp4"))[0]
LEFT_EYE_RAW_VIDEO_PATH = list(EYE_SYNCHRONIZED_VIDEOS_FOLDER.glob("eye1*.mp4"))[0]
LEFT_EYE_TIMESTAMPS_NPY_PATH = list(EYE_TIMESTAMPS_FOLDER.glob("eye1_timestamps*.npy"))[0]

# Mocap data paths
MOCAP_DATA_FOLDER = CLIP_FOLDER / "mocap_data"
MOCAP_ANNOTATED_VIDEOS_FOLDER = MOCAP_DATA_FOLDER / "annotated_videos"
MOCAP_SYNCHRONIZED_VIDEOS_FOLDER = MOCAP_DATA_FOLDER / "synchronized_videos"
MOCAP_TIMESTAMPS_FOLDER = MOCAP_SYNCHRONIZED_VIDEOS_FOLDER
MOCAP_OUTPUT_DATA_FOLDER = MOCAP_DATA_FOLDER / "output_data" / "dlc_output"

TOPDOWN_VIDEO_NAME = "24676894"

TOPDOWN_ANNOTATED_VIDEO_PATH = list(MOCAP_ANNOTATED_VIDEOS_FOLDER.glob(f"{TOPDOWN_VIDEO_NAME}*.mp4"))[0]
TOPDOWN_RAW_VIDEO_PATH = list(MOCAP_SYNCHRONIZED_VIDEOS_FOLDER.glob(f"{TOPDOWN_VIDEO_NAME}*.mp4"))[0]
TOPDOWN_TIMESTAMPS_NPY_PATH = list(MOCAP_TIMESTAMPS_FOLDER.glob(f"{TOPDOWN_VIDEO_NAME}*utc*.npy"))[0]


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
    resized_width: int
    resized_height: int
    timestamps_array: np.ndarray
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
        resized_width = int(width * RESIZE_FACTOR)
        resized_height = int(height * RESIZE_FACTOR)

        print(f"{data_name} video info: {width}x{height} @ {framerate} FPS, "
              f"{frame_count} frames, {duration_seconds:.1f}s duration")
        if RESIZE_FACTOR != 1.0:
            print(f"Resizing to: {resized_width}x{resized_height}")

        # Load timestamps
        timestamps_array = np.load(timestamps_npy_path).astype(np.float64) / 1e9  # Convert from nanoseconds to seconds
        if len(timestamps_array) != frame_count:
            raise ValueError(
                f"Expected {frame_count} timestamps, but found {len(timestamps_array)} in NPY data.")

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
            resized_width=resized_width,
            resized_height=resized_height,
            timestamps_array=timestamps_array,
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


def create_rerun_recording(recording_name: str,
                           left_eye_video_data: EyeVideoData,
                           right_eye_video_data: EyeVideoData,
                           topdown_mocap_video: MocapVideoData,
                           ) -> None:
    """Process both eye videos and visualize them with Rerun."""
    # Initialize Rerun
    rr.init(recording_name, spawn=True)

    videos_blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(name="Right Eye Video (Annotated)",
                              origin=f"/right_eye/video/annotated",
                              visual_bounds=VisualBounds2D.from_fields(
                                  range=Range2D(
                                      x_range=(0, right_eye_video_data.resized_width),
                                      y_range=(0, right_eye_video_data.resized_height)
                                  )
                              ),
                              visible=False
                              ),
            rrb.Spatial2DView(name="Right Eye Video (Raw)",
                              origin=f"/right_eye/video/raw",
                              visual_bounds=VisualBounds2D.from_fields(
                                  range=Range2D(
                                      x_range=(0, right_eye_video_data.resized_width),
                                      y_range=(0, right_eye_video_data.resized_height)
                                  )
                              ),
                              ),
        ),
        rrb.Vertical(
            rrb.Spatial2DView(name="Left Eye Video (Annotated)",
                              origin=f"/left_eye/video/annotated",
                              visual_bounds=VisualBounds2D.from_fields(
                                  range=Range2D(
                                      x_range=(0, left_eye_video_data.resized_width),
                                      y_range=(0, left_eye_video_data.resized_height)
                                  )
                              ),
                              visible=False
                              ),
            rrb.Spatial2DView(name="Left Eye Video (Raw)",
                              origin=f"/left_eye/video/raw",
                              visual_bounds=VisualBounds2D.from_fields(
                                  range=Range2D(
                                      x_range=(0, left_eye_video_data.resized_width),
                                      y_range=(0, left_eye_video_data.resized_height)
                                  )
                              ),
                              )),
        rrb.Vertical(
            rrb.Spatial2DView(name="TopDown Mocap Video(Annotated)",
                              origin=f"/mocap_video/top_down/annotated",
                              visual_bounds=VisualBounds2D.from_fields(
                                  range=Range2D(
                                      x_range=(0, topdown_mocap_video.resized_width),
                                      y_range=(0, topdown_mocap_video.resized_height)
                                  )
                              ),
                              ),
            rrb.Spatial2DView(name="TopDown Mocap Video(Raw)",
                              origin=f"/mocap_video/top_down/raw",
                              visual_bounds=VisualBounds2D.from_fields(
                                  range=Range2D(
                                      x_range=(0, topdown_mocap_video.resized_width),
                                      y_range=(0, topdown_mocap_video.resized_height)
                                  )
                              ),
                              visible=False
                              ),

        ))
    eye_timeseries_blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.TimeSeriesView(name="Right Eye Horizontal Position",
                               contents=[f"+ right_eye/pupil_x_line",
                                         f"+ right_eye/pupil_x_dots"],
                               axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal)),
            rrb.TimeSeriesView(name="Left Eye Horizontal Position",
                               contents=[f"+ left_eye/pupil_x_line",
                                         f"+ left_eye/pupil_x_dots"],
                               axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal),
                               ),

            rrb.TimeSeriesView(name="Right Eye Vertical Position",
                               contents=[f"+ right_eye/pupil_y_line",
                                         f"+ right_eye/pupil_y_dots"],
                               axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal)),

            rrb.TimeSeriesView(name="Left Eye Vertical Position",
                               contents=[f"+ left_eye/pupil_y_line",
                                         f"+ left_eye/pupil_y_dots"],
                               axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal)),
        ))
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            videos_blueprint,
            eye_timeseries_blueprint
        ),
        rrb.BlueprintPanel(state="expanded"),
    )

    rr.send_blueprint(blueprint)

    right_eye_horizontal_color = [255, 0, 0]
    right_eye_vertical_color = [255, 0, 255]
    left_eye_horizontal_color = [0, 255, 255]
    left_eye_vertical_color = [65, 85, 255]

    def process_video_frame(frame: np.ndarray,
                            resize_width: int,
                            resize_height: int,
                            flip_horizontal: bool = False,
                            jpeg_quality: int = 80) -> bytes:
        """Process a single video frame."""
        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        # Resize if needed
        if RESIZE_FACTOR != 1.0:
            frame = cv2.resize(frame, (resize_width, resize_height))

        # Encode to JPEG
        return cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes()

    def process_video_frames(video_cap: cv2.VideoCapture,
                             resize_width: int,
                             resize_height: int,
                             flip_horizontal: bool = False) -> list[bytes]:
        """Process a batch of video frames."""
        encoded_frames = []
        success = True
        while success:
            success, frame = video_cap.read()
            if not success:
                continue
            encoded_frames.append(process_video_frame(frame=frame,
                                                      resize_width=resize_width,
                                                      resize_height=resize_height,
                                                      flip_horizontal=flip_horizontal))

        return encoded_frames

    def process_video(video_data: VideoData, entity_path: str, flip_horizontal: bool = False):
        """Process a video and send it to Rerun."""
        print(f"Processing {video_data.data_name} video...")

        # Log video stream
        for video_type in ["annotated", "raw"]:
            encoded_frames = process_video_frames(
                video_cap=video_data.raw_vid_cap if video_type == "raw" else video_data.annotated_vid_cap,
                resize_width=video_data.resized_width,
                resize_height=video_data.resized_height,
                flip_horizontal=flip_horizontal
            )

            rr.send_columns(
                entity_path=f"{entity_path}/{video_type}",
                indexes=[rr.TimeColumn("time", duration=video_data.timestamps_array)],
                columns=rr.EncodedImage.columns(
                    blob=encoded_frames,
                    media_type=['image/jpeg'] * len(encoded_frames))
            )

    # Process pupil tracking data for both eyes
    for eye, prefix, horizontal_color, vertical_color in [
        (right_eye_video_data, "right_eye", right_eye_horizontal_color, right_eye_vertical_color),
        (left_eye_video_data, "left_eye", left_eye_horizontal_color, left_eye_vertical_color)
    ]:
        print(f"Processing {eye.data_name} pupil tracking data...")

        for data_type, color, data in [
            ("pupil_x_line", horizontal_color, eye.pupil_x),
            ("pupil_x_dots", horizontal_color, eye.pupil_x),

            ("pupil_y_line", vertical_color, eye.pupil_y),
            ("pupil_y_dots", vertical_color, eye.pupil_y)
        ]:
            entity_path = f"{prefix}/{data_type}"
            print(f"Logging {entity_path}...")
            # Set up visualization properties
            if "line" in data_type:
                rr.log(entity_path,
                       rr.SeriesLines(colors=color,
                                      names=prefix,
                                      widths=2),
                       static=True)
            else:  # dots
                rr.log(entity_path,
                       rr.SeriesPoints(colors=color,
                                       names=prefix,
                                       markers="circle",
                                       marker_sizes=2),
                       static=True)

                # Send data
            rr.send_columns(
                entity_path=entity_path,
                indexes=[rr.TimeColumn("time", duration=eye.timestamps_array)],
                columns=rr.Scalars.columns(scalars=data),
            )
        # Log static data for time series

        # Process video
        process_video(
            video_data=eye,
            entity_path=f"{prefix}/video",
            flip_horizontal=(prefix == "left_eye")  # Mirror left eye
        )

    # Process mocap video
    process_video(video_data=topdown_mocap_video,
                  entity_path="mocap_video/top_down")

    print(f"Processing complete! Rerun recording '{recording_name}' is ready.")


def main_rerun_viewer_maker():
    """Main function to run the eye tracking visualization."""

    # Create eye data models
    left_eye = EyeVideoData.create(
        annotated_video_path=LEFT_EYE_ANNOTATED_VIDEO_PATH,
        raw_video_path=LEFT_EYE_RAW_VIDEO_PATH,
        timestamps_npy_path=LEFT_EYE_TIMESTAMPS_NPY_PATH,
        data_csv_path=EYE_DATA_CSV_PATH,
        data_name="Left Eye"
    )
    right_eye = EyeVideoData.create(
        annotated_video_path=RIGHT_EYE_ANNOTATED_VIDEO_PATH,
        raw_video_path=RIGHT_EYE_RAW_VIDEO_PATH,
        timestamps_npy_path=RIGHT_EYE_TIMESTAMPS_NPY_PATH,
        data_csv_path=EYE_DATA_CSV_PATH,
        data_name="Right Eye"
    )

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=TOPDOWN_ANNOTATED_VIDEO_PATH,
        raw_video_path=TOPDOWN_RAW_VIDEO_PATH,
        timestamps_npy_path=TOPDOWN_TIMESTAMPS_NPY_PATH,
        data_name="TopDown Mocap",
    )
    left_eye.load_pupil_data()
    right_eye.load_pupil_data()

    recording_start_time = np.min([
        float(left_eye.timestamps_array[0]),
        float(right_eye.timestamps_array[0]),
        float(topdown_mocap_video.timestamps_array[0]),
    ])

    left_eye.timestamps_array -= recording_start_time
    right_eye.timestamps_array -= recording_start_time
    topdown_mocap_video.timestamps_array -= recording_start_time
    topdown_mocap_video.timestamps_array += .2  # adjust for desync offset
    # Process and visualize the eye videos
    create_rerun_recording(left_eye_video_data=left_eye,
                           right_eye_video_data=right_eye,
                           topdown_mocap_video=topdown_mocap_video,
                           recording_name=RECORDING_NAME)


if __name__ == "__main__":
    main_rerun_viewer_maker()
