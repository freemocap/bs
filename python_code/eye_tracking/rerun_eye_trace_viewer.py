"""Video encode images using av and stream them to Rerun."""

import av
import numpy as np
import numpy.typing as npt
import rerun as rr
from pathlib import Path
import cv2

VIDEO_PATH = r"C:\Users\jonma\Downloads\eye1.mp4"
CSV_PATH = r"C:\Users\jonma\Downloads\eye1DLC_Resnet50_dlc_pupil_tracking_shuffle1_snapshot_090.csv"
GOOD_PUPIL_POINT = "pupil_outer"

# Load video
if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
video_name = str(Path(VIDEO_PATH).stem)
vid_cap = cv2.VideoCapture(VIDEO_PATH)
framerate = vid_cap.get(cv2.CAP_PROP_FPS)
width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = int(frame_count / framerate)
frame_duration = 1.0 / framerate
print(f"Video info: {width}x{height} @ {framerate} FPS, {frame_count} frames, {duration_seconds}s duration")


#Load pupil data from CSV
import pandas as pd
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
pupil_df = pd.read_csv(CSV_PATH, header=[0,1])
if GOOD_PUPIL_POINT not in pupil_df.columns:
    raise ValueError(f"Expected column '{GOOD_PUPIL_POINT}' not found in CSV data.")
pupil_outer_x = pupil_df["pupil_outer", "x"]
pupil_outer_y = pupil_df["pupil_outer", "y"]

if len(pupil_outer_x) != frame_count:
    raise ValueError(f"Expected {frame_count} pupil points, but found {len(pupil_outer_x)} in CSV data.")


rr.init("eye_tracking_test", spawn=True)

# Setup encoding pipeline.
av.logging.set_level(av.logging.VERBOSE)
container = av.open("/dev/null", "w", format="h264")  # Use AnnexB H.264 stream.
stream = container.add_stream("libx264", rate=int(framerate))
stream.width = width
stream.height = height
# TODO(#10090): Rerun Video Streams don't support b-frames yet.
# Note that b-frames are generally not recommended for low-latency streaming and may make logging more complex.
stream.max_b_frames = 0

# Log codec only once as static data (it naturally never changes). This isn't strictly necessary, but good practice.
rr.log(video_name, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
rr.log("pupil_x_line", rr.SeriesLines(colors=[125, 0, 0], names="Horizonal Pupil Position", widths=2), static=True)
rr.log("pupil_y_line", rr.SeriesLines(colors=[0, 62, 125], names="Vertical Pupil Position ", widths=2), static=True)
rr.log("pupil_x_dots", rr.SeriesPoints(colors=[255, 0, 0], names="Horizonal Pupil Position", markers="circle", marker_sizes=2), static=True)
rr.log("pupil_y_dots", rr.SeriesPoints(colors=[0, 125, 255], names="Vertical Pupil Position", markers="circle", marker_sizes=2), static=True)

# Generate frames and stream them directly to Rerun.
for frame_number in range(frame_count):
    ret, img = vid_cap.read()
    if not ret:
        print(f"Failed to read frame {frame_number}, stopping.")
        break
    
    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    for packet in stream.encode(frame):
        if packet.pts is None:
            continue
        rr.set_time("time", duration=float(packet.pts * packet.time_base))
        rr.log(video_name, rr.VideoStream.from_fields(sample=bytes(packet)))
    
    rr.set_time("time", duration=frame_number * frame_duration)
    rr.log("pupil_x_line", rr.Scalars(pupil_outer_x[frame_number]))
    rr.log("pupil_y_line", rr.Scalars(pupil_outer_y[frame_number]))
    rr.log("pupil_x_dots", rr.Scalars(pupil_outer_x[frame_number]))
    rr.log("pupil_y_dots", rr.Scalars(pupil_outer_y[frame_number]))

# Flush stream.
for packet in stream.encode():
    if packet.pts is None:
        continue
    rr.set_time("time", duration=float(packet.pts * packet.time_base))
    rr.log(video_name, rr.VideoStream.from_fields(sample=bytes(packet)))
