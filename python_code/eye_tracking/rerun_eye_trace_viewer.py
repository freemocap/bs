"""Video encode images using av and stream them to Rerun with optimized performance."""

import av
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import cv2
import pandas as pd
import time

# Configuration
VIDEO_PATH = r"C:\Users\jonma\Downloads\eye0(3).mp4"
CSV_PATH = r"C:\Users\jonma\Downloads\eye0DLC_Resnet50_pupil_tracking_ferret_757_EyeCameras_P43_E15__1_shuffle1_snapshot_030.csv"
GOOD_PUPIL_POINT = "pupil_outer"
RESIZE_FACTOR = 1.0  # Resize video to 75% of original size
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)

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

# Calculate new dimensions if resizing
new_width = int(width * RESIZE_FACTOR)
new_height = int(height * RESIZE_FACTOR)

print(f"Video info: {width}x{height} @ {framerate} FPS, {frame_count} frames, {duration_seconds}s duration")
if RESIZE_FACTOR != 1.0:
    print(f"Resizing to: {new_width}x{new_height}")

# Load pupil data from CSV
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
pupil_df = pd.read_csv(CSV_PATH, header=[0,1])
if GOOD_PUPIL_POINT not in pupil_df.columns:
    raise ValueError(f"Expected column '{GOOD_PUPIL_POINT}' not found in CSV data.")
pupil_outer_x = pupil_df["pupil_outer", "x"]
pupil_outer_y = pupil_df["pupil_outer", "y"]

if len(pupil_outer_x) != frame_count:
    raise ValueError(f"Expected {frame_count} pupil points, but found {len(pupil_outer_x)} in CSV data.")

# Convert to numpy arrays for faster processing
pupil_x_array = pupil_outer_x.to_numpy()
pupil_y_array = pupil_outer_y.to_numpy()
time_array = np.arange(frame_count) * frame_duration

# Initialize Rerun
rr.init("eeye_tracking_simple", spawn=True)
rr.set_time("time", duration=0.0)  # Set initial time to 0

# Define a blueprint with separate views 
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.Spatial2DView(name="Video View", origin=f"/{video_name}"),
        rrb.Vertical(
            rrb.TimeSeriesView(name="Horizontal Pupil Position", contents=[
                "+ /pupil_x_line",
                "+ /pupil_x_dots"
            ]),
            rrb.TimeSeriesView(name="Vertical Pupil Position", contents=[
                "+ /pupil_y_line", 
                "+ /pupil_y_dots"
            ]),
        ),
    ),
    rrb.BlueprintPanel(state="expanded"),
)
rr.send_blueprint(blueprint)

# Setup optimized encoding pipeline
av.logging.set_level(av.logging.ERROR)  # Reduce logging noise
container = av.open("/dev/null", "w", format="h264")
stream = container.add_stream("libx264", rate=int(framerate))
stream.width = new_width
stream.height = new_height
stream.max_b_frames = 0  # Keep this for compatibility with Rerun
stream.pix_fmt = "yuv420p"  # More efficient pixel format
stream.options = {
    "crf": str(COMPRESSION_LEVEL),  # Higher = more compression
    "preset": "slow",               # Slower = better compression
    "tune": "zerolatency",          # Good for streaming
    "profile": "baseline",          # More compatible profile
}

# Log static data
rr.log(video_name, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
rr.log("pupil_x_line", rr.SeriesLines(colors=[125, 0, 0], names="Horizontal Pupil Position", widths=2), 
       static=True)
rr.log("pupil_y_line", rr.SeriesLines(colors=[0, 62, 125], names="Vertical Pupil Position", widths=2), 
       static=True)
rr.log("pupil_x_dots", rr.SeriesPoints(colors=[255, 0, 0], names="Horizontal Pupil Position", markers="circle", marker_sizes=2), 
       static=True)
rr.log("pupil_y_dots", rr.SeriesPoints(colors=[0, 125, 255], names="Vertical Pupil Position", markers="circle", marker_sizes=2), 
       static=True)

# Process frames in chunks for better performance
start_time = time.time()
for frame_number in range(0, frame_count):
    ret, img = vid_cap.read()
    if not ret:
        print(f"Failed to read frame {frame_number}, stopping.")
        break
    
    # Resize if needed
    if RESIZE_FACTOR != 1.0:
        img = cv2.resize(img, (new_width, new_height))
    
    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    for packet in stream.encode(frame):
        if packet.pts is None:
            continue
        rr.set_time("time", duration=float(packet.pts * packet.time_base))
        rr.log(video_name, rr.VideoStream.from_fields(sample=bytes(packet)))

    # Use a single time point for all data at this frame
    frame_time = time_array[frame_number]
    rr.set_time("time", duration=frame_time)
    
    # Log all data for this frame at once
    rr.log("pupil_x_line", rr.Scalars(pupil_x_array[frame_number]))
    rr.log("pupil_y_line", rr.Scalars(pupil_y_array[frame_number]))
    rr.log("pupil_x_dots", rr.Scalars(pupil_x_array[frame_number]))
    rr.log("pupil_y_dots", rr.Scalars(pupil_y_array[frame_number]))


# Flush stream
for packet in stream.encode():
    if packet.pts is None:
        continue
    rr.set_time("time", duration=float(packet.pts * packet.time_base))
    rr.log(video_name, rr.VideoStream.from_fields(sample=bytes(packet)))

print(f"Processing complete in {time.time() - start_time:.1f}s")