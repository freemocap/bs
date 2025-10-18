"""Video encode images using av and stream them to Rerun with optimized performance."""
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# add python_code to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from python_code.eye_data_cleanup.process_video_for_rerun import process_video_for_rerun
from python_code.eye_data_cleanup.rerun_video_data import RerunEyeVideoDataset

# Configuration
RESIZE_FACTOR: float = 1.0  # Resize video to this factor (1.0 = no resize)

# Entity path constants - simplified and consistent
EYE_ROOT: str = "eye"
VIDEO_PATH: str = f"{EYE_ROOT}/video"
PUPIL_OUTLINE_PATH: str = f"{VIDEO_PATH}/pupil_outline"
PUPIL_POINTS_PATH: str = f"{VIDEO_PATH}/pupil_points"
TIMESERIES_ROOT: str = f"{EYE_ROOT}/tracking"


def create_eye_rerun_view(
    *,
    recording_name: str,
    eye_dataset: RerunEyeVideoDataset,
    save_path: str | None = None,
) -> None:
    """Process eye video and visualize with Rerun using modern API patterns."""

    # Initialize Rerun and spawn viewer
    rr.init(application_id=recording_name, spawn=True)

    # Optionally save to file
    if save_path:
        rr.save(path=save_path)

    # Define colors for horizontal and vertical tracking
    horizontal_color: list[int] = [255, 0, 0]  # Red
    vertical_color: list[int] = [255, 0, 255]  # Magenta

    # Create blueprint using modern patterns from examples
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            # Video view with overlays
            rrb.Spatial2DView(
                name="Eye Video",
                origin=VIDEO_PATH,
                # Video, pupil outline, and pupil points are all children of VIDEO_PATH
            ),
            # Timeseries views for pupil tracking
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Horizontal Position",
                    origin=TIMESERIES_ROOT,
                    contents=[
                        f"+ $origin/pupil_x",
                    ],
                    overrides={
                        f"{TIMESERIES_ROOT}/pupil_x": rr.SeriesLines.from_fields(
                            colors=[horizontal_color],
                            names=["Pupil X"],
                            widths=[2.0],
                        ),
                    },
                ),
                rrb.TimeSeriesView(
                    name="Vertical Position",
                    origin=TIMESERIES_ROOT,
                    contents=[
                        f"+ $origin/pupil_y",
                    ],
                    overrides={
                        f"{TIMESERIES_ROOT}/pupil_y": rr.SeriesLines.from_fields(
                            colors=[vertical_color],
                            names=["Pupil Y"],
                            widths=[2.0],
                        ),
                    },
                ),
            ),
            row_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )

    rr.send_blueprint(blueprint=blueprint)

    # Process and log data
    print(f"Processing {eye_dataset.data_name} data...")

    # Prepare timestamps ONCE
    t0: float = eye_dataset.video.timestamps[0]
    timestamps_seconds: np.ndarray = np.array([(t - t0) / 1e9 for t in eye_dataset.video.timestamps])
    n_timestamps: int = len(timestamps_seconds)

    print(f"Video has {n_timestamps} timestamps")
    print(f"Pupil data has {len(eye_dataset.pupil_mean_x)} values")

    # Ensure pupil data matches timestamp length
    pupil_x_data: np.ndarray = eye_dataset.pupil_mean_x
    pupil_y_data: np.ndarray = eye_dataset.pupil_mean_y

    # Truncate or pad pupil data to match timestamps if needed
    if len(pupil_x_data) != n_timestamps:
        print(f"WARNING: Pupil data length ({len(pupil_x_data)}) doesn't match timestamps ({n_timestamps})")
        if len(pupil_x_data) > n_timestamps:
            pupil_x_data = pupil_x_data[:n_timestamps]
            pupil_y_data = pupil_y_data[:n_timestamps]
        else:
            # Pad with NaN
            pad_length: int = n_timestamps - len(pupil_x_data)
            pupil_x_data = np.concatenate([pupil_x_data, np.full(shape=pad_length, fill_value=np.nan)])
            pupil_y_data = np.concatenate([pupil_y_data, np.full(shape=pad_length, fill_value=np.nan)])

    # Log pupil tracking data using send_columns (efficient bulk logging)
    times_column = [rr.TimeColumn("time", duration=timestamps_seconds)]

    print(f"Logging horizontal tracking data to {TIMESERIES_ROOT}/pupil_x...")
    rr.send_columns(
        entity_path=f"{TIMESERIES_ROOT}/pupil_x",
        indexes=times_column,
        columns=rr.Scalars.columns(scalars=pupil_x_data),
    )

    print(f"Logging vertical tracking data to {TIMESERIES_ROOT}/pupil_y...")
    rr.send_columns(
        entity_path=f"{TIMESERIES_ROOT}/pupil_y",
        indexes=times_column,
        columns=rr.Scalars.columns(scalars=pupil_y_data),
    )

    # Log pupil outline and points as video overlays
    print("Logging pupil outline and points...")
    log_pupil_overlays(eye_dataset=eye_dataset, n_timestamps=n_timestamps, times_column=times_column)

    # Process and log video
    print(f"Processing video...")
    process_video_for_rerun(
        video=eye_dataset.video,
        entity_path=VIDEO_PATH,
        flip_horizontal=False
    )

    print(f"Processing complete! Rerun recording '{recording_name}' is ready.")


def log_pupil_overlays(
    *,
    eye_dataset: RerunEyeVideoDataset,
    n_timestamps: int,
    times_column: list[rr.TimeColumn],
) -> None:
    """
    Log the pupil outline (p1-p8 markers) and points as overlays on the video.

    These are logged as children of VIDEO_PATH so they appear as overlays.
    """
    # Get the trajectory data as numpy array: shape (n_frames, n_points, 2)
    trajectory_array: np.ndarray = eye_dataset.pixel_trajectories.to_array()
    n_frames: int = trajectory_array.shape[0]

    # Handle length mismatch
    if n_frames != n_timestamps:
        print(f"WARNING: Trajectory frames ({n_frames}) doesn't match timestamps ({n_timestamps})")
        if n_frames > n_timestamps:
            trajectory_array = trajectory_array[:n_timestamps]
            n_frames = n_timestamps
        else:
            # Pad with NaN
            pad_length: int = n_timestamps - n_frames
            pad_shape: tuple[int, int, int] = (pad_length, trajectory_array.shape[1], trajectory_array.shape[2])
            nan_pad: np.ndarray = np.full(shape=pad_shape, fill_value=np.nan)
            trajectory_array = np.concatenate([trajectory_array, nan_pad], axis=0)
            n_frames = n_timestamps

    # Extract p1-p8 points for each frame to create the pupil outline
    pupil_point_indices: list[int] = [eye_dataset.landmarks[f"p{i}"] for i in range(1, 9)]

    # Create line strips and points for each frame
    line_strips_per_frame: list[np.ndarray] = []
    pupil_points_per_frame: list[np.ndarray] = []

    for frame_idx in range(n_frames):
        # Get pupil points for this frame (8 points with x,y coordinates)
        pupil_points: np.ndarray = trajectory_array[frame_idx, pupil_point_indices, :]

        # Create closed loop by appending first point at the end
        closed_pupil_outline: np.ndarray = np.vstack([pupil_points, pupil_points[0:1, :]])

        line_strips_per_frame.append(closed_pupil_outline)
        pupil_points_per_frame.append(pupil_points)

    # Log the pupil outline as LineStrips2D
    print(f"Sending pupil outline to {PUPIL_OUTLINE_PATH}...")
    rr.send_columns(
        entity_path=PUPIL_OUTLINE_PATH,
        indexes=times_column,
        columns=rr.LineStrips2D.columns(
            strips=line_strips_per_frame,
            colors=[[0, 255, 0]] * n_frames,  # Green color for pupil outline
            radii=[2.0] * n_frames  # Line thickness
        )
    )

    # Log individual pupil points as Points2D
    print(f"Sending pupil points to {PUPIL_POINTS_PATH}...")
    rr.send_columns(
        entity_path=PUPIL_POINTS_PATH,
        indexes=times_column,
        columns=rr.Points2D.columns(
            positions=pupil_points_per_frame,
            colors=[[255, 0, 0]] * n_frames,  # Red color for points
            radii=[3.0] * n_frames
        )
    )


if __name__ == "__main__":
    _recording_name: str = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name: str = "0m_37s-1m_37s"
    recording_name_clip: str = _recording_name + "_" + clip_name
    base_path: Path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37")
    video_path: Path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4")
    timestamps_npy_path: Path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy")
    csv_path: Path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye0_clipped_4354_11523DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv")

    rerun_eye_video_dataset: RerunEyeVideoDataset = RerunEyeVideoDataset.create(
        data_name=f"{recording_name_clip}_eye_videos",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
    )

    # Optional: save to .rrd file
    save_file: Path = base_path / f"{_recording_name}_eye_tracking.rrd"

    create_eye_rerun_view(
        recording_name=_recording_name,
        eye_dataset=rerun_eye_video_dataset,
        save_path=str(save_file)
    )

    print(f"\nViewer launched! Recording saved to: {save_file}")
    print("Close the Rerun viewer window when done.")