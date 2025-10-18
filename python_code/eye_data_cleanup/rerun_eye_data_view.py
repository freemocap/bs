"""Video encode images using av and stream them to Rerun with optimized performance."""
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb


from rerun.blueprint import VisualBounds2D
from rerun.blueprint.archetypes import TimeAxis
from rerun.blueprint.components import LinkAxis
from rerun.datatypes import Range2D

# add python_code to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from python_code.eye_data_cleanup.process_video_for_rerun import process_video_for_rerun
from python_code.eye_data_cleanup.rerun_video_data import RerunEyeVideoDataset

# Configuration
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)n


def create_eye_rerun_view(recording_name: str,
                          eye_dataset: RerunEyeVideoDataset,
                          ) -> None:
    """Process both eye videos and visualize them with Rerun."""
    # Initialize Rerun
    rr.init(recording_name, spawn=True)

    videos_blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(name="Eye Video",
                              origin=f"/eye/video",
                              visual_bounds=VisualBounds2D.from_fields(
                                  range=Range2D(
                                      x_range=(0, eye_dataset.video.width),
                                      y_range=(0, eye_dataset.video.height)
                                  )
                              ),
                              visible=True
                              ),
        ),
    )
    eye_timeseries_blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.TimeSeriesView(name=" Eye Horizontal Position",
                               contents=[f"+ eye/pupil_x_line",
                                         f"+eye/pupil_x_dots"],
                               axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal),
                               axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                               ),
            rrb.TimeSeriesView(name="Eye Vertical Position",
                               contents=[f"+ eye/pupil_y_line",
                                         f"+ eye/pupil_y_dots"],
                               axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal),
                               axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                               ),
        ))
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            videos_blueprint,
            eye_timeseries_blueprint
        ),
        rrb.BlueprintPanel(state="expanded"),
    )

    rr.send_blueprint(blueprint)

    horizontal_color = [255, 0, 0]
    vertical_color = [255, 0, 255]

    # Process pupil tracking data for both eyes

    print(f"Processing {eye_dataset.data_name} pupil tracking data...")
    for data_type, color, data in [
        ("pupil_x_line", horizontal_color, eye_dataset.pupil_mean_x),
        ("pupil_x_dots", horizontal_color, eye_dataset.pupil_mean_x),

        ("pupil_y_line", vertical_color, eye_dataset.pupil_mean_y),
        ("pupil_y_dots", vertical_color, eye_dataset.pupil_mean_y),
    ]:
        entity_path = f"eye/{data_type}"
        print(f"Logging {entity_path}...")
        # Set up visualization properties
        if "line" in data_type:
            rr.log(entity_path,
                   rr.SeriesLines(colors=color,
                                  names='eye',
                                  widths=2),
                   static=True)
        else:  # dots
            rr.log(entity_path,
                   rr.SeriesPoints(colors=color,
                                   names='eye',
                                   markers="circle",
                                   marker_sizes=2),
                   static=True)

            # Send data

        ts = []
        t0 = eye_dataset.video.timestamps[0]
        for t in eye_dataset.video.timestamps:
            ts.append((t - t0)/1e9)  # Convert to seconds
        print(f"Sending time series data to {entity_path} with {len(ts)} timestamps...")
        rr.send_columns(
            entity_path=entity_path,
            indexes=[rr.TimeColumn("time", duration=ts)],
            columns=rr.Scalars.columns(scalars=data),
        )
    # Log static data for time series

    # Process video
    process_video_for_rerun(
        video=eye_dataset.video,
        entity_path=f"eye/video",
        flip_horizontal=False
    )

    print(f"Processing complete! Rerun recording '{recording_name}' is ready.")


if __name__ == "__main__":
    _recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name = "0m_37s-1m_37s"
    recording_name_clip = _recording_name + "_" + clip_name
    base_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37")
    video_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4")
    timestamps_npy_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy")
    csv_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye0_clipped_4354_11523DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv")

    rerun_eye_video_dataset = RerunEyeVideoDataset.create(
        data_name=f"{recording_name_clip}_eye_videos",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
    )
    create_eye_rerun_view(recording_name=_recording_name,
                          eye_dataset=rerun_eye_video_dataset
                          )
