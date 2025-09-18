"""Video encode images using av and stream them to Rerun with optimized performance."""

from pathlib import Path
import numpy as np
from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
from python_code.rerun_viewer.rerun_utils.video_data import EyeVideoData, MocapVideoData
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.blueprint.archetypes import TimeAxis
from rerun.blueprint.components import LinkAxis
from rerun.datatypes import Range2D

# Configuration
GOOD_PUPIL_POINT = "p2"
RESIZE_FACTOR = 1.0  # Resize video to this factor (1.0 = no resize)
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)n


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
                                axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal),
                                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                                ),
            rrb.TimeSeriesView(name="Left Eye Horizontal Position",
                                contents=[f"+ left_eye/pupil_x_line",
                                            f"+ left_eye/pupil_x_dots"],
                                axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal),
                                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                                ),
            rrb.TimeSeriesView(name="Right Eye Vertical Position",
                                contents=[f"+ right_eye/pupil_y_line",
                                            f"+ right_eye/pupil_y_dots"],
                                axis_x=TimeAxis.from_fields(link=LinkAxis.LinkToGlobal),
                                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                                ),

            rrb.TimeSeriesView(name="Left Eye Vertical Position",
                                contents=[f"+ left_eye/pupil_y_line",
                                            f"+ left_eye/pupil_y_dots"],
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

    right_eye_horizontal_color = [255, 0, 0]
    right_eye_vertical_color = [255, 0, 255]
    left_eye_horizontal_color = [0, 255, 255]
    left_eye_vertical_color = [65, 85, 255]

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
            # flip_horizontal=(prefix == "left_eye")  # Mirror left eye
            flip_horizontal=False
        )

    # Process mocap video
    process_video(video_data=topdown_mocap_video,
                  entity_path="mocap_video/top_down")

    print(f"Processing complete! Rerun recording '{recording_name}' is ready.")


def main_rerun_viewer_maker(recording_folder: RecordingFolder):
    """Main function to run the eye tracking visualization."""

    # Create eye data models
    left_eye = EyeVideoData.create(
        annotated_video_path=recording_folder.left_eye_annotated_video_path,
        raw_video_path=recording_folder.left_eye_video_path,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy_path,
        data_csv_path=recording_folder.eye_data_csv_path,
        data_name="Left Eye"
    )
    right_eye = EyeVideoData.create(
        annotated_video_path=recording_folder.right_eye_annotated_video_path,
        raw_video_path=recording_folder.right_eye_video_path,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy_path,
        data_csv_path=recording_folder.eye_data_csv_path,
        data_name="Right Eye"
    )

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=recording_folder.topdown_annotated_video_path,
        raw_video_path=recording_folder.topdown_video_path,
        timestamps_npy_path=recording_folder.topdown_timestamps_npy_path,
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
    # Process and visualize the eye videos
    create_rerun_recording(left_eye_video_data=left_eye,
                           right_eye_video_data=right_eye,
                           topdown_mocap_video=topdown_mocap_video,
                           recording_name=recording_folder.recording_name,)


if __name__ == "__main__":
    recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name)
    main_rerun_viewer_maker(recording_folder=recording_folder)
