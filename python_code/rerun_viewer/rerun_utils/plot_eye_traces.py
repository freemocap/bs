from pathlib import Path
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.video_data import AlignedEyeVideoData, EyeVideoData

def plot_eye_traces(right_eye_video_data: EyeVideoData, left_eye_video_data: EyeVideoData, entity_path: str = ""):
    right_eye_horizontal_color = [255, 0, 0]
    right_eye_vertical_color = [255, 0, 255]
    left_eye_horizontal_color = [0, 255, 255]
    left_eye_vertical_color = [65, 85, 255]

    right_eye_video_data.load_pupil_means()
    left_eye_video_data.load_pupil_means()

    # Process pupil tracking data for both eyes
    for eye, prefix, horizontal_color, vertical_color in [
        (right_eye_video_data, "right_eye", right_eye_horizontal_color, right_eye_vertical_color),
        (left_eye_video_data, "left_eye", left_eye_horizontal_color, left_eye_vertical_color)
    ]:
        print(f"Processing {eye.data_name} pupil tracking data...")
        for data_type, color, data in [
            ("pupil_x_line", horizontal_color, eye.pupil_mean_x),
            ("pupil_x_dots", horizontal_color, eye.pupil_mean_x),

            ("pupil_y_line", vertical_color, eye.pupil_mean_y),
            ("pupil_y_dots", vertical_color, eye.pupil_mean_y)
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
                indexes=[rr.TimeColumn("time", duration=eye.timestamps)],
                columns=rr.Scalars.columns(scalars=data),
            )

def get_eye_trace_views(entity_path: str = ""):
    views = rrb.Vertical(
            rrb.TimeSeriesView(origin=entity_path, 
                name="Right Eye Horizontal Position",
                contents=[f"+ right_eye/pupil_x_line",
                        f"+ right_eye/pupil_x_dots"],
                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                ),
            rrb.TimeSeriesView(origin=entity_path,
                name="Left Eye Horizontal Position",
                contents=[f"+ left_eye/pupil_x_line",
                            f"+ left_eye/pupil_x_dots"],
                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                ),
            rrb.TimeSeriesView(origin=entity_path,
                name="Right Eye Vertical Position",
                contents=[f"+ right_eye/pupil_y_line",
                            f"+ right_eye/pupil_y_dots"],
                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                ),
            rrb.TimeSeriesView(origin=entity_path,
                name="Left Eye Vertical Position",
                contents=[f"+ left_eye/pupil_y_line",
                            f"+ left_eye/pupil_y_dots"],
                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                ),
        )
    return views

if __name__ == "__main__":
    from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
    from datetime import datetime

    recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name, base_recordings_folder=Path("/home/scholl-lab/ferret_recordings"))
    # recording_folder = RecordingFolder.create_full_recording(recording_name, base_recordings_folder="/home/scholl-lab/ferret_recordings")

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

    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    eye_plots_entity_path = "/eye_plots"

    views = get_eye_trace_views(entity_path=eye_plots_entity_path)

    blueprint = rrb.Horizontal(views)

    rr.send_blueprint(blueprint)

    plot_eye_traces(right_eye_video_data=right_eye, left_eye_video_data=left_eye, entity_path=eye_plots_entity_path)
