from pathlib import Path
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.video_data import MocapVideoData

def get_mocap_video_view(mocap_video: MocapVideoData, entity_path: str = "/"):
    view = rrb.Vertical(
        rrb.Spatial2DView(
            name="TopDown Mocap Video(Annotated)",
            origin=f"{entity_path}/annotated",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, mocap_video.resized_width),
                    y_range=(0, mocap_video.resized_height),
                )
            ),
        ),
        rrb.Spatial2DView(
            name="TopDown Mocap Video(Raw)",
            origin=f"{entity_path}/raw",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(
                    x_range=(0, mocap_video.resized_width),
                    y_range=(0, mocap_video.resized_height),
                )
            ),
            visible=False,
        ),
    )
    return view

if __name__ == "__main__":
    from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
    from datetime import datetime

    recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name, base_recordings_folder=Path("/home/scholl-lab/ferret_recordings"))
    # recording_folder = RecordingFolder.create_full_recording(recording_name, base_recordings_folder="/home/scholl-lab/ferret_recordings")

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=recording_folder.topdown_annotated_video_path,
        raw_video_path=recording_folder.topdown_video_path,
        timestamps_npy_path=recording_folder.topdown_timestamps_npy_path,
        data_name="TopDown Mocap",
    )

    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    mocap_video_entity_path = "/mocap_video"

    view = get_mocap_video_view(mocap_video=topdown_mocap_video, entity_path=mocap_video_entity_path)

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)

    process_video(topdown_mocap_video, entity_path=mocap_video_entity_path)
