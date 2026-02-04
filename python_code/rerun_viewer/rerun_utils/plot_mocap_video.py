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
    from python_code.utilities.folder_utilities.recording_folder import RecordingFolder, BaslerCamera
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-18_ferret_420_E09/full_recording"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)
    recording_folder.check_triangulation(enforce_toy=False, enforce_annotated=True)

    topdown_synchronized_video = recording_folder.get_synchronized_video_by_name(BaslerCamera.TOPDOWN.value)
    topdown_annotated_video = recording_folder.get_annotated_video_by_name(BaslerCamera.TOPDOWN.value)
    topdown_timestamps_npy = recording_folder.get_timestamp_by_name(BaslerCamera.TOPDOWN.value)

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=topdown_annotated_video,
        raw_video_path=topdown_synchronized_video,
        timestamps_npy_path=topdown_timestamps_npy,
        data_name="TopDown Mocap",
    )

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    mocap_video_entity_path = "/mocap_video"

    view = get_mocap_video_view(mocap_video=topdown_mocap_video, entity_path=mocap_video_entity_path)

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)

    process_video(topdown_mocap_video, entity_path=mocap_video_entity_path)
