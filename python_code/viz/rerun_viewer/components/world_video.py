from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D
from pydantic import BaseModel

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.viz.rerun_viewer.utils.process_videos import process_video
from python_code.viz.rerun_viewer.utils.video_data import WorldCameraVideoData


class WorldVideoContext(BaseModel):
    entity_path: str = "/world_video"


def required_data_available(recording_folder: RecordingFolder, context: "WorldVideoContext") -> bool:
    return recording_folder.pupil_world_video is not None and recording_folder.pupil_world_timestamps_npy is not None


def get_world_video_view(world_video: WorldCameraVideoData, entity_path: str = "/"):
    return rrb.Spatial2DView(
        name="Pupil World Camera",
        origin=f"{entity_path}/raw",
        visual_bounds=VisualBounds2D.from_fields(
            range=Range2D(
                x_range=(0, world_video.resized_width),
                y_range=(0, world_video.resized_height),
            )
        ),
    )


def plot_world_video(recording_folder: RecordingFolder, entity_path: str = "/world_video"):
    context = WorldVideoContext(entity_path=entity_path)
    if not required_data_available(recording_folder, context):
        print("Skipping world video: required data not found")
        return
    world_video = WorldCameraVideoData.create_from_world(
        raw_video_path=recording_folder.pupil_world_video,
        timestamps_npy_path=recording_folder.pupil_world_timestamps_npy,
    )
    process_video(world_video, entity_path=entity_path, include_annotated=False)


if __name__ == "__main__":
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)

    world_video = WorldCameraVideoData.create_from_world(
        raw_video_path=recording_folder.pupil_world_video,
        timestamps_npy_path=recording_folder.pupil_world_timestamps_npy,
    )

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    entity_path = "/world_video"
    view = get_world_video_view(world_video=world_video, entity_path=entity_path)
    rr.send_blueprint(rrb.Horizontal(view))

    plot_world_video(recording_folder=recording_folder, entity_path=entity_path)
