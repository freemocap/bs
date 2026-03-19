"""Viewer that tiles mocap, eye, and (optionally) pupil world camera videos, aligned by timestamp."""
from pathlib import Path
from datetime import datetime

import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.plot_mocap_video import get_mocap_video_view
from python_code.rerun_viewer.rerun_utils.plot_world_video import get_world_video_view
from python_code.rerun_viewer.rerun_utils.process_videos import process_video_frames
from python_code.rerun_viewer.rerun_utils.video_data import (
    MocapVideoData,
    VideoData,
    WorldCameraVideoData,
)
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder, BaslerCamera


def _send_video_aligned(
    video_data: VideoData,
    entity_path: str,
    global_start: float,
    include_annotated: bool = True,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> None:
    """Send video frames to Rerun using timestamps relative to global_start for alignment."""
    timestamps = video_data.timestamps - global_start
    video_types = ["raw", "annotated"] if include_annotated else ["raw"]
    for video_type in video_types:
        encoded_frames = process_video_frames(
            video_cap=video_data.raw_vid_cap if video_type == "raw" else video_data.annotated_vid_cap,
            resize_factor=video_data.resize_factor,
            resize_width=video_data.resized_width,
            resize_height=video_data.resized_height,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )
        rr.send_columns(
            entity_path=f"{entity_path}/{video_type}",
            indexes=[rr.TimeColumn("time", duration=timestamps)],
            columns=rr.EncodedImage.columns(
                blob=encoded_frames,
                media_type=["image/jpeg"] * len(encoded_frames),
            ),
        )


def main(recording_folder: RecordingFolder) -> None:
    # --- Load mocap video (topdown) ---
    topdown_synchronized_video = recording_folder.get_synchronized_video_by_name(BaslerCamera.TOPDOWN.value)
    topdown_annotated_video = recording_folder.get_annotated_video_by_name(BaslerCamera.TOPDOWN.value)
    topdown_timestamps_npy = recording_folder.get_timestamp_by_name(BaslerCamera.TOPDOWN.value)

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=topdown_annotated_video,
        raw_video_path=topdown_synchronized_video,
        timestamps_npy_path=topdown_timestamps_npy,
        data_name="TopDown Mocap",
    )

    # --- Load eye videos ---
    left_eye = VideoData.create(
        annotated_video_path=recording_folder.left_eye_stabilized_canvas,
        raw_video_path=recording_folder.left_eye_stabilized_canvas,
        timestamps_npy_path=recording_folder.left_eye_timestamps_npy,
        data_name="Left Eye",
    )

    right_eye = VideoData.create(
        annotated_video_path=recording_folder.right_eye_stabilized_canvas,
        raw_video_path=recording_folder.right_eye_stabilized_canvas,
        timestamps_npy_path=recording_folder.right_eye_timestamps_npy,
        data_name="Right Eye",
    )

    # --- Optionally load world camera ---
    world_video = None
    if recording_folder.pupil_world_video and recording_folder.pupil_world_timestamps_npy:
        world_video = WorldCameraVideoData.create_from_world(
            raw_video_path=recording_folder.pupil_world_video,
            timestamps_npy_path=recording_folder.pupil_world_timestamps_npy,
        )

    # --- Compute global start time for alignment ---
    all_first_timestamps = [
        float(topdown_mocap_video.timestamps[0]),
        float(left_eye.timestamps[0]),
        float(right_eye.timestamps[0]),
    ]
    if world_video is not None:
        all_first_timestamps.append(float(world_video.timestamps[0]))
    global_start = min(all_first_timestamps)

    # --- Init Rerun ---
    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    # --- Entity paths ---
    mocap_entity_path = "/mocap_video"
    eye_entity_path = "/eye_videos"
    world_entity_path = "/world_video"

    # --- Build blueprint ---
    mocap_view = get_mocap_video_view(mocap_video=topdown_mocap_video, entity_path=mocap_entity_path)

    eye_views = [
        rrb.Spatial2DView(
            name="Right Eye",
            origin=f"{eye_entity_path}/right_eye",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(x_range=(0, right_eye.resized_width), y_range=(0, right_eye.resized_height))
            ),
        ),
        rrb.Spatial2DView(
            name="Left Eye",
            origin=f"{eye_entity_path}/left_eye",
            visual_bounds=VisualBounds2D.from_fields(
                range=Range2D(x_range=(0, left_eye.resized_width), y_range=(0, left_eye.resized_height))
            ),
        ),
    ]

    if world_video is not None:
        world_view = get_world_video_view(world_video=world_video, entity_path=world_entity_path)
        top_row = rrb.Horizontal(mocap_view, world_view)
    else:
        top_row = rrb.Horizontal(mocap_view)

    blueprint = rrb.Vertical(
        top_row,
        rrb.Horizontal(*eye_views),
    )
    rr.send_blueprint(blueprint)

    # --- Log data ---
    _send_video_aligned(topdown_mocap_video, entity_path=mocap_entity_path, global_start=global_start, include_annotated=True)
    _send_video_aligned(left_eye, entity_path=f"{eye_entity_path}/left_eye", global_start=global_start, include_annotated=False)
    _send_video_aligned(right_eye, entity_path=f"{eye_entity_path}/right_eye", global_start=global_start, include_annotated=False)

    if world_video is not None:
        _send_video_aligned(world_video, entity_path=world_entity_path, global_start=global_start, include_annotated=False)


if __name__ == "__main__":
    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)
    recording_folder.check_eye_postprocessing()

    main(recording_folder)
