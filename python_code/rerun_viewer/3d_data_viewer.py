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
COMPRESSION_LEVEL = 28  # CRF value (18-28 is good, higher = more compression)


def create_rerun_recording(recording_name: str,
                           data_3d: np.ndarray,
                           topdown_mocap_video: MocapVideoData,
                           ) -> None:
    """Process both eye videos and visualize them with Rerun."""
    # Initialize Rerun
    rr.init(recording_name, spawn=True)

    blueprint = rrb.Horizontal(
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
        ),
        rrb.Spatial3DView(name="3D Data",
                          origin=f"/3d_view",
        ))

    rr.send_blueprint(blueprint)

    color = [255, 0, 255]
    print(f"Processing 3d data data...")
    for i in range(data_3d.shape[0]):
        rr.set_time("time", duration=topdown_mocap_video.timestamps_array[i])
        frame_data = data_3d[i, :, :]
        rr.log("/3d_view", rr.Points3D(frame_data, radii=0.06))

    # Process mocap video
    process_video(video_data=topdown_mocap_video,
                  entity_path="mocap_video/top_down")

    print(f"Processing complete! Rerun recording '{recording_name}' is ready.")


def main_rerun_viewer_maker(recording_folder: RecordingFolder, data_3d: np.ndarray):
    """Main function to run the eye tracking visualization."""

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=recording_folder.topdown_annotated_video_path,
        raw_video_path=recording_folder.topdown_video_path,
        timestamps_npy_path=recording_folder.topdown_timestamps_npy_path,
        data_name="TopDown Mocap",
    )

    recording_start_time = np.min([
        float(topdown_mocap_video.timestamps_array[0]),
    ])

    topdown_mocap_video.timestamps_array -= recording_start_time
    # Process and visualize the eye videos
    create_rerun_recording(data_3d=data_3d,
                           topdown_mocap_video=topdown_mocap_video,
                           recording_name=recording_folder.recording_name,)


if __name__ == "__main__":
    recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name)

    data_3d_path = recording_folder.mocap_data_folder / "output_data" / "dlc" / "dlc_body_rigid_3d_xyz.npy"
    data_3d= np.load(data_3d_path)
    main_rerun_viewer_maker(recording_folder=recording_folder, data_3d=data_3d)
