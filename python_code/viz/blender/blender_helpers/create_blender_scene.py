import bpy
import numpy as np
from freemocap_blender_addon.core_functions.load_videos.load_videos import add_videos_to_scene
from freemocap_blender_addon.core_functions.setup_scene.clear_scene import clear_scene
from freemocap_blender_addon.core_functions.setup_scene.scene_objects.ground_plane.create_ground_plane import \
    create_ground_plane, GroundPlaneConfig

from python_code.viz.blender.blender_helpers.add_cameras import add_cameras
from python_code.viz.blender.blender_helpers.blender_recording_model import BlenderRecording
from python_code.viz.blender.blender_helpers.create_arena import create_arena
from python_code.viz.blender.blender_helpers.load_kinematics_object_bpy import (
    load_eye_kinematics_bpy,
    load_gaze_kinematics_bpy,
    load_rigid_body_kinematics_bpy,
)
from python_code.viz.blender.blender_helpers.load_simple_object.load_simple_object_bpy import load_simple_object_bpy
from python_code.viz.blender.blender_helpers.set_scene_parameters import set_scene_parameters

def create_blender_scene(recording: BlenderRecording):
    print("=" * 70)
    print("CREATE BLENDER SCENE")
    print("=" * 70)
    print(f"Recording name: {recording.name}")
    print(f"Recording path: {recording.recording_path}")
    print(f"Frame count: {recording.frame_count}")
    print(f"Number of cameras: {len(recording.data.calibration.cameras)}")
    print(f"Number of videos: {len(recording.videos.mocap_videos.videos)}")
    timestamps = recording.data.timestamps
    print(f"Timestamp range: {timestamps[0]:.6f} to {timestamps[-1]:.6f} seconds")
    print(f"Timestamp mean delta: {np.mean(np.diff(timestamps)):.6f} seconds")
    framerate = float(np.mean(np.diff(timestamps))) ** -1
    print(f"Computed framerate: {framerate:.3f} frames_per_second")

    print("\n\n\n--- Clearing existing scene ---")
    clear_scene()
    print("Scene cleared.")

    print("\n\n\n--- Setting scene parameters ---")
    set_scene_parameters(recording=recording)

    print("\n--- Creating arena ---")
    create_arena()

    create_ground_plane(config=GroundPlaneConfig(size=1))

    print("\n\n\n--- Loading Top Down Video as groundplane---")
    add_videos_to_scene(videos_directory=str(recording.folder.display_videos), video_scale=.5)
    # load_top_down_video_as_groundplane(video=VideoHelper.create(video_path=recording.folder.topdown_mocap_display_video,
    #                                                             timestamps_npy_path=recording.folder.common_timestamps))
    print("\n--- Adding cameras ---")
    add_cameras(cameras=recording.data.calibration.cameras,
                mocap_videos=recording.videos.mocap_videos)

    print("\n\n--- Loading Toy Object ---")
    load_simple_object_bpy(simple_object=recording.data.toy)


    print("\n\n--- Loading Skull & Spine Object ---")
    load_simple_object_bpy(simple_object=recording.data.skull_and_spine)

    print("\n\n--- Loading Skull RigidBodyKinematics Object ---")
    skull_frame_empty = load_rigid_body_kinematics_bpy(rbk=recording.data.skull_kinematics)



    print("\n\n--- Loading Right Eye Kinematics ---")
    load_eye_kinematics_bpy(eye_kinematics=recording.data.right_eye_kinematics)

    print("\n\n--- Loading Left Eye Kinematics ---")
    load_eye_kinematics_bpy(eye_kinematics=recording.data.left_eye_kinematics)



    print("\n\n--- Loading Right Gaze Kinematics ---")
    load_gaze_kinematics_bpy(
        gaze_kinematics=recording.data.right_gaze_kinematics,
        skull_frame_empty=skull_frame_empty,
    )

    print("\n\n--- Loading Left Gaze Kinematics ---")
    load_gaze_kinematics_bpy(
        gaze_kinematics=recording.data.left_gaze_kinematics,
        skull_frame_empty=skull_frame_empty,
    )

    print("\n" + "=" * 70)
    print("CREATE BLENDER SCENE COMPLETE")
    print("=" * 70)





