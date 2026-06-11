import bpy
import numpy as np
from freemocap_blender_addon.core_functions.empties.creation.create_empty_from_trajectory import \
    create_keyframed_empty_from_3d_trajectory_data
from freemocap_blender_addon.core_functions.setup_scene.clear_scene import clear_scene
from freemocap_blender_addon.core_functions.setup_scene.make_parent_empties import create_parent_empty

from python_code.kinematics_core.keypoint_trajectories import KeypointTrajectories
from python_code.viz.blender.blender_helpers.add_cameras import add_cameras
from python_code.viz.blender.blender_helpers.blender_recording_model import BlenderRecording
from python_code.viz.blender.blender_helpers.create_arena import create_arena
from python_code.viz.blender.blender_helpers.set_scene_parameters import set_scene_parameters


def load_keypoint_trajectories_bpy(keypoint_trajectories: KeypointTrajectories, parent_empty: bpy.types.Object):
    for name in keypoint_trajectories.keypoint_names:
        create_keyframed_empty_from_3d_trajectory_data(
            trajectory_name=name,
            trajectory_fr_xyz=keypoint_trajectories[name]*.001,
            parent_object=parent_empty,
            empty_scale=0.01,
            empty_type="SPHERE"
        )

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

    print("\n\n\n--- Creating recording parent empty ---")
    parent_empty = create_parent_empty(name=recording.name, display_scale=0.1, type="ARROWS")
    print(f"Parent empty '{recording.name}' created (ARROWS, display_scale=0.1)")

    print("\n--- Creating arena ---")
    create_arena()

    print("\n--- Adding cameras ---")
    add_cameras(cameras=recording.data.calibration.cameras,
                mocap_videos=recording.videos.mocap_videos)

    load_keypoint_trajectories_bpy(keypoint_trajectories=recording.data.toy_trajectories,
                               parent_empty=parent_empty)

    load_keypoint_trajectories_bpy(keypoint_trajectories=recording.data.skull_and_spine_trajectories,
                               parent_empty=parent_empty)


    # load_skull_and_spine(recording=recording)
    #
    # load_skull_kinematics()
    #
    # load_eye_kinematics()

    print("\n" + "=" * 70)
    print("CREATE BLENDER SCENE COMPLETE")
    print("=" * 70)





