import numpy as np
from freemocap.core.pipeline.posthoc.video_group_helper import VideoGroupHelper, VideoHelper
from freemocap.core.tasks.calibration.shared.camera_model import CameraModel
from freemocap_blender_addon.core_functions.setup_scene.make_parent_empties import create_parent_empty

from python_code.viz.blender.blender_helpers.blender_recording_model import BlenderRecording


def create_blender_scene(recording: BlenderRecording):
    import bpy
    # set start/end frame to match recording
    # set framerate to match recording
    # create parent data empty (arrows empty, 10cm, at origin)

    set_scene_parameters(recording=recording)
    create_parent_empty(name=recording.name,
                        display_scale=0.1,
                        type="ARROWS")

    create_arena()
    add_cameras(cameras=recording.data.calibration.cameras,
                mocap_videos=recording.videos.mocap_videos)


def set_scene_parameters(recording: BlenderRecording, start_frame:int=0, end_frame:int=None):
    start_frame = 0
    end_frame = len(recording.data.calibration.cameras) if end_frame is None else end_frame

    #set blender scene start and end frames

    framerate = (np.mean(np.diff(recording.data.timestamps))) ** -1

    #set blender scene framerate




def create_arena():
    #1m cube of cylindrical posts (1cm diameter) centered on origin (bottom face @ Z=0)
    pass


def add_cameras(cameras:list[CameraModel], mocap_videos:VideoGroupHelper):
    # Load camera data from calibration_result
    # loop through each and do `add_camera` on each camera
    for camera, video in zip(cameras, mocap_videos.videos.values()):
        add_camera(camera=camera, video=video)
    pass


def add_camera(camera:CameraModel, video:VideoHelper):
    # Place camera position/orientation based on camera_model result
    # Load videos as plane, scale/translate/rotate them to align w/ near-clipping plane of each camera
    # (so if you look through the camera view, the image is properly aligned per the camera's extrinsics)
    pass
