import numpy as np
from freemocap.core.pipeline.posthoc.video_group_helper import VideoGroupHelper, VideoHelper
from freemocap.core.tasks.calibration.shared.camera_model import CameraModel
from freemocap_blender_addon.core_functions.setup_scene.make_parent_empties import create_parent_empty
from freemocap_blender_addon.core_functions.setup_scene.clear_scene import clear_scene

from python_code.viz.blender.blender_helpers.blender_recording_model import BlenderRecording


def create_blender_scene(recording: BlenderRecording):
    import bpy
    print("Clearing scene...")
    clear_scene()

    set_scene_parameters(recording=recording)
    create_parent_empty(name=recording.name,
                        display_scale=0.1,
                        type="ARROWS")

    create_arena()
    add_cameras(cameras=recording.data.calibration.cameras,
                mocap_videos=recording.videos.mocap_videos)


def set_scene_parameters(recording: BlenderRecording, start_frame: int = 0, end_frame: int | None = None):
    import bpy

    if end_frame is None:
        end_frame = recording.frame_count - 1  # Blender frame range is inclusive (0-based)

    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame

    framerate = float(np.mean(np.diff(recording.data.timestamps))) ** -1
    bpy.context.scene.render.fps = int(round(framerate))




def create_arena():
    import bpy

    # --- parent empty ---
    arena_empty = bpy.data.objects.new("arena", None)
    arena_empty.empty_display_type = "PLAIN_AXES"
    arena_empty.empty_display_size = 0.02
    bpy.context.collection.objects.link(arena_empty)

    # --- shared dark matte material ---
    mat_name = "arena_bar_material"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8

    # --- 1m cube, bottom face at Z=0 ---
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0, 0, 0.5),  # center at Z=0.5 → spans Z=0 to Z=1.0
        align='WORLD',
        enter_editmode=False,
    )
    cube = bpy.context.active_object
    cube.name = "arena_cube"
    cube.data.materials.append(mat)
    cube.parent = arena_empty

    # --- Wireframe modifier: replace faces with 1cm-thick edge geometry ---
    wire = cube.modifiers.new(name="Wireframe", type='WIREFRAME')
    wire.thickness = 0.01       # 1 cm edge thickness
    wire.use_replace = True     # discard faces, keep only edges
    wire.use_boundary = True

    return arena_empty


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
