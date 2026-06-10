import numpy as np
from freemocap.core.pipeline.posthoc.video_group_helper import VideoGroupHelper, VideoHelper
from freemocap.core.tasks.calibration.shared.camera_model import CameraModel
from freemocap_blender_addon.core_functions.setup_scene.make_parent_empties import create_parent_empty
from freemocap_blender_addon.core_functions.setup_scene.clear_scene import clear_scene

from python_code.viz.blender.blender_helpers.blender_recording_model import BlenderRecording
import bpy

import math
from mathutils import Matrix, Quaternion


# ═══════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _log_header(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")

def _log_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  ▸ {title}")
    print(f"{'─' * 50}")

def _log_kv(key: str, value, unit: str = ""):
    unit_str = f" {unit}" if unit else ""
    print(f"    {key:30s} = {value}{unit_str}")

def _log_matrix(label: str, matrix: np.ndarray):
    print(f"    {label}:")
    for row in matrix:
        print(f"      [ {row[0]: 12.6f}  {row[1]: 12.6f}  {row[2]: 12.6f} ]")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def create_blender_scene(recording: BlenderRecording):
    _log_header("CREATE BLENDER SCENE")
    _log_kv("recording_path", recording.recording_path)
    _log_kv("recording_name", recording.name)
    _log_kv("frame_count", recording.frame_count)
    _log_kv("n_cameras", len(recording.data.calibration.cameras))
    _log_kv("timestamp_range",
            f"{recording.data.timestamps[0]:.3f} → {recording.data.timestamps[-1]:.3f}",
            "sec")
    _log_kv("n_mocap_videos", len(recording.videos.mocap_videos.videos))

    _log_section("Step 1: clear existing scene")
    print("    calling clear_scene()...")
    clear_scene()
    print("    ✓ scene cleared")

    _log_section("Step 2: set scene parameters")
    set_scene_parameters(recording=recording)

    _log_section("Step 3: create recording parent empty")
    create_parent_empty(name=recording.name,
                        display_scale=0.1,
                        type="ARROWS")
    print(f"    ✓ parent empty '{recording.name}' created (ARROWS, scale=0.1)")

    _log_section("Step 4: create arena")
    create_arena()

    _log_section("Step 5: add cameras")
    add_cameras(cameras=recording.data.calibration.cameras,
                mocap_videos=recording.videos.mocap_videos)

    _log_header("CREATE BLENDER SCENE — DONE")


# ═══════════════════════════════════════════════════════════════════════════════
# Scene parameters (frame range, fps)
# ═══════════════════════════════════════════════════════════════════════════════

def set_scene_parameters(recording: BlenderRecording, start_frame: int = 0, end_frame: int | None = None):
    _log_section("set_scene_parameters")

    # --- frame range ---
    _log_kv("input start_frame", start_frame)
    _log_kv("input end_frame", end_frame)
    _log_kv("recording.frame_count", recording.frame_count)

    if end_frame is None:
        end_frame = recording.frame_count - 1  # Blender frame range is inclusive (0-based)
        _log_kv("→ computed end_frame", end_frame, "(frame_count - 1, inclusive)")

    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    print(f"    ✓ scene.frame_start = {start_frame}")
    print(f"    ✓ scene.frame_end   = {end_frame}")

    # --- framerate ---
    timestamps = recording.data.timestamps
    _log_kv("n_timestamps", len(timestamps))
    _log_kv("timestamp[0]", f"{timestamps[0]:.6f}", "sec")
    _log_kv("timestamp[-1]", f"{timestamps[-1]:.6f}", "sec")

    time_diffs = np.diff(timestamps)
    _log_kv("mean Δt", f"{np.mean(time_diffs):.6f}", "sec")
    _log_kv("min  Δt", f"{np.min(time_diffs):.6f}", "sec")
    _log_kv("max  Δt", f"{np.max(time_diffs):.6f}", "sec")
    _log_kv("std  Δt", f"{np.std(time_diffs):.6f}", "sec")

    framerate = float(np.mean(time_diffs)) ** -1
    framerate_rounded = int(round(framerate))
    _log_kv("→ computed framerate", f"{framerate:.3f}", "fps")
    _log_kv("→ rounded framerate", framerate_rounded, "fps")

    bpy.context.scene.render.fps = framerate_rounded
    print(f"    ✓ scene.render.fps  = {framerate_rounded}")


# ═══════════════════════════════════════════════════════════════════════════════
# Arena (1m³ cube wireframe)
# ═══════════════════════════════════════════════════════════════════════════════

def create_arena():
    _log_section("create_arena")

    # --- parent empty ---
    _log_kv("arena parent", "arena (PLAIN_AXES, display_size=0.02)")
    arena_empty = bpy.data.objects.new("arena", None)
    arena_empty.empty_display_type = "PLAIN_AXES"
    arena_empty.empty_display_size = 0.02
    bpy.context.collection.objects.link(arena_empty)
    print("    ✓ arena parent empty created")

    # --- material ---
    mat_name = "arena_bar_material"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
        print(f"    ✓ material '{mat_name}' created (RGB=0.02, Roughness=0.8)")
    else:
        print(f"    ✓ material '{mat_name}' reused from existing")

    # --- cube ---
    cube_size = 1.0
    cube_location = (0, 0, 0.5)
    _log_kv("cube size", cube_size, "m")
    _log_kv("cube location", f"{cube_location}", "(center at Z=0.5 → spans Z=0 to Z=1.0)")

    bpy.ops.mesh.primitive_cube_add(
        size=cube_size,
        location=cube_location,
        align='WORLD',
        enter_editmode=False,
    )
    cube = bpy.context.active_object
    cube.name = "arena_cube"
    cube.data.materials.append(mat)
    cube.parent = arena_empty
    print(f"    ✓ cube '{cube.name}' created and parented to 'arena'")

    # --- wireframe modifier ---
    wire_thickness = 0.01  # 1 cm
    _log_kv("wireframe thickness", wire_thickness, "m (1 cm)")
    _log_kv("wireframe use_replace", True)
    _log_kv("wireframe use_boundary", True)

    wire = cube.modifiers.new(name="Wireframe", type='WIREFRAME')
    wire.thickness = wire_thickness
    wire.use_replace = True
    wire.use_boundary = True
    print(f"    ✓ wireframe modifier applied (thickness={wire_thickness}m)")

    print("    ✓ create_arena — DONE")
    return arena_empty


# ═══════════════════════════════════════════════════════════════════════════════
# Camera placement
# ═══════════════════════════════════════════════════════════════════════════════

def add_cameras(cameras: list[CameraModel], mocap_videos: VideoGroupHelper):
    _log_section("add_cameras")
    _log_kv("n_cameras", len(cameras))

    # --- camera list ---
    print(f"\n    cameras from calibration:")
    for i, cam in enumerate(cameras):
        print(f"      [{i}] id='{cam.id}'  index={cam.index}  "
              f"image_size={cam.image_size[0]}x{cam.image_size[1]}")

    # --- parent empty ---
    cameras_parent = bpy.data.objects.new("mocap_cameras", None)
    cameras_parent.empty_display_type = "CONE"
    cameras_parent.empty_display_size = 0.02
    bpy.context.collection.objects.link(cameras_parent)
    print(f"\n    ✓ parent empty 'mocap_cameras' created (CONE, display_size=0.02)")

    # --- place each camera ---
    print()
    for i, camera in enumerate(cameras):
        print(f"    ── camera {i+1}/{len(cameras)} ──")
        add_camera(camera=camera, parent=cameras_parent)

    print(f"\n    ✓ add_cameras — DONE ({len(cameras)} cameras placed)")


def add_camera(camera: CameraModel, parent: bpy.types.Object | None = None):
    _log_kv("camera id", camera.id)
    _log_kv("camera index", camera.index)
    _log_kv("image size", f"{camera.image_size[0]} x {camera.image_size[1]}", "px")

    # ═══ intrinsics ═══
    _log_section(f"intrinsics — {camera.id}")
    intr = camera.intrinsics
    _log_kv("fx", f"{intr.fx:.3f}", "px")
    _log_kv("fy", f"{intr.fy:.3f}", "px")
    _log_kv("cx", f"{intr.cx:.3f}", "px")
    _log_kv("cy", f"{intr.cy:.3f}", "px")
    _log_kv("k1", f"{intr.k1:.6f}")
    _log_kv("k2", f"{intr.k2:.6f}")
    _log_kv("p1", f"{intr.p1:.6f}")
    _log_kv("p2", f"{intr.p2:.6f}")

    # ═══ extrinsics ═══
    _log_section(f"extrinsics — {camera.id}")
    extr = camera.extrinsics
    _log_kv("quaternion_wxyz",
            f"[{extr.quaternion_wxyz[0]:.6f}, {extr.quaternion_wxyz[1]:.6f}, "
            f"{extr.quaternion_wxyz[2]:.6f}, {extr.quaternion_wxyz[3]:.6f}]")
    _log_kv("translation (tvec)",
            f"[{extr.translation[0]:.3f}, {extr.translation[1]:.3f}, {extr.translation[2]:.3f}]", "mm")

    _log_kv("rotation_matrix (R, world→cam)",
            f"shape={extr.rotation_matrix.shape}")
    _log_matrix("R (world→cam)", extr.rotation_matrix)

    # ═══ world_position ═══
    _log_section(f"world_position — {camera.id}")
    stored_pos = camera.world_position
    computed_pos = extr.world_position

    _log_kv("stored   world_position",
            f"[{stored_pos[0]:.3f}, {stored_pos[1]:.3f}, {stored_pos[2]:.3f}]", "mm")
    _log_kv("computed world_position (-Rᵀ·t)",
            f"[{computed_pos[0]:.3f}, {computed_pos[1]:.3f}, {computed_pos[2]:.3f}]", "mm")

    delta_pos = stored_pos - computed_pos
    _log_kv("Δ (stored − computed)",
            f"[{delta_pos[0]:.6f}, {delta_pos[1]:.6f}, {delta_pos[2]:.6f}]", "mm")
    _log_kv("‖Δ‖", f"{np.linalg.norm(delta_pos):.6f}", "mm")

    if not np.allclose(stored_pos, computed_pos, atol=1e-3):
        print(f"\n    ⚠ WARNING: stored and computed world_position differ!")
        print(f"       Using stored (from TOML): {stored_pos}")
        print(f"       Computed (from extrinsics): {computed_pos}")
    else:
        print(f"    ✓ stored ≈ computed (within 0.001mm tolerance)")

    # --- convert to Blender meters ---
    location = [float(coord) / 1000.0 for coord in stored_pos]
    _log_kv("→ Blender location",
            f"[{location[0]:.6f}, {location[1]:.6f}, {location[2]:.6f}]", "m")

    # ═══ world_orientation ═══
    _log_section(f"world_orientation — {camera.id}")
    stored_ori = camera.world_orientation
    computed_ori = extr.world_orientation

    _log_matrix("stored   world_orientation (cam→world)", stored_ori)
    _log_matrix("computed world_orientation (Rᵀ)", computed_ori)

    ori_delta = stored_ori - computed_ori
    _log_kv("‖stored − computed‖_F", f"{np.linalg.norm(ori_delta):.6f}")

    # --- rotation matrix → quaternion ---
    rot_matrix = Matrix(stored_ori.tolist())
    rot_quat = rot_matrix.to_quaternion()
    _log_kv("→ rot_quat (w,x,y,z)",
            f"[{rot_quat.w:.6f}, {rot_quat.x:.6f}, {rot_quat.y:.6f}, {rot_quat.z:.6f}]")

    # --- 180° X flip ---
    rot_flip = Quaternion((1.0, 0.0, 0.0), math.pi)
    _log_kv("rot_flip (180° around X)",
            f"[{rot_flip.w:.6f}, {rot_flip.x:.6f}, {rot_flip.y:.6f}, {rot_flip.z:.6f}]")

    rot_quat_fixed = rot_quat @ rot_flip
    _log_kv("→ rot_quat_fixed (w,x,y,z)",
            f"[{rot_quat_fixed.w:.6f}, {rot_quat_fixed.x:.6f}, "
            f"{rot_quat_fixed.y:.6f}, {rot_quat_fixed.z:.6f}]")

    # --- euler angles for readability ---
    euler = rot_quat_fixed.to_euler()
    _log_kv("  = euler (XYZ)",
            f"[{math.degrees(euler.x):.3f}°, {math.degrees(euler.y):.3f}°, "
            f"{math.degrees(euler.z):.3f}°]")

    # ═══ create Blender camera ═══
    _log_section(f"Blender object — {camera.id}")
    bpy.ops.object.camera_add(location=location)
    camera_obj = bpy.context.active_object
    camera_obj.name = f"camera_{camera.id}"
    camera_obj.show_name = True
    camera_obj.scale = (0.3, 0.3, 0.3)
    _log_kv("object name", camera_obj.name)
    _log_kv("object location",
            f"[{camera_obj.location.x:.6f}, {camera_obj.location.y:.6f}, "
            f"{camera_obj.location.z:.6f}]", "m")
    _log_kv("object scale",
            f"[{camera_obj.scale.x:.1f}, {camera_obj.scale.y:.1f}, "
            f"{camera_obj.scale.z:.1f}]", "(viewport visual only)")

    # --- rotation ---
    camera_obj.rotation_mode = 'QUATERNION'
    camera_obj.rotation_quaternion = rot_quat_fixed
    _log_kv("rotation_mode", camera_obj.rotation_mode)
    _log_kv("rotation_quaternion",
            f"[{camera_obj.rotation_quaternion.w:.6f}, "
            f"{camera_obj.rotation_quaternion.x:.6f}, "
            f"{camera_obj.rotation_quaternion.y:.6f}, "
            f"{camera_obj.rotation_quaternion.z:.6f}]")

    # --- sensor and lens ---
    camera_obj.data.sensor_width = 36.0
    width_px, height_px = camera.image_size
    f_mm = camera.intrinsics.fx * (36.0 / max(width_px, height_px))
    camera_obj.data.lens_unit = 'MILLIMETERS'
    camera_obj.data.lens = f_mm

    _log_kv("sensor_width", f"{camera_obj.data.sensor_width}", "mm")
    _log_kv("lens_unit", camera_obj.data.lens_unit)
    _log_kv("computed f_mm", f"{f_mm:.3f}", "mm")
    _log_kv("→ lens (focal length)", f"{camera_obj.data.lens:.3f}", "mm")

    # --- parent ---
    if parent is not None:
        camera_obj.parent = parent
        _log_kv("parent", parent.name)

    print(f"\n    ✓ camera '{camera_obj.name}' placed and configured")
    return camera_obj
