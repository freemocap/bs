import os
import tomllib

import numpy as np


def load_calibration_data(calibration_path: str) -> dict[str, object]:
    with open(calibration_path, "rb") as f:
        return tomllib.load(f)


def rodriguez_to_euler(r: np.ndarray) -> mathutils.Euler:
    """
    Convert a Rodriguez rotation vector directly to Euler angles without using quaternions.
    """
    # Add type checking and conversion
    if not isinstance(r, np.ndarray):
        print(f"Warning: rotation is not a numpy array, type: {type(r)}")
        r = np.array(r, dtype=float)

    # Ensure r is a 1D array with 3 elements
    if r.ndim > 1:
        print(f"Warning: rotation has shape {r.shape}, flattening")
        r = r.flatten()

    if len(r) != 3:
        print(f"Error: rotation vector should have 3 elements, has {len(r)}")
        return mathutils.Euler((0, 0, 0), "XYZ")  # Return identity as fallback

    print(f"Converting rotation vector {r} to Euler angles...")

    # Calculate the angle of rotation
    theta = np.linalg.norm(r)
    if abs(theta) < 1e-6:
        return mathutils.Euler((0, 0, 0), "XYZ")  # Identity rotation

    # Normalize the rotation axis
    axis = r / theta

    # Convert to rotation matrix (Rodrigues' formula)
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c
    x, y, z = axis

    # Construct the rotation matrix
    R = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )

    try:
        # Convert rotation matrix to Euler angles
        # Extract Euler angles from rotation matrix
        # This is based on the XYZ rotation order
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        if sy > 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock case
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        euler = mathutils.Euler((x, y, z), "XYZ")
        print(f"Converted to Euler angles: {euler}")
        return euler
    except Exception as e:
        print(f"Error converting rotation matrix to Euler: {e}")
        return mathutils.Euler((0, 0, 0), "XYZ")  # Return identity as fallback


def clear_scene() -> None:
    # First clear all animation data
    for obj in bpy.data.objects:
        if obj.animation_data:
            obj.animation_data_clear()

    # Delete all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Also clear materials, textures, and images that might be lingering
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    for image in bpy.data.images:
        bpy.data.images.remove(image)


def set_scene_frame_range_from_video(camera_video_map: dict[str, object]) -> None:
    if not camera_video_map:
        return

    print("Setting scene frame range based on video duration...")
    first_video_path = next(iter(camera_video_map.values()))
    video_filename = os.path.basename(first_video_path)

    # Load the video if not already loaded
    if video_filename not in bpy.data.images:
        video = bpy.data.images.load(first_video_path)
    else:
        video = bpy.data.images[video_filename]

    # Set scene frame range based on video duration
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = video.frame_duration

    print(f"Set frame range to 1-{video.frame_duration} based on {video_filename}")


def set_render_resolution(width: int = 1024, height: int = 1024) -> None:
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
