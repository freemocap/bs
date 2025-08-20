import glob
import os
import tomllib
from pathlib import Path

import bpy
import numpy as np

from python_code.blender_stuff.blender_helpers.blender_utilities import rodriguez_to_euler

BASLER_SENSOR_WIDTH = 11.26  # Basler camera sensor width in mm


def load_calibration_data(calibration_path: str) -> dict[str, dict[str, object]]:
    with open(calibration_path, "rb") as f:
        calibration_by_camera =  tomllib.load(f)
    print(f"Loaded calibration data from {calibration_path} for cameras: {list(calibration_by_camera.keys())}")
    return calibration_by_camera

def create_mocap_camera_objects(
        calibration_by_camera = dict[str, object],
        sensor_width_mm: float = BASLER_SENSOR_WIDTH) -> dict[str, bpy.types.Object]:

    camera_objects: dict[str, bpy.types.Object] = {}

    for camera_id, camera_calibration in calibration_by_camera.items():
        if camera_id == "metadata":
            continue  # Skip metadata entry

        print(f"Creating camera {camera_id} object...")

        # Debug: print the structure of camera_calibration
        print(f"Camera calibration keys: {list(camera_calibration.keys())}")

        # Check if required keys exist
        required_keys = ["name", "size", "matrix", "rotation", "translation"]
        for key in required_keys:
            if key not in camera_calibration:
                print(
                    f"Error: Missing required key '{key}' in camera calibration for {camera_id}"
                )
                continue

        camera_name = camera_calibration["name"]
        camera_resolution = camera_calibration["size"]
        camera_matrix = camera_calibration["matrix"]

        # Debug: print rotation data
        print(f"Rotation data for {camera_id}: {camera_calibration['rotation']}")

        try:
            # Convert mm to m for translation
            camera_translation = [
                t * 0.001 for t in camera_calibration["translation"]
            ]  # Convert mm to m

            # Create camera data first
            camera_data = bpy.data.cameras.new(name=f"CameraData_{camera_name}")

            # Then create the camera object with the data
            camera_object = bpy.data.objects.new(f"Camera_{camera_name}", camera_data)

            # Link the camera to the scene
            bpy.context.collection.objects.link(camera_object)

            # Set the camera location
            camera_object.location = camera_translation

            # Set scale for visibility
            camera_object.scale = (0.1, 0.1, 0.1)

            # Process rotation safely
            rotation_data = camera_calibration["rotation"]
            if isinstance(rotation_data, list):
                rotation_data = np.array(rotation_data, dtype=float)

            # Get Euler angles directly
            camera_euler = rodriguez_to_euler(r=rotation_data)

            # Apply a 180-degree rotation around Y to match Blender's coordinate system
            # We'll do this directly in Euler space
            camera_object.rotation_mode = "XYZ"
            camera_object.rotation_euler = camera_euler

            # Apply the Y-flip directly
            camera_object.rotation_euler.y += np.radians(180)

            # Extract camera parameters from the intrinsic matrix
            fx = camera_matrix[0][0]
            fy = camera_matrix[1][1]
            cx = camera_matrix[0][2]
            cy = camera_matrix[1][2]

            # Calculate focal length in mm
            focal_length_mm = (fx / camera_resolution[0]) * sensor_width_mm
            camera_object.data.lens = focal_length_mm

            # Set the sensor size
            camera_object.data.sensor_width = sensor_width_mm

            # Calculate and set the shift to match the principal point
            shift_x = (cx - camera_resolution[0] / 2) / camera_resolution[0]
            shift_y = (cy - camera_resolution[1] / 2) / camera_resolution[1]
            camera_object.data.shift_x = shift_x
            camera_object.data.shift_y = shift_y

            camera_object.data.clip_start = 0.01

            # Store camera object for later reference
            camera_objects[camera_id] = camera_object

            print(f"Created camera {camera_name} at position {camera_translation}")
            print(f"  - Focal length: {focal_length_mm:.2f}mm")
            print(f"  - Sensor width: {sensor_width_mm}mm")
            print(f"  - Shift X: {shift_x:.4f}, Shift Y: {shift_y:.4f}")

        except Exception as e:
            print(f"Error creating camera {camera_id}: {e}")
            import traceback

            traceback.print_exc()
            # Continue with next camera
            continue

    return camera_objects


def create_video_material(video_path: str, name: str) -> bpy.types.Material:
    print(f"Creating material for video {name}...")
    material = bpy.data.materials.new(name=f"Video_Material_{name}")
    material.use_nodes = True

    # Clear existing nodes
    nodes = material.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    # Create nodes
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    mix_node = nodes.new(type="ShaderNodeMixShader")
    emission_node = nodes.new(type="ShaderNodeEmission")
    transparent_node = nodes.new(type="ShaderNodeBsdfTransparent")
    texture_node = nodes.new(type="ShaderNodeTexImage")

    # Load the video
    texture_node.image = bpy.data.images.load(video_path)
    texture_node.image.source = "MOVIE"

    # Set up auto-refresh for the movie
    texture_node.image_user.use_auto_refresh = True
    texture_node.image_user.frame_duration = texture_node.image.frame_duration

    # Set mix factor (adjust this value to control transparency)
    mix_node.inputs[0].default_value = 0.7  # 0.7 means 70% video, 30% transparent

    # Connect nodes
    links = material.node_tree.links
    links.new(texture_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], mix_node.inputs[2])
    links.new(transparent_node.outputs["BSDF"], mix_node.inputs[1])
    links.new(mix_node.outputs["Shader"], output_node.inputs["Surface"])

    # Position nodes
    output_node.location = (400, 0)
    mix_node.location = (200, 0)
    emission_node.location = (0, -100)
    transparent_node.location = (0, 100)
    texture_node.location = (-200, -100)

    # Set material properties for transparency
    material.blend_method = "BLEND"

    return material


def map_cameras_to_videos(
        calibration_by_camera: dict[str, object], videos_path: str
) -> dict[str, object]:
    # Find all mp4 files in the synchronized_videos directory
    video_files = glob.glob(str(Path(videos_path) / "*.mp4"))
    print(f"Found {len(video_files)} video files in {videos_path}")

    camera_video_map: dict[str, object] = {}

    # Process each camera and find matching videos
    for camera_id, camera_calibration in calibration_by_camera.items():
        if camera_id == "metadata":
            continue  # Skip metadata entry

        print(f"Finding video for camera {camera_id}...")
        # Get the camera name from the calibration data
        camera_name = camera_calibration["name"]

        # Find a video file that contains the camera name
        matching_video = None
        for video_path in video_files:
            video_filename = os.path.basename(video_path)
            if camera_name in video_filename:
                matching_video = video_path
                break

        if matching_video:
            camera_video_map[camera_id] = matching_video
            print(
                f"Matched camera {camera_name} with video {os.path.basename(matching_video)}"
            )
        else:
            print(f"Warning: No matching video found for camera {camera_name}")

    return camera_video_map


def create_video_planes(
        camera_objects: dict[str, bpy.types.Object],
        camera_video_map: dict[str, object],
        calibration_by_camera: dict[str, object],
        plane_distance: float = 1.0,
) -> None:
    camera_num = -1
    for camera_id, video_path in camera_video_map.items():
        camera_num += 1
        print(f"Creating video plane for camera {camera_id}...")
        camera_calibration = calibration_by_camera[camera_id]
        camera_name = camera_calibration["name"]
        camera_resolution = camera_calibration["size"]
        camera_matrix = camera_calibration["matrix"]

        # Get the camera object we created earlier
        camera_object = camera_objects[camera_id]

        # Extract camera parameters from the intrinsic matrix
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]

        # Calculate the dimensions of the image plane at the specified distance
        width = 2 * plane_distance * (camera_resolution[0] / (2 * fx))
        height = 2 * plane_distance * (camera_resolution[1] / (2 * fy))

        # Create a plane for the video
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        plane = bpy.context.active_object
        plane.name = f"VideoPlane_{camera_name}"

        # Scale the plane to match the calculated dimensions
        plane.scale = (width / 2, height / 2, 1.0)

        # Parent the plane to the camera
        # plane.parent = camera_object

        # Position the plane directly in front of the camera at the standard distance
        plane.location = (camera_num, 0, -plane_distance)

        # Rotate the plane to face the camera
        # plane.rotation_euler = (np.radians(180), 0, 0)

        # Create and assign material with the video
        material = create_video_material(video_path=video_path, name=camera_name)
        plane.data.materials.append(material)

        print(f"Created geometrically correct video plane for camera {camera_name}")
        print(f"  - Distance: {plane_distance}m")
        print(f"  - Plane dimensions: {width:.4f}m × {height:.4f}m")
