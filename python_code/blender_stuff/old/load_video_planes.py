import os
import bpy
import mathutils
import tomllib
from pathlib import Path
import glob
import numpy as np

# Load Camera Calibration Data
camera_calibration_toml_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\2025-04-28-calibration\2025-04-28_camera_calibration.toml"

# Path to the annotated videos folder (in the same directory as the calibration file)
calibration_dir = Path(camera_calibration_toml_path).parent
videos_dir = calibration_dir / "annotated_videos"

# Load camera calibration data
with open(camera_calibration_toml_path, "rb") as f:
    calibration_by_camera = tomllib.load(f)

# Function to convert rodriguez rotation to Euler angles
def rodriguez_to_euler(r):
    import numpy as np
    theta = np.linalg.norm(r)
    if theta < 1e-6:
        return mathutils.Euler((0, 0, 0), 'XYZ')  # Identity rotation
    
    # First convert to quaternion
    r = r / theta
    w = np.cos(theta / 2)
    x = r[0] * np.sin(theta / 2)
    y = r[1] * np.sin(theta / 2)
    z = r[2] * np.sin(theta / 2)
    quat = mathutils.Quaternion([w, x, y, z])
    
    # Then convert quaternion to Euler angles
    euler = quat.to_euler('XYZ')
    
    return euler

# Create a rotation to flip the camera direction (180 degrees around Y and Z axes)
flip_euler = mathutils.Euler((0, np.radians(180), np.radians(180)), 'XYZ')

# Create a material for the video planes
def create_video_material(video_path, name):
    material = bpy.data.materials.new(name=f"Video_Material_{name}")
    material.use_nodes = True
    # material.use_backface_culling = True

    # Clear existing nodes
    nodes = material.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = nodes.new(type='ShaderNodeEmission')
    texture_node = nodes.new(type='ShaderNodeTexImage')
    
    # Load the video
    texture_node.image = bpy.data.images.load(video_path)
    texture_node.image.source = 'MOVIE'
    
    # Set up auto-refresh for the movie
    texture_node.image_user.use_auto_refresh = True
    texture_node.image_user.frame_duration = texture_node.image.frame_duration
    
    # Connect nodes
    links = material.node_tree.links
    links.new(texture_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    
    # Position nodes
    output_node.location = (300, 0)
    emission_node.location = (100, 0)
    texture_node.location = (-100, 0)
    
    return material

# Find all mp4 files in the annotated_videos directory
video_files = glob.glob(str(videos_dir / "*.mp4"))
print(f"Found {len(video_files)} video files in {videos_dir}")

# Create a dictionary to map camera names to video files
camera_video_map = {}

# Process each camera and find matching videos
for camera_id, camera_calibration in calibration_by_camera.items():
    if camera_id == "metadata":
        continue  # Skip metadata entry
    
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
        print(f"Matched camera {camera_name} with video {os.path.basename(matching_video)}")
    else:
        print(f"Warning: No matching video found for camera {camera_name}")

# Process each camera with a matching video
for camera_id, video_path in camera_video_map.items():
    camera_calibration = calibration_by_camera[camera_id]
    camera_name = camera_calibration["name"]
    camera_resolution = camera_calibration["size"]
    camera_matrix = camera_calibration["matrix"]
    camera_euler = rodriguez_to_euler(camera_calibration["rotation"])
    
    # Apply the flip to the camera euler angles
    # We need to convert to quaternion, multiply, then convert back to euler
    camera_quat = camera_euler.to_quaternion()
    flip_quat = flip_euler.to_quaternion()
    final_quat = camera_quat @ flip_quat
    final_euler = final_quat.to_euler('XYZ')
    
    # Convert translation from mm to m for Blender
    camera_translation = [t * 0.001 for t in camera_calibration["translation"]]
    
    # Calculate focal length in mm (assuming 35mm sensor)
    focal_length_pixels = camera_matrix[0][0]  # Assuming square pixels and fx = fy
    sensor_width_mm = 11.26 # sensor size - https://www.baslerweb.com/en-us/shop/aca2040-90umnir/
    focal_length_mm = (focal_length_pixels / camera_resolution[0]) * sensor_width_mm
    
    # Create a plane for the video
    bpy.ops.mesh.primitive_plane_add(
        size=1.0,
        location=camera_translation
    )
    plane = bpy.context.active_object
    plane.name = f"VideoPlane_{camera_name}"
    
    # Set the rotation mode to Euler and apply the rotation
    plane.rotation_mode = 'XYZ'
    plane.rotation_euler = final_euler
    
    # Calculate the aspect ratio of the video
    aspect_ratio = camera_resolution[0] / camera_resolution[1]
    
    # Scale the plane to match the video aspect ratio
    # We'll make the plane width match the aspect ratio while keeping height at 1
    plane.scale = (aspect_ratio*.22, .22, .22)
    
    # Flip the normals of the plane to face the camera
    # Select the plane
    bpy.context.view_layer.objects.active = plane
    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Select all vertices
    bpy.ops.mesh.select_all(action='SELECT')
    # Flip normals
    bpy.ops.mesh.flip_normals()
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    # Create and assign material with the video
    material = create_video_material(video_path, camera_name)
    plane.data.materials.append(material)
    
    # Position the plane slightly in front of the camera
    # Create a local offset vector (z-axis in camera space) 
    # We need to use the rotation to transform the offset
    local_offset = mathutils.Vector((0, 0, -0.05))  # 10cm in front of camera
    
    # Transform to world space and apply
    rotation_matrix = final_euler.to_matrix()
    world_offset = rotation_matrix @ local_offset
    plane.location = mathutils.Vector(camera_translation) + world_offset
    
    print(f"Created video plane for camera {camera_name}")

# Set the scene frame range to match the videos
# Get the first video to determine frame range
if camera_video_map:
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

#
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
print("Video planes setup complete!")