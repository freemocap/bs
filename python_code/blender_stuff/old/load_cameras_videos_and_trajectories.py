import os
import bpy
import mathutils
import tomllib
from pathlib import Path
import glob
import numpy as np
import colorsys

BASLER_SENSOR_WIDTH = 11.26  # Basler camera sensor width in mm


def validate_paths(
    camera_calibration_path: str, videos_path: str, trajectories_path: str
) -> None:
    if not Path(camera_calibration_path).is_file():
        raise FileNotFoundError(
            f"Camera calibration file not found: {camera_calibration_path}"
        )

    if not Path(videos_path).is_dir():
        raise FileNotFoundError(
            f"Synchronized videos directory not found: {videos_path}"
        )

    if not Path(trajectories_path).is_file():
        raise FileNotFoundError(f"Trajectory data file not found: {trajectories_path}")


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


def create_camera_objects(
    calibration_by_camera: dict[str, object],
    sensor_width_mm: float = BASLER_SENSOR_WIDTH,
) -> dict[str, bpy.types.Object]:
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
    for camera_id, video_path in camera_video_map.items():
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
        plane.parent = camera_object

        # Position the plane directly in front of the camera at the standard distance
        plane.location = (0, 0, -plane_distance)

        # Rotate the plane to face the camera
        # plane.rotation_euler = (np.radians(180), 0, 0)

        # Create and assign material with the video
        material = create_video_material(video_path=video_path, name=camera_name)
        plane.data.materials.append(material)

        print(f"Created geometrically correct video plane for camera {camera_name}")
        print(f"  - Distance: {plane_distance}m")
        print(f"  - Plane dimensions: {width:.4f}m × {height:.4f}m")


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


def create_ray_visualization(
    camera_obj: bpy.types.Object,
    image_points: list[tuple[float, float]] = [(0.5, 0.5)],
    length: float = 2.0,
) -> None:
    print(f"Creating ray visualizations for camera {camera_obj.name}...")
    for i, (u, v) in enumerate(image_points):
        # Convert normalized coordinates to camera space
        x = (u - 0.5) * 2  # Convert [0,1] to [-1,1]
        y = (v - 0.5) * 2  # Convert [0,1] to [-1,1]

        # Create a curve for the ray
        curve_data = bpy.data.curves.new(f"RayCurve_{i}", "CURVE")
        curve_data.dimensions = "3D"

        polyline = curve_data.splines.new("POLY")
        polyline.points.add(1)  # Two points: camera origin and target

        # Start at camera origin
        polyline.points[0].co = (0, 0, 0, 1)

        # End at the target point
        polyline.points[1].co = (x * length, y * length, -length, 1)

        # Create the curve object
        curve_obj = bpy.data.objects.new(f"Ray_{camera_obj.name}_{i}", curve_data)

        # Link to scene and parent to camera
        bpy.context.collection.objects.link(curve_obj)
        curve_obj.parent = camera_obj

        # Set curve appearance
        curve_data.bevel_depth = 0.005
        curve_obj.data.materials.append(
            bpy.data.materials.new(name=f"Ray_Material_{i}")
        )
        curve_obj.data.materials[0].diffuse_color = (1, 0, 0, 1)  # Red


def load_trajectory_data(trajectories_path: str) -> tuple[np.ndarray, int, int]:
    trajectory_data = np.load(trajectories_path, allow_pickle=True)
    trajectory_data *= 0.001  # Convert mm to m
    number_of_frames = trajectory_data.shape[0]
    number_of_markers = trajectory_data.shape[1]

    return trajectory_data, number_of_frames, number_of_markers


def process_trajectory_data(
    trajectory_data: np.ndarray, number_of_markers: int
) -> tuple[dict[int, list[list[float]]], dict[int, object]]:
    # Load the trajectory data into a dictionary
    trajectories: dict[int, list[list[float]]] = {
        idx: trajectory_data[:, idx, :].tolist() for idx in range(number_of_markers)
    }

    # Determine % isnan for each trajectory
    nan_percentages: dict[int, object] = {
        idx: np.isnan(trajectory_data[:, idx, :]).mean() * 100
        for idx in range(number_of_markers)
    }
    print(f"NaN percentages: {nan_percentages}")

    return trajectories, nan_percentages


def create_trajectory_objects(
    trajectories: dict[int, list[list[float]]], number_of_frames: int
) -> None:
    number_of_markers = len(trajectories)
    print(
        f"Creating {number_of_markers} empties, each with {number_of_frames} frames..."
    )
    marker_index_to_name = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
    }
    # create virtual markers between the eye and ear markers
    eye_center_xyz = np.mean(
        [
            trajectories[marker_index_to_name["left_eye"]],
            trajectories[marker_index_to_name["right_eye"]],
        ],
        axis=0,
    )
    ear_center_xyz = np.mean(
        [
            trajectories[marker_index_to_name["left_ear"]],
            trajectories[marker_index_to_name["right_ear"]],
        ],
        axis=0,
    )
    trajectories_by_name = {
        name: trajectories[idx] for name, idx in marker_index_to_name.items()
    }
    trajectories_by_name["eye_center"] = eye_center_xyz
    trajectories_by_name["ear_center"] = ear_center_xyz
    for name, trajectories in trajectories_by_name.items():
        index = marker_index_to_name.get(name, None)
        print(f"Creating empty for marker {name} at index {index}...")
        empty = bpy.data.objects.new(f"Marker_{name}_empty", None)
        bpy.context.collection.objects.link(empty)
        empty.empty_display_size = 0.0025

        if name == "nose":
            empty.empty_display_type = "SPHERE"
            empty.empty_display_size = 0.02

        # Create a small sphere and parent it to the empty
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=(0, 0, 0))
        sphere = bpy.context.object
        sphere.name = f"Marker_{name}_sphere"
        sphere.parent = empty

        # Add a simple color material to the sphere
        material = bpy.data.materials.new(name=f"Marker_{name}_material")

        # Generate a unique color based on marker index
        if name == "nose":
            hue = 0  # Red for the first marker
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # full saturation
            material.diffuse_color = (r, g, b, 1.0)

        else:
            hue = (index * 0.1) % 1.0  # Cycle through hues

            r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 1.0)
            material.diffuse_color = (r, g, b, 1.0)

        # Assign material to sphere
        if sphere.data.materials:
            sphere.data.materials[0] = material
        else:
            sphere.data.materials.append(material)
        sphere.show_in_front = True
        # Add keyframes for each frame
        for frame in range(number_of_frames):
            if np.isnan(trajectories[index][frame]).any():
                continue
            empty.location = mathutils.Vector(trajectories[index][frame])
            empty.keyframe_insert(data_path="location", frame=frame)


def main() -> None:
    # Define paths

    # # #2025-04-28
    # camera_calibration_toml_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\ferret_9C04_NoImplant_P35_E3\clips\2025_04_28.ferret_9C04_NoImplant_P35_E3.fr300-1200.ears_nose_eyes\2025-04-28-calibration_camera_calibration.toml"
    # trajectories_path =            r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\ferret_9C04_NoImplant_P35_E3\clips\2025_04_28.ferret_9C04_NoImplant_P35_E3.fr300-1200.ears_nose_eyes\output_data\dlc_body_rigid_3d_xyz.npy"
    # synchronized_videos_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\ferret_9C04_NoImplant_P35_E3\clips\2025_04_28.ferret_9C04_NoImplant_P35_E3.fr300-1200.ears_nose_eyes\annotated_videos"

    # # #2025-05-04
    camera_calibration_toml_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\ferret_9C04_NoImplant_P35_E3\clips\2025_04_28.ferret_9C04_NoImplant_P35_E3.fr300-1200.ears_nose_eyes\2025-04-28-calibration_camera_calibration.toml"
    trajectories_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-05-04\ferret_9C04_NoImplant_P41_E9\clips\model_toy_clipped_5_4\output_data_iteration_2_4_28_aligned_calibration\dlc_body_rigid_3d_xyz.npy"
    synchronized_videos_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-05-04\ferret_9C04_NoImplant_P41_E9\clips\model_toy_clipped_5_4\annotated_videos"

    # Validate paths
    validate_paths(
        camera_calibration_path=camera_calibration_toml_path,
        videos_path=synchronized_videos_path,
        trajectories_path=trajectories_path,
    )

    # Load calibration data
    calibration_by_camera = load_calibration_data(
        calibration_path=camera_calibration_toml_path
    )

    # Clear scene
    clear_scene()

    # Create camera objects
    camera_objects = create_camera_objects(calibration_by_camera=calibration_by_camera)

    # Map cameras to videos
    camera_video_map = map_cameras_to_videos(
        calibration_by_camera=calibration_by_camera,
        videos_path=synchronized_videos_path,
    )

    # Create video planes
    STANDARD_PLANE_DISTANCE = 1.0  # 1 meter
    create_video_planes(
        camera_objects=camera_objects,
        camera_video_map=camera_video_map,
        calibration_by_camera=calibration_by_camera,
        plane_distance=STANDARD_PLANE_DISTANCE,
    )

    # Set scene frame range from video
    set_scene_frame_range_from_video(camera_video_map=camera_video_map)

    # Set render resolution
    set_render_resolution(width=1024, height=1024)

    # Optional: Add ray visualizations for each camera
    # for camera_id, camera_obj in camera_objects.items():
    #     create_ray_visualization(
    #         camera_obj=camera_obj,
    #         image_points=[
    #             (0, 0),      # Bottom-left
    #             (1, 0),      # Bottom-right
    #             (1, 1),      # Top-right
    #             (0, 1),      # Top-left
    #             (0.5, 0.5)   # Center
    #         ],
    #         length=STANDARD_PLANE_DISTANCE * 1.5
    #     )

    print("Geometrically correct video planes setup complete!")

    # Load and process trajectory data
    trajectory_data, number_of_frames, number_of_markers = load_trajectory_data(
        trajectories_path=trajectories_path
    )

    # Set scene frame range for trajectories
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = number_of_frames - 1

    # Process trajectory data
    trajectories, nan_percentages = process_trajectory_data(
        trajectory_data=trajectory_data, number_of_markers=number_of_markers
    )

    # Create trajectory objects
    create_trajectory_objects(
        trajectories=trajectories, number_of_frames=number_of_frames
    )


main()
bpy.context.space_data.shading.type = "MATERIAL"
print("Done!")
