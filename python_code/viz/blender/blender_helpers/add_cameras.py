import math

import bpy
from freemocap.core.pipeline.posthoc.video_group_helper import VideoGroupHelper, VideoHelper
from freemocap.core.tasks.calibration.shared.camera_model import CameraModel
from mathutils import Matrix, Quaternion


def add_cameras(cameras: list[CameraModel]):#, mocap_videos: VideoGroupHelper):
    print(f"Creating parent empty 'mocap_cameras' (CONE, display_size=0.02)")
    # cameras_parent = bpy.data.objects.new("mocap_cameras", None)
    # cameras_parent.empty_display_type = "CONE"
    # cameras_parent.empty_display_size = 0.02
    # bpy.context.collection.objects.link(cameras_parent)

    print(f"Placing {len(cameras)} cameras:")
    for i, camera in enumerate(cameras):
        # video = mocap_videos.videos.get(camera.id)
        print(f"\n{'─' * 50}")
        print(f"Camera {i + 1} of {len(cameras)}:")
        print(f"{'─' * 50}")
        print(camera)
        add_camera(camera=camera)#, video=video, parent=cameras_parent)

    print(f"\nAll {len(cameras)} cameras placed.")


def add_camera(camera: CameraModel, video: VideoHelper | None = None, parent: bpy.types.Object | None = None):
    # position: millimeters → meters
    world_position_millimeters = camera.world_position
    location_meters = [float(coord) / 1000.0 for coord in world_position_millimeters]
    print(f"  Blender location (meters): [{location_meters[0]:.6f}, {location_meters[1]:.6f}, {location_meters[2]:.6f}]")

    # orientation: 3×3 world_orientation matrix → quaternion → 180° X flip
    world_orientation_matrix = camera.world_orientation
    rot_matrix = Matrix(world_orientation_matrix.tolist())
    rot_quaternion = rot_matrix.to_quaternion()
    print(f"  world_orientation → quaternion: "
          f"[w={rot_quaternion.w:.6f}, x={rot_quaternion.x:.6f}, "
          f"y={rot_quaternion.y:.6f}, z={rot_quaternion.z:.6f}]")

    rot_flip = Quaternion((1.0, 0.0, 0.0), math.pi)
    print(f"  180° X-axis flip quaternion: "
          f"[w={rot_flip.w:.6f}, x={rot_flip.x:.6f}, "
          f"y={rot_flip.y:.6f}, z={rot_flip.z:.6f}]")

    rot_quaternion_fixed = rot_quaternion @ rot_flip
    print(f"  Final quaternion (world_orientation @ 180° X flip): "
          f"[w={rot_quaternion_fixed.w:.6f}, x={rot_quaternion_fixed.x:.6f}, "
          f"y={rot_quaternion_fixed.y:.6f}, z={rot_quaternion_fixed.z:.6f}]")

    euler = rot_quaternion_fixed.to_euler()
    print(f"  Final Euler angles (XYZ degrees): "
          f"[{math.degrees(euler.x):.3f}, {math.degrees(euler.y):.3f}, {math.degrees(euler.z):.3f}]")

    # create Blender camera
    bpy.ops.object.camera_add(location=location_meters)
    camera_object = bpy.context.active_object
    camera_object.name = f"camera_{camera.id}"
    camera_object.show_name = True
    camera_object.scale = (0.3, 0.3, 0.3)
    print(f"  Created Blender camera: '{camera_object.name}'")

    camera_object.rotation_mode = 'QUATERNION'
    camera_object.rotation_quaternion = rot_quaternion_fixed
    print(f"  Rotation mode: {camera_object.rotation_mode}")
    print(f"  Rotation quaternion set on object: "
          f"[w={camera_object.rotation_quaternion.w:.6f}, "
          f"x={camera_object.rotation_quaternion.x:.6f}, "
          f"y={camera_object.rotation_quaternion.y:.6f}, "
          f"z={camera_object.rotation_quaternion.z:.6f}]")

    # sensor and lens — match camera shape to actual video aspect ratio
    sensor_width_millimeters = 36.0
    width_pixels, height_pixels = camera.image_size
    sensor_height_millimeters = sensor_width_millimeters * height_pixels / width_pixels
    focal_length_millimeters = camera.intrinsics.fx * (sensor_width_millimeters / max(width_pixels, height_pixels))
    camera_object.data.sensor_width = sensor_width_millimeters
    camera_object.data.sensor_height = sensor_height_millimeters
    camera_object.data.sensor_fit = 'HORIZONTAL'
    camera_object.data.lens_unit = 'MILLIMETERS'
    camera_object.data.lens = focal_length_millimeters
    print(f"  Sensor: {sensor_width_millimeters:.3f} x {sensor_height_millimeters:.3f} mm (from image {width_pixels}x{height_pixels})")
    print(f"  Sensor fit: {camera_object.data.sensor_fit}")
    print(f"  Focal length (fx): {camera.intrinsics.fx:.6f} pixels")
    print(f"  Computed focal length: {focal_length_millimeters:.6f} millimeters")
    print(f"  Lens unit: {camera_object.data.lens_unit}")
    print(f"  Lens (focal length) set to: {camera_object.data.lens:.6f} millimeters")

    if parent is not None:
        camera_object.parent = parent
        print(f"  Parented to: '{parent.name}'")

    print(f"  Camera '{camera_object.name}' placed successfully.")

    # load video plane
    if video is not None:
        load_video_as_plane(video=video, camera_object=camera_object, camera=camera)

    return camera_object


def load_video_as_plane(video: VideoHelper, camera_object: bpy.types.Object, camera: CameraModel):
    video_path = str(video.video_path)
    width_pixels, height_pixels = camera.image_size
    sensor_width_millimeters = camera_object.data.sensor_width
    focal_length_millimeters = camera_object.data.lens

    # place plane far enough from camera to be large and visible in the viewport
    plane_distance_meters = .3

    # compute frustum size at this distance
    sensor_width_meters = sensor_width_millimeters / 1000.0
    focal_length_meters = focal_length_millimeters / 1000.0
    frustum_width_meters = plane_distance_meters * sensor_width_meters / focal_length_meters
    frustum_height_meters = frustum_width_meters * height_pixels / width_pixels

    print(f"\n  --- Video plane ---")
    print(f"  Video path: {video_path}")
    print(f"  Video: {video.metadata.width}x{video.metadata.height}, "
          f"{video.metadata.fps} frames_per_second, {video.metadata.frame_count} frames")
    print(f"  Plane distance from camera: {plane_distance_meters} meters")
    print(f"  Frustum at this distance: {frustum_width_meters:.3f} x {frustum_height_meters:.3f} meters")
    print(f"  Sensor: {sensor_width_millimeters} x {camera_object.data.sensor_height:.3f} mm")
    print(f"  Focal length: {focal_length_millimeters:.3f} mm")
    print(f"  Image: {width_pixels} x {height_pixels} pixels")

    # create plane
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = f"video_plane_{camera.id}"

    # load image
    image = bpy.data.images.load(filepath=video_path)
    print(f"  Loaded image: {image.name} ({image.size[0]}x{image.size[1]})")

    # create emission material with image texture
    material_name = f"video_material_{camera.id}"
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.image = image
    texture_node.location = (-300, 0)

    emission_node = nodes.new('ShaderNodeEmission')
    emission_node.location = (0, 0)

    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (300, 0)

    links.new(texture_node.outputs['Color'], emission_node.inputs['Color'])
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    plane.data.materials.append(material)

    # position and scale the plane
    plane.location = (0, 0, -plane_distance_meters)
    plane.scale = (frustum_width_meters, frustum_height_meters, 1.0)
    plane.parent = camera_object

    print(f"  Plane '{plane.name}': position=(0, 0, -{plane_distance_meters}), "
          f"scale=({frustum_width_meters:.3f}, {frustum_height_meters:.3f}, 1.0)")
    print(f"  Parented to '{camera_object.name}'")