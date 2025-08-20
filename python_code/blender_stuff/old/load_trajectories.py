import os
import bpy
import mathutils
import tomllib
from pathlib import Path
import glob
import numpy as np
import colorsys

BASLER_SENSOR_WIDTH = 11.26  # Basler camera sensor width in mm


def validate_paths(trajectories_path: str) -> None:
    if not Path(trajectories_path).is_file():
        raise FileNotFoundError(f"Trajectory data file not found: {trajectories_path}")


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


def create_trajectory_objects(
    trajectories: dict[str, list[list[float]]], number_of_frames: int
) -> None:
    number_of_markers = len(trajectories)
    print(
        f"Creating {number_of_markers} empties, each with {number_of_frames} frames..."
    )
    # create virtual markers between the eye and ear markers

    trajectories["eye_center"] = np.mean(
        [
            trajectories["left_eye"],
            trajectories["right_eye"],
        ],
        axis=0,
    )
    trajectories["ear_center"] = np.mean(
        [
            trajectories["left_ear"],
            trajectories["right_ear"],
        ],
        axis=0,
    )
    for index, item in enumerate(trajectories.items()):
        name, trajectory = item
        print(f"Creating empty for marker {name} at index {index}...")
        empty = bpy.data.objects.new(f"Marker_{name}_empty", None)
        bpy.context.collection.objects.link(empty)
        empty.empty_display_size = 0.001

        if name == "nose":
            empty.empty_display_type = "ARROWS"
            empty.empty_display_size = 0.02
        if name == "toy":
            empty.empty_display_type = "CUBE"
            empty.empty_display_size = 0.04

        # # Create a small sphere and parent it to the empty
        # bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=(0, 0, 0))
        # sphere = bpy.context.object
        # sphere.name = f"Marker_{name}_sphere"
        # sphere.parent = empty

        # # Add a simple color material to the sphere
        # material = bpy.data.materials.new(name=f"Marker_{name}_material")

        # # Generate a unique color based on marker index
        # if name == "nose":
        #     hue = 0  # Red for the first marker
        #     r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # full saturation
        #     material.diffuse_color = (r, g, b, 1.0)

        # else:
        #     # hue = (index * 0.1) % 1.0  # Cycle through hues
        #     # Use a different color for each marker
        #     hue = (index / number_of_markers) % 1.0  # Cycle through hues

        #     r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 1.0)
        #     material.diffuse_color = (r, g, b, 1.0)

        # # Assign material to sphere
        # if sphere.data.materials:
        #     sphere.data.materials[0] = material
        # else:
        #     sphere.data.materials.append(material)
        # sphere.show_in_front = True
        # Add keyframes for each frame
        for frame_number, frame_xyz in enumerate(trajectory):
            if np.isnan(frame_xyz).any():
                print(
                    f"Skipping frame {frame_xyz} for marker {name} due to NaN values."
                )
                continue
            empty.location = mathutils.Vector(frame_xyz)
            empty.keyframe_insert(data_path="location", frame=frame_number)


def main() -> None:
    # # 2025-04-28
    trajectories_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\ferret_9C04_NoImplant_P35_E3\clips\2025_04_28.ferret_9C04_NoImplant_P35_E3.fr300-1200.ears_nose_eyes\output_data\dlc_body_rigid_3d_xyz_rigid_aligned.npy"

    trajectory_names = [
        "nose",
        "right_eye",
        "right_ear",
        "left_eye",
        "left_ear",
        "toy"
    ]

    # # # #2025-05-04
    # trajectories_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-05-04\ferret_9C04_NoImplant_P41_E9\clips\model_toy_clipped_5_4\output_data_iteration_4_4_28_calibration_aligned\dlc_body_rigid_3d_xyz.npy"
    # trajectory_names = [
    #     "right_ear",
    #     "left_ear",
    #     "toy",
    #     "nose",
    #     "left_eye",
    #     "right_eye",
    # ]

    # Validate paths
    validate_paths(
        trajectories_path=trajectories_path,
    )

    # Clear scene
    # clear_scene()

    print("Geometrically correct video planes setup complete!")

    trajectory_data = np.load(trajectories_path, allow_pickle=True)
    trajectory_data *= 0.001  # Convert mm to m
    number_of_frames = trajectory_data.shape[0]
    number_of_markers = trajectory_data.shape[1]
    # Set scene frame range for trajectories
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = number_of_frames - 1

    trajectories_by_name = {
        name: trajectory_data[:, index, :]
        for index, name in enumerate(trajectory_names)
    }

    # Create trajectory objects
    create_trajectory_objects(
        trajectories=trajectories_by_name, number_of_frames=number_of_frames
    )


main()
bpy.context.space_data.shading.type = "MATERIAL"
print("Done!")
