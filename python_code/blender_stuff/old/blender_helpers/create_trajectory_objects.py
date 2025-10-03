import colorsys

import numpy as np
import pandas as pd

import bpy
import mathutils

from python_code.blender_stuff.blender_helpers.freemocap_csv_data_helper import filter_by_keypoint_df, \
    filter_by_trajectory_df


def load_trajectory_data_npy(trajectories_numpy_path: str) -> tuple[np.ndarray, int, int]:
    trajectory_data = np.load(trajectories_numpy_path, allow_pickle=True)
    trajectory_data *= 0.001  # Convert mm to m
    number_of_frames = trajectory_data.shape[0]
    number_of_markers = trajectory_data.shape[1]

    return trajectory_data, number_of_frames, number_of_markers


def load_keypoint_csv_as_xyz_arrays(filepath: str,scale_data:float=1, trajectory_type: str = 'rigid_3d_xyz') -> tuple[dict[str, np.ndarray], int, int]:
    """
    Load keypoint data from a CSV file and return a dictionary of numpy arrays
    where each array has shape (number_of_frames, 3) for xyz coordinates.

    Args:
        filepath: Path to the CSV file
        trajectory_type: Type of trajectory data to use (default: 'rigid_3d_xyz')

    Returns:
        Dictionary where keys are keypoint names and values are numpy arrays of shape (num_frames, 3)
        num_frames: Total number of frames in the data
        number_of_markers: Total number of unique keypoints in the data
    """
    # Load the data
    df = pd.read_csv(filepath)
    # scale the data
    df[['x', 'y', 'z']] *= scale_data

    # Filter by trajectory type
    # df = filter_by_trajectory_df(df, trajectory_type)

    # Get all unique keypoints and frames
    keypoints = df['keypoint'].unique()
    frames = df['frame'].unique()
    frames.sort()  # Ensure frames are in order
    number_of_frames = len(frames)
    number_of_keypoints = len(keypoints)
    result = {}

    # For each keypoint, create an array of shape (num_frames, 3)
    for keypoint in keypoints:
        keypoint_df = filter_by_keypoint_df(df, keypoint)

        # Initialize array with NaN values
        xyz_array = np.full((number_of_frames, 3), np.nan)

        # Fill in the values
        for _, row in keypoint_df.iterrows():
            frame_idx = np.where(frames == row['frame'])[0][0]
            xyz_array[frame_idx] = [row['x'], row['y'], row['z']]

        result[keypoint] = xyz_array

    return result, number_of_frames, number_of_keypoints

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
