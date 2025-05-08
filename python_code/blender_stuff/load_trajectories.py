import tomllib
import bpy
import mathutils
import numpy as np

trajectory_data = np.load(r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\2025-04-28-calibration\output_data\charuco_3d_xyz.npy")

# Load Trajectory Data
trajectory_data *= 0.001  
number_of_frames = trajectory_data.shape[0]
number_of_markers = trajectory_data.shape[1]

# set scene start and end frames
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = number_of_frames - 1
# set the current frame to 0

# load the trajectory data into a dictionary
# where the keys are the marker indices and the values are lists of 3D coordinates
trajectories = {
    idx: trajectory_data[:, idx, :].tolist()
    for idx in range(number_of_markers)
}

# determine % isnan for each trajectory
nan_percentages = {
    idx: np.isnan(trajectory_data[:, idx, :]).mean() * 100
    for idx in range(number_of_markers)
}
print(f"NaN percentages: {nan_percentages}")

# create empties for each marker
print(f"Creating {number_of_markers} empties, each with {number_of_frames} frames...")
for idx in range(number_of_markers):
    print(f"Creating empty for marker {idx}...")
    empty = bpy.data.objects.new(f"Marker_{idx}_empty", None)
    bpy.context.collection.objects.link(empty)
    empty.empty_display_size = 0.005
    if idx == 0:
        empty.empty_display_type = 'ARROWS'
        empty.empty_display_size = 0.02
    # else:
    #     empty.empty_display_type = 'SPHERE'
    # create a small sphere and set a copy-location constraint to the empty
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=(0, 0, 0))
    sphere = bpy.context.object
    sphere.name = f"Marker_{idx}_sphere"
    sphere.parent = empty

    for frame in range(number_of_frames):
        if np.isnan(trajectories[idx][frame]).any():
            continue
        empty.location = mathutils.Vector(trajectories[idx][frame]) #convert mm to m 
        empty.keyframe_insert(data_path="location", frame=frame)
    

