import tomllib
import bpy
import mathutils
import numpy as np

# Load Camera Calibration Data
camera_calibration_toml_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\2025-04-28-calibration\2025-04-28_camera_calibration.toml"
# load camera calibration data
with open(camera_calibration_toml_path, "rb") as f:
    calibration_by_camera = tomllib.load(f)

# Convert rodriguez rotation to Euler angles
def rodriguez_to_euler(r):
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

# Create a rotation to flip the camera direction (180 degrees around Y-axis)
flip_euler = mathutils.Euler((0, np.radians(180), 0), 'XYZ')

for camera_id, camera_calibration in calibration_by_camera.items():
    if camera_id == "metadata":
        continue  # Skip metadata entry
    camera_name = camera_calibration["name"]
    camera_resolution = camera_calibration["size"]
    camera_matrix = camera_calibration["matrix"]
    camera_distortions = camera_calibration["distortions"]
    camera_euler = rodriguez_to_euler(camera_calibration["rotation"])
    # get focal length from camera matrix
    focal_length = camera_matrix[0][0]  # Assuming square pixels and fx = fy
    
    # Apply the flip to the camera euler angles
    # We need to convert to quaternion, multiply, then convert back to euler
    camera_quat = camera_euler.to_quaternion()
    flip_quat = flip_euler.to_quaternion()
    final_quat = camera_quat @ flip_quat
    final_euler = final_quat.to_euler('XYZ')
    
    camera_translation = [t * 0.001 for t in camera_calibration["translation"]]

    # Create a new camera object
    bpy.ops.object.camera_add(location=camera_translation)
    camera_object = bpy.context.object
    camera_object.name = f"Camera_{camera_name}"
    
    camera_object.scale = (.2, .2, .2)  # Scale the camera object for visibility
    
    # Set the rotation mode to Euler and apply the rotation
    camera_object.rotation_mode = 'XYZ'
    camera_object.rotation_euler = final_euler

        # Calculate focal length in mm (assuming 35mm sensor)
    focal_length_pixels = camera_matrix[0][0]  # Assuming square pixels and fx = fy
    sensor_width_mm = 11.26 # sensor size - https://www.baslerweb.com/en-us/shop/aca2040-90umnir/
    focal_length_mm = (focal_length_pixels / camera_resolution[0]) * sensor_width_mm
    camera_object.data.lens = focal_length_mm  # Set the focal length