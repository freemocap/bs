import numpy as np
import cv2
import tomllib
from itertools import combinations

def load_camera_data(file_path):
    """Load camera data from a TOML file."""
    with open(file_path, "rb") as f:
        data = tomllib.load(f)
    
    # Extract camera positions and rotations
    cameras = {}
    for key, value in data.items():
        if key.startswith("cam_"):
            cameras[key] = {
                "name": value["name"],
                "translation": np.array(value["translation"]),
                "rvec": np.array(value["rotation"])
            }
    
    return cameras

def get_transformation_matrix(rvec, tvec):
    """Convert rotation vector and translation vector to 4x4 transformation matrix."""
    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = tvec
    
    return T

def calculate_relative_poses(cameras, reference_cam="cam_0"):
    """Calculate relative poses of all cameras with respect to a reference camera."""
    # Get reference camera transformation
    ref_cam = cameras[reference_cam]
    ref_T = get_transformation_matrix(ref_cam["rvec"], ref_cam["translation"])
    ref_T_inv = np.linalg.inv(ref_T)
    
    # Calculate relative poses
    relative_poses = {}
    for cam_id, cam_data in cameras.items():
        # Get this camera's transformation
        cam_T = get_transformation_matrix(cam_data["rvec"], cam_data["translation"])
        
        # Calculate relative transformation (reference to this camera)
        rel_T = np.matmul(ref_T_inv, cam_T)
        
        # Extract rotation and translation from transformation matrix
        rel_R = rel_T[:3, :3]
        rel_t = rel_T[:3, 3]
        
        # Convert rotation matrix back to rotation vector
        rel_rvec, _ = cv2.Rodrigues(rel_R)
        
        relative_poses[cam_id] = {
            "name": cam_data["name"],
            "rel_rvec": rel_rvec.flatten(),
            "rel_tvec": rel_t
        }
    
    return relative_poses

def main():
    # Load camera data from both files
    aligned_calibration_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\2025-04-28-calibration\2025-04-28-calibration_camera_calibration_aligned.toml"
    original_calibration_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\2025-04-28-calibration\2025-04-28-calibration_camera_calibration.toml"

    original_data = load_camera_data(original_calibration_path)
    aligned_data = load_camera_data(aligned_calibration_path)
    
    # Calculate relative poses with respect to cam_0
    original_relative = calculate_relative_poses(original_data)
    aligned_relative = calculate_relative_poses(aligned_data)
    
    # Print results
    print("Relative Camera Poses (with respect to cam_0)")
    print("=" * 80)
    print("Original Calibration:")
    for cam_id, pose in original_relative.items():
        if cam_id != "cam_0":
            print(f"{cam_id}: Rotation: {pose['rel_rvec']} Translation: {pose['rel_tvec']}")
    
    print("\nAligned Calibration:")
    for cam_id, pose in aligned_relative.items():
        if cam_id != "cam_0":
            print(f"{cam_id}: Rotation: {pose['rel_rvec']} Translation: {pose['rel_tvec']}")
    
    # Compare relative poses
    print("\nDifferences in Relative Poses:")
    for cam_id in original_relative.keys():
        if cam_id != "cam_0":
            orig = original_relative[cam_id]
            align = aligned_relative[cam_id]
            
            rot_diff = np.linalg.norm(orig['rel_rvec'] - align['rel_rvec'])
            trans_diff = np.linalg.norm(orig['rel_tvec'] - align['rel_tvec'])
            
            print(f"{cam_id}: Rotation diff: {rot_diff:.6f} rad, Translation diff: {trans_diff:.2f} mm")

if __name__ == "__main__":
    main()