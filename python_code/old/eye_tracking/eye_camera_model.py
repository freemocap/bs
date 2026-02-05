"""Camera model for eye tracking with pinhole projection."""

import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    
    focal_length_mm: float
    """Focal length in millimeters"""
    
    sensor_width_mm: float
    """Sensor width in millimeters"""
    
    sensor_height_mm: float
    """Sensor height in millimeters"""
    
    image_width_px: int
    """Image width in pixels"""
    
    image_height_px: int
    """Image height in pixels"""
    
    def __post_init__(self) -> None:
        """Compute derived parameters."""
        # Pixel size in mm
        self.pixel_size_x = self.sensor_width_mm / self.image_width_px
        self.pixel_size_y = self.sensor_height_mm / self.image_height_px
        
        # Principal point (image center)
        self.cx = self.image_width_px / 2.0
        self.cy = self.image_height_px / 2.0
        
        # Focal length in pixels
        self.fx = self.focal_length_mm / self.pixel_size_x
        self.fy = self.focal_length_mm / self.pixel_size_y
    
    @classmethod
    def from_eye_camera_spec(cls) -> "CameraIntrinsics":
        """
        Create camera intrinsics for the eye camera.
        
        Specs:
        - Image sensor: 1.15mm x 1.15mm at 192x192 pixels
        - Focal length: 1.7mm
        """
        return cls(
            focal_length_mm=1.7,
            sensor_width_mm=1.15,
            sensor_height_mm=1.15,
            image_width_px=192,
            image_height_px=192
        )


def project_point_to_image(
    *,
    point_3d: np.ndarray,
    camera: CameraIntrinsics
) -> np.ndarray:
    """
    Project 3D point to 2D image using pinhole camera model.
    
    Args:
        point_3d: (3,) or (N, 3) 3D point(s) in camera frame
        camera: Camera intrinsics
        
    Returns:
        (2,) or (N, 2) image coordinates in pixels
    """
    point_3d = np.atleast_2d(point_3d)
    
    # Pinhole projection: [x/z, y/z]
    x_proj = point_3d[:, 0] / point_3d[:, 2]
    y_proj = point_3d[:, 1] / point_3d[:, 2]
    
    # Convert to pixel coordinates
    u = camera.fx * x_proj + camera.cx
    v = camera.fy * y_proj + camera.cy
    
    result = np.stack(arrays=[u, v], axis=1)
    
    return result[0] if len(point_3d) == 1 else result


def unproject_pixel_to_ray(
    *,
    pixel: np.ndarray,
    camera: CameraIntrinsics
) -> np.ndarray:
    """
    Unproject 2D pixel to 3D ray direction.
    
    Args:
        pixel: (2,) or (N, 2) pixel coordinates
        camera: Camera intrinsics
        
    Returns:
        (3,) or (N, 3) normalized ray direction
    """
    pixel = np.atleast_2d(pixel)
    
    # Convert to normalized coordinates
    x = (pixel[:, 0] - camera.cx) / camera.fx
    y = (pixel[:, 1] - camera.cy) / camera.fy
    
    # Ray direction (normalized)
    rays = np.stack(arrays=[x, y, np.ones(shape=len(pixel))], axis=1)
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    
    return rays[0] if len(pixel) == 1 else rays


def compute_reprojection_error(
    *,
    points_3d: np.ndarray,
    observed_pixels: np.ndarray,
    camera: CameraIntrinsics
) -> np.ndarray:
    """
    Compute reprojection error in pixels.
    
    Args:
        points_3d: (N, 3) 3D points
        observed_pixels: (N, 2) observed pixel coordinates
        camera: Camera intrinsics
        
    Returns:
        (N,) reprojection errors in pixels
    """
    projected = project_point_to_image(point_3d=points_3d, camera=camera)
    errors = np.linalg.norm(projected - observed_pixels, axis=1)
    return errors
