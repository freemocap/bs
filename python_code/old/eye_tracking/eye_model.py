"""Eyeball geometry model for 3D eye tracking."""

import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class EyeballModel:
    """Model of the eyeball as a sphere with a pupil."""
    
    radius_mm: float = 12.0
    """Eyeball radius in millimeters (typical human eye ~11-12mm)"""
    
    pupil_radius_mm: float = 2.0
    """Pupil radius in millimeters (typical 2-4mm)"""


def create_pupil_circle_3d(
    *,
    n_points: int = 8,
    radius_mm: float = 2.0
) -> np.ndarray:
    """
    Create 3D points on a circle representing the pupil.
    
    The circle lies in the XY plane at Z=0.
    
    Args:
        n_points: Number of points on the circle
        radius_mm: Pupil radius in millimeters
        
    Returns:
        (n_points, 3) 3D coordinates of circle points
    """
    angles = np.linspace(start=0, stop=2*np.pi, num=n_points, endpoint=False)
    
    x = radius_mm * np.cos(angles)
    y = radius_mm * np.sin(angles)
    z = np.zeros(shape=n_points)
    
    return np.stack(arrays=[x, y, z], axis=1)


def transform_pupil_to_eyeball(
    *,
    pupil_points_local: np.ndarray,
    eyeball_center: np.ndarray,
    eyeball_rotation: np.ndarray,
    eyeball_radius: float
) -> np.ndarray:
    """
    Transform pupil circle from local coordinates to eyeball surface.
    
    Steps:
    1. Pupil starts as circle in XY plane at origin
    2. Translate forward along Z by eyeball_radius (pupil on sphere surface)
    3. Rotate by eyeball_rotation (eye gaze direction)
    4. Translate to eyeball_center
    
    Args:
        pupil_points_local: (N, 3) pupil points in local coordinates
        eyeball_center: (3,) eyeball center position
        eyeball_rotation: (3, 3) rotation matrix
        eyeball_radius: Eyeball radius in mm
        
    Returns:
        (N, 3) transformed pupil points in world coordinates
    """
    # Translate pupil to sphere surface (forward along +Z)
    pupil_on_surface = pupil_points_local.copy()
    pupil_on_surface[:, 2] += eyeball_radius
    
    # Rotate according to eye orientation
    pupil_rotated = (eyeball_rotation @ pupil_on_surface.T).T
    
    # Translate to eyeball center
    pupil_world = pupil_rotated + eyeball_center
    
    return pupil_world


def fit_ellipse_to_points(
    *,
    points: np.ndarray
) -> dict[str, float]:
    """
    Fit ellipse to 2D points using least squares.
    
    Ellipse equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    
    Args:
        points: (N, 2) 2D points
        
    Returns:
        Dictionary with ellipse parameters:
        - center: (cx, cy)
        - axes: (major_axis, minor_axis)
        - angle: rotation angle in radians
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Build design matrix
    D = np.column_stack(arrays=[
        x*x,
        x*y,
        y*y,
        x,
        y,
        np.ones(shape=len(x))
    ])
    
    # Solve: D @ [A, B, C, D, E, F] = 0, with constraint 4AC - B^2 = 1
    S = D.T @ D
    C = np.zeros(shape=(6, 6))
    C[0, 2] = 2
    C[2, 0] = 2
    C[1, 1] = -1
    
    # Generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S) @ C)
    
    # Find positive eigenvalue
    pos_idx = np.where(eigvals > 0)[0][0]
    coeffs = eigvecs[:, pos_idx]
    
    A, B, C, D, E, F = coeffs
    
    # Compute ellipse parameters
    denom = B**2 - 4*A*C
    cx = (2*C*D - B*E) / denom
    cy = (2*A*E - B*D) / denom
    
    num = 2 * (A*E**2 + C*D**2 - B*D*E + denom*F) * (A + C + np.sqrt((A-C)**2 + B**2))
    den1 = denom * (np.sqrt((A-C)**2 + B**2) - (A+C))
    den2 = denom * (-np.sqrt((A-C)**2 + B**2) - (A+C))
    
    major_axis = np.sqrt(num / den1) if den1 != 0 else 0
    minor_axis = np.sqrt(num / den2) if den2 != 0 else 0
    
    angle = 0.5 * np.arctan2(B, A - C) if A != C else 0
    
    return {
        "center": (cx, cy),
        "axes": (major_axis, minor_axis),
        "angle": angle
    }


def estimate_gaze_from_ellipse(
    *,
    ellipse_center: tuple[float, float],
    camera_center: tuple[float, float],
    focal_length_px: float
) -> np.ndarray:
    """
    Rough estimate of gaze direction from ellipse center displacement.
    
    Args:
        ellipse_center: (cx, cy) ellipse center in pixels
        camera_center: (cx, cy) camera principal point in pixels
        focal_length_px: Focal length in pixels
        
    Returns:
        (3,) approximate gaze direction vector (normalized)
    """
    dx = ellipse_center[0] - camera_center[0]
    dy = ellipse_center[1] - camera_center[1]
    
    # Convert to normalized coordinates
    x = dx / focal_length_px
    y = dy / focal_length_px
    
    # Gaze direction (approximate)
    gaze = np.array([x, y, 1.0])
    gaze = gaze / np.linalg.norm(gaze)
    
    return gaze
