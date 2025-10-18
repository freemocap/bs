"""Simple ellipse fitting for pupil outlines using OpenCV."""

import numpy as np
import cv2
from pydantic import BaseModel


class EllipseParams(BaseModel):
    """Parameters defining an ellipse."""
    center_x: float
    center_y: float
    semi_major: float  # a (half of major axis)
    semi_minor: float  # b (half of minor axis)
    rotation: float  # radians

    def to_array(self) -> np.ndarray:
        """Convert to parameter array [cx, cy, a, b, theta]."""
        return np.array([
            self.center_x,
            self.center_y,
            self.semi_major,
            self.semi_minor,
            self.rotation
        ])

    @classmethod
    def from_array(cls, *, arr: np.ndarray) -> "EllipseParams":
        """Create from parameter array."""
        return cls(
            center_x=float(arr[0]),
            center_y=float(arr[1]),
            semi_major=float(arr[2]),
            semi_minor=float(arr[3]),
            rotation=float(arr[4])
        )

    def generate_points(self, *, n_points: int = 100) -> np.ndarray:
        """Generate points along the ellipse for visualization.

        Returns:
            (n_points, 2) array of x,y coordinates
        """
        theta = np.linspace(start=0, stop=2*np.pi, num=n_points)

        # Parametric ellipse in local coordinates
        x_local = self.semi_major * np.cos(theta)
        y_local = self.semi_minor * np.sin(theta)

        # Rotate and translate to world coordinates
        cos_t = np.cos(self.rotation)
        sin_t = np.sin(self.rotation)

        x = self.center_x + x_local * cos_t - y_local * sin_t
        y = self.center_y + x_local * sin_t + y_local * cos_t

        return np.column_stack([x, y])


def fit_ellipse_to_points(*, points: np.ndarray) -> EllipseParams:
    """Fit an ellipse to points using OpenCV.

    Args:
        points: (N, 2) array of x,y coordinates

    Returns:
        Fitted ellipse parameters

    Raises:
        ValueError: If fewer than 5 valid points
    """
    # Filter out NaN points
    valid_mask = ~np.isnan(points).any(axis=1)
    valid_points = points[valid_mask]

    if len(valid_points) < 5:
        raise ValueError(f"Need at least 5 valid points for ellipse fitting, got {len(valid_points)}")

    # Fit ellipse using OpenCV
    ellipse = cv2.fitEllipse(points=valid_points.astype(np.float32))
    (cx, cy), (width, height), angle = ellipse

    # OpenCV returns:
    # - (cx, cy): center
    # - (width, height): FULL axes lengths (we need semi-axes)
    # - angle: rotation of the WIDTH axis in degrees (0-360)

    # Important: OpenCV's angle is for the width axis, not necessarily the major axis!
    # If width < height, the major axis is perpendicular to the angle
    if width > height:
        # Width is the major axis
        semi_major = float(width / 2)
        semi_minor = float(height / 2)
        rotation = float(np.deg2rad(angle))
    else:
        # Height is the major axis, so rotate by 90 degrees
        semi_major = float(height / 2)
        semi_minor = float(width / 2)
        rotation = float(np.deg2rad(angle + 90))

    return EllipseParams(
        center_x=float(cx),
        center_y=float(cy),
        semi_major=semi_major,
        semi_minor=semi_minor,
        rotation=rotation
    )