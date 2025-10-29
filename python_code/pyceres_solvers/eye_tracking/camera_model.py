"""Pinhole camera model for eye tracking."""

import numpy as np
from pydantic import BaseModel, Field, computed_field
from numpydantic import NDArray, Shape
from typing import Any


class CameraIntrinsics(BaseModel):
    """Camera intrinsic parameters."""

    model_config = {"arbitrary_types_allowed": True}

    focal_length_mm: float = Field(gt=0)
    sensor_width_mm: float = Field(gt=0)
    sensor_height_mm: float = Field(gt=0)
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)

    @computed_field
    @property
    def pixel_size_x(self) -> float:
        """Pixel size in X direction (mm/pixel)."""
        return self.sensor_width_mm / self.image_width_px

    @computed_field
    @property
    def pixel_size_y(self) -> float:
        """Pixel size in Y direction (mm/pixel)."""
        return self.sensor_height_mm / self.image_height_px

    @computed_field
    @property
    def cx(self) -> float:
        """Principal point X coordinate (pixels)."""
        return self.image_width_px / 2.0

    @computed_field
    @property
    def cy(self) -> float:
        """Principal point Y coordinate (pixels)."""
        return self.image_height_px / 2.0

    @computed_field
    @property
    def fx(self) -> float:
        """Focal length in X direction (pixels)."""
        return self.focal_length_mm / self.pixel_size_x

    @computed_field
    @property
    def fy(self) -> float:
        """Focal length in Y direction (pixels)."""
        return self.focal_length_mm / self.pixel_size_y

    @classmethod
    def create_pupil_labs_camera(cls) -> "CameraIntrinsics":
        """Create Pupil Labs eye camera specifications."""
        return cls(
            focal_length_mm=1.7,
            sensor_width_mm=1.15,
            sensor_height_mm=1.15,
            image_width_px=400,
            image_height_px=400,
        )


# Pupil Labs eye camera specifications
PUPIL_LABS_CAMERA = CameraIntrinsics.create_pupil_labs_camera()


def project_point(
    *,
    point_3d: NDArray[Shape["*, 3"], np.floating[Any]],
    camera: CameraIntrinsics
) -> NDArray[Shape["*, 2"], np.floating[Any]]:
    """
    Project 3D point(s) to 2D image using pinhole model.

    Args:
        point_3d: (3,) or (N, 3) point(s) in camera frame (mm)
        camera: Camera intrinsics

    Returns:
        (2,) or (N, 2) pixel coordinates
    """
    point_3d = np.atleast_2d(point_3d)

    # Pinhole projection: [x/z, y/z]
    x_norm = point_3d[:, 0] / point_3d[:, 2]
    y_norm = point_3d[:, 1] / point_3d[:, 2]

    # Convert to pixels
    u = camera.fx * x_norm + camera.cx
    v = camera.fy * y_norm + camera.cy

    result = np.stack(arrays=[u, v], axis=1)
    return result[0] if len(point_3d) == 1 else result