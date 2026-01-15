"""Pydantic model for rigid body reference geometry with coordinate frame definitions."""
import json
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from python_code.ferret_gaze.kinematics_core.quaternion_model import Quaternion


class AxisType(str, Enum):
    """Whether an axis definition is exact or approximate."""

    EXACT = "exact"
    APPROXIMATE = "approximate"


class AxisDefinition(BaseModel):
    """Definition of a coordinate axis direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    markers: list[str]  # Marker(s) defining direction (mean if multiple)
    type: AxisType  # Whether this axis is exact or approximate

    @field_validator("markers")
    @classmethod
    def markers_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("markers list cannot be empty")
        return v


class CoordinateFrameDefinition(BaseModel):
    """
    Definition of the body-fixed coordinate frame.

    Exactly TWO of x_axis, y_axis, z_axis must be defined.
    One must be 'exact' and one must be 'approximate'.
    The third axis is computed via cross product to define a right-handed coordinate system (X × Y = Z, Y × Z = X, Z × X = Y).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin_markers: list[str]  # Markers whose mean defines the origin
    x_axis: AxisDefinition | None = None
    y_axis: AxisDefinition | None = None
    z_axis: AxisDefinition | None = None

    @field_validator("origin_markers")
    @classmethod
    def origin_markers_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("origin_markers list cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_axis_definitions(self) -> "CoordinateFrameDefinition":
        """Validate that exactly 2 axes are defined, one exact and one approximate."""
        defined_axes: list[tuple[str, AxisDefinition]] = []
        for axis_name in ("x_axis", "y_axis", "z_axis"):
            axis_def = getattr(self, axis_name)
            if axis_def is not None:
                defined_axes.append((axis_name, axis_def))

        # Check exactly 2 defined
        if len(defined_axes) != 2:
            raise ValueError(
                f"Exactly 2 axes must be defined, got {len(defined_axes)}: "
                f"{[name for name, _ in defined_axes]}"
            )

        # Check one exact, one approximate
        types = [axis_def.type for _, axis_def in defined_axes]
        if AxisType.EXACT not in types:
            raise ValueError("One axis must be marked as 'exact'")
        if AxisType.APPROXIMATE not in types:
            raise ValueError("One axis must be marked as 'approximate'")

        return self

    def get_defined_axes(self) -> tuple[str, str]:
        """Return the names of the two defined axes (exact first, approximate second)."""
        exact_axis: str | None = None
        approx_axis: str | None = None
        for axis_name in ("x_axis", "y_axis", "z_axis"):
            axis_def = getattr(self, axis_name)
            if axis_def is not None and axis_def.type == AxisType.EXACT:
                exact_axis = axis_name
            elif axis_def is not None and axis_def.type == AxisType.APPROXIMATE:
                approx_axis = axis_name
        if exact_axis is None or approx_axis is None:
            raise ValueError("Could not find exact and approximate axes")
        return exact_axis, approx_axis

    def get_computed_axis(self) -> str:
        """Return the name of the axis that will be computed via cross product."""
        defined = {
            name
            for name in ("x_axis", "y_axis", "z_axis")
            if getattr(self, name) is not None
        }
        computed = {"x_axis", "y_axis", "z_axis"} - defined
        return computed.pop()


class MarkerPosition(BaseModel):
    """Position of a single marker in the reference frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float
    z: float

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


class ReferenceGeometry(BaseModel):
    """
    Complete reference geometry specification for a rigid body.

    Includes marker positions and coordinate frame definition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: Literal["mm", "m"]
    coordinate_frame: CoordinateFrameDefinition
    markers: dict[str, MarkerPosition]

    @property
    def marker_local_positions_array(self) -> NDArray[np.float64]:
        """Return marker positions as an (N, 3) array."""
        return np.array(
            [marker.to_array() for marker in self.markers.values()], dtype=np.float64
        )

    @model_validator(mode="after")
    def validate_marker_references(self) -> "ReferenceGeometry":
        """Validate that all referenced markers exist in the markers dict."""
        marker_names = set(self.markers.keys())

        # Check origin markers
        for marker in self.coordinate_frame.origin_markers:
            if marker not in marker_names:
                raise ValueError(
                    f"Origin marker '{marker}' not found in markers. "
                    f"Available: {sorted(marker_names)}"
                )

        # Check axis markers
        for axis_name in ("x_axis", "y_axis", "z_axis"):
            axis_def = getattr(self.coordinate_frame, axis_name)
            if axis_def is not None:
                for marker in axis_def.markers:
                    if marker not in marker_names:
                        raise ValueError(
                            f"Axis '{axis_name}' references marker '{marker}' "
                            f"not found in markers. Available: {sorted(marker_names)}"
                        )

        return self

    @classmethod
    def from_json_file(cls, path: Path | str) -> "ReferenceGeometry":
        """Load reference geometry from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json_file(self, path: Path | str) -> None:
        """Save reference geometry to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def get_marker_positions(self) -> dict[str, NDArray[np.float64]]:
        """Return marker positions as numpy arrays."""
        return {name: pos.to_array() for name, pos in self.markers.items()}

    def get_marker_array(self) -> tuple[list[str], NDArray[np.float64]]:
        """Return marker names and positions as (names, (N, 3) array)."""
        names = list(self.markers.keys())
        positions = np.array([self.markers[name].to_array() for name in names])
        return names, positions

    def compute_basis_vectors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the orthonormal basis vectors and origin point.

        The basis is constructed as follows:
        1. The "exact" axis is computed directly from marker positions
        2. The "approximate" axis is orthogonalized via Gram-Schmidt
        3. The third axis is computed via cross product for right-handedness

        Returns:
            basis_vectors: (3, 3) array where rows are [x_axis, y_axis, z_axis]
            origin_point: (3,) origin position
        """
        marker_positions = self.get_marker_positions()

        # Compute origin
        origin_points = [marker_positions[m] for m in self.coordinate_frame.origin_markers]
        origin = np.mean(origin_points, axis=0)

        # Get exact and approximate axis definitions
        exact_axis_name, approx_axis_name = self.coordinate_frame.get_defined_axes()
        computed_axis_name = self.coordinate_frame.get_computed_axis()

        exact_def = getattr(self.coordinate_frame, exact_axis_name)
        approx_def = getattr(self.coordinate_frame, approx_axis_name)

        # Compute exact axis direction (mean of markers if multiple)
        exact_points = [marker_positions[m] for m in exact_def.markers]
        exact_target = np.mean(exact_points, axis=0)
        exact_vec = exact_target - origin
        exact_norm = np.linalg.norm(exact_vec)
        if exact_norm < 1e-10:
            raise ValueError(
                f"Exact axis '{exact_axis_name}' has near-zero length. "
                f"Check that markers are not at the origin."
            )
        exact_vec = exact_vec / exact_norm

        # Compute approximate axis direction, then orthogonalize
        approx_points = [marker_positions[m] for m in approx_def.markers]
        approx_target = np.mean(approx_points, axis=0)
        approx_vec = approx_target - origin
        # Gram-Schmidt: remove component along exact axis
        approx_vec = approx_vec - np.dot(approx_vec, exact_vec) * exact_vec
        approx_norm = np.linalg.norm(approx_vec)
        if approx_norm < 1e-10:
            raise ValueError(
                f"Approximate axis '{approx_axis_name}' is parallel to exact axis. "
                f"Choose markers that define a different direction."
            )
        approx_vec = approx_vec / approx_norm

        # Compute third axis via cross product
        # Need to determine the correct cross product order for right-handed system
        computed_vec = _compute_third_axis(
            exact_axis_name=exact_axis_name,
            approx_axis_name=approx_axis_name,
            computed_axis_name=computed_axis_name,
            exact_vec=exact_vec,
            approx_vec=approx_vec,
        )

        # Assemble basis matrix (rows = basis vectors)
        # The vectors are named by HOW they were computed (exact/approx/computed),
        # but we assign them to the correct COORDINATE axis (x/y/z) based on the config
        basis = np.zeros((3, 3), dtype=np.float64)
        axis_index = {"x_axis": 0, "y_axis": 1, "z_axis": 2}

        basis[axis_index[exact_axis_name]] = exact_vec
        basis[axis_index[approx_axis_name]] = approx_vec
        basis[axis_index[computed_axis_name]] = computed_vec

        return basis, origin


def _compute_third_axis(
    exact_axis_name: str,
    approx_axis_name: str,
    computed_axis_name: str,
    exact_vec: NDArray[np.float64],
    approx_vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the third axis via cross product, ensuring right-handed coordinate system.

    For a right-handed system: X × Y = Z, Y × Z = X, Z × X = Y

    The algorithm determines which cross product order to use based on the
    cyclic ordering of the defined axes.
    """
    # Map axis names to indices for cyclic ordering
    axis_order = {"x_axis": 0, "y_axis": 1, "z_axis": 2}

    exact_idx = axis_order[exact_axis_name]
    approx_idx = axis_order[approx_axis_name]

    # Determine cross product based on cyclic ordering
    # Right-hand rule: (i) × (i+1) = (i+2) in cyclic order
    # So we need: cross(a, b) where (a_idx + 1) % 3 == b_idx gives positive c
    if (exact_idx + 1) % 3 == approx_idx:
        # exact × approx = computed
        computed_vec = np.cross(exact_vec, approx_vec)
    elif (approx_idx + 1) % 3 == exact_idx:
        # approx × exact = computed
        computed_vec = np.cross(approx_vec, exact_vec)
    else:
        # The two defined axes are not adjacent in cyclic order
        # This means computed is between them
        # e.g., x and z defined, y computed: z × x = y
        if (exact_idx + 2) % 3 == approx_idx:
            computed_vec = np.cross(approx_vec, exact_vec)
        else:
            computed_vec = np.cross(exact_vec, approx_vec)

    norm = np.linalg.norm(computed_vec)
    if norm < 1e-10:
        raise ValueError("Cross product resulted in zero vector - axes may be parallel")
    return computed_vec / norm


class StaticPose(BaseModel):
    """
    A rigid body's pose at a specific instant in time (position + orientation only).

    This is a lightweight pose class for geometric computations without velocity info.
    For full kinematic state including velocities, use RigidBodyPose from kinematics_core.

    Combines reference geometry with a specific position and orientation
    to represent the spatial configuration of a rigid body at one timestamp.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    reference_geometry: ReferenceGeometry
    timestamp: float
    position_xyz_mm: NDArray[np.float64]  # (3,) world-frame position of origin
    orientation: Quaternion  # orientation quaternion

    @field_validator("position_xyz_mm")
    @classmethod
    def validate_position_shape(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if v.shape != (3,):
            raise ValueError(f"position_xyz_mm must have shape (3,), got {v.shape}")
        return v

    @property
    def world_basis_vectors(self) -> NDArray[np.float64]:
        """
        Get the basis vectors transformed to world frame.

        Returns:
            (3, 3) array where rows are [x_axis, y_axis, z_axis] in world frame
        """
        return self.orientation.to_rotation_matrix()

    @property
    def world_keypoints(self) -> dict[str, NDArray[np.float64]]:
        """
        Compute world-frame positions of all markers.

        Returns:
            Dict mapping marker name to (3,) world position
        """
        marker_positions = self.reference_geometry.get_marker_positions()
        result: dict[str, NDArray[np.float64]] = {}

        for name, local_pos in marker_positions.items():
            world_pos = self.position_xyz_mm + self.orientation.rotate_vector(local_pos)
            result[name] = world_pos

        return result

    def get_world_keypoint(self, marker_name: str) -> NDArray[np.float64]:
        """Get world-frame position of a specific marker."""
        if marker_name not in self.reference_geometry.markers:
            raise KeyError(
                f"Marker '{marker_name}' not found. "
                f"Available: {sorted(self.reference_geometry.markers.keys())}"
            )
        local_pos = self.reference_geometry.markers[marker_name].to_array()
        return self.position_xyz_mm + self.orientation.rotate_vector(local_pos)

    @property
    def world_origin(self) -> NDArray[np.float64]:
        """World-frame position of the body origin."""
        return self.position_xyz_mm.copy()

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Get the 3x3 rotation matrix."""
        return self.orientation.to_rotation_matrix()

    @property
    def homogeneous_transform(self) -> NDArray[np.float64]:
        """
        Get the 4x4 homogeneous transformation matrix.

        Transforms points from body frame to world frame:
            world_point = T @ [local_point; 1]
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position_xyz_mm
        return T

    def euler_angles_xyz(self) -> tuple[float, float, float]:
        """Get (roll, pitch, yaw) in radians."""
        return self.orientation.to_euler_xyz()

    @classmethod
    def from_arrays(
        cls,
        reference_geometry: ReferenceGeometry,
        timestamp: float,
        position: NDArray[np.float64],
        quaternion_wxyz: NDArray[np.float64],
    ) -> "StaticPose":
        """
        Construct from numpy arrays.

        Args:
            reference_geometry: The reference geometry
            timestamp: Time in seconds
            position: (3,) position array
            quaternion_wxyz: (4,) quaternion as [w, x, y, z]
        """
        quat = Quaternion(
            w=float(quaternion_wxyz[0]),
            x=float(quaternion_wxyz[1]),
            y=float(quaternion_wxyz[2]),
            z=float(quaternion_wxyz[3]),
        )
        return cls(
            reference_geometry=reference_geometry,
            timestamp=timestamp,
            position_xyz_mm=position.astype(np.float64),
            orientation=quat,
        )


# Example JSON structure for reference
EXAMPLE_JSON = """
{
  "units": "mm",
  "coordinate_frame": {
    "origin_markers": ["left_eye", "right_eye"],
    "x_axis": {
      "markers": ["nose"],
      "type": "exact"
    },
    "y_axis": {
      "markers": ["left_eye"],
      "type": "approximate"
    }
  },
  "markers": {
    "nose": {"x": 18.125, "y": 0.0, "z": 0.0},
    "left_eye": {"x": -0.178, "y": 11.866, "z": 0.0},
    "right_eye": {"x": 0.178, "y": -11.866, "z": 0.0}
  }
}
"""
