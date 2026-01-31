from typing import TYPE_CHECKING

import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from python_code.kinematics_core.quaternion_model import Quaternion
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry


if TYPE_CHECKING:
    from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics

class RigidBodyState(BaseModel):
    """
    Complete state of a rigid body at a single instant in time.

    This is a "horizontal slice" of the kinematics data - one observation.
    Includes position, velocity, orientation, and angular velocity.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    reference_geometry: ReferenceGeometry
    timestamp: float

    # Position and velocity of the origin
    position: NDArray[np.float64]  # (3,) in mm
    velocity: NDArray[np.float64]  # (3,) in mm/s
    acceleration: NDArray[np.float64]  # (3,) in mm/s²

    # Orientation and angular velocity
    orientation: Quaternion
    angular_velocity_global: NDArray[np.float64]  # (3,) rad/s in world frame
    angular_velocity_local: NDArray[np.float64]  # (3,) rad/s in body frame

    angular_acceleration_global: NDArray[np.float64] | None = None  # (3,) rad/s² in world frame
    angular_acceleration_local: NDArray[np.float64] | None = None  # (3,) rad/s² in body frame



    @classmethod
    def from_kinematics_and_frame_number(cls,
        kin: "RigidBodyKinematics",
        frame_number: int,
    ) -> "RigidBodyState":
        """Create RigidBodyState from kinematics data."""
        return cls(
            reference_geometry=kin.reference_geometry,
            timestamp=kin.timestamps[frame_number],
            position=kin.position_xyz[frame_number],
            velocity=kin.velocity_xyz[frame_number],
            acceleration=kin.acceleration_xyz[frame_number],
            orientation=kin.get_quaternion(frame_number),
            angular_velocity_global=kin.angular_velocity_global[frame_number],
            angular_velocity_local=kin.angular_velocity_local[frame_number],
            angular_acceleration_global=kin.angular_acceleration_global[frame_number],
            angular_acceleration_local=kin.angular_acceleration_local[frame_number],
        )


    @field_validator("position",
                     "velocity",
                     "angular_velocity_global",
                     "angular_velocity_local")
    @classmethod
    def validate_vec3_shape(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if v.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {v.shape}")
        return v

    @property
    def basis_vectors(self) -> NDArray[np.float64]:
        """(3, 3) rotation matrix. Columns are body-frame basis vectors in world frame."""
        return self.orientation.to_rotation_matrix()

    @property
    def basis_x(self) -> NDArray[np.float64]:
        """Body X-axis direction in world frame."""
        return self.basis_vectors[:, 0]

    @property
    def basis_y(self) -> NDArray[np.float64]:
        """Body Y-axis direction in world frame."""
        return self.basis_vectors[:, 1]

    @property
    def basis_z(self) -> NDArray[np.float64]:
        """Body Z-axis direction in world frame."""
        return self.basis_vectors[:, 2]

    @property
    def keypoints(self) -> dict[str, NDArray[np.float64]]:
        """World-frame positions of all keypoints."""
        keypoint_positions = self.reference_geometry.get_keypoint_positions()
        return {
            name: self.position + self.orientation.rotate_vector(local_pos)
            for name, local_pos in keypoint_positions.items()
        }

    def get_keypoint(self, name: str) -> NDArray[np.float64]:
        """Get world-frame position of a specific keypoint."""
        if name not in self.reference_geometry.keypoints:
            raise KeyError(
                f"Keypoint '{name}' not found. "
                f"Available: {sorted(self.reference_geometry.keypoints.keys())}"
            )
        local_pos = self.reference_geometry.keypoints[name].to_array()
        return self.position + self.orientation.rotate_vector(local_pos)

    @property
    def speed(self) -> float:
        """Linear speed (magnitude of velocity)."""
        return float(np.linalg.norm(self.velocity))

    @property
    def angular_speed(self) -> float:
        """Angular speed (magnitude of angular velocity)."""
        return float(np.linalg.norm(self.angular_velocity_global))

    @property
    def euler_angles(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) in radians."""
        return self.orientation.to_euler_xyz()

    @property
    def homogeneous_transform(self) -> NDArray[np.float64]:
        """4x4 transformation matrix from body to world frame."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.basis_vectors
        T[:3, 3] = self.position
        return T
