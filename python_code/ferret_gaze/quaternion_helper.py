"""Unit quaternion class for 3D rotations."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Quaternion:
    """Unit quaternion for rotations. Convention: scalar-first [w, x, y, z]."""

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:
            raise ValueError("Cannot normalize zero quaternion")
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    @classmethod
    def from_rotation_matrix(cls, R: NDArray[np.float64]) -> "Quaternion":
        R = np.asarray(R, dtype=np.float64)
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return cls(w=w, x=x, y=y, z=z)

    def conjugate(self) -> "Quaternion":
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def inverse(self) -> "Quaternion":
        return self.conjugate()

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """
        Multiply two quaternions (Hamilton product). The result represents the composition of rotations.
        :param other:
        :return:
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            x=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )


    def rotate_vector(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(v, dtype=np.float64)
        u = np.array([self.x, self.y, self.z])
        uv = np.cross(u, v)
        uuv = np.cross(u, uv)
        return v + 2.0 * (self.w * uv + uuv)

    def to_axis_angle(self) -> tuple[NDArray[np.float64], float]:
        w_clamped = np.clip(self.w, -1.0, 1.0)
        angle = 2.0 * np.arccos(abs(w_clamped))
        sin_half = np.sqrt(1.0 - w_clamped**2)
        if sin_half < 1e-10:
            return np.array([1.0, 0.0, 0.0]), 0.0
        axis = np.array([self.x, self.y, self.z]) / sin_half
        if self.w < 0:
            axis = -axis
        return axis, angle

    def to_euler_xyz(self) -> tuple[float, float, float]:
        """Returns (roll, pitch, yaw) in radians.

        Convention: ZYX intrinsic (aerospace) / XYZ extrinsic.
        Rotation order: yaw(Z) → pitch(Y) → roll(X) in body frame.
        """
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def to_rotation_matrix(self) -> NDArray[np.float64]:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ])