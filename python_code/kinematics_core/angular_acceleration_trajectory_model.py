import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from python_code.kinematics_core.timeseries_model import Timeseries


class AngularAccelerationTrajectory(BaseModel):
    """
    Angular acceleration tracked over time, with both global and local representations.

    Units: rad/s²
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    global_xyz: NDArray[np.float64]  # (N, 3) in world frame
    local_xyz: NDArray[np.float64]  # (N, 3) in body frame

    roll_index: int = 0
    pitch_index: int = 1
    yaw_index: int = 2

    @model_validator(mode="after")
    def validate_shapes(self) -> "AngularAccelerationTrajectory":
        n = len(self.timestamps)
        if self.global_xyz.shape != (n, 3):
            raise ValueError(f"global_xyz shape {self.global_xyz.shape} != ({n}, 3)")
        if self.local_xyz.shape != (n, 3):
            raise ValueError(f"local_xyz shape {self.local_xyz.shape} != ({n}, 3)")
        return self

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def global_roll(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.global_roll",
            timestamps=self.timestamps,
            values=self.global_xyz[:, self.roll_index],
        )

    @property
    def global_pitch(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.global_pitch",
            timestamps=self.timestamps,
            values=self.global_xyz[:, self.pitch_index],
        )

    @property
    def global_yaw(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.global_yaw",
            timestamps=self.timestamps,
            values=self.global_xyz[:, self.yaw_index],
        )

    @property
    def local_roll(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.local_roll",
            timestamps=self.timestamps,
            values=self.local_xyz[:, self.roll_index],
        )

    @property
    def local_pitch(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.local_pitch",
            timestamps=self.timestamps,
            values=self.local_xyz[:, self.pitch_index],
        )

    @property
    def local_yaw(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.local_yaw",
            timestamps=self.timestamps,
            values=self.local_xyz[:, self.yaw_index],
        )

    @property
    def global_magnitude(self) -> Timeseries:
        mags = np.linalg.norm(self.global_xyz, axis=1)
        return Timeseries(
            name=f"|{self.name}|",
            timestamps=self.timestamps,
            values=mags,
        )

    @property
    def local_magnitude(self) -> Timeseries:
        mags = np.linalg.norm(self.local_xyz, axis=1)
        return Timeseries(
            name=f"|{self.name}_local|",
            timestamps=self.timestamps,
            values=mags,
        )