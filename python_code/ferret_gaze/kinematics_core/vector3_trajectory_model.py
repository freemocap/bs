import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from python_code.ferret_gaze.kinematics_core.timeseries_model import Timeseries


class Vector3Trajectory(BaseModel):
    """
    A 3D vector quantity tracked over time.

    Examples: position, velocity, angular velocity.
    Composed of three Timeseries (x, y, z components).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    values: NDArray[np.float64]  # (N, 3)

    @model_validator(mode="after")
    def validate_shape(self) -> "Vector3Trajectory":
        expected = (len(self.timestamps), 3)
        if self.values.shape != expected:
            raise ValueError(f"values shape {self.values.shape} != {expected}")
        return self

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> NDArray[np.float64]:
        return self.values[idx]

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def x(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.x", timestamps=self.timestamps, values=self.values[:, 0])

    @property
    def y(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.y", timestamps=self.timestamps, values=self.values[:, 1])

    @property
    def z(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.z", timestamps=self.timestamps, values=self.values[:, 2])

    @property
    def magnitude(self) -> Timeseries:
        """Compute magnitude (norm) at each timestamp."""
        mags = np.linalg.norm(self.values, axis=1)
        return Timeseries(name=f"|{self.name}|", timestamps=self.timestamps, values=mags)

    def differentiate(self) -> "Vector3Trajectory":
        """Compute time derivative of each component."""
        dx = self.x.differentiate()
        dy = self.y.differentiate()
        dz = self.z.differentiate()
        return Vector3Trajectory(
            name=f"d({self.name})/dt",
            timestamps=self.timestamps,
            values=np.column_stack([dx.values, dy.values, dz.values]),
        )
