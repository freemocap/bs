import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator


class Timeseries(BaseModel):
    """
    A single scalar quantity tracked over time.

    This is the most granular time-varying data: one value per timestamp.
    Examples: x-position, roll angle, speed.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    values: NDArray[np.float64]  # (N,)

    @model_validator(mode="after")
    def validate_lengths(self) -> "Timeseries":
        if self.timestamps.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"timestamps length {self.timestamps.shape[0]} != "
                f"values length {self.values.shape[0]}"
            )
        return self

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> float:
        return float(self.values[idx])

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def mean_dt(self) -> float:
        return self.duration / (self.n_frames - 1) if self.n_frames > 1 else 0.0

    def differentiate(self) -> "Timeseries":
        """Compute time derivative using central differences."""
        n = len(self.values)
        derivative = np.zeros(n, dtype=np.float64)

        # Use central differences for interior points, forward/backward differences for endpoints
        for i in range(n):
            if i == 0:
                dt = self.timestamps[1] - self.timestamps[0]
                derivative[i] = (self.values[1] - self.values[0]) / dt
            elif i == n - 1:
                dt = self.timestamps[i] - self.timestamps[i - 1]
                derivative[i] = (self.values[i] - self.values[i - 1]) / dt
            else:
                dt = self.timestamps[i + 1] - self.timestamps[i - 1]
                derivative[i] = (self.values[i + 1] - self.values[i - 1]) / dt

        return Timeseries(
            name=f"d({self.name})/dt",
            timestamps=self.timestamps,
            values=derivative,
        )

    def interpolate(self, target_timestamps: NDArray[np.float64]) -> "Timeseries":
        """Linearly interpolate to new timestamps."""
        interpolated = np.interp(target_timestamps, self.timestamps, self.values)
        return Timeseries(
            name=self.name,
            timestamps=target_timestamps,
            values=interpolated,
        )
