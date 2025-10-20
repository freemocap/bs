from typing import Any, Self

import numpy as np
from pydantic import Field, model_validator
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from python_code.eye_analysis.csv_io import ABaseModel, logger


class Trajectory2D(ABaseModel):
    """A 2D trajectory (x, y positions over time).

    Attributes:
        name: Identifier for this trajectory
        data: (n_frames, 2) x,y positions over time
        confidence: Optional (n_frames,) confidence scores [0-1]
        metadata: Optional additional data
    """

    name: str
    data: np.ndarray  # shape: (n_frames, 2)
    framerate: float
    confidence: np.ndarray | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate trajectory data."""
        if self.data.ndim != 2 or self.data.shape[1] != 2:
            raise ValueError(f"Data must be (n_frames, 2), got shape {self.data.shape}")

        if self.confidence is not None:
            if len(self.confidence) != len(self.data):
                raise ValueError(
                    f"Confidence length {len(self.confidence)} != data length {len(self.data)}"
                )

        return self

    @property
    def n_frames(self) -> int:
        """Number of frames in trajectory."""
        return len(self.data)

    @property
    def x(self) -> np.ndarray:
        """X coordinates (n_frames,)."""
        return self.data[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Y coordinates (n_frames,)."""
        return self.data[:, 1]

    def is_valid(self, *, min_confidence: float | None = None) -> np.ndarray:
        """Get mask of valid frames."""
        not_nan = ~np.isnan(self.data[:, 0])

        if self.confidence is None or min_confidence is None:
            return not_nan

        above_threshold = self.confidence >= min_confidence
        return above_threshold & not_nan

    def interpolate_missing(self, *, method: str = "linear") -> "Trajectory2D":
        """Interpolate missing (NaN) values."""
        data_interp = self.data.copy()

        for axis in range(2):
            values = self.data[:, axis]
            valid_mask = ~np.isnan(values)

            if np.sum(valid_mask) < 2:
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_values = values[valid_mask]

            interp_func = interp1d(
                valid_indices,
                valid_values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate"
            )

            missing_mask = np.isnan(values)
            if np.any(missing_mask):
                missing_indices = np.where(missing_mask)[0]
                data_interp[missing_indices, axis] = interp_func(missing_indices)

        return Trajectory2D(
            name=self.name,
            data=data_interp,
            framerate=self.framerate,
            confidence=self.confidence,
            metadata=self.metadata
        )

    def apply_butterworth_filter(
        self,
        *,
        cutoff_freq: float = 6.0,
        order: int = 4
    ) -> "Trajectory2D":
        """Apply Butterworth low-pass filter to trajectory."""
        nyquist = self.framerate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False)

        filtered_data = self.data.copy()

        for axis in range(2):
            values = self.data[:, axis]
            valid_mask = ~np.isnan(values)
            n_valid = np.sum(valid_mask)

            if n_valid > 2 * order:
                if not np.any(np.isnan(values)):
                    filtered_data[:, axis] = filtfilt(b=b, a=a, x=values)
                else:
                    logger.warning(f"Cannot filter '{self.name}' axis {axis}: contains NaN")

        return Trajectory2D(
            name=self.name,
            data=filtered_data,
            confidence=self.confidence,
            framerate=self.framerate,
            metadata={**self.metadata, 'filtered': True}
        )

    def create_cleaned(
        self,
        *,
        interpolation_method: str = "linear",
        butterworth_cutoff: float = 6.0,
        butterworth_order: int = 4
    ) -> "Trajectory2D":
        """Create cleaned version: interpolate + Butterworth filter."""
        interpolated = self.interpolate_missing(method=interpolation_method)
        cleaned = interpolated.apply_butterworth_filter(
            cutoff_freq=butterworth_cutoff,
            order=butterworth_order
        )
        return cleaned


class TrajectoryPair(ABaseModel):
    """Holds both raw and cleaned versions of a trajectory.

    Attributes:
        raw: Raw trajectory data
        cleaned: Cleaned trajectory data (interpolated + filtered)
    """

    raw: Trajectory2D
    cleaned: Trajectory2D

    @property
    def name(self) -> str:
        """Trajectory name."""
        return self.raw.name

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.raw.n_frames


class TrajectoryDataset(ABaseModel):
    """Collection of 2D trajectories with both raw and cleaned versions.

    Access patterns:
        dataset.raw['p1']  # Raw Trajectory2D for p1
        dataset.cleaned['p1']  # Cleaned Trajectory2D for p1
        dataset.pairs['p1']  # TrajectoryPair with both versions
        dataset.pairs['p1'].raw  # Raw version
        dataset.pairs['p1'].cleaned  # Cleaned version

    Attributes:
        pairs: Dictionary mapping marker names to TrajectoryPair instances
        frame_indices: Frame numbers (may not start at 0)
        metadata: Optional dataset-level metadata
    """

    pairs: dict[str, TrajectoryPair]
    frame_indices: np.ndarray
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate dataset."""
        if len(self.pairs) == 0:
            raise ValueError("Dataset must contain at least one trajectory")

        n_frames_list = [pair.n_frames for pair in self.pairs.values()]
        if len(set(n_frames_list)) > 1:
            raise ValueError(f"All trajectories must have same length, got {n_frames_list}")

        if len(self.frame_indices) != n_frames_list[0]:
            raise ValueError(
                f"Frame indices length {len(self.frame_indices)} != trajectory length {n_frames_list[0]}"
            )

        return self

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.frame_indices)

    @property
    def marker_names(self) -> list[str]:
        """Get list of marker names."""
        return list(self.pairs.keys())

    @property
    def n_markers(self) -> int:
        """Number of markers."""
        return len(self.pairs)

    @property
    def raw(self) -> dict[str, Trajectory2D]:
        """Access raw trajectories: dataset.raw['marker_name']"""
        return {name: pair.raw for name, pair in self.pairs.items()}

    @property
    def cleaned(self) -> dict[str, Trajectory2D]:
        """Access cleaned trajectories: dataset.cleaned['marker_name']"""
        return {name: pair.cleaned for name, pair in self.pairs.items()}

    def to_array(self, *, marker_names: list[str] | None = None, use_cleaned: bool = False) -> np.ndarray:
        """Convert to numpy array.

        Args:
            marker_names: Optional list of markers to include (default: all)
            use_cleaned: If True, use cleaned data; otherwise use raw

        Returns:
            (n_frames, n_markers, 2) array of x,y positions
        """
        if marker_names is None:
            marker_names = self.marker_names

        missing = set(marker_names) - set(self.marker_names)
        if missing:
            raise ValueError(f"Markers not in dataset: {missing}")

        if use_cleaned:
            arrays = [self.pairs[name].cleaned.data for name in marker_names]
        else:
            arrays = [self.pairs[name].raw.data for name in marker_names]

        return np.stack(arrays, axis=1)

    def __str__(self) -> str:
        """Human-readable dataset summary."""
        lines = [
            f"TrajectoryDataset:",
            f"  Frames:  {self.n_frames}",
            f"  Markers: {self.n_markers}",
            f"  Marker names: {', '.join(self.marker_names[:5])}"
        ]
        if self.n_markers > 5:
            lines.append(f"                ... and {self.n_markers - 5} more")

        return "\n".join(lines)
