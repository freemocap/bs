from functools import cached_property
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy._typing import NDArray
from pydantic import ConfigDict, model_validator

from python_code.eye_analysis.data_models.abase_model import FrozenABaseModel


class KeypointTrajectories(FrozenABaseModel):
    """
    Stores positions for a set of keypoints across all frames in a single
    (N, M, 3) array for efficient memory access and computation.

    Access individual keypoint trajectories by name using indexing:
        trajectories["nose"]  # Returns (N, 3) array
    """

    keypoint_names: tuple[str, ...]  # Immutable, ordered
    timestamps: NDArray[np.float64]  # (N,)
    trajectories_fr_id_xyz: NDArray[np.float64]  # (N, M, 3)

    @cached_property
    def _name_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.keypoint_names)}

    @model_validator(mode="after")
    def validate_shapes(self) -> "KeypointTrajectories":
        n_frames = len(self.timestamps)
        n_keypoints = len(self.keypoint_names)
        expected_shape = (n_frames, n_keypoints, 3)
        if self.trajectories_fr_id_xyz.shape != expected_shape:
            raise ValueError(
                f"rotated_positions shape {self.trajectories_fr_id_xyz.shape} != "
                f"expected {expected_shape}"
            )
        return self

    def __getitem__(self, keypoint_name: str) -> NDArray[np.float64]:
        """
        Get (N, 3) trajectory for a specific keypoint.

        Args:
            keypoint_name: Name of the keypoint

        Returns:
            (N, 3) array of rotated positions over time

        Raises:
            KeyError: If keypoint_name not found
        """
        if keypoint_name not in self._name_to_index:
            raise KeyError(
                f"Keypoint '{keypoint_name}' not found. "
                f"Available: {sorted(self.keypoint_names)}"
            )
        idx = self._name_to_index[keypoint_name]
        return self.trajectories_fr_id_xyz[:, idx, :]

    def __contains__(self, keypoint_name: str) -> bool:
        return keypoint_name in self._name_to_index

    def __len__(self) -> int:
        return len(self.keypoint_names)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoint_names)

    @classmethod
    def from_tidy_csv(
        cls,
        csv_path: Path,
        *,
        frame_column: str = "frame",
        timestamp_column: str = "timestamp",
        trajectory_column: str = "trajectory",
        component_column: str = "component",
        value_column: str = "value",
    ) -> "KeypointTrajectories":
        """
        Load keypoint trajectories from a tidy-format CSV.

        Tidy format has one row per (frame, keypoint, component), e.g.:

            frame,timestamp,trajectory,component,value,units
            0,0.0,nose,x,12.5,mm
            0,0.0,nose,y,-3.2,mm
            0,0.0,nose,z,500.0,mm
            ...

        Column names are customizable via keyword arguments with these defaults.
        """
        import polars as pl

        df = pl.read_csv(csv_path)

        keypoint_names = sorted(df[trajectory_column].unique().to_list())
        n_frames = df[frame_column].n_unique()

        # Extract timestamps from the first keypoint's x-component
        first_keypoint = df.filter(
            (pl.col(trajectory_column) == keypoint_names[0])
            & (pl.col(component_column) == "x")
        ).sort(frame_column)
        timestamps = first_keypoint[timestamp_column].to_numpy().astype(np.float64)

        trajectories = np.zeros((n_frames, len(keypoint_names), 3), dtype=np.float64)
        for i, kp in enumerate(keypoint_names):
            wide = (
                df.filter(pl.col(trajectory_column) == kp)
                .pivot(
                    index=frame_column,
                    on=component_column,
                    values=value_column,
                    aggregate_function="first",
                )
                .sort(frame_column)
            )
            trajectories[:, i, :] = wide.select(["x", "y", "z"]).to_numpy()

        return cls(
            keypoint_names=tuple(keypoint_names),
            timestamps=timestamps,
            trajectories_fr_id_xyz=trajectories,
        )
