from functools import cached_property

import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator


class KeypointTrajectories(BaseModel):
    """
    Pre-computed keypoint trajectories from quaternion rotations.

    Stores rotated positions for all keypoints across all frames in a single
    (N, M, 3) array for efficient memory access and computation.

    Access individual keypoint trajectories by name using indexing:
        trajectories["nose"]  # Returns (N, 3) array
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

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
