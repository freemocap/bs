"""Rigid body topology definitions using Pydantic v2."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StickFigureTopology(BaseModel):
    """
    Define which basic stick-figure connections between sets of keypoint trajectories

    This class specifies:
    - Which keypoints belong to the rigid body
    - Which pairs should maintain fixed distances (constraints)
    - Which edges to display in visualization
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    keypoint_names: list[str]
    """Names of keypoints that belong to this rigid body"""

    rigid_edges: list[tuple[str, str]]
    """Pairs of keypoint names that should maintain fixed distance during optimization"""

    display_edges: list[tuple[str, str]] | None = None
    """Edges to display in visualization (defaults to rigid_edges if None)"""

    name: str = "rigid_body"
    """Descriptive name for this rigid body configuration"""

    @field_validator("keypoint_names")
    @classmethod
    def keypoint_names_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("keypoint_names cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_edges(self) -> "StickFigureTopology":
        """Validate that all edge keypoints exist in keypoint_names."""
        keypoint_set = set(self.keypoint_names)
        for i, j in self.rigid_edges:
            if i not in keypoint_set:
                raise ValueError(f"Rigid edge keypoint '{i}' not in keypoint_names: {self.keypoint_names}")
            if j not in keypoint_set:
                raise ValueError(f"Rigid edge keypoint '{j}' not in keypoint_names: {self.keypoint_names}")

        if self.display_edges is not None:
            for i, j in self.display_edges:
                if i not in keypoint_set:
                    raise ValueError(f"Display edge keypoint '{i}' not in keypoint_names: {self.keypoint_names}")
                if j not in keypoint_set:
                    raise ValueError(f"Display edge keypoint '{j}' not in keypoint_names: {self.keypoint_names}")

        return self

    @property
    def rigid_edges_as_index_pairs(self) -> list[tuple[int, int]]:
        """Convert rigid edges from keypoint names to index pairs."""
        return [(self.name_to_index(i), self.name_to_index(j)) for i, j in self.rigid_edges]

    @property
    def display_edges_resolved(self) -> list[tuple[str, str]]:
        """Get display edges, defaulting to rigid_edges if not set."""
        if self.display_edges is None:
            return list(self.rigid_edges)
        return list(self.display_edges)

    def name_to_index(self, name: str) -> int:
        """Convert keypoint name to index."""
        try:
            return self.keypoint_names.index(name)
        except ValueError:
            raise ValueError(f"Marker name '{name}' not found in keypoint_names: {self.keypoint_names}")

    def index_to_name(self, index: int) -> str:
        """Convert keypoint index to name."""
        if index < 0 or index >= len(self.keypoint_names):
            raise IndexError(f"Marker index {index} out of range for keypoint_names: {self.keypoint_names}")
        return self.keypoint_names[index]


    def save_json(self, filepath: Path) -> None:
        """Save topology to JSON file."""
        self_dict = self.model_dump()
        for key, value in self_dict.items():
            if isinstance(value, float) and np.abs(value) < 1e-10:
                self_dict[key] = 0.0  # Squish small number
        with open(filepath, "w") as f:
            json.dump(self_dict, fp=f, indent=2)

    @classmethod
    def load_json(cls, filepath: Path) -> "StickFigureTopology":
        """Load topology from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(fp=f)
        return cls(**data)

    def validate_data(self, trajectory_dict: dict[str, NDArray[np.float64]]) -> None:
        """
        Validate that trajectory data contains all required keypoints.

        Args:
            trajectory_dict: Dictionary mapping keypoint names to trajectories

        Raises:
            ValueError: If any keypoints are missing
        """
        missing = set(self.keypoint_names) - set(trajectory_dict.keys())
        if missing:
            raise ValueError(f"Missing {len(missing)} keypoints in data: {sorted(missing)}")

    def extract_trajectories(
        self,
        trajectory_dict: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        Extract and order trajectories according to topology.

        Args:
            trajectory_dict: Maps keypoint names to (n_frames, 3) arrays

        Returns:
            (n_frames, n_keypoints, 3) ordered trajectory array
        """
        self.validate_data(trajectory_dict=trajectory_dict)

        trajectories = [trajectory_dict[name] for name in self.keypoint_names]
        return np.stack(trajectories, axis=1)

    def __repr__(self) -> str:
        return (
            f"RigidBodyTopology(name='{self.name}', "
            f"keypoints={len(self.keypoint_names)}, "
            f"edges={len(self.rigid_edges)})"
        )
