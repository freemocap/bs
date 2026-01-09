"""Rigid body topology definitions."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class RigidBodyTopology:
    """
    Define which markers form a rigid body and how they're connected.

    This class specifies:
    - Which markers belong to the rigid body
    - Which pairs should maintain fixed distances (constraints)
    - Which edges to display in visualization
    """

    marker_names: list[str]
    """Names of markers that belong to this rigid body"""

    rigid_edges: list[tuple[str, str]]
    """Pairs of marker indices that should maintain fixed distance during optimization"""

    display_edges: list[tuple[str, str]] | None = None
    """Edges to display in visualization (defaults to rigid_edges if None)"""

    name: str = "rigid_body"
    """Descriptive name for this rigid body configuration"""

    @property
    def rigid_edges_as_index_pairs(self) -> list[tuple[int, int]]:
        """Convert rigid edges from marker names to index pairs."""
        return [(self.name_to_index(i), self.name_to_index(j)) for i, j in self.rigid_edges]
    def __post_init__(self) -> None:
        """Initialize display edges if not provided."""
        if self.display_edges is None:
            self.display_edges = self.rigid_edges.copy()

        # Validation
        for i, j in self.rigid_edges:
            if not(i in self.marker_names) or not(j in self.marker_names):
                raise ValueError(f"Rigid edge ({i}, {j}) contains marker not in marker_names: {self.marker_names}")

    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "marker_names": self.marker_names,
            "rigid_edges": self.rigid_edges,
            "display_edges": self.display_edges,
        }

    def name_to_index(self, name: str) -> int:
        """Convert marker name to index."""
        try:
            return self.marker_names.index(name)
        except ValueError:
            raise ValueError(f"Marker name '{name}' not found in marker_names: {self.marker_names}")

    def index_to_name(self, index: int) -> str:
        """Convert marker index to name."""
        if index < 0 or index >= len(self.marker_names):
            raise IndexError(f"Marker index {index} out of range for marker_names: {self.marker_names}")
        return self.marker_names[index]

    @classmethod
    def from_dict(cls, *, data: dict[str, object]) -> "RigidBodyTopology":
        """Create topology from dictionary."""
        return cls(
            name=str(data["name"]),
            marker_names=list(data["marker_names"]),
            rigid_edges=list(data["rigid_edges"]),
            display_edges=list(data.get("display_edges")) if data.get("display_edges") else None,
        )

    def save_json(self, *, filepath: Path) -> None:
        """Save topology to JSON file."""
        with open(filepath, "w") as f:
            json.dump(obj=self.to_dict(), fp=f, indent=2)

    @classmethod
    def load_json(cls, *, filepath: Path) -> "RigidBodyTopology":
        """Load topology from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(fp=f)
        return cls.from_dict(data=data)




    def validate_data(self, *, trajectory_dict: dict[str, np.ndarray]) -> None:
        """
        Validate that trajectory data contains all required markers.

        Args:
            trajectory_dict: Dictionary mapping marker names to trajectories

        Raises:
            ValueError: If any markers are missing
        """
        missing = set(self.marker_names) - set(trajectory_dict.keys())
        if missing:
            raise ValueError(
                f"Missing {len(missing)} markers in data: {sorted(missing)}"
            )

    def extract_trajectories(
            self,
            *,
            trajectory_dict: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Extract and order trajectories according to topology.

        Args:
            trajectory_dict: Maps marker names to (n_frames, 3) arrays

        Returns:
            (n_frames, n_markers, 3) ordered trajectory array
        """
        self.validate_data(trajectory_dict=trajectory_dict)

        trajectories = [trajectory_dict[name] for name in self.marker_names]
        return np.stack(trajectories, axis=1)

    def __repr__(self) -> str:
        return (
            f"RigidBodyTopology(name='{self.name}', "
            f"markers={len(self.marker_names)}, "
            f"edges={len(self.rigid_edges)})"
        )