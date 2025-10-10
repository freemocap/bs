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

    rigid_edges: list[tuple[int, int]]
    """Pairs of marker indices that should maintain fixed distance during optimization"""

    display_edges: list[tuple[int, int]] | None = None
    """Edges to display in visualization (defaults to rigid_edges if None)"""

    name: str = "rigid_body"
    """Descriptive name for this rigid body configuration"""

    def __post_init__(self) -> None:
        """Initialize display edges if not provided."""
        if self.display_edges is None:
            self.display_edges = self.rigid_edges.copy()

        # Validation
        n_markers = len(self.marker_names)
        for i, j in self.rigid_edges:
            if i < 0 or i >= n_markers or j < 0 or j >= n_markers:
                raise ValueError(
                    f"Invalid edge ({i}, {j}): indices must be in range [0, {n_markers})"
                )

    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "marker_names": self.marker_names,
            "rigid_edges": self.rigid_edges,
            "display_edges": self.display_edges,
        }

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

    @classmethod
    def from_marker_names(
            cls,
            *,
            marker_names: list[str],
            name: str = "auto_generated",
            edge_strategy: str = "full"
    ) -> "RigidBodyTopology":
        """
        Create topology automatically from marker names.

        Args:
            marker_names: List of marker names
            name: Name for this topology
            edge_strategy: Strategy for creating edges:
                - "full": Connect all pairs (n*(n-1)/2 edges)
                - "minimal": Create minimal spanning tree
                - "skeleton": Connect adjacent markers in sequence

        Returns:
            RigidBodyTopology instance
        """
        n = len(marker_names)

        if edge_strategy == "full":
            # All pairwise connections
            rigid_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        elif edge_strategy == "minimal":
            # Minimal spanning tree (star pattern from first marker)
            rigid_edges = [(0, i) for i in range(1, n)]
        elif edge_strategy == "skeleton":
            # Sequential connections
            rigid_edges = [(i, i + 1) for i in range(n - 1)]
        else:
            raise ValueError(f"Unknown edge_strategy: {edge_strategy}")

        return cls(
            marker_names=marker_names,
            rigid_edges=rigid_edges,
            name=name
        )

    def compute_reference_distances(
            self,
            *,
            reference_geometry: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise distances for rigid edges.

        Args:
            reference_geometry: (n_markers, 3) reference positions

        Returns:
            (n_markers, n_markers) distance matrix (0 for non-rigid pairs)
        """
        n = len(self.marker_names)
        distances = np.zeros((n, n))

        for i, j in self.rigid_edges:
            dist = np.linalg.norm(reference_geometry[i] - reference_geometry[j])
            distances[i, j] = dist
            distances[j, i] = dist

        return distances

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