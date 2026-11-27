"""Smoke test for the rigid_edges -> estimate_distance_matrix partial-rigidity path
(solver_improvement_plan.md item 6, Option A).

Scenario: a rigid cluster of 4 points (a, b, c, d) undergoes rigid body motion, plus a
5th point (e) attached via a hinge — fixed distance to 'a' only, free to wobble in
every other direction, so its distance to b/c/d varies frame to frame and is not a
real rigid-body constant.

Without rigid_edges (legacy all-pairs behavior), e's inconsistent distances to b/c/d
get fed into the same joint MDS solve as the honest a/b/c/d distances, which can
distort the reconstructed geometry of the rigid cluster itself (not just e's own
position) through the shared double-centering step. With rigid_edges restricted to
the honest pairs (all of a/b/c/d, plus a-e), e's bad distances are never computed
directly — b-e/c-e/d-e are filled via shortest path through 'a' instead — so the
rigid cluster's reconstructed geometry should closely match a clean baseline computed
from a/b/c/d alone.
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.transform import Rotation

from python_code.rigid_body_solver.core.calculate_reference_geometry import (
    estimate_distance_matrix,
    estimate_reference_geometry,
)


def _generate_synthetic_data(rng: np.random.Generator, n_frames: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Returns (data_without_e, data_with_e), both (n_frames, n_keypoints, 3)."""
    # Reference ("body frame") positions, noise-free, not collinear/degenerate.
    core_reference = np.array(
        [
            [0.0, 0.0, 0.0],  # a
            [2.0, 0.0, 0.0],  # b
            [0.0, 2.0, 0.0],  # c
            [0.0, 0.0, 2.0],  # d
        ]
    )
    hinge_distance = 1.5

    core_frames = np.zeros((n_frames, 4, 3))
    e_frames = np.zeros((n_frames, 1, 3))

    for t in range(n_frames):
        R = Rotation.random(random_state=rng).as_matrix()
        translation = rng.normal(size=3) * 10.0
        core_world = (R @ core_reference.T).T + translation
        core_frames[t] = core_world

        wobble = rng.normal(size=3)
        wobble = wobble / np.linalg.norm(wobble)
        e_frames[t, 0] = core_world[0] + wobble * hinge_distance  # fixed distance to 'a'

    data_without_e = core_frames
    data_with_e = np.concatenate([core_frames, e_frames], axis=1)
    return data_without_e, data_with_e


def main() -> None:
    rng = np.random.default_rng(0)
    data_without_e, data_with_e = _generate_synthetic_data(rng)

    core_names = ["a", "b", "c", "d"]
    all_names = ["a", "b", "c", "d", "e"]

    # Clean baseline: reconstruct the rigid cluster with no corrupting marker present.
    clean_geometry, clean_aligned = estimate_reference_geometry(
        original_data=data_without_e,
        keypoint_names=core_names,
        origin_keypoints=["a"],
        x_axis_keypoint="b",
        y_axis_keypoint="c",
        units="mm",
    )

    # Fixed path: rigid_edges excludes the hinge marker's bad (b-e, c-e, d-e) pairs.
    from itertools import combinations

    rigid_edges = list(combinations(core_names, 2)) + [("a", "e")]
    fixed_geometry, fixed_aligned = estimate_reference_geometry(
        original_data=data_with_e,
        keypoint_names=all_names,
        origin_keypoints=["a"],
        x_axis_keypoint="b",
        y_axis_keypoint="c",
        units="mm",
        rigid_edges=rigid_edges,
    )

    # Legacy path: no rigid_edges given -> all pairs computed directly, including the
    # hinge marker's inconsistent distances to b/c/d.
    corrupted_geometry, corrupted_aligned = estimate_reference_geometry(
        original_data=data_with_e,
        keypoint_names=all_names,
        origin_keypoints=["a"],
        x_axis_keypoint="b",
        y_axis_keypoint="c",
        units="mm",
    )

    fixed_core_error = float(np.max(np.abs(fixed_aligned[:4] - clean_aligned)))
    corrupted_core_error = float(np.max(np.abs(corrupted_aligned[:4] - clean_aligned)))

    print(f"Fixed (rigid_edges) core position max abs error vs. clean baseline:     {fixed_core_error:.4f} mm")
    print(f"Legacy (all-pairs) core position max abs error vs. clean baseline:      {corrupted_core_error:.4f} mm")

    # Shortest-path-filled distances for a 5-point graph aren't a perfect Euclidean
    # embedding (e's true position isn't fixed in the body frame at all), so some
    # small MDS coupling into the rigid cluster is still expected on the fixed path —
    # the point of this check is that it's no worse than feeding e's literally
    # inconsistent raw median distances into the same joint solve.
    assert fixed_core_error <= corrupted_core_error + 1e-9, (
        "rigid_edges path should not distort the rigid cluster more than the legacy all-pairs path"
    )

    # Deterministic, MDS-independent check of the actual mechanism: estimate_distance_matrix
    # with rigid_edges must (a) compute excluded-pair distances identically to a plain
    # shortest-path fill over the directly-computed rigid_edges distances (not the raw,
    # inconsistent median), and (b) leave the directly-specified pairs untouched.
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    raw_distances = estimate_distance_matrix(original_data=data_with_e, use_median=True)
    filled_distances = estimate_distance_matrix(
        original_data=data_with_e, use_median=True, keypoint_names=all_names, rigid_edges=rigid_edges
    )

    rigid_edge_indices = {(name_to_idx[i], name_to_idx[j]) for i, j in rigid_edges}
    known_distances = np.full_like(raw_distances, np.nan)
    for i, j in rigid_edge_indices:
        known_distances[i, j] = known_distances[j, i] = raw_distances[i, j]
    np.fill_diagonal(known_distances, 0.0)
    expected_filled = shortest_path(np.where(np.isnan(known_distances), 0.0, known_distances), directed=False)

    assert np.allclose(filled_distances, expected_filled), (
        "estimate_distance_matrix(rigid_edges=...) should match an independent shortest-path fill"
    )
    for i, j in rigid_edge_indices:
        assert np.isclose(filled_distances[i, j], raw_distances[i, j]), (
            "a directly-specified rigid_edges pair should keep its raw computed distance"
        )

    b_idx, e_idx = name_to_idx["b"], name_to_idx["e"]
    assert not np.isclose(raw_distances[b_idx, e_idx], filled_distances[b_idx, e_idx]), (
        "excluded pair (b, e) should be shortest-path filled, not the raw median distance"
    )

    # A rigid_edges set that leaves a keypoint disconnected (no path to it at all)
    # cannot produce a complete distance matrix for MDS, and must raise rather than
    # silently returning a nonsensical (inf/NaN) geometry.
    disconnected_edges = list(combinations(core_names, 2))  # no edge touches 'e' at all
    try:
        estimate_distance_matrix(
            original_data=data_with_e,
            use_median=True,
            keypoint_names=all_names,
            rigid_edges=disconnected_edges,
        )
        raise AssertionError("disconnected rigid_edges should have raised ValueError")
    except ValueError:
        pass

    print("PASS")


if __name__ == "__main__":
    main()
