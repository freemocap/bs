import logging
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import shortest_path

from python_code.kinematics_core.reference_geometry_model import MarkerPosition, CoordinateFrameDefinition, \
    AxisDefinition, AxisType, ReferenceGeometry

logger = logging.getLogger(__name__)

def define_body_frame(
    *,
    reference_geometry: NDArray[np.float64],
    keypoint_names: list[str],
    origin_keypoints: list[str],
    x_axis_keypoint: str,
    y_axis_keypoint: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Define a body-fixed coordinate frame using Gram-Schmidt orthogonalization."""
    name_to_idx = {name: i for i, name in enumerate(keypoint_names)}

    origin_indices = [name_to_idx[name] for name in origin_keypoints]
    origin_point = reference_geometry[origin_indices].mean(axis=0)

    p_x = reference_geometry[name_to_idx[x_axis_keypoint]]
    p_y = reference_geometry[name_to_idx[y_axis_keypoint]]

    x_axis = p_x - origin_point
    x_axis = x_axis / np.linalg.norm(x_axis)

    v_y = p_y - origin_point
    y_axis = v_y - np.dot(v_y, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        raise ValueError("x and y keypoints are nearly collinear with the origin")
    y_axis = y_axis / y_norm

    # z_axis = np.cross(y_axis, x_axis)  # TODO: Double check this - appears to create a left hand coordinate system
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    basis_vectors = np.array([x_axis, y_axis, z_axis])

    return basis_vectors, origin_point


def transform_to_body_frame(
    *,
    reference_geometry: NDArray[np.float64],
    basis_vectors: NDArray[np.float64],
    origin_point: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform reference geometry to body frame."""
    centered_geometry = reference_geometry - origin_point
    transformed_geometry = (basis_vectors @ centered_geometry.T).T
    return transformed_geometry

def _fill_missing_distances_via_shortest_path(
    distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fill in NaN (unknown) pairwise distances via shortest path through the known
    (rigid-edge) distances, treating the distance matrix as a weighted graph.

    Requires the known distances to span a connected graph over all keypoints —
    otherwise there's no path to infer an unknown pair's distance from, and MDS
    (which needs a complete matrix) would have nothing principled to reconstruct it
    with.
    """
    n_keypoints = distances.shape[0]
    graph = np.where(np.isnan(distances), 0.0, distances)
    full_distances = shortest_path(graph, method="D", directed=False)

    if np.isinf(full_distances).any():
        raise ValueError(
            "rigid_edges does not connect every keypoint — cannot reconstruct a "
            "complete distance matrix for MDS. Provide a rigid_edges set whose graph "
            "spans all keypoints."
        )

    np.fill_diagonal(full_distances, 0.0)
    return full_distances


def estimate_distance_matrix(
    *,
    original_data: NDArray[np.float64],
    use_median: bool = True,
    keypoint_names: list[str] | None = None,
    rigid_edges: list[tuple[str, str]] | None = None,
) -> NDArray[np.float64]:
    """Estimate the true rigid body distance matrix from original trajectories.

    When `rigid_edges` is given, distances are only computed for keypoint pairs
    present in it (both name orderings match) — all other pairs are filled via
    shortest-path through the known rigid distances (see
    `_fill_missing_distances_via_shortest_path`), so non-rigid marker pairs don't
    corrupt the MDS reconstruction with a distance that isn't actually fixed.
    `keypoint_names` is required in that case, to map edge names to indices.
    """
    n_frames, n_keypoints, _ = original_data.shape

    if rigid_edges is not None:
        if keypoint_names is None:
            raise ValueError("keypoint_names is required when rigid_edges is given")
        name_to_idx = {name: i for i, name in enumerate(keypoint_names)}
        pairs = {
            (min(name_to_idx[a], name_to_idx[b]), max(name_to_idx[a], name_to_idx[b]))
            for a, b in rigid_edges
        }
    else:
        pairs = set(combinations(range(n_keypoints), 2))

    distances = np.full((n_keypoints, n_keypoints), np.nan, dtype=np.float64)

    for i, j in pairs:
        frame_distances = np.linalg.norm(
            original_data[:, i, :] - original_data[:, j, :],
            axis=1,
        )
        if use_median:
            distances[i, j] = distances[j, i] = np.nanmedian(frame_distances)
        else:
            distances[i, j] = distances[j, i] = np.nanmean(frame_distances)

    np.fill_diagonal(distances, 0.0)

    if rigid_edges is not None and len(pairs) < n_keypoints * (n_keypoints - 1) // 2:
        distances = _fill_missing_distances_via_shortest_path(distances)

    return distances


def reconstruct_from_distances(
    *,
    distance_matrix: NDArray[np.float64],
    n_dims: int = 3,
) -> NDArray[np.float64]:
    """Reconstruct point coordinates from distance matrix using Classical MDS."""
    n_keypoints = distance_matrix.shape[0]

    D_squared = distance_matrix**2
    H = np.eye(n_keypoints) - np.ones((n_keypoints, n_keypoints)) / n_keypoints
    B = -0.5 * H @ D_squared @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = eigenvalues[:n_dims]
    eigenvectors = eigenvectors[:, :n_dims]

    eigenvalues = np.maximum(eigenvalues, 0)
    coordinates = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    return coordinates

def estimate_reference_geometry(
    *,
    original_data: NDArray[np.float64],
    keypoint_names: list[str],
    origin_keypoints: list[str],
    x_axis_keypoint: str,
    y_axis_keypoint: str,
    units: str = "mm",
display_edges: list[tuple[str, str]] | None = None,
        rigid_edges: list[tuple[str, str]] | None = None,
) -> tuple[ReferenceGeometry, NDArray[np.float64]]:
    """
    Estimate reference geometry and return as a ReferenceGeometry model.

    Returns:
        reference_geometry: ReferenceGeometry pydantic model
        aligned_positions: (n_keypoints, 3) aligned positions as numpy array
    """
    logger.info("Estimating distance matrix from data...")
    if rigid_edges is None:
        rigid_edges = list(combinations(keypoint_names, 2))
    distance_matrix = estimate_distance_matrix(
        original_data=original_data,
        use_median=True,
        keypoint_names=keypoint_names,
        rigid_edges=rigid_edges,
    )

    logger.info("Reconstructing geometry from distances (Classical MDS)...")
    mds_geometry = reconstruct_from_distances(distance_matrix=distance_matrix, n_dims=3)

    logger.info("Defining body frame:")
    logger.info(f"  Origin: mean of {origin_keypoints}")
    logger.info(f"  X-axis: towards '{x_axis_keypoint}'")
    logger.info(f"  Y-axis: towards '{y_axis_keypoint}'")

    basis_vectors, origin_point = define_body_frame(
        reference_geometry=mds_geometry,
        keypoint_names=keypoint_names,
        origin_keypoints=origin_keypoints,
        x_axis_keypoint=x_axis_keypoint,
        y_axis_keypoint=y_axis_keypoint,
    )

    aligned_geometry = transform_to_body_frame(
        reference_geometry=mds_geometry,
        basis_vectors=basis_vectors,
        origin_point=origin_point,
    )

    # ── Correct MDS Z-axis sign ambiguity ──────────────────────
    # Classical MDS (``np.linalg.eigh``) produces eigenvectors with
    # arbitrary sign — the 3rd eigenvector (Z dimension) has a 50 %
    # chance of pointing the wrong way.  Distances alone cannot
    # resolve this: a geometry and its mirror image share the same
    # distance matrix.
    #
    # We break the ambiguity by checking a known dorsal keypoint
    # against the *original* trajectory data, which lives in a
    # physically meaningful world frame (Z ≈ up).
    _dorsal_candidate_names: list[str] = [
        n for n in keypoint_names
        if n not in {*origin_keypoints, x_axis_keypoint, y_axis_keypoint}
    ]
    if _dorsal_candidate_names:
        _dorsal_name: str = _dorsal_candidate_names[0]
        _dorsal_idx: int = keypoint_names.index(_dorsal_name)
        _origin_idxs: list[int] = [
            keypoint_names.index(n) for n in origin_keypoints
        ]
        _x_idx: int = keypoint_names.index(x_axis_keypoint)
        _y_idx: int = keypoint_names.index(y_axis_keypoint)

        # Collect signed distance of the dorsal keypoint from the
        # eye-nose plane, averaged over all valid frames.
        _world_dots: list[float] = []
        for _f in range(original_data.shape[0]):
            _pts = original_data[_f]
            if np.isnan(_pts[[*_origin_idxs, _x_idx, _y_idx, _dorsal_idx]]).any():
                continue
            _origin = _pts[_origin_idxs].mean(axis=0)
            _v1 = _pts[_x_idx] - _origin
            _v2 = _pts[_y_idx] - _origin
            _plane_normal = np.cross(_v1, _v2)
            _world_dots.append(
                float(np.dot(_pts[_dorsal_idx] - _origin, _plane_normal)),
            )

        if _world_dots:
            _expected_sign: int = 1 if np.mean(_world_dots) > 0 else -1
            _current_z: float = float(aligned_geometry[_dorsal_idx, 2])
            if _expected_sign != 0 and np.sign(_current_z) != _expected_sign:
                logger.warning(
                    "MDS produced a Z-mirrored geometry "
                    "(%s Z=%.2f, expected sign %+d).  Flipping Z axis.",
                    _dorsal_name, _current_z, _expected_sign,
                )
                aligned_geometry[:, 2] *= -1.0

    # Build the ReferenceGeometry pydantic model
    keypoints = {
        name: MarkerPosition(
            x=float(aligned_geometry[i, 0]),
            y=float(aligned_geometry[i, 1]),
            z=float(aligned_geometry[i, 2]),
        )
        for i, name in enumerate(keypoint_names)
    }

    coordinate_frame = CoordinateFrameDefinition(
        origin_keypoints=origin_keypoints,
        x_axis=AxisDefinition(keypoints=[x_axis_keypoint], type=AxisType.EXACT),
        y_axis=AxisDefinition(keypoints=[y_axis_keypoint], type=AxisType.APPROXIMATE),
    )

    reference_geometry_model = ReferenceGeometry(
        units=units,
        coordinate_frame=coordinate_frame,
        keypoints=keypoints,
        display_edges=display_edges,
        rigid_edges=rigid_edges
    )

    return reference_geometry_model, aligned_geometry
