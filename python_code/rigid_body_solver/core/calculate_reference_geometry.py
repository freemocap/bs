import logging

import numpy as np
from numpy.typing import NDArray

from python_code.ferret_gaze.kinematics_core.reference_geometry_model import MarkerPosition, CoordinateFrameDefinition, \
    AxisDefinition, AxisType, ReferenceGeometry

logger = logging.getLogger(__name__)

def define_body_frame(
    *,
    reference_geometry: NDArray[np.float64],
    marker_names: list[str],
    origin_markers: list[str],
    x_axis_marker: str,
    y_axis_marker: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Define a body-fixed coordinate frame using Gram-Schmidt orthogonalization."""
    name_to_idx = {name: i for i, name in enumerate(marker_names)}

    origin_indices = [name_to_idx[name] for name in origin_markers]
    origin_point = reference_geometry[origin_indices].mean(axis=0)

    p_x = reference_geometry[name_to_idx[x_axis_marker]]
    p_y = reference_geometry[name_to_idx[y_axis_marker]]

    x_axis = p_x - origin_point
    x_axis = x_axis / np.linalg.norm(x_axis)

    v_y = p_y - origin_point
    y_axis = v_y - np.dot(v_y, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.cross(y_axis, x_axis)
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

def estimate_distance_matrix(
    *,
    original_data: NDArray[np.float64],
    use_median: bool = True,
) -> NDArray[np.float64]:
    """Estimate the true rigid body distance matrix from original trajectories."""
    n_frames, n_markers, _ = original_data.shape
    distances = np.zeros((n_markers, n_markers), dtype=np.float64)

    for i in range(n_markers):
        for j in range(i + 1, n_markers):
            frame_distances = np.linalg.norm(
                original_data[:, i, :] - original_data[:, j, :],
                axis=1,
            )
            if use_median:
                distances[i, j] = distances[j, i] = np.nanmedian(frame_distances)
            else:
                distances[i, j] = distances[j, i] = np.nanmean(frame_distances)

    return distances


def reconstruct_from_distances(
    *,
    distance_matrix: NDArray[np.float64],
    n_dims: int = 3,
) -> NDArray[np.float64]:
    """Reconstruct point coordinates from distance matrix using Classical MDS."""
    n_markers = distance_matrix.shape[0]

    D_squared = distance_matrix**2
    H = np.eye(n_markers) - np.ones((n_markers, n_markers)) / n_markers
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
    marker_names: list[str],
    origin_markers: list[str],
    x_axis_marker: str,
    y_axis_marker: str,
    units: str = "mm",
) -> tuple[ReferenceGeometry, NDArray[np.float64]]:
    """
    Estimate reference geometry and return as a ReferenceGeometry model.

    Returns:
        reference_geometry: ReferenceGeometry pydantic model
        aligned_positions: (n_markers, 3) aligned positions as numpy array
    """
    logger.info("Estimating distance matrix from data...")
    distance_matrix = estimate_distance_matrix(original_data=original_data, use_median=True)

    logger.info("Reconstructing geometry from distances (Classical MDS)...")
    mds_geometry = reconstruct_from_distances(distance_matrix=distance_matrix, n_dims=3)

    logger.info("Defining body frame:")
    logger.info(f"  Origin: mean of {origin_markers}")
    logger.info(f"  X-axis: towards '{x_axis_marker}'")
    logger.info(f"  Y-axis: towards '{y_axis_marker}'")

    basis_vectors, origin_point = define_body_frame(
        reference_geometry=mds_geometry,
        marker_names=marker_names,
        origin_markers=origin_markers,
        x_axis_marker=x_axis_marker,
        y_axis_marker=y_axis_marker,
    )

    aligned_geometry = transform_to_body_frame(
        reference_geometry=mds_geometry,
        basis_vectors=basis_vectors,
        origin_point=origin_point,
    )

    # Build the ReferenceGeometry pydantic model
    markers = {
        name: MarkerPosition(
            x=float(aligned_geometry[i, 0]),
            y=float(aligned_geometry[i, 1]),
            z=float(aligned_geometry[i, 2]),
        )
        for i, name in enumerate(marker_names)
    }

    coordinate_frame = CoordinateFrameDefinition(
        origin_markers=origin_markers,
        x_axis=AxisDefinition(markers=[x_axis_marker], type=AxisType.EXACT),
        y_axis=AxisDefinition(markers=[y_axis_marker], type=AxisType.APPROXIMATE),
    )

    reference_geometry_model = ReferenceGeometry(
        units=units,
        coordinate_frame=coordinate_frame,
        markers=markers,
    )

    return reference_geometry_model, aligned_geometry
