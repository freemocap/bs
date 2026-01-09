"""Reference geometry saving and trajectory verification utilities."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def estimate_distance_matrix(
        *,
        original_data: np.ndarray,
        use_median: bool = True
) -> np.ndarray:
    """
    Estimate the true rigid body distance matrix from original trajectories.

    For each pair of markers, compute distances across all frames and take
    the median (or mean) to get the best estimate of the rigid distance.

    Args:
        original_data: (n_frames, n_markers, 3) measured positions
        use_median: If True, use median distance; if False, use mean

    Returns:
        distances: (n_markers, n_markers) estimated rigid distances
    """
    n_frames, n_markers, _ = original_data.shape
    distances = np.zeros((n_markers, n_markers))

    for i in range(n_markers):
        for j in range(i + 1, n_markers):
            # Compute distance in every frame
            frame_distances = np.linalg.norm(
                original_data[:, i, :] - original_data[:, j, :],
                axis=1
            )
            # Take median as best estimate (robust to outliers)
            if use_median:
                distances[i, j] = distances[j, i] = np.median(frame_distances)
            else:
                distances[i, j] = distances[j, i] = np.mean(frame_distances)

    return distances


def reconstruct_from_distances(
        *,
        distance_matrix: np.ndarray,
        n_dims: int = 3
) -> np.ndarray:
    """
    Reconstruct point coordinates from distance matrix using Classical MDS.

    This solves for positions that best match the given pairwise distances,
    which is exactly what we want for rigid body estimation!

    Args:
        distance_matrix: (n_markers, n_markers) pairwise distances
        n_dims: Number of dimensions (typically 3)

    Returns:
        coordinates: (n_markers, n_dims) reconstructed positions
    """
    n_markers = distance_matrix.shape[0]

    # Square the distances
    D_squared = distance_matrix ** 2

    # Double centering (Classical MDS)
    H = np.eye(n_markers) - np.ones((n_markers, n_markers)) / n_markers
    B = -0.5 * H @ D_squared @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top n_dims components
    eigenvalues = eigenvalues[:n_dims]
    eigenvectors = eigenvectors[:, :n_dims]

    # Reconstruct coordinates
    # Handle negative eigenvalues (shouldn't happen for valid distances, but numerical issues)
    eigenvalues = np.maximum(eigenvalues, 0)
    coordinates = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    return coordinates


def define_body_frame(
        *,
        reference_geometry: np.ndarray,
        marker_names: list[str],
        origin_markers: list[str],
        x_axis_marker: str,
        y_axis_marker: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Define a body-fixed coordinate frame using Gram-Schmidt orthogonalization.

    The frame is defined by:
    - Origin: mean of origin_markers (e.g., head_center from eyes/ears)
    - X-axis: points EXACTLY from origin to x_axis_marker (no adjustment)
    - Y-axis: points generally towards y_axis_marker, adjusted to be perpendicular to X
    - Z-axis: Y × X (perpendicular to both, points up for anatomical coords)

    Note: Only X points exactly at its target marker. Y is orthogonalized via Gram-Schmidt
    to ensure all three axes are mutually perpendicular (orthonormal basis).

    Args:
        reference_geometry: (n_markers, 3) marker positions
        marker_names: List of marker names
        origin_markers: Names of markers whose mean defines the origin
        x_axis_marker: Name of marker that defines X-axis direction (exact)
        y_axis_marker: Name of marker that defines Y-axis direction (approximate)

    Returns:
        basis: (3, 3) orthonormal basis vectors as rows [x, y, z]
        centered_geometry: (n_markers, 3) geometry in new frame
        origin_point: (3,) the computed origin point
    """
    # Get marker indices
    name_to_idx = {name: i for i, name in enumerate(marker_names)}

    # Compute origin as mean of specified markers
    origin_indices = [name_to_idx[name] for name in origin_markers]
    origin_point = reference_geometry[origin_indices].mean(axis=0)

    # Get direction markers
    p_x = reference_geometry[name_to_idx[x_axis_marker]]
    p_y = reference_geometry[name_to_idx[y_axis_marker]]

    # X-axis: points EXACTLY from origin to target (no adjustment)
    x_axis = p_x - origin_point
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Y-axis: points generally towards target, but adjusted to be perpendicular to X
    # Using Gram-Schmidt: remove the X component from the vector towards target
    v_y = p_y - origin_point
    y_axis = v_y - np.dot(v_y, x_axis) * x_axis  # Project out X component
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Z-axis: perpendicular to both X and Y (Y × X makes it point up/dorsal)
    z_axis = np.cross(y_axis, x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Rotation matrix: rows are the basis vectors
    # When we multiply: basis @ point, we project the point onto these new axes
    basis_vectors = np.array([x_axis, y_axis, z_axis])

    return basis_vectors, origin_point


def transform_to_body_frame(
        *,
        reference_geometry: np.ndarray,
        basis_vectors: np.ndarray,
        origin_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform reference geometry to body frame using the computed basis and origin.

    Args:
        reference_geometry: (n_markers, 3) original positions
        basis_vectors: (3, 3) orthonormal basis vectors as rows [x, y, z]
        origin_point: (3,) the computed origin point

    :returns
        transformed_geometry: (n_markers, 3) geometry in new body frame
        transformation_matrix: (4, 4) homogeneous transformation matrix from original to body frame
    """

    # Transform all points to body-fixed frame:
    # 1. Center at origin (subtract head_center)
    centered_geometry = reference_geometry - origin_point
    # 2. Rotate so anatomical axes become standard axes (+X, +Y, +Z)
    transformed_geometry = (basis_vectors @ centered_geometry.T).T

    # Result: transformed geometry where:
    # - Origin is at (0, 0, 0) [head_center]
    # - +X points towards nose (forward/rostral)
    # - +Y points left (lateral)
    # - +Z points up (dorsal)

    # Calculate alignment transformation, i.e. the transformation matrix to take the original geometry to the new body frame
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = basis_vectors
    transformation_matrix[:3, 3] = -basis_vectors @ origin_point

    return transformed_geometry, transformation_matrix


def estimate_rigid_body_reference_geometry(
        *,
        original_data: np.ndarray,
        marker_names: list[str],
        origin_markers: list[str],
        x_axis_marker: str,
        y_axis_marker: str,
        display_edges: list[tuple[str, str]],
        show_alignment_plots: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate reference geometry using distance matrix + MDS + body frame alignment.

    This properly accounts for rigid body constraints by:
    1. Estimating true distances from all frames (median across time)
    2. Reconstructing geometry from distance matrix (MDS)
    3. Aligning to a meaningful body-fixed coordinate frame (Gram-Schmidt)

    Args:
        original_data: (n_frames, n_markers, 3) measured positions
        marker_names: List of marker names
        origin_markers: Names of markers whose mean defines the origin
        x_axis_marker: Name of marker that defines X-axis direction
        y_axis_marker: Name of marker that defines Y-axis direction

    Returns:
        aligned_reference: (n_markers, 3) in body-fixed frame
        original_reference: (n_markers, 3) in original MDS frame
        basis: (3, 3) orthonormal basis vectors as rows [x, y, z]
        origin_point: (3,) the computed origin point
    """
    n_frames, n_markers, _ = original_data.shape

    logger.info(f"  Estimating distance matrix from {n_frames} frames...")
    distance_matrix = estimate_distance_matrix(original_data=original_data, use_median=True)

    # Report distance statistics
    rigid_distances = distance_matrix[np.triu_indices(n_markers, k=1)]
    logger.info(f"    Distance range: [{rigid_distances.min():.3f}, {rigid_distances.max():.3f}]")

    logger.info("  Reconstructing geometry from distances (Classical MDS)...")
    reference_geometry = reconstruct_from_distances(distance_matrix=distance_matrix, n_dims=3)

    (original_basis_vectors,
     origin_point) = define_body_frame(reference_geometry=reference_geometry,
                                                             marker_names=marker_names,
                                                             origin_markers=origin_markers,
                                                             x_axis_marker=x_axis_marker,
                                                             y_axis_marker=y_axis_marker
                                                             )
    (aligned_reference_geometry,
     aligned_basis_vectors,
     alignment_transformation_matrix) = align_reference_geometry_to_body_frame(reference_geometry=reference_geometry,
                                                                               marker_names=marker_names,
                                                                               origin_markers=origin_markers,
                                                                               x_axis_marker=x_axis_marker,
                                                                               y_axis_marker=y_axis_marker
                                                                               )
    # Plot the estimated reference geometry
    logger.info("\nPlotting reference geometry (close window to continue)...")

    def name_to_index(name: str) -> int:
        if marker_names is None:
            raise ValueError("marker_names must be provided to use name_to_index")
        return marker_names.index(name)

    display_edges_as_indices = [(name_to_index(i), name_to_index(j)) for i, j in display_edges]
    aligned_origin_point = aligned_reference_geometry[[name_to_index(name) for name in origin_markers]].mean(axis=0)
    plot_reference_geometry(
        aligned_geometry=aligned_reference_geometry,
        original_geometry=reference_geometry,
        original_basis_vectors=original_basis_vectors,
        aligned_basis_vectors=aligned_basis_vectors,
        original_origin_point=origin_point,
        aligned_origin_point=aligned_origin_point,
        marker_names=marker_names,
        display_edges=display_edges_as_indices,
        origin_markers=origin_markers,
        x_axis_marker=x_axis_marker,
        y_axis_marker=y_axis_marker
    )
    return aligned_reference_geometry, aligned_basis_vectors, alignment_transformation_matrix


def align_reference_geometry_to_body_frame(
        *,
        reference_geometry: np.ndarray,
        marker_names: list[str],
        origin_markers: list[str],
        x_axis_marker: str,
        y_axis_marker: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info(f"  Defining body frame:")
    logger.info(f"    Origin: mean of {origin_markers}")
    logger.info(f"    X-axis: towards '{x_axis_marker}'")
    logger.info(f"    Y-axis: towards '{y_axis_marker}'")

    basis_vectors, origin_point = define_body_frame(
        reference_geometry=reference_geometry,
        marker_names=marker_names,
        origin_markers=origin_markers,
        x_axis_marker=x_axis_marker,
        y_axis_marker=y_axis_marker
    )

    aligned_reference_geometry, alignment_transformation_matrix = transform_to_body_frame(
        reference_geometry=reference_geometry,
        basis_vectors=basis_vectors,
        origin_point=origin_point
    )
    aligned_basis_vectors = basis_vectors @ alignment_transformation_matrix[:3, :3].T
    logger.info(f"  Original  Body frame basis vectors:")
    logger.info(f"      X: [{basis_vectors[0, 0]:7.4f}, {basis_vectors[0, 1]:7.4f}, {basis_vectors[0, 2]:7.4f}]")
    logger.info(f"      Y: [{basis_vectors[1, 0]:7.4f}, {basis_vectors[1, 1]:7.4f}, {basis_vectors[1, 2]:7.4f}]")
    logger.info(f"      Z: [{basis_vectors[2, 0]:7.4f}, {basis_vectors[2, 1]:7.4f}, {basis_vectors[2, 2]:7.4f}]")

    logger.info(f"  Aligned  Body frame basis vectors (should be near eye(3)):")
    logger.info(
        f"      X: [{aligned_basis_vectors[0, 0]:7.4f}, {aligned_basis_vectors[0, 1]:7.4f}, {aligned_basis_vectors[0, 2]:7.4f}]")
    logger.info(
        f"      Y: [{aligned_basis_vectors[1, 0]:7.4f}, {aligned_basis_vectors[1, 1]:7.4f}, {aligned_basis_vectors[1, 2]:7.4f}]")
    logger.info(
        f"      Z: [{aligned_basis_vectors[2, 0]:7.4f}, {aligned_basis_vectors[2, 1]:7.4f}, {aligned_basis_vectors[2, 2]:7.4f}]")

    # Verify orthonormality
    dot_xy = np.dot(basis_vectors[0], basis_vectors[1])
    dot_xz = np.dot(basis_vectors[0], basis_vectors[2])
    dot_yz = np.dot(basis_vectors[1], basis_vectors[2])
    logger.info(f"  Original Basis Vectors  Orthogonality check (should be ~0):")
    logger.info(f"      X·Y = {dot_xy:.2e}, X·Z = {dot_xz:.2e}, Y·Z = {dot_yz:.2e}")

    transformed_dot_xy = np.dot(aligned_basis_vectors[0], aligned_basis_vectors[1])
    transformed_dot_xz = np.dot(aligned_basis_vectors[0], aligned_basis_vectors[2])
    transformed_dot_yz = np.dot(aligned_basis_vectors[1], aligned_basis_vectors[2])
    logger.info(f"  Aligned Basis Vectors  Orthogonality check (should be ~0):")
    logger.info(f"      X·Y = {transformed_dot_xy:.2e}, X·Z = {transformed_dot_xz:.2e}, Y·Z = {transformed_dot_yz:.2e}")

    # Verify transformation worked correctly
    x_reference_marker = aligned_reference_geometry[marker_names.index(x_axis_marker)]
    y_reference_marker = aligned_reference_geometry[marker_names.index(y_axis_marker)]
    target_origin = aligned_reference_geometry[[marker_names.index(name) for name in origin_markers]].mean(axis=0)

    logger.info(f"  Aligned reference geometry to body frame:")
    logger.info(
        f"    Geometry Origin is: [{target_origin[0]:.3f}, {target_origin[1]:.3f}, {target_origin[2]:.3f}] (should be [0, 0, 0])")
    logger.info(
        f"    {x_axis_marker} position: [{x_reference_marker[0]:.3f}, {x_reference_marker[1]:.3f}, {x_reference_marker[2]:.3f}] (should be at +X)")
    logger.info(
        f"    {y_axis_marker} position: [{y_reference_marker[0]:.3f}, {y_reference_marker[1]:.3f}, {y_reference_marker[2]:.3f}] (should be at +Y-ish)")

    return aligned_reference_geometry, aligned_basis_vectors, alignment_transformation_matrix


def save_reference_geometry_json(
        *,
        filepath: Path,
        reference_geometry: np.ndarray,
        marker_names: list[str],
        scale: float = 1.0,
        units: str = "mm"
) -> None:
    """
    Save reference geometry as JSON mapping marker names to xyz coordinates.

    Args:
        filepath: Output JSON file path
        reference_geometry: (n_markers, 3) reference positions in body frame (meters)
        marker_names: List of marker names
        units: Units to save ("mm" or "m")
    """

    geometry_dict = {}
    for marker_name, position in zip(marker_names, reference_geometry):
        geometry_dict[marker_name] = {
            "x": float(position[0] * scale),
            "y": float(position[1] * scale),
            "z": float(position[2] * scale),
            "units": units
        }

    output = {
        "units": units,
        "coordinate_system": {
            "description": "Body-fixed frame",
            "origin": "Mean of origin markers",
            "x_axis": "Points from origin to x_axis_marker",
            "y_axis": "Points generally toward y_axis_marker (orthogonalized)",
            "z_axis": "Cross product of y and x axes"
        },
        "markers": geometry_dict,
        "n_markers": len(marker_names)
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved reference geometry to: {filepath}")
    logger.info(f"  Units: {units}")
    logger.info(f"  Markers: {len(marker_names)}")


def plot_reference_geometry(
        original_geometry: np.ndarray,
        original_basis_vectors: np.ndarray,
        original_origin_point: np.ndarray,

        aligned_geometry: np.ndarray,
        aligned_basis_vectors: np.ndarray,
        aligned_origin_point: np.ndarray,

        marker_names: list[str],
        display_edges: list[tuple[int, int]],
        origin_markers: list[str],
        x_axis_marker: str,
        y_axis_marker: str
) -> None:
    """Plot original and transformed reference geometry side by side."""
    fig = plt.figure(figsize=(20, 9))

    name_to_idx = {name: i for i, name in enumerate(marker_names)}
    frame_marker_names = origin_markers + [x_axis_marker, y_axis_marker]
    frame_indices = [name_to_idx[name] for name in frame_marker_names if name in name_to_idx]

    # =========================================================================
    # LEFT: Original MDS geometry with calculated basis vectors
    # =========================================================================
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot edges
    for i, j in display_edges:
        i = name_to_idx[marker_names[i]]
        j = name_to_idx[marker_names[j]]
        points = original_geometry[[i, j]]
        ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'gray', linewidth=1, alpha=0.4)

    # Plot markers
    ax1.scatter(original_geometry[:, 0], original_geometry[:, 1], original_geometry[:, 2],
                c='blue', s=100, edgecolors='black', linewidth=1)

    # Highlight frame-defining markers
    frame_pos_orig = original_geometry[frame_indices]
    ax1.scatter(frame_pos_orig[:, 0], frame_pos_orig[:, 1], frame_pos_orig[:, 2],
                c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                label='Frame-defining markers', zorder=10)

    # Label markers
    for i, (x, y, z) in enumerate(original_geometry):
        label = marker_names[i]
        if marker_names[i] in origin_markers:
            label += ' (origin)'
        elif marker_names[i] == x_axis_marker:
            label += ' (X)'
        elif marker_names[i] == y_axis_marker:
            label += ' (Y)'
        ax1.text(x, y, z, f'  {label}', fontsize=7)

    # Plot origin point
    ax1.scatter([original_origin_point[0]], [original_origin_point[1]], [original_origin_point[2]],
                c='yellow', s=300, marker='X', edgecolors='black', linewidth=2,
                label='Computed origin', zorder=11)

    # Plot calculated basis vectors FROM origin point
    scale = np.max(np.abs(original_geometry)) * 0.4
    ax1.quiver(original_origin_point[0], original_origin_point[1], original_origin_point[2],
               original_basis_vectors[0, 0] * scale, original_basis_vectors[0, 1] * scale,
               original_basis_vectors[0, 2] * scale,
               color='red', arrow_length_ratio=0.15, linewidth=3, label='X-basis')
    ax1.quiver(original_origin_point[0], original_origin_point[1], original_origin_point[2],
               original_basis_vectors[1, 0] * scale, original_basis_vectors[1, 1] * scale,
               original_basis_vectors[1, 2] * scale,
               color='green', arrow_length_ratio=0.15, linewidth=3, label='Y-basis')
    ax1.quiver(original_origin_point[0], original_origin_point[1], original_origin_point[2],
               original_basis_vectors[2, 0] * scale, original_basis_vectors[2, 1] * scale,
               original_basis_vectors[2, 2] * scale,
               color='blue', arrow_length_ratio=0.15, linewidth=3, label='Z-basis')

    # Formatting
    max_range = np.max(np.abs(original_geometry)) * 1.2
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([-max_range, max_range])
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlabel('X (MDS frame)', fontsize=10)
    ax1.set_ylabel('Y (MDS frame)', fontsize=10)
    ax1.set_zlabel('Z (MDS frame)', fontsize=10)
    ax1.set_title('Original: MDS Reconstruction\n(arbitrary orientation)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # RIGHT: Transformed geometry in body frame
    # =========================================================================
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot edges
    for i, j in display_edges:
        i = name_to_idx[marker_names[i]]
        j = name_to_idx[marker_names[j]]
        points = aligned_geometry[[i, j]]
        ax2.plot(points[:, 0], points[:, 1], points[:, 2], 'gray', linewidth=1, alpha=0.4)

    # Plot markers
    ax2.scatter(aligned_geometry[:, 0], aligned_geometry[:, 1], aligned_geometry[:, 2],
                c='blue', s=100, edgecolors='black', linewidth=1)

    # Highlight frame-defining markers
    frame_pos_trans = aligned_geometry[frame_indices]
    ax2.scatter(frame_pos_trans[:, 0], frame_pos_trans[:, 1], frame_pos_trans[:, 2],
                c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                label='Frame-defining markers', zorder=10)

    # Label markers
    for i, (x, y, z) in enumerate(aligned_geometry):
        label = marker_names[i]
        if marker_names[i] in origin_markers:
            label += ' (origin)'
        elif marker_names[i] == x_axis_marker:
            label += ' (X)'
        elif marker_names[i] == y_axis_marker:
            label += ' (Y)'
        ax2.text(x, y, z, f'  {label}', fontsize=7)

    # Plot origin (now at 0,0,0)
    ax2.scatter([aligned_origin_point[0]], [aligned_origin_point[1]], [aligned_origin_point[2]], c='yellow', s=300,
                marker='X', edgecolors='black',
                linewidth=2, label='Aligned Origin', zorder=11)

    # Plot standard unit vectors (in body frame)
    scale = np.max(np.abs(original_geometry)) * 0.4
    ax2.quiver(aligned_origin_point[0], aligned_origin_point[1], aligned_origin_point[2],
               aligned_basis_vectors[0, 0] * scale,
               aligned_basis_vectors[0, 1] * scale,
               aligned_basis_vectors[0, 2] * scale,
               color='red', arrow_length_ratio=0.15, linewidth=3, label='X-basis')
    ax2.quiver(aligned_origin_point[0], aligned_origin_point[1], aligned_origin_point[2],
               aligned_basis_vectors[1, 0] * scale,
               aligned_basis_vectors[1, 1] * scale,
               aligned_basis_vectors[1, 2] * scale,
               color='green', arrow_length_ratio=0.15, linewidth=3, label='Y-basis')
    ax2.quiver(aligned_origin_point[0], aligned_origin_point[1], aligned_origin_point[2],
               aligned_basis_vectors[2, 0] * scale,
               aligned_basis_vectors[2, 1] * scale,
               aligned_basis_vectors[2, 2] * scale,
               color='blue', arrow_length_ratio=0.15, linewidth=3, label='Z-basis')
    # Formatting
    max_range = np.max(np.abs(aligned_geometry)) * 1.2
    ax2.set_xlim([-max_range, max_range])
    ax2.set_ylim([-max_range, max_range])
    ax2.set_zlim([-max_range, max_range])
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlabel('X (body frame)', fontsize=10)
    ax2.set_ylabel('Y (body frame)', fontsize=10)
    ax2.set_zlabel('Z (body frame)', fontsize=10)
    ax2.set_title('Transformed: Body-Fixed Frame\n(anatomically aligned)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)


def verify_trajectory_reconstruction(
        *,
        reference_geometry: np.ndarray,
        rotations: np.ndarray,
        translations: np.ndarray,
        reconstructed: np.ndarray,
        marker_names: list[str],
        n_frames_to_check: int = 5,
        tolerance: float = 1e-6
) -> bool:
    """
    Verify that reconstructed trajectories match: world = R @ reference + t

    Args:
        reference_geometry: (n_markers, 3) reference positions in body frame
        rotations: (n_frames, 3, 3) rotation matrices
        translations: (n_frames, 3) translation vectors
        reconstructed: (n_frames, n_markers, 3) reconstructed positions
        marker_names: List of marker names
        n_frames_to_check: Number of frames to verify
        tolerance: Maximum allowed error (meters)

    Returns:
        True if verification passes, False otherwise
    """
    n_frames = min(n_frames_to_check, rotations.shape[0])
    n_markers = reference_geometry.shape[0]

    logger.info("=" * 80)
    logger.info("VERIFYING TRAJECTORY RECONSTRUCTION")
    logger.info("=" * 80)
    logger.info(f"Formula: world = R @ reference + t")
    logger.info(f"Checking {n_frames} frames, {n_markers} markers")
    logger.info(f"Tolerance: {tolerance * 1000:.4f} mm")

    max_error = 0.0
    max_error_frame = -1
    max_error_marker = -1

    for frame_idx in range(n_frames):
        R = rotations[frame_idx]
        t = translations[frame_idx]

        # Recompute reconstruction: world = R @ reference + t
        expected = (R @ reference_geometry.T).T + t
        actual = reconstructed[frame_idx]

        # Compute errors
        errors = np.linalg.norm(expected - actual, axis=1)

        # Track maximum error
        frame_max_error = errors.max()
        if frame_max_error > max_error:
            max_error = frame_max_error
            max_error_frame = frame_idx
            max_error_marker = errors.argmax()

        # Log first few frames
        if frame_idx < 3:
            logger.info(f"\nFrame {frame_idx}:")
            logger.info(f"  Max error: {frame_max_error * 1000:.6f} mm (marker: {marker_names[errors.argmax()]})")
            logger.info(f"  Mean error: {errors.mean() * 1000:.6f} mm")

    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Frames checked: {n_frames}")
    logger.info(f"Max error: {max_error * 1000:.6f} mm")
    logger.info(f"  Frame: {max_error_frame}")
    logger.info(f"  Marker: {marker_names[max_error_marker]}")

    if max_error < tolerance:
        logger.info(f"✓ PASS: All errors < {tolerance * 1000:.4f} mm")
        return True
    else:
        logger.warning(f"✗ FAIL: Max error {max_error * 1000:.6f} mm exceeds tolerance {tolerance * 1000:.4f} mm")
        return False


def print_reference_geometry_summary(
        *,
        reference_geometry: np.ndarray,
        marker_names: list[str],
        units: str = "mm",
        scale: float = 1.0
) -> None:
    """
    Print a summary of the reference geometry.

    Args:
        reference_geometry: (n_markers, 3) positions
        marker_names: List of marker names
        units: Display units ("mm" or "m")
    """

    logger.info("=" * 80)
    logger.info("REFERENCE GEOMETRY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Units: {units}")
    logger.info(f"Coordinate system: Body-fixed frame")
    logger.info("\nMarker positions:")
    logger.info(f"{'Name':<20} {'X':>10} {'Y':>10} {'Z':>10}")
    logger.info("-" * 80)

    for name, pos in zip(marker_names, reference_geometry):
        x, y, z = pos * scale
        logger.info(f"{name:<20} {x:>10.3f} {y:>10.3f} {z:>10.3f}")

    # Compute bounding box
    min_vals = reference_geometry.min(axis=0) * scale
    max_vals = reference_geometry.max(axis=0) * scale
    size = max_vals - min_vals

    logger.info("\nBounding box:")
    logger.info(f"  X: [{min_vals[0]:.3f}, {max_vals[1]:.3f}] {units} (size: {size[0]:.3f} {units})")
    logger.info(f"  Y: [{min_vals[1]:.3f}, {max_vals[1]:.3f}] {units} (size: {size[1]:.3f} {units})")
    logger.info(f"  Z: [{min_vals[2]:.3f}, {max_vals[2]:.3f}] {units} (size: {size[2]:.3f} {units})")
