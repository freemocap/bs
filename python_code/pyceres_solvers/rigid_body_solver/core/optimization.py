"""Bundle adjustment optimization - jointly optimize reference geometry AND poses."""

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def estimate_distance_matrix(
    *,
    noisy_data: np.ndarray,
    use_median: bool = True
) -> np.ndarray:
    """
    Estimate the true rigid body distance matrix from noisy trajectories.

    For each pair of markers, compute distances across all frames and take
    the median (or mean) to get the best estimate of the rigid distance.

    Args:
        noisy_data: (n_frames, n_markers, 3) measured positions
        use_median: If True, use median distance; if False, use mean

    Returns:
        distances: (n_markers, n_markers) estimated rigid distances
    """
    n_frames, n_markers, _ = noisy_data.shape
    distances = np.zeros((n_markers, n_markers))

    for i in range(n_markers):
        for j in range(i + 1, n_markers):
            # Compute distance in every frame
            frame_distances = np.linalg.norm(
                noisy_data[:, i, :] - noisy_data[:, j, :],
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    p_origin = reference_geometry[origin_indices].mean(axis=0)

    # Get direction markers
    p_x = reference_geometry[name_to_idx[x_axis_marker]]
    p_y = reference_geometry[name_to_idx[y_axis_marker]]

    # X-axis: points EXACTLY from origin to nose (no adjustment)
    x_axis = p_x - p_origin
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Y-axis: points generally towards left_ear, but adjusted to be perpendicular to X
    # Using Gram-Schmidt: remove the X component from the vector towards left_ear
    v_y = p_y - p_origin
    y_axis = v_y - np.dot(v_y, x_axis) * x_axis  # Project out X component
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Z-axis: perpendicular to both X and Y (Y × X makes it point up/dorsal)
    z_axis = np.cross(y_axis, x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Rotation matrix: rows are the basis vectors
    # When we multiply: basis @ point, we project the point onto these new axes
    basis = np.array([x_axis, y_axis, z_axis])

    # Transform all points to body-fixed frame:
    # 1. Center at origin (subtract head_center)
    centered = reference_geometry - p_origin
    # 2. Rotate so anatomical axes become standard axes (+X, +Y, +Z)
    transformed = (basis @ centered.T).T

    # Result: transformed geometry where:
    # - Origin is at (0, 0, 0) [head_center]
    # - +X points towards nose (forward/rostral)
    # - +Y points left (lateral)
    # - +Z points up (dorsal)

    return basis, transformed, p_origin


def estimate_reference_rigid(
    *,
    noisy_data: np.ndarray,
    marker_names: list[str],
    origin_markers: list[str],
    x_axis_marker: str,
    y_axis_marker: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate reference geometry using distance matrix + MDS + body frame alignment.

    This properly accounts for rigid body constraints by:
    1. Estimating true distances from all frames (median across time)
    2. Reconstructing geometry from distance matrix (MDS)
    3. Aligning to a meaningful body-fixed coordinate frame (Gram-Schmidt)

    Args:
        noisy_data: (n_frames, n_markers, 3) measured positions
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
    n_frames, n_markers, _ = noisy_data.shape

    logger.info(f"  Estimating distance matrix from {n_frames} frames...")
    distance_matrix = estimate_distance_matrix(noisy_data=noisy_data, use_median=True)

    # Report distance statistics
    rigid_distances = distance_matrix[np.triu_indices(n_markers, k=1)]
    logger.info(f"    Distance range: [{rigid_distances.min():.3f}, {rigid_distances.max():.3f}]")

    logger.info("  Reconstructing geometry from distances (Classical MDS)...")
    original_reference = reconstruct_from_distances(distance_matrix=distance_matrix, n_dims=3)

    logger.info(f"  Defining body frame:")
    logger.info(f"    Origin: mean of {origin_markers}")
    logger.info(f"    X-axis: towards '{x_axis_marker}'")
    logger.info(f"    Y-axis: towards '{y_axis_marker}'")

    basis, aligned_reference, origin_point = define_body_frame(
        reference_geometry=original_reference,
        marker_names=marker_names,
        origin_markers=origin_markers,
        x_axis_marker=x_axis_marker,
        y_axis_marker=y_axis_marker
    )

    logger.info(f"    Body frame basis vectors:")
    logger.info(f"      X: [{basis[0, 0]:7.4f}, {basis[0, 1]:7.4f}, {basis[0, 2]:7.4f}]")
    logger.info(f"      Y: [{basis[1, 0]:7.4f}, {basis[1, 1]:7.4f}, {basis[1, 2]:7.4f}]")
    logger.info(f"      Z: [{basis[2, 0]:7.4f}, {basis[2, 1]:7.4f}, {basis[2, 2]:7.4f}]")

    # Verify orthonormality
    dot_xy = np.dot(basis[0], basis[1])
    dot_xz = np.dot(basis[0], basis[2])
    dot_yz = np.dot(basis[1], basis[2])
    logger.info(f"    Orthogonality check (should be ~0):")
    logger.info(f"      X·Y = {dot_xy:.2e}, X·Z = {dot_xz:.2e}, Y·Z = {dot_yz:.2e}")

    # Verify transformation worked correctly
    nose_idx = marker_names.index(x_axis_marker)
    left_ear_idx = marker_names.index(y_axis_marker)
    nose_pos = aligned_reference[nose_idx]
    left_ear_pos = aligned_reference[left_ear_idx]

    logger.info(f"  Transformed reference geometry to body frame:")
    logger.info(f"    Origin is now at: [0, 0, 0]")
    logger.info(f"    {x_axis_marker} position: [{nose_pos[0]:.3f}, {nose_pos[1]:.3f}, {nose_pos[2]:.3f}] (should be at +X)")
    logger.info(f"    {y_axis_marker} position: [{left_ear_pos[0]:.3f}, {left_ear_pos[1]:.3f}, {left_ear_pos[2]:.3f}] (should be at +Y)")
    logger.info(f"    Using this transformed geometry for optimization...")

    return aligned_reference, original_reference, basis, origin_point


def plot_reference_geometry(
    transformed_geometry: np.ndarray,
    original_geometry: np.ndarray,
    basis: np.ndarray,
    origin_point: np.ndarray,
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
    ax1.scatter([origin_point[0]], [origin_point[1]], [origin_point[2]],
                c='yellow', s=300, marker='X', edgecolors='black', linewidth=2,
                label='Computed origin', zorder=11)

    # Plot calculated basis vectors FROM origin point
    scale = np.max(np.abs(original_geometry)) * 0.4
    ax1.quiver(origin_point[0], origin_point[1], origin_point[2],
               basis[0, 0] * scale, basis[0, 1] * scale, basis[0, 2] * scale,
               color='red', arrow_length_ratio=0.15, linewidth=3, label='X-basis')
    ax1.quiver(origin_point[0], origin_point[1], origin_point[2],
               basis[1, 0] * scale, basis[1, 1] * scale, basis[1, 2] * scale,
               color='green', arrow_length_ratio=0.15, linewidth=3, label='Y-basis')
    ax1.quiver(origin_point[0], origin_point[1], origin_point[2],
               basis[2, 0] * scale, basis[2, 1] * scale, basis[2, 2] * scale,
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
        points = transformed_geometry[[i, j]]
        ax2.plot(points[:, 0], points[:, 1], points[:, 2], 'gray', linewidth=1, alpha=0.4)

    # Plot markers
    ax2.scatter(transformed_geometry[:, 0], transformed_geometry[:, 1], transformed_geometry[:, 2],
                c='blue', s=100, edgecolors='black', linewidth=1)

    # Highlight frame-defining markers
    frame_pos_trans = transformed_geometry[frame_indices]
    ax2.scatter(frame_pos_trans[:, 0], frame_pos_trans[:, 1], frame_pos_trans[:, 2],
                c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                label='Frame-defining markers', zorder=10)

    # Label markers
    for i, (x, y, z) in enumerate(transformed_geometry):
        label = marker_names[i]
        if marker_names[i] in origin_markers:
            label += ' (origin)'
        elif marker_names[i] == x_axis_marker:
            label += ' (X)'
        elif marker_names[i] == y_axis_marker:
            label += ' (Y)'
        ax2.text(x, y, z, f'  {label}', fontsize=7)

    # Plot origin (now at 0,0,0)
    ax2.scatter([0], [0], [0], c='yellow', s=300, marker='X', edgecolors='black',
                linewidth=2, label='Origin (0,0,0)', zorder=11)

    # Plot standard unit vectors (in body frame)
    scale = np.max(np.abs(transformed_geometry)) * 0.4
    ax2.quiver(0, 0, 0, scale, 0, 0,
               color='red', arrow_length_ratio=0.15, linewidth=3, label='X (forward)')
    ax2.quiver(0, 0, 0, 0, scale, 0,
               color='green', arrow_length_ratio=0.15, linewidth=3, label='Y (left)')
    ax2.quiver(0, 0, 0, 0, 0, scale,
               color='blue', arrow_length_ratio=0.15, linewidth=3, label='Z (up)')

    # Formatting
    max_range = np.max(np.abs(transformed_geometry)) * 1.2
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


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    max_iter: int = 300
    lambda_data: float = 100.0
    lambda_rigid: float = 500.0
    lambda_rot_smooth: float = 200.0
    lambda_trans_smooth: float = 200.0
    function_tolerance: float = 1e-9
    gradient_tolerance: float = 1e-11
    parameter_tolerance: float = 1e-10


@dataclass
class OptimizationResult:
    """Results from optimization."""
    quaternions: np.ndarray  # (n_frames, 4) [w, x, y, z]
    rotations: np.ndarray  # (n_frames, 3, 3)
    translations: np.ndarray  # (n_frames, 3)
    reconstructed: np.ndarray  # (n_frames, n_markers, 3)
    reference_geometry: np.ndarray  # (n_markers, 3)
    initial_cost: float
    final_cost: float
    success: bool
    iterations: int
    time_seconds: float



# =============================================================================
# COST FUNCTIONS
# =============================================================================

class MeasurementFactorBA(pyceres.CostFunction):
    """Data fitting: measured point should match transformed reference."""

    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        marker_idx: int,
        n_markers: int,
        weight: float = 100.0
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.marker_idx = marker_idx
        self.n_markers = n_markers
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3, n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]
        reference_flat = parameters[2]

        start_idx = self.marker_idx * 3
        reference_point = reference_flat[start_idx:start_idx + 3]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        predicted = R @ reference_point + translation
        residuals[:] = self.weight * (self.measured_point - predicted)

        if jacobians is not None:
            eps = 1e-8

            if jacobians[0] is not None:
                for j in range(4):
                    quat_plus = quat.copy()
                    quat_plus[j] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    predicted_plus = R_plus @ reference_point + translation
                    residual_plus = self.weight * (self.measured_point - predicted_plus)
                    for i in range(3):
                        jacobians[0][i * 4 + j] = (residual_plus[i] - residuals[i]) / eps

            if jacobians[1] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[1][i * 3 + j] = -self.weight if i == j else 0.0

            if jacobians[2] is not None:
                jacobians[2][:] = 0.0
                start_idx = self.marker_idx * 3
                for i in range(3):
                    for j in range(3):
                        jacobians[2][i * (self.n_markers * 3) + (start_idx + j)] = -self.weight * R[i, j]

        return True


class RigidBodyFactorBA(pyceres.CostFunction):
    """Rigid body constraint: edge length in reference geometry."""

    def __init__(
        self,
        *,
        marker_i: int,
        marker_j: int,
        n_markers: int,
        target_distance: float,
        weight: float = 100.0
    ) -> None:
        super().__init__()
        self.marker_i = marker_i
        self.marker_j = marker_j
        self.n_markers = n_markers
        self.target_dist = target_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        reference_flat = parameters[0]

        ref_i = reference_flat[self.marker_i * 3:(self.marker_i + 1) * 3]
        ref_j = reference_flat[self.marker_j * 3:(self.marker_j + 1) * 3]

        diff = ref_i - ref_j
        current_dist = np.linalg.norm(diff)
        residuals[0] = self.weight * (current_dist - self.target_dist)

        if jacobians is not None and jacobians[0] is not None:
            eps = 1e-8
            jacobians[0][:] = 0.0

            for k in range(3):
                ref_flat_plus = reference_flat.copy()
                ref_flat_plus[self.marker_i * 3 + k] += eps
                ref_i_plus = ref_flat_plus[self.marker_i * 3:(self.marker_i + 1) * 3]
                ref_j_plus = ref_flat_plus[self.marker_j * 3:(self.marker_j + 1) * 3]
                diff_plus = ref_i_plus - ref_j_plus
                dist_plus = np.linalg.norm(diff_plus)
                residual_plus = self.weight * (dist_plus - self.target_dist)
                jacobians[0][self.marker_i * 3 + k] = (residual_plus - residuals[0]) / eps

            for k in range(3):
                ref_flat_plus = reference_flat.copy()
                ref_flat_plus[self.marker_j * 3 + k] += eps
                ref_i_plus = ref_flat_plus[self.marker_i * 3:(self.marker_i + 1) * 3]
                ref_j_plus = ref_flat_plus[self.marker_j * 3:(self.marker_j + 1) * 3]
                diff_plus = ref_i_plus - ref_j_plus
                dist_plus = np.linalg.norm(diff_plus)
                residual_plus = self.weight * (dist_plus - self.target_dist)
                jacobians[0][self.marker_j * 3 + k] = (residual_plus - residuals[0]) / eps

        return True


class SoftDistanceFactorBA(pyceres.CostFunction):
    """Soft distance constraint between measured point and reference point."""

    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        marker_idx_on_body: int,
        n_markers: int,
        median_distance: float,
        weight: float = 10.0
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.marker_idx = marker_idx_on_body
        self.n_markers = n_markers
        self.median_dist = median_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([4, 3, n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]
        reference_flat = parameters[2]

        start_idx = self.marker_idx * 3
        ref_point = reference_flat[start_idx:start_idx + 3]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        transformed_ref = R @ ref_point + translation
        diff = self.measured_point - transformed_ref
        current_dist = np.linalg.norm(diff)
        residuals[0] = self.weight * (current_dist - self.median_dist)

        if jacobians is not None:
            eps = 1e-8

            if jacobians[0] is not None:
                for j in range(4):
                    quat_plus = quat.copy()
                    quat_plus[j] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    transformed_plus = R_plus @ ref_point + translation
                    diff_plus = self.measured_point - transformed_plus
                    dist_plus = np.linalg.norm(diff_plus)
                    residual_plus = self.weight * (dist_plus - self.median_dist)
                    jacobians[0][j] = (residual_plus - residuals[0]) / eps

            if jacobians[1] is not None:
                if current_dist > 1e-10:
                    grad = -self.weight * diff / current_dist
                    jacobians[1][:] = grad
                else:
                    jacobians[1][:] = 0.0

            if jacobians[2] is not None:
                jacobians[2][:] = 0.0
                if current_dist > 1e-10:
                    start_idx = self.marker_idx * 3
                    grad_ref = self.weight * (diff / current_dist) @ R
                    for i in range(3):
                        jacobians[2][start_idx + i] = grad_ref[i]

        return True


class RotationSmoothnessFactor(pyceres.CostFunction):
    """Rotation smoothness between consecutive frames."""

    def __init__(self, *, weight: float = 500.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(4)
        self.set_parameter_block_sizes([4, 4])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat_t = parameters[0]
        quat_t1 = parameters[1]

        if np.dot(quat_t, quat_t1) < 0:
            quat_t1_corrected = -quat_t1.copy()
        else:
            quat_t1_corrected = quat_t1.copy()

        residuals[:] = self.weight * (quat_t1_corrected - quat_t)

        if jacobians is not None:
            if jacobians[0] is not None:
                for i in range(4):
                    for j in range(4):
                        jacobians[0][i * 4 + j] = -self.weight if i == j else 0.0
            if jacobians[1] is not None:
                sign = -1.0 if np.dot(quat_t, quat_t1) < 0 else 1.0
                for i in range(4):
                    for j in range(4):
                        jacobians[1][i * 4 + j] = self.weight * sign if i == j else 0.0

        return True


class TranslationSmoothnessFactor(pyceres.CostFunction):
    """Translation smoothness between consecutive frames."""

    def __init__(self, *, weight: float = 200.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        trans_t = parameters[0]
        trans_t1 = parameters[1]
        residuals[:] = self.weight * (trans_t1 - trans_t)

        if jacobians is not None:
            if jacobians[0] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[0][i * 3 + j] = -self.weight if i == j else 0.0
            if jacobians[1] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[1][i * 3 + j] = self.weight if i == j else 0.0

        return True


class ReferenceAnchorFactor(pyceres.CostFunction):
    """Soft anchor to prevent reference from drifting too far."""

    def __init__(self, *, initial_reference: np.ndarray, weight: float = 10.0) -> None:
        super().__init__()
        self.initial_ref = initial_reference.copy()
        self.weight = weight
        n_params = len(initial_reference)
        self.set_num_residuals(n_params)
        self.set_parameter_block_sizes([n_params])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        reference = parameters[0]
        residuals[:] = self.weight * (reference - self.initial_ref)

        if jacobians is not None and jacobians[0] is not None:
            n = len(residuals)
            for i in range(n):
                for j in range(n):
                    jacobians[0][i * n + j] = self.weight if i == j else 0.0

        return True


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def optimize_rigid_body(
    *,
    noisy_data: np.ndarray,
    rigid_edges: list[tuple[int, int]],
    reference_distances: np.ndarray,
    config: OptimizationConfig,
    marker_names: list[str] | None = None,
    display_edges: list[tuple[int, int]] | None = None,
    soft_edges: list[tuple[int, int]] | None = None,
    soft_distances: np.ndarray | None = None,
    lambda_soft: float = 10.0,
    body_frame_origin_markers: list[str] | None = None,
    body_frame_x_axis_marker: str | None = None,
    body_frame_y_axis_marker: str | None = None
) -> OptimizationResult:
    """
    Bundle adjustment: jointly optimize reference geometry AND poses.

    Args:
        noisy_data: (n_frames, n_markers, 3) measured positions
        rigid_edges: List of (i, j) pairs that should remain rigid
        reference_distances: (n_markers, n_markers) initial distance estimates
        config: OptimizationConfig
        marker_names: Marker names for visualization
        display_edges: Edges to show in visualization (defaults to rigid_edges)
        soft_edges: Optional list of soft edges
        soft_distances: Optional soft edge distances
        lambda_soft: Weight for soft constraints
        body_frame_origin_markers: Marker names whose mean defines the origin
        body_frame_x_axis_marker: Marker name that defines X-axis direction
        body_frame_y_axis_marker: Marker name that defines Y-axis direction

    Returns:
        OptimizationResult with optimized reference geometry and poses
    """
    n_frames, n_markers, _ = noisy_data.shape

    # Set defaults
    if marker_names is None:
        marker_names = [f"M{i}" for i in range(n_markers)]
    if display_edges is None:
        display_edges = rigid_edges
    if body_frame_origin_markers is None:
        body_frame_origin_markers = [marker_names[0]]
    if body_frame_x_axis_marker is None:
        body_frame_x_axis_marker = marker_names[1] if len(marker_names) > 1 else marker_names[0]
    if body_frame_y_axis_marker is None:
        body_frame_y_axis_marker = marker_names[2] if len(marker_names) > 2 else marker_names[0]

    logger.info("="*80)
    logger.info("BUNDLE ADJUSTMENT OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Frames: {n_frames}, Markers: {n_markers}, Rigid edges: {len(rigid_edges)}")

    # =========================================================================
    # INITIALIZE REFERENCE GEOMETRY USING DISTANCE MATRIX + MDS
    # =========================================================================
    logger.info("\nInitializing reference geometry from rigid distance matrix...")
    reference_geometry, original_geometry, basis, origin_point = estimate_reference_rigid(
        noisy_data=noisy_data,
        marker_names=marker_names,
        origin_markers=body_frame_origin_markers,
        x_axis_marker=body_frame_x_axis_marker,
        y_axis_marker=body_frame_y_axis_marker
    )
    # reference_geometry is now in body-fixed frame:
    # - Origin at (0,0,0) [head_center]
    # - +X towards nose, +Y towards left, +Z up
    # This transformed geometry is used for ALL optimization below
    reference_params = reference_geometry.flatten().copy()

    # Plot the estimated reference geometry
    logger.info("\nPlotting reference geometry (close window to continue)...")
    plot_reference_geometry(
        transformed_geometry=reference_geometry,
        original_geometry=original_geometry,
        basis=basis,
        origin_point=origin_point,
        marker_names=marker_names,
        display_edges=display_edges,
        origin_markers=body_frame_origin_markers,
        x_axis_marker=body_frame_x_axis_marker,
        y_axis_marker=body_frame_y_axis_marker
    )

    # =========================================================================
    # INITIALIZE POSES
    # =========================================================================
    logger.info("Initializing poses...")
    poses: list[tuple[np.ndarray, np.ndarray]] = []

    for frame_idx in range(n_frames):
        quat_ceres = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        translation = np.mean(noisy_data[frame_idx], axis=0)
        poses.append((quat_ceres, translation))

    # =========================================================================
    # BUILD OPTIMIZATION PROBLEM
    # =========================================================================
    logger.info("\nBuilding optimization problem...")
    problem = pyceres.Problem()

    # Add reference geometry as parameters
    problem.add_parameter_block(reference_params, n_markers * 3)

    # Add pose parameters
    pose_params: list[tuple[np.ndarray, np.ndarray]] = []
    for quat, trans in poses:
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(trans, 3)
        problem.set_manifold(quat, pyceres.QuaternionManifold())
        pose_params.append((quat, trans))

    # DATA FITTING
    logger.info("  Adding measurement factors...")
    for frame_idx in range(n_frames):
        quat, trans = pose_params[frame_idx]
        for point_idx in range(n_markers):
            cost = MeasurementFactorBA(
                measured_point=noisy_data[frame_idx, point_idx],
                marker_idx=point_idx,
                n_markers=n_markers,
                weight=config.lambda_data
            )
            problem.add_residual_block(cost, None, [quat, trans, reference_params])

    # RIGID BODY CONSTRAINTS
    logger.info(f"  Adding {len(rigid_edges)} rigid body constraints...")
    for i, j in rigid_edges:
        cost = RigidBodyFactorBA(
            marker_i=i,
            marker_j=j,
            n_markers=n_markers,
            target_distance=reference_distances[i, j],
            weight=config.lambda_rigid
        )
        problem.add_residual_block(cost, None, [reference_params])

    # SOFT EDGES
    if soft_edges is not None and soft_distances is not None:
        logger.info(f"  Adding {len(soft_edges)} soft constraints...")
        for frame_idx in range(n_frames):
            quat, trans = pose_params[frame_idx]
            for i, j in soft_edges:
                cost = SoftDistanceFactorBA(
                    measured_point=noisy_data[frame_idx, j],
                    marker_idx_on_body=i,
                    n_markers=n_markers,
                    median_distance=soft_distances[i, j],
                    weight=lambda_soft
                )
                problem.add_residual_block(cost, None, [quat, trans, reference_params])

    # SMOOTHNESS
    logger.info("  Adding smoothness factors...")
    for frame_idx in range(n_frames - 1):
        quat_t, trans_t = pose_params[frame_idx]
        quat_t1, trans_t1 = pose_params[frame_idx + 1]

        rot_cost = RotationSmoothnessFactor(weight=config.lambda_rot_smooth)
        problem.add_residual_block(rot_cost, None, [quat_t, quat_t1])

        trans_cost = TranslationSmoothnessFactor(weight=config.lambda_trans_smooth)
        problem.add_residual_block(trans_cost, None, [trans_t, trans_t1])

    # REFERENCE ANCHOR
    logger.info("  Adding reference anchor...")
    anchor_cost = ReferenceAnchorFactor(
        initial_reference=reference_params,
        weight=config.lambda_data * 0.1
    )
    problem.add_residual_block(anchor_cost, None, [reference_params])

    logger.info(f"\n  Total residual blocks: {problem.num_residual_blocks()}")
    logger.info(f"  Total parameters: {problem.num_parameters()}")

    # =========================================================================
    # SOLVE
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("SOLVING")
    logger.info("="*80)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = config.max_iter
    options.function_tolerance = config.function_tolerance
    options.gradient_tolerance = config.gradient_tolerance
    options.parameter_tolerance = config.parameter_tolerance

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"  Status: {summary.termination_type}")
    logger.info(f"  Initial cost: {summary.initial_cost:.2f}")
    logger.info(f"  Final cost: {summary.final_cost:.2f}")
    logger.info(f"  Iterations: {summary.num_successful_steps}")
    logger.info(f"  Time: {summary.total_time_in_seconds:.2f}s")

    # =========================================================================
    # EXTRACT RESULTS
    # =========================================================================
    optimized_reference = reference_params.reshape(n_markers, 3)

    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))
    reconstructed = np.zeros((n_frames, n_markers, 3))
    quaternions = np.zeros((n_frames, 4))

    for frame_idx in range(n_frames):
        quat_ceres = pose_params[frame_idx][0]
        trans = pose_params[frame_idx][1]

        quat_scipy = np.array([quat_ceres[1], quat_ceres[2], quat_ceres[3], quat_ceres[0]])
        R = Rotation.from_quat(quat_scipy).as_matrix()

        quaternions[frame_idx] = quat_scipy
        rotations[frame_idx] = R
        translations[frame_idx] = trans
        reconstructed[frame_idx] = (R @ optimized_reference.T).T + trans

    # Note: All results are in body-fixed frame:
    # - reference_geometry: head geometry with origin at head_center, +X=forward, +Y=left, +Z=up
    # - rotations/translations: body pose relative to this frame
    # - reconstructed: markers in world coordinates (= R @ reference + t)

    return OptimizationResult(
        quaternions=np.array([pose_params[i][0] for i in range(n_frames)]),
        rotations=rotations,
        translations=translations,
        reconstructed=reconstructed,
        reference_geometry=optimized_reference,
        initial_cost=summary.initial_cost,
        final_cost=summary.final_cost,
        success=summary.termination_type == pyceres.TerminationType.CONVERGENCE,
        iterations=summary.num_successful_steps,
        time_seconds=summary.total_time_in_seconds
    )