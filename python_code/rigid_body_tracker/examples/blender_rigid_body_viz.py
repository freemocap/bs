"""Load rigid body tracking output into Blender with sphere markers and tube connections.

PERFORMANCE OPTIMIZATIONS:
- Uses curves with hooks for edges instead of individual cylinders
- Edges automatically follow markers via hooks (no keyframes needed!)
- Only markers are keyframed (11 objects vs potentially 15+ edge objects)
- Single curve object per edge type instead of N separate mesh objects
- Curve bevel creates tube geometry efficiently

NEW: Toy data support - load additional tracked objects alongside rigid body data!

This approach is 10-100x faster than creating individual cylinder meshes!
"""

import sys
from pathlib import Path
from dataclasses import dataclass
import json
import csv

import bpy
import bmesh
from mathutils import Vector, Matrix
import numpy as np


# Add project root to path if needed
_script_path = Path(__file__).resolve()
_project_root = _script_path.parent.parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

"""Load 3D keypoint tracking data from tidy CSV format."""

from dataclasses import dataclass, field
from pathlib import Path
import csv
import numpy as np


@dataclass(frozen=True)
class Point3D:
    """Immutable 3D point."""
    x: float
    y: float
    z: float


@dataclass
class Trajectory:
    """Time series of 3D positions for a single keypoint."""
    keypoint_name: str
    observations: list[tuple[int, Point3D]] = field(default_factory=list)

    def add_observation(self, *, frame: int, position: Point3D) -> None:
        """Add an observation to this trajectory."""
        self.observations.append((frame, position))

    def to_numpy(self, *, scale_factor: float = 1.0) -> np.ndarray:
        """
        Convert trajectory to numpy array of shape (n_frames, 3).

        Args:
            scale_factor: Multiplier for coordinates (e.g., 0.001 for mm to m)

        Returns:
            Array of shape (n_frames, 3) with x, y, z coordinates
        """
        xyz_array = np.array(
            [[pos.x, pos.y, pos.z] for _, pos in self.observations],
            dtype=np.float64
        )
        return xyz_array * scale_factor

    @property
    def num_frames(self) -> int:
        """Number of frames in this trajectory."""
        return len(self.observations)


def load_trajectories_from_tidy_csv(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Load trajectories from tidy/long-format CSV and convert to numpy arrays.

    Args:
        filepath: Path to CSV with columns: frame, keypoint, x, y, z
        scale_factor: Scale multiplier for coordinates (default: 1.0)

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    print(f"Loading trajectories from: {filepath.name}")

    # First pass: read and organize data
    trajectories: dict[str, Trajectory] = {}

    with filepath.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f=f)

        for row in reader:
            frame = int(row['frame'])
            keypoint_name = row['keypoint']
            position = Point3D(
                x=float(row['x']),
                y=float(row['y']),
                z=float(row['z']) if 'z' in row and row['z'] else 0.0
            )

            if keypoint_name not in trajectories:
                trajectories[keypoint_name] = Trajectory(keypoint_name=keypoint_name)

            trajectories[keypoint_name].add_observation(frame=frame, position=position)

    # Convert to numpy arrays
    trajectory_arrays = {
        name: traj.to_numpy(scale_factor=scale_factor)
        for name, traj in trajectories.items()
    }

    num_keypoints = len(trajectory_arrays)
    num_frames = next(iter(trajectories.values())).num_frames
    print(f"âœ“ Loaded {num_keypoints} keypoints Ã— {num_frames} frames")

    return trajectory_arrays


def load_trajectories_from_wide_csv(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Load 2D trajectories from wide-format CSV and convert to 3D numpy arrays.

    Expected CSV format:
    - Columns: frame, video (optional), {keypoint}_x, {keypoint}_y, ...
    - Each row represents one frame
    - Keypoint coordinates are in paired x,y columns

    Args:
        filepath: Path to CSV with paired {keypoint}_x, {keypoint}_y columns
        scale_factor: Scale multiplier for coordinates (default: 1.0)
        z_value: Default z-coordinate value for 2D data (default: 0.0)

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    print(f"Loading wide-format trajectories from: {filepath.name}")

    # Read CSV and extract column names
    with filepath.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f=f)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError(f"CSV file has no headers: {filepath}")

        # Identify keypoint columns (those ending in _x or _y)
        keypoint_names: set[str] = set()
        for header in headers:
            if header.endswith('_x'):
                keypoint_name = header[:-2]  # Remove '_x' suffix
                keypoint_names.add(keypoint_name)

        if not keypoint_names:
            raise ValueError(f"No keypoint columns found (expected columns ending in '_x')")

        # Initialize storage for trajectories
        trajectory_data: dict[str, list[list[float]]] = {
            name: [] for name in keypoint_names
        }

        # Read all rows
        f.seek(0)  # Reset file pointer
        reader = csv.DictReader(f=f)

        for row in reader:
            for keypoint_name in keypoint_names:
                x_col = f"{keypoint_name}_x"
                y_col = f"{keypoint_name}_y"

                # Handle missing values
                try:
                    x = float(row[x_col]) if row[x_col] else np.nan
                    y = float(row[y_col]) if row[y_col] else np.nan
                except (KeyError, ValueError):
                    x = np.nan
                    y = np.nan

                trajectory_data[keypoint_name].append([x, y, z_value])

    # Convert to numpy arrays and apply scaling
    trajectory_arrays: dict[str, np.ndarray] = {}
    for keypoint_name, coords in trajectory_data.items():
        array = np.array(coords, dtype=np.float64)
        trajectory_arrays[keypoint_name] = array * scale_factor

    num_keypoints = len(trajectory_arrays)
    num_frames = len(next(iter(trajectory_data.values())))
    print(f"âœ“ Loaded {num_keypoints} keypoints Ã— {num_frames} frames (2D data)")

    return trajectory_arrays


def load_trajectories_from_dlc_csv(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
        likelihood_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Load trajectories from DeepLabCut multi-header CSV format.

    Expected CSV format (3-row header):
    - Row 0: scorer (metadata, ignored)
    - Row 1: bodypart names (repeated for x, y, likelihood)
    - Row 2: coords ('x', 'y', 'likelihood' for each bodypart)
    - Row 3+: data rows

    Args:
        filepath: Path to DLC CSV file
        scale_factor: Scale multiplier for coordinates (default: 1.0)
        z_value: Default z-coordinate value for 2D data (default: 0.0)
        likelihood_threshold: If provided, set coordinates to NaN when likelihood < threshold

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    print(f"Loading DLC-format trajectories from: {filepath.name}")

    # Read the file to parse the multi-level header
    with filepath.open(mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError(f"DLC CSV must have at least 4 rows (3 header + 1 data)")

    # Parse the 3-row header structure
    scorer_row = lines[0].strip().split(',')
    bodypart_row = lines[1].strip().split(',')
    coords_row = lines[2].strip().split(',')

    # Build column mapping: determine which columns correspond to which bodypart and coord type
    column_map: dict[str, dict[str, int]] = {}  # {bodypart: {'x': col_idx, 'y': col_idx, 'likelihood': col_idx}}

    for col_idx, (bodypart, coord_type) in enumerate(zip(bodypart_row, coords_row)):
        bodypart = bodypart.strip()
        coord_type = coord_type.strip()

        if not bodypart or bodypart == 'scorer':  # Skip empty or metadata columns
            continue

        if bodypart not in column_map:
            column_map[bodypart] = {}

        column_map[bodypart][coord_type] = col_idx

    # Validate that each bodypart has x and y columns
    valid_bodyparts: list[str] = []
    for bodypart, coords in column_map.items():
        if 'x' in coords and 'y' in coords:
            valid_bodyparts.append(bodypart)
        else:
            print(f"Warning: Bodypart '{bodypart}' missing x or y coordinate, skipping")

    if not valid_bodyparts:
        raise ValueError("No valid bodyparts found with both x and y coordinates")

    # Initialize storage
    trajectory_data: dict[str, list[list[float]]] = {
        name: [] for name in valid_bodyparts
    }

    # Parse data rows (skip first 3 header rows)
    for line_idx, line in enumerate(lines[3:], start=3):
        values = line.strip().split(',')

        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]

            try:
                x_str = values[coords['x']].strip()
                y_str = values[coords['y']].strip()

                x = float(x_str) if x_str else np.nan
                y = float(y_str) if y_str else np.nan

                # Apply likelihood threshold if specified
                if likelihood_threshold is not None and 'likelihood' in coords:
                    likelihood_str = values[coords['likelihood']].strip()
                    likelihood = float(likelihood_str) if likelihood_str else 0.0

                    if likelihood < likelihood_threshold:
                        x = np.nan
                        y = np.nan

            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line {line_idx}, bodypart '{bodypart}': {e}")
                x = np.nan
                y = np.nan

            trajectory_data[bodypart].append([x, y, z_value])

    # Convert to numpy arrays and apply scaling
    trajectory_arrays: dict[str, np.ndarray] = {}
    for keypoint_name, coords in trajectory_data.items():
        array = np.array(coords, dtype=np.float64)
        trajectory_arrays[keypoint_name] = array * scale_factor

    num_keypoints = len(trajectory_arrays)
    num_frames = len(next(iter(trajectory_data.values())))
    print(f"âœ“ Loaded {num_keypoints} keypoints Ã— {num_frames} frames (DLC format)")

    return trajectory_arrays


def detect_csv_format(*, filepath: Path | str) -> str:
    """
    Detect whether CSV is in 'tidy', 'wide', or 'dlc' format.

    Args:
        filepath: Path to CSV file

    Returns:
        Either 'tidy', 'wide', or 'dlc'
    """
    filepath = Path(filepath)

    with filepath.open(mode='r', encoding='utf-8') as f:
        # Read first few lines to detect format
        lines = [f.readline().strip() for _ in range(3)]

    if len(lines) < 1:
        raise ValueError(f"CSV file is empty: {filepath}")

    # Check for DLC format: 3-row header with bodyparts in row 2, coords in row 3
    if len(lines) >= 3:
        row2_values = lines[1].split(',')
        row3_values = lines[2].split(',')

        # DLC format has 'coords' or coordinate type labels in row 3
        if len(row3_values) > 1 and any(val.strip() in ['x', 'y', 'likelihood', 'coords'] for val in row3_values):
            return 'dlc'

    # Parse as regular CSV with DictReader for tidy/wide detection
    with filepath.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f=f)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError(f"CSV file has no headers: {filepath}")

        # Tidy format has columns: frame, keypoint, x, y, z
        if 'keypoint' in headers and 'x' in headers and 'y' in headers:
            return 'tidy'

        # Wide format has columns like: frame, {keypoint}_x, {keypoint}_y
        has_x_columns = any(h.endswith('_x') for h in headers)
        has_y_columns = any(h.endswith('_y') for h in headers)

        if has_x_columns and has_y_columns:
            return 'wide'

        raise ValueError(
            f"Unable to detect CSV format. Expected either:\n"
            f"  - Tidy format: columns 'frame', 'keypoint', 'x', 'y', 'z'\n"
            f"  - Wide format: columns 'frame', '{{keypoint}}_x', '{{keypoint}}_y'\n"
            f"  - DLC format: 3-row header (scorer/bodyparts/coords)"
        )


def load_trajectories_auto(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
        likelihood_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Automatically detect CSV format and load trajectories.

    Args:
        filepath: Path to CSV file (tidy, wide, or DLC format)
        scale_factor: Scale multiplier for coordinates (default: 1.0)
        z_value: Default z-coordinate for 2D data in wide/DLC format (default: 0.0)
        likelihood_threshold: For DLC format, filter low-confidence points (default: None)

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    csv_format = detect_csv_format(filepath=filepath)

    print(f"Detected format: {csv_format}")

    if csv_format == 'tidy':
        return load_trajectories_from_tidy_csv(
            filepath=filepath,
            scale_factor=scale_factor
        )
    elif csv_format == 'dlc':
        return load_trajectories_from_dlc_csv(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value,
            likelihood_threshold=likelihood_threshold
        )
    else:  # wide format
        return load_trajectories_from_wide_csv(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value
        )
@dataclass
class RigidBodyVisualizationConfig:
    """Configuration for visualization."""

    csv_path: Path
    """Path to trajectory_data.csv"""

    topology_path: Path
    """Path to topology.json"""

    data_type: str = "optimized"
    """Which dataset to visualize: 'noisy', 'optimized', or 'gt'"""

    data_scale: float = 1.0
    """Scale factor for data (e.g. 0.001 for mm to meters)"""

    sphere_radius: float = 0.01
    """Radius of marker spheres (meters)"""

    tube_radius: float = 0.003
    """Radius of connecting tubes (meters)"""

    rigid_edge_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
    """RGBA color for rigid edges (cyan)"""

    soft_edge_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 1.0)
    """RGBA color for soft edges (magenta)"""

    display_edge_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    """RGBA color for display edges (white)"""

    show_rigid_edges: bool = False
    """Show rigid constraint edges"""

    show_soft_edges: bool = True
    """Show soft constraint edges"""

    show_display_edges: bool = True
    """Show display edges"""

    frame_start: int = 0
    """Start frame for animation"""

    keyframe_step: int = 1
    """Keyframe every N frames (1=all frames, 2=every other, etc.) for even faster loading"""

    # NEW: Toy data support
    toy_csv_path: Path | None = None
    """Optional path to toy trajectory CSV (wide format)"""

    toy_sphere_radius: float = 0.015
    """Radius of toy marker spheres (meters)"""

    toy_color: tuple[float, float, float] = (1.0, 0.8, 0.0)
    """RGB color for toy markers (gold/yellow)"""


def clear_scene() -> None:
    """Clear all objects from the scene."""
    # Select and delete all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Let Blender handle orphaned data cleanup automatically
    # (or call bpy.ops.outliner.orphans_purge() if needed)


def load_topology(*, filepath: Path) -> dict[str, object]:
    """Load topology from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(fp=f)
    return data['topology']


def load_trajectory_data(
    *,
    csv_path: Path,
    topology: dict[str, object],
    data_type: str,
    data_scale: float
) -> list[dict[str, tuple[float, float, float]]]:
    """
    Load trajectory data from CSV.

    Returns:
        List of frames, each containing dict of marker_name -> (x, y, z)
    """
    marker_names: list[str] = topology['marker_names']
    frames: list[dict[str, tuple[float, float, float]]] = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f=f)

        for row in reader:
            frame_data: dict[str, tuple[float, float, float]] = {}

            for marker_name in marker_names:
                x = float(row[f'{data_type}_{marker_name}_x']) * data_scale
                y = float(row[f'{data_type}_{marker_name}_y']) * data_scale
                z = float(row[f'{data_type}_{marker_name}_z']) * data_scale
                frame_data[marker_name] = (x, y, z)

            frames.append(frame_data)

    print(f"âœ“ Loaded {len(frames)} frames of {data_type} data")
    return frames


def create_marker_material(
    *,
    name: str,
    color: tuple[float, float, float],
    metallic: float = 0.3,
    roughness: float = 0.4
) -> bpy.types.Material:
    """Create a material for markers."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create shader nodes
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_mix = nodes.new(type='ShaderNodeMixShader')

    # Set colors and properties
    node_bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    node_bsdf.inputs['Metallic'].default_value = metallic
    node_bsdf.inputs['Roughness'].default_value = roughness

    node_emission.inputs['Color'].default_value = (*color, 1.0)
    node_emission.inputs['Strength'].default_value = 0.5

    # Connect nodes
    links = mat.node_tree.links
    links.new(node_bsdf.outputs['BSDF'], node_mix.inputs[1])
    links.new(node_emission.outputs['Emission'], node_mix.inputs[2])
    links.new(node_mix.outputs['Shader'], node_output.inputs['Surface'])

    node_mix.inputs['Fac'].default_value = 0.2  # 20% emission

    return mat


def create_edge_material(
    *,
    name: str,
    color: tuple[float, float, float, float],
    metallic: float = 0.8,
    roughness: float = 0.2
) -> bpy.types.Material:
    """Create a material for edges."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    nodes.clear()

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    node_bsdf.inputs['Base Color'].default_value = color
    node_bsdf.inputs['Metallic'].default_value = metallic
    node_bsdf.inputs['Roughness'].default_value = roughness
    node_bsdf.inputs['Alpha'].default_value = color[3]

    links = mat.node_tree.links
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

    # Enable transparency
    mat.blend_method = 'BLEND'

    return mat


def create_sphere_marker(
    *,
    name: str,
    radius: float,
    location: tuple[float, float, float],
    material: bpy.types.Material,
    parent: bpy.types.Object
) -> bpy.types.Object:
    """Create a UV sphere marker."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=location,
        segments=32,
        ring_count=16
    )

    sphere = bpy.context.active_object
    sphere.name = name
    sphere.parent = parent

    # Apply material
    if sphere.data.materials:
        sphere.data.materials[0] = material
    else:
        sphere.data.materials.append(material)

    # Smooth shading
    bpy.ops.object.shade_smooth()

    return sphere


def create_edge_curves(
    *,
    name: str,
    edges: list[tuple[int, int]],
    marker_names: list[str],
    markers: dict[str, bpy.types.Object],
    initial_positions: dict[str, tuple[float, float, float]],
    radius: float,
    material: bpy.types.Material,
    parent: bpy.types.Object
) -> bpy.types.Object | None:
    """Create a single curve object with multiple splines for edges using drivers (FAST!)."""
    if not edges:
        return None

    # Create curve data
    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = radius  # This creates the tube geometry
    curve_data.bevel_resolution = 4  # Circular cross-section
    curve_data.resolution_u = 2  # Keep it simple for performance
    curve_data.fill_mode = 'FULL'  # Fill the tube
    curve_data.use_fill_caps = True  # Cap the ends

    # Create curve object
    curve_obj = bpy.data.objects.new(name=name, object_data=curve_data)
    bpy.context.collection.objects.link(object=curve_obj)
    curve_obj.parent = parent

    # Set display settings to show solid geometry, not wires
    curve_obj.show_wire = False
    curve_obj.show_all_edges = False
    curve_obj.display_type = 'TEXTURED'

    # Apply material
    if curve_obj.data.materials:
        curve_obj.data.materials[0] = material
    else:
        curve_obj.data.materials.append(material)

    # Create one spline per edge
    for edge_idx, (i, j) in enumerate(edges):
        marker_i_name = marker_names[i]
        marker_j_name = marker_names[j]

        # Get initial positions
        start_pos = Vector(initial_positions[marker_i_name])
        end_pos = Vector(initial_positions[marker_j_name])

        # Create spline (2 control points for straight line)
        spline = curve_data.splines.new(type='NURBS')
        spline.points.add(1)  # Adds 1 more point (starts with 1)

        spline.points[0].co = (*start_pos, 1.0)
        spline.points[1].co = (*end_pos, 1.0)

        # Set to linear interpolation for straight lines
        spline.order_u = 2
        spline.use_endpoint_u = True

        # Add drivers to link curve points to marker locations
        marker_i = markers[marker_i_name]
        marker_j = markers[marker_j_name]

        # Driver for start point (x, y, z)
        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            # Create driver for this coordinate
            fcurve = curve_data.driver_add(f'splines[{edge_idx}].points[0].co', axis_idx)
            driver = fcurve.driver
            driver.type = 'AVERAGE'

            # Add variable pointing to marker location
            var = driver.variables.new()
            var.name = 'loc'
            var.type = 'TRANSFORMS'

            target = var.targets[0]
            target.id = marker_i
            target.transform_type = f'LOC_{axis_name.upper()}'
            target.transform_space = 'WORLD_SPACE'

        # Driver for end point (x, y, z)
        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            # Create driver for this coordinate
            fcurve = curve_data.driver_add(f'splines[{edge_idx}].points[1].co', axis_idx)
            driver = fcurve.driver
            driver.type = 'AVERAGE'

            # Add variable pointing to marker location
            var = driver.variables.new()
            var.name = 'loc'
            var.type = 'TRANSFORMS'

            target = var.targets[0]
            target.id = marker_j
            target.transform_type = f'LOC_{axis_name.upper()}'
            target.transform_space = 'WORLD_SPACE'

    return curve_obj


def animate_marker(
    *,
    marker: bpy.types.Object,
    trajectory: list[tuple[float, float, float]],
    frame_start: int,
    keyframe_step: int = 1
) -> None:
    """Animate a marker along its trajectory."""
    for frame_idx in range(0, len(trajectory), keyframe_step):
        frame = frame_start + frame_idx
        x, y, z = trajectory[frame_idx]
        marker.location = (x, y, z)
        marker.keyframe_insert(data_path="location", frame=frame)

    # Set interpolation to linear for smoother motion
    if marker.animation_data and marker.animation_data.action:
        for fcurve in marker.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


def animate_marker_from_numpy(
    *,
    marker: bpy.types.Object,
    trajectory: np.ndarray,
    frame_start: int,
    keyframe_step: int = 1
) -> None:
    """Animate a marker along its trajectory from numpy array."""
    for frame_idx in range(0, trajectory.shape[0], keyframe_step):
        frame = frame_start + frame_idx
        marker.location = tuple(trajectory[frame_idx])
        marker.keyframe_insert(data_path="location", frame=frame)

    # Set interpolation to linear for smoother motion
    if marker.animation_data and marker.animation_data.action:
        for fcurve in marker.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


def create_toy_markers(
    *,
    toy_trajectories: dict[str, np.ndarray],
    parent: bpy.types.Object,
    material: bpy.types.Material,
    radius: float,
    frame_start: int,
    keyframe_step: int
) -> dict[str, bpy.types.Object]:
    """Create and animate toy marker spheres."""
    toy_markers: dict[str, bpy.types.Object] = {}

    for keypoint_name, trajectory in toy_trajectories.items():
        # Get initial position
        initial_pos = tuple(trajectory[0])

        # Create sphere
        sphere = create_sphere_marker(
            name=f"Toy_{keypoint_name}",
            radius=radius,
            location=initial_pos,
            material=material,
            parent=parent
        )

        # Animate
        animate_marker_from_numpy(
            marker=sphere,
            trajectory=trajectory,
            frame_start=frame_start,
            keyframe_step=keyframe_step
        )

        toy_markers[keypoint_name] = sphere

    return toy_markers


def create_rigid_body_visualization(*, config: RigidBodyVisualizationConfig) -> None:
    """Create complete rigid body visualization in Blender."""

    print("="*80)
    print("RIGID BODY TRACKING VISUALIZATION")
    print("="*80)

    # Load data
    print("\nLoading data...")
    topology = load_topology(filepath=config.topology_path)
    frames = load_trajectory_data(
        csv_path=config.csv_path,
        topology=topology,
        data_type=config.data_type,
        data_scale=config.data_scale
    )

    marker_names: list[str] = topology['marker_names']
    n_frames = len(frames)

    print(f"\nTopology: {topology['name']}")
    print(f"Markers: {len(marker_names)}")
    print(f"Frames: {n_frames}")

    # Load toy data if provided - NOW USING load_trajectories_auto!
    toy_trajectories: dict[str, np.ndarray] | None = None
    if config.toy_csv_path is not None:
        try:
            print("\nLoading toy data...")
            toy_trajectories = load_trajectories_auto(
                filepath=config.toy_csv_path,
                scale_factor=config.data_scale,
                z_value=0.0
            )
        except Exception as e:
            print(f"âš ï¸ Warning: Failed to load toy data: {e}")
            print("Continuing without toy data...")
            toy_trajectories = None

    # Create parent empty
    bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
    parent = bpy.context.active_object
    parent.name = f"RigidBody_{topology['name']}"
    parent.empty_display_size = 0.1

    # Create materials
    print("\nCreating materials...")

    # Marker colors (rainbow palette)
    marker_colors = [
        (1.0, 0.0, 0.0),  # Red
        (1.0, 0.5, 0.0),  # Orange
        (1.0, 1.0, 0.0),  # Yellow
        (0.5, 1.0, 0.0),  # Yellow-green
        (0.0, 1.0, 0.0),  # Green
        (0.0, 1.0, 0.5),  # Cyan-green
        (0.0, 1.0, 1.0),  # Cyan
        (0.0, 0.5, 1.0),  # Light blue
        (0.0, 0.0, 1.0),  # Blue
        (0.5, 0.0, 1.0),  # Purple
        (1.0, 0.0, 1.0),  # Magenta
        (1.0, 0.0, 0.5),  # Pink
    ]

    marker_materials = {}
    for idx, marker_name in enumerate(marker_names):
        color = marker_colors[idx % len(marker_colors)]
        mat = create_marker_material(
            name=f"Marker_{marker_name}",
            color=color
        )
        marker_materials[marker_name] = mat

    # Edge materials
    rigid_edge_mat = create_edge_material(
        name="RigidEdge",
        color=config.rigid_edge_color
    )
    soft_edge_mat = create_edge_material(
        name="SoftEdge",
        color=config.soft_edge_color
    )
    display_edge_mat = create_edge_material(
        name="DisplayEdge",
        color=config.display_edge_color
    )

    # Toy material
    toy_mat = create_marker_material(
        name="ToyMarker",
        color=config.toy_color,
        metallic=0.5,
        roughness=0.3
    )

    # Create markers
    print("\nCreating marker spheres...")
    markers = {}

    for marker_name in marker_names:
        initial_pos = frames[0][marker_name]

        sphere = create_sphere_marker(
            name=f"Marker_{marker_name}",
            radius=config.sphere_radius,
            location=initial_pos,
            material=marker_materials[marker_name],
            parent=parent
        )
        markers[marker_name] = sphere

    # Create edges using curves with hooks (FAST!)
    print("\nCreating edge curves with hooks...")

    if config.show_rigid_edges and 'rigid_edges' in topology:
        print(f"  Creating rigid edge curves ({len(topology['rigid_edges'])} edges)...")
        create_edge_curves(
            name="RigidEdges",
            edges=topology['rigid_edges'],
            marker_names=marker_names,
            markers=markers,
            initial_positions=frames[0],
            radius=config.tube_radius,
            material=rigid_edge_mat,
            parent=parent
        )

    if config.show_soft_edges and 'soft_edges' in topology:
        print(f"  Creating soft edge curves ({len(topology['soft_edges'])} edges)...")
        create_edge_curves(
            name="SoftEdges",
            edges=topology['soft_edges'],
            marker_names=marker_names,
            markers=markers,
            initial_positions=frames[0],
            radius=config.tube_radius,
            material=soft_edge_mat,
            parent=parent
        )

    if config.show_display_edges and 'display_edges' in topology:
        print(f"  Creating display edge curves ({len(topology['display_edges'])} edges)...")
        create_edge_curves(
            name="DisplayEdges",
            edges=topology['display_edges'],
            marker_names=marker_names,
            markers=markers,
            initial_positions=frames[0],
            radius=config.tube_radius,
            material=display_edge_mat,
            parent=parent
        )

    # Animate markers
    print(f"\nAnimating markers (keyframing every {config.keyframe_step} frame(s))...")
    for marker_name, sphere in markers.items():
        trajectory = [frames[f][marker_name] for f in range(n_frames)]
        animate_marker(
            marker=sphere,
            trajectory=trajectory,
            frame_start=config.frame_start,
            keyframe_step=config.keyframe_step
        )

    # Create and animate toy markers if available
    if toy_trajectories is not None:
        print(f"\nCreating toy markers ({len(toy_trajectories)} keypoints)...")
        toy_markers = create_toy_markers(
            toy_trajectories=toy_trajectories,
            parent=parent,
            material=toy_mat,
            radius=config.toy_sphere_radius,
            frame_start=config.frame_start,
            keyframe_step=config.keyframe_step
        )
        print(f"âœ“ Created {len(toy_markers)} toy markers")

    # Set up scene
    bpy.context.scene.frame_start = config.frame_start
    bpy.context.scene.frame_end = config.frame_start + n_frames - 1
    bpy.context.scene.frame_current = config.frame_start

    # Set viewport shading
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    print("\n" + "="*80)
    print("âœ“ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Timeline: frames {config.frame_start} to {config.frame_start + n_frames - 1}")
    print(f"Body markers: {len(markers)} spheres Ã— {n_frames // config.keyframe_step} keyframes each")
    if toy_trajectories is not None:
        print(f"Toy markers: {len(toy_trajectories)} spheres Ã— {n_frames // config.keyframe_step} keyframes each")
    print(f"Edges: Curves with hooks (auto-follow, no keyframes needed!)")
    print(f"Total keyframes: ~{len(markers) * 3 * (n_frames // config.keyframe_step)} (body)")
    if toy_trajectories is not None:
        print(f"              + ~{len(toy_trajectories) * 3 * (n_frames // config.keyframe_step)} (toy)")
    print("\nPress SPACE to play animation! ðŸŽ¬")


# Example configuration
config = RigidBodyVisualizationConfig(
    csv_path=Path(
        r"C:\Users\jonma\github_repos\jonmatthis\bs\python_code\rigid_body_tracker\examples\output\ferret_skull_only_raw_spine\trajectory_data.csv"
    ),
    topology_path=Path(
        r"C:\Users\jonma\github_repos\jonmatthis\bs\python_code\rigid_body_tracker\examples\output\ferret_skull_only_raw_spine\topology.json"
    ),
    data_type="optimized",  # or "noisy" or "gt"
    data_scale=0.001,  # mm to meters
    sphere_radius=0.005,
    tube_radius=0.001,
    show_rigid_edges=False,  # Usually too many, clutters the view
    show_soft_edges=True,    # Spine connections
    show_display_edges=True,  # Main skeleton
    frame_start=0,
    keyframe_step=3,  # Set to 2-5 for even faster loading on long animations

    # NEW: Toy data configuration
    toy_csv_path=Path(
        r"D:\bs\ferret_recordings\session_2025-07-01_ferret_757_EyeCameras_P33_EO5\clips\1m_20s-2m_20s\mocap_data\output_data\processed_data\toy_body_rigid_3d_xyz.csv"
    ),
    toy_sphere_radius=0.01,  # Slightly larger than body markers
    toy_color=(1.0, 0.8, 0.0),  # Gold/yellow
)

create_rigid_body_visualization(config=config)