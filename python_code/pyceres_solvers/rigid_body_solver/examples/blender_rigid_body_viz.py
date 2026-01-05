"""Load rigid body tracking + eye tracking into Blender with 3D eyeball visualization.

NEW: Eye tracking support - visualize reconstructed gaze as animated 3D eyeball!
"""

import sys
from pathlib import Path
from dataclasses import dataclass
import json
import csv

import bpy
import bmesh
from mathutils import Vector, Matrix, Euler
import numpy as np


# [Previous imports and utility functions remain the same]
# ... [load_trajectories_auto and related functions] ...

@dataclass
class EyeTrackingConfig:
    """Configuration for eye tracking visualization."""

    csv_path: Path
    """Path to eye_tracking_results.csv"""

    parent_marker_name: str
    """Name of skull marker to attach eyeball to (e.g., 'M8' for right eye)"""

    track_to_marker_name: str
    """Marker to point AWAY from (e.g., opposite eye or head center)"""

    up_marker_name: str
    """Marker to use for up direction (e.g., 'base' or top of head)"""

    eyeball_radius: float = 0.012
    """Eyeball radius in meters (default: 12mm)"""

    pupil_radius: float = 0.002
    """Pupil radius in meters (default: 2mm)"""

    iris_radius: float = 0.005
    """Iris radius in meters (default: 5mm)"""

    local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset from parent marker in local space (meters)"""

    eyeball_color: tuple[float, float, float] = (0.95, 0.95, 0.95)
    """RGB color for eyeball (white)"""

    iris_color: tuple[float, float, float] = (0.3, 0.5, 0.8)
    """RGB color for iris (blue)"""

    pupil_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """RGB color for pupil (black)"""

    show_gaze_arrow: bool = True
    """Show gaze direction arrow"""

    gaze_arrow_length: float = 0.05
    """Length of gaze arrow in meters"""

    gaze_arrow_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    """RGBA color for gaze arrow (red)"""


@dataclass
class RigidBodyVisualizationConfig:
    """Configuration for visualization."""

    csv_path: Path
    """Path to trajectory_data.csv"""

    topology_path: Path
    """Path to topology.json"""

    data_type: str = "optimized"
    """Which dataset to visualize: 'original', 'optimized', or 'gt'"""

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
    """Keyframe every N frames (1=all frames, 2=every other, etc.)"""

    toy_csv_path: Path | None = None
    """Optional path to toy trajectory CSV"""

    toy_sphere_radius: float = 0.015
    """Radius of toy marker spheres (meters)"""

    toy_color: tuple[float, float, float] = (1.0, 0.8, 0.0)
    """RGB color for toy markers (gold/yellow)"""

    # NEW: Eye tracking support
    eye_tracking_configs: list[EyeTrackingConfig] | None = None
    """List of eye tracking configurations (one per eye)"""


def load_eye_tracking_data(
    *,
    csv_path: Path,
    data_scale: float = 1.0
) -> list[dict[str, float]]:
    """
    Load eye tracking results from CSV.

    Returns:
        List of frames with eye tracking data
    """
    frames: list[dict[str, float]] = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f=f)

        for row in reader:
            frame_data = {
                'frame': int(row['frame']),
                'gaze_azimuth_rad': float(row['gaze_azimuth_rad']),
                'gaze_elevation_rad': float(row['gaze_elevation_rad']),
                'gaze_x': float(row['gaze_x']),
                'gaze_y': float(row['gaze_y']),
                'gaze_z': float(row['gaze_z']),
                'eyeball_x_mm': float(row['eyeball_x_mm']) * data_scale,
                'eyeball_y_mm': float(row['eyeball_y_mm']) * data_scale,
                'eyeball_z_mm': float(row['eyeball_z_mm']) * data_scale,
                'reprojection_error_px': float(row['reprojection_error_px'])
            }
            frames.append(frame_data)

    print(f"✓ Loaded {len(frames)} frames of eye tracking data")
    return frames


def create_eyeball_with_pupil(
    *,
    name: str,
    eyeball_radius: float,
    pupil_radius: float,
    iris_radius: float,
    location: tuple[float, float, float],
    eyeball_material: bpy.types.Material,
    iris_material: bpy.types.Material,
    pupil_material: bpy.types.Material,
    parent: bpy.types.Object
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    """
    Create a 3D eyeball with iris and pupil.

    Returns:
        Tuple of (eyeball_sphere, iris, pupil) objects
    """
    # Create eyeball sphere
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=eyeball_radius,
        location=location,
        segments=32,
        ring_count=16
    )

    eyeball = bpy.context.active_object
    eyeball.name = name
    eyeball.parent = parent

    if eyeball.data.materials:
        eyeball.data.materials[0] = eyeball_material
    else:
        eyeball.data.materials.append(eyeball_material)

    bpy.ops.object.shade_smooth()

    # Create iris (ring on eyeball surface)
    bpy.ops.mesh.primitive_circle_add(
        vertices=32,
        radius=iris_radius,
        location=(0, 0, eyeball_radius * 0.98),
        fill_type='NGON'
    )

    iris = bpy.context.active_object
    iris.name = f"{name}_Iris"
    iris.parent = eyeball
    iris.parent_type = 'OBJECT'

    if iris.data.materials:
        iris.data.materials[0] = iris_material
    else:
        iris.data.materials.append(iris_material)

    bpy.ops.object.shade_smooth()

    # Create pupil (circle on iris)
    bpy.ops.mesh.primitive_circle_add(
        vertices=16,
        radius=pupil_radius,
        location=(0, 0, eyeball_radius * 0.99),
        fill_type='NGON'
    )

    pupil = bpy.context.active_object
    pupil.name = f"{name}_Pupil"
    pupil.parent = eyeball
    pupil.parent_type = 'OBJECT'

    if pupil.data.materials:
        pupil.data.materials[0] = pupil_material
    else:
        pupil.data.materials.append(pupil_material)

    bpy.ops.object.shade_smooth()

    return eyeball, iris, pupil


def create_gaze_arrow(
    *,
    name: str,
    length: float,
    color: tuple[float, float, float, float],
    parent: bpy.types.Object
) -> bpy.types.Object:
    """Create a gaze direction arrow."""
    # Create arrow from curve
    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.001  # Thin arrow
    curve_data.bevel_resolution = 4
    curve_data.fill_mode = 'FULL'

    # Create spline for arrow shaft
    spline = curve_data.splines.new(type='NURBS')
    spline.points.add(1)

    spline.points[0].co = (0, 0, 0, 1.0)
    spline.points[1].co = (0, 0, length, 1.0)

    spline.order_u = 2
    spline.use_endpoint_u = True

    # Create curve object
    arrow = bpy.data.objects.new(name=name, object_data=curve_data)
    bpy.context.collection.objects.link(object=arrow)
    arrow.parent = parent

    # Create material
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    nodes.clear()

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_emission = nodes.new(type='ShaderNodeEmission')

    node_emission.inputs['Color'].default_value = color
    node_emission.inputs['Strength'].default_value = 2.0

    mat.node_tree.links.new(node_emission.outputs['Emission'], node_output.inputs['Surface'])

    if arrow.data.materials:
        arrow.data.materials[0] = mat
    else:
        arrow.data.materials.append(mat)

    return arrow


def animate_eyeball_gaze(
    *,
    eyeball: bpy.types.Object,
    eye_frames: list[dict[str, float]],
    frame_start: int,
    keyframe_step: int = 1
) -> None:
    """
    Animate eyeball rotation based on gaze angles.

    The gaze angles are in camera space, so we need to:
    1. Convert azimuth/elevation to rotation
    2. Apply rotation to eyeball (which is parented to skull marker)
    """
    for frame_idx in range(0, len(eye_frames), keyframe_step):
        frame = frame_start + frame_idx
        eye_data = eye_frames[frame_idx]

        # Get gaze angles (these are in camera space)
        azimuth_rad = eye_data['gaze_azimuth_rad']
        elevation_rad = eye_data['gaze_elevation_rad']

        # Convert to Euler angles
        # Azimuth rotates around Y (vertical)
        # Elevation rotates around X (horizontal)
        # The eyeball looks along +Z in neutral pose
        euler = Euler((elevation_rad, azimuth_rad, 0.0), 'XYZ')
        eyeball.rotation_euler = euler

        eyeball.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Set interpolation to linear
    if eyeball.animation_data and eyeball.animation_data.action:
        for fcurve in eyeball.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


def create_eye_visualization(
    *,
    config: EyeTrackingConfig,
    parent_marker: bpy.types.Object,
    track_to_marker: bpy.types.Object,
    up_marker: bpy.types.Object,
    rigid_body_parent: bpy.types.Object,
    frame_start: int,
    keyframe_step: int,
    data_scale: float
) -> bpy.types.Object:
    """
    Create and animate eye tracking visualization with proper orientation constraints.

    Strategy:
    1. Create base empty at eye marker location (follows head movement)
    2. Use constraints to orient it away from opposite eye, up towards head top
    3. Create eyeball as child of base empty
    4. Apply gaze rotations to eyeball (relative to base orientation)

    This way the eye naturally points outward and rotates with the head,
    plus the gaze tracking rotations are added on top.

    Returns:
        The eyeball object
    """
    print(f"\nCreating eye visualization from {config.csv_path.name}...")

    # Load eye tracking data
    eye_frames = load_eye_tracking_data(
        csv_path=config.csv_path,
        data_scale=data_scale
    )

    # Create materials
    eyeball_mat = create_marker_material(
        name=f"Eye_{config.parent_marker_name}_Eyeball",
        color=config.eyeball_color,
        metallic=0.1,
        roughness=0.3
    )

    iris_mat = create_marker_material(
        name=f"Eye_{config.parent_marker_name}_Iris",
        color=config.iris_color,
        metallic=0.2,
        roughness=0.4
    )

    pupil_mat = create_marker_material(
        name=f"Eye_{config.parent_marker_name}_Pupil",
        color=config.pupil_color,
        metallic=0.0,
        roughness=0.8
    )

    # ========================================================================
    # STEP 1: Create base orientation empty
    # This empty will be constrained to look AWAY from the opposite eye
    # and have its up direction point towards the head top marker
    # ========================================================================

    bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
    eye_base_empty = bpy.context.active_object
    eye_base_empty.name = f"EyeBase_{config.parent_marker_name}"
    eye_base_empty.empty_display_size = config.eyeball_radius * 0.5

    # Parent to skull marker (so it moves with the head)
    eye_base_empty.parent = parent_marker
    eye_base_empty.parent_type = 'OBJECT'

    # Apply local offset
    eye_base_empty.location = config.local_offset

    # ========================================================================
    # STEP 2: Add constraints for proper orientation
    # ========================================================================

    # Damped Track constraint: Point AWAY from opposite eye marker
    # (Track -Z axis, since we want the BACK of the eye to point towards opposite eye,
    #  meaning the FRONT (+Z) points away)
    constraint_track = eye_base_empty.constraints.new(type='DAMPED_TRACK')
    constraint_track.target = track_to_marker
    constraint_track.track_axis = 'TRACK_NEGATIVE_Z'  # Back of eye towards target
    constraint_track.name = "TrackAwayFromOppositeEye"

    print(f"  ✓ Added constraint: back of eye points towards {config.track_to_marker_name}")

    # Locked Track constraint: Keep up direction pointing towards head top
    constraint_up = eye_base_empty.constraints.new(type='LOCKED_TRACK')
    constraint_up.target = up_marker
    constraint_up.track_axis = 'TRACK_Y'  # Y axis points up
    constraint_up.lock_axis = 'LOCK_Z'     # Lock Z (viewing direction)
    constraint_up.name = "UpTowardsHeadTop"

    print(f"  ✓ Added constraint: up direction towards {config.up_marker_name}")

    # ========================================================================
    # STEP 3: Create eyeball geometry as child of orientation empty
    # ========================================================================

    # Create another empty for the eyeball rotation
    bpy.ops.object.empty_add(type='SPHERE', location=(0, 0, 0))
    eye_rotation_empty = bpy.context.active_object
    eye_rotation_empty.name = f"EyeRotation_{config.parent_marker_name}"
    eye_rotation_empty.empty_display_size = config.eyeball_radius * 0.3
    eye_rotation_empty.parent = eye_base_empty
    eye_rotation_empty.parent_type = 'OBJECT'

    # Create eyeball with iris and pupil
    eyeball, iris, pupil = create_eyeball_with_pupil(
        name=f"Eyeball_{config.parent_marker_name}",
        eyeball_radius=config.eyeball_radius,
        pupil_radius=config.pupil_radius,
        iris_radius=config.iris_radius,
        location=(0, 0, 0),
        eyeball_material=eyeball_mat,
        iris_material=iris_mat,
        pupil_material=pupil_mat,
        parent=eye_rotation_empty
    )

    # Create gaze arrow if requested
    if config.show_gaze_arrow:
        gaze_arrow = create_gaze_arrow(
            name=f"GazeArrow_{config.parent_marker_name}",
            length=config.gaze_arrow_length,
            color=config.gaze_arrow_color,
            parent=eyeball
        )
        print(f"  ✓ Created gaze arrow")

    # ========================================================================
    # STEP 4: Animate eyeball rotation (relative to base orientation)
    # ========================================================================

    print(f"  Animating eyeball rotation...")
    animate_eyeball_gaze(
        eyeball=eye_rotation_empty,  # Animate the rotation empty, not the eyeball mesh
        eye_frames=eye_frames,
        frame_start=frame_start,
        keyframe_step=keyframe_step
    )

    print(f"  ✓ Eye visualization complete for {config.parent_marker_name}")
    print(f"    - Base orientation constrained to skull markers")
    print(f"    - Gaze rotations applied relative to base")

    return eyeball


def clear_scene() -> None:
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


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
    """Load trajectory data from CSV."""
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

    print(f"✓ Loaded {len(frames)} frames of {data_type} data")
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

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_mix = nodes.new(type='ShaderNodeMixShader')

    node_bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    node_bsdf.inputs['Metallic'].default_value = metallic
    node_bsdf.inputs['Roughness'].default_value = roughness

    node_emission.inputs['Color'].default_value = (*color, 1.0)
    node_emission.inputs['Strength'].default_value = 0.5

    links = mat.node_tree.links
    links.new(node_bsdf.outputs['BSDF'], node_mix.inputs[1])
    links.new(node_emission.outputs['Emission'], node_mix.inputs[2])
    links.new(node_mix.outputs['Shader'], node_output.inputs['Surface'])

    node_mix.inputs['Fac'].default_value = 0.2

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

    if sphere.data.materials:
        sphere.data.materials[0] = material
    else:
        sphere.data.materials.append(material)

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
    """Create edge curves with drivers."""
    if not edges:
        return None

    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4
    curve_data.resolution_u = 2
    curve_data.fill_mode = 'FULL'
    curve_data.use_fill_caps = True

    curve_obj = bpy.data.objects.new(name=name, object_data=curve_data)
    bpy.context.collection.objects.link(object=curve_obj)
    curve_obj.parent = parent

    curve_obj.show_wire = False
    curve_obj.show_all_edges = False
    curve_obj.display_type = 'TEXTURED'

    if curve_obj.data.materials:
        curve_obj.data.materials[0] = material
    else:
        curve_obj.data.materials.append(material)

    for edge_idx, (i, j) in enumerate(edges):
        marker_i_name = marker_names[i]
        marker_j_name = marker_names[j]

        start_pos = Vector(initial_positions[marker_i_name])
        end_pos = Vector(initial_positions[marker_j_name])

        spline = curve_data.splines.new(type='NURBS')
        spline.points.add(1)

        spline.points[0].co = (*start_pos, 1.0)
        spline.points[1].co = (*end_pos, 1.0)

        spline.order_u = 2
        spline.use_endpoint_u = True

        marker_i = markers[marker_i_name]
        marker_j = markers[marker_j_name]

        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            fcurve = curve_data.driver_add(f'splines[{edge_idx}].points[0].co', axis_idx)
            driver = fcurve.driver
            driver.type = 'AVERAGE'

            var = driver.variables.new()
            var.name = 'loc'
            var.type = 'TRANSFORMS'

            target = var.targets[0]
            target.id = marker_i
            target.transform_type = f'LOC_{axis_name.upper()}'
            target.transform_space = 'WORLD_SPACE'

        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            fcurve = curve_data.driver_add(f'splines[{edge_idx}].points[1].co', axis_idx)
            driver = fcurve.driver
            driver.type = 'AVERAGE'

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

    if marker.animation_data and marker.animation_data.action:
        for fcurve in marker.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


def create_rigid_body_visualization(*, config: RigidBodyVisualizationConfig) -> None:
    """Create complete rigid body visualization with eye tracking."""

    print("="*80)
    print("RIGID BODY + EYE TRACKING VISUALIZATION")
    print("="*80)

    # Load rigid body data
    print("\nLoading rigid body data...")
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

    # Create parent empty
    bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
    parent = bpy.context.active_object
    parent.name = f"RigidBody_{topology['name']}"
    parent.empty_display_size = 0.1

    # Create materials
    print("\nCreating materials...")

    marker_colors = [
        (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 1.0, 0.5), (0.0, 1.0, 1.0), (0.0, 0.5, 1.0),
        (0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.5),
    ]

    marker_materials = {}
    for idx, marker_name in enumerate(marker_names):
        color = marker_colors[idx % len(marker_colors)]
        mat = create_marker_material(name=f"Marker_{marker_name}", color=color)
        marker_materials[marker_name] = mat

    rigid_edge_mat = create_edge_material(name="RigidEdge", color=config.rigid_edge_color)
    soft_edge_mat = create_edge_material(name="SoftEdge", color=config.soft_edge_color)
    display_edge_mat = create_edge_material(name="DisplayEdge", color=config.display_edge_color)

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

    # Create edges
    print("\nCreating edge curves...")

    if config.show_rigid_edges and 'rigid_edges' in topology:
        print(f"  Creating rigid edges ({len(topology['rigid_edges'])} edges)...")
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
        print(f"  Creating soft edges ({len(topology['soft_edges'])} edges)...")
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
        print(f"  Creating display edges ({len(topology['display_edges'])} edges)...")
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
    print(f"\nAnimating markers...")
    for marker_name, sphere in markers.items():
        trajectory = [frames[f][marker_name] for f in range(n_frames)]
        animate_marker(
            marker=sphere,
            trajectory=trajectory,
            frame_start=config.frame_start,
            keyframe_step=config.keyframe_step
        )

    # Create eye visualizations
    if config.eye_tracking_configs is not None:
        print("\n" + "="*80)
        print("ADDING EYE TRACKING VISUALIZATIONS")
        print("="*80)

        for eye_config in config.eye_tracking_configs:
            # Check parent marker exists
            if eye_config.parent_marker_name not in markers:
                print(f"⚠ Warning: Parent marker '{eye_config.parent_marker_name}' not found, skipping eye")
                continue

            # Check track_to marker exists
            if eye_config.track_to_marker_name not in markers:
                print(f"⚠ Warning: Track-to marker '{eye_config.track_to_marker_name}' not found, skipping eye")
                continue

            # Check up marker exists
            if eye_config.up_marker_name not in markers:
                print(f"⚠ Warning: Up marker '{eye_config.up_marker_name}' not found, skipping eye")
                continue

            parent_marker = markers[eye_config.parent_marker_name]
            track_to_marker = markers[eye_config.track_to_marker_name]
            up_marker = markers[eye_config.up_marker_name]

            create_eye_visualization(
                config=eye_config,
                parent_marker=parent_marker,
                track_to_marker=track_to_marker,
                up_marker=up_marker,
                rigid_body_parent=parent,
                frame_start=config.frame_start,
                keyframe_step=config.keyframe_step,
                data_scale=config.data_scale
            )

    # Set up scene
    bpy.context.scene.frame_start = config.frame_start
    bpy.context.scene.frame_end = config.frame_start + n_frames - 1
    bpy.context.scene.frame_current = config.frame_start

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Timeline: frames {config.frame_start} to {config.frame_start + n_frames - 1}")
    print(f"Markers: {len(markers)} spheres × {n_frames // config.keyframe_step} keyframes")
    if config.eye_tracking_configs:
        print(f"Eyes: {len(config.eye_tracking_configs)} eyeball(s) with animated gaze")
    print("\nPress SPACE to play animation! 🎬")


# ============================================================================
# EXAMPLE USAGE - FERRET EYE TRACKING
# ============================================================================

"""
HOW THE EYE ORIENTATION SYSTEM WORKS:

The eye visualization uses a constraint-based hierarchy to ensure proper orientation:

1. EYE MARKER (skull) 
   ↓ (parent)
2. EYE BASE EMPTY (constrained orientation)
   - Damped Track: Back of eye (-Z) points towards opposite eye marker
   - Locked Track: Up direction (+Y) points towards head top marker
   - Result: Eye naturally faces outward from skull
   ↓ (parent)
3. EYE ROTATION EMPTY (gaze animation)
   - Animated with azimuth/elevation from eye tracking
   - Rotations are relative to the base orientation
   ↓ (parent)
4. EYEBALL MESH (visual geometry)
   - Sphere with iris and pupil
   - Gaze arrow shows final viewing direction

This hierarchy ensures:
- Eye moves with the skull (parented to marker)
- Eye points outward by default (constraints on base empty)
- Eye rotates with head movements (constraints follow parent)
- Gaze tracking adds movements on top (rotation empty animation)
- Final gaze = head orientation × base orientation × eye rotation ✓

KEY MARKERS TO IDENTIFY:
- parent_marker_name: Marker at this eye's location (e.g., "right_eye")
- track_to_marker_name: Marker to point AWAY from (opposite eye or head center)
- up_marker_name: Marker defining "up" direction (head top, "base", etc.)

FERRET EYE DIMENSIONS:
- Axial length: ~7mm (front-to-back diameter of eyeball)
- Eyeball radius: 3.5mm (half of axial length)
- Pupil: ~1-1.5mm radius (dilated in low light)
- Iris: ~2.5-3mm radius
"""

# Example configuration with ferret eye tracking
config = RigidBodyVisualizationConfig(
    csv_path=Path(
        r"/python_code/pyceres_solvers\examples\output\2025-07-11_ferret_757_EyeCameras_P43_E15__1_0m_37s-1m_37s\trajectory_data.csv"
    ),
    topology_path=Path(
        r"/python_code/pyceres_solvers\examples\output\2025-07-11_ferret_757_EyeCameras_P43_E15__1_0m_37s-1m_37s\topology.json"
    ),
    data_type="optimized",
    data_scale=0.001,  # mm to meters
    sphere_radius=0.005,
    tube_radius=0.001,
    show_rigid_edges=False,
    show_soft_edges=True,
    show_display_edges=True,
    frame_start=0,
    keyframe_step=3,

    # Eye tracking configurations
    eye_tracking_configs=[
        EyeTrackingConfig(
            csv_path=Path(
                r"/python_code/pyceres_solvers/eye_tracking\output\eye_tracking_demo\eye_tracking_results.csv"),
            parent_marker_name="right_eye",     # Right eye marker
            track_to_marker_name="left_eye",    # Left eye marker (point away from this)
            up_marker_name="base",              # Head top/base marker for up direction
            eyeball_radius=0.0035,              # 3.5mm - half of 7mm axial length
            pupil_radius=0.001,                 # 1mm - ferret pupil
            iris_radius=0.0025,                 # 2.5mm - ferret iris
            local_offset=(0.0, 0.0, 0.0),       # Adjust if needed to position eye correctly
            show_gaze_arrow=True,
            gaze_arrow_length=0.03,             # 30mm arrow (shorter for smaller ferret eye)
            gaze_arrow_color=(1.0, 0.0, 0.0, 1.0)
        ),
        # Add left eye if you have eye1 data:
        # EyeTrackingConfig(
        #     csv_path=Path(r"C:\Users\jonma\github_repos\jonmatthis\bs\python_code\eye_tracking\output\eye_tracking_demo\eye1_tracking_results.csv"),
        #     parent_marker_name="left_eye",      # Left eye marker
        #     track_to_marker_name="right_eye",   # Right eye marker (point away from this)
        #     up_marker_name="base",              # Same head top marker
        #     eyeball_radius=0.0035,
        #     pupil_radius=0.001,
        #     iris_radius=0.0025,
        #     local_offset=(0.0, 0.0, 0.0),
        #     show_gaze_arrow=True,
        #     gaze_arrow_length=0.03,
        #     gaze_arrow_color=(0.0, 1.0, 0.0, 1.0)  # Green arrow for left eye
        # ),
    ]
)

# Run the visualization
# clear_scene()
create_rigid_body_visualization(config=config)