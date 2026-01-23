"""
Ferret Eye Blender Visualization Core
=====================================

Shared module for building eye visualizations in Blender 4.4+.

This module provides:
- Data structures for eye geometry and kinematics
- Data loading functions
- Blender utility functions (materials, meshes, animation)
- Core eye building function

Import this module from the visualization scripts.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
import numpy as np
from bpy_extras import anim_utils


# ============================================================================
# CONSTANTS
# ============================================================================

MM_TO_M = 0.001
TARGET_FPS = 90
NUM_PUPIL_POINTS = 8

# Visualization sizes (in meters)
EYE_SPHERE_RADIUS = 0.001  # 1mm for eye markers
PUPIL_POINT_RADIUS = 0.0008  # 0.8mm for pupil points
EYE_BONE_RADIUS = 0.0004  # 0.4mm for bones
GAZE_ARROW_LENGTH = 0.012  # 12mm gaze arrow
SOCKET_AXIS_SIZE = 0.008  # 8mm for socket frame arrows
EYEBALL_AXIS_SIZE = 0.006  # 6mm for eyeball frame arrows
SOCKET_SPHERE_RADIUS = 0.0012  # 1.2mm for socket landmarks


# ============================================================================
# COLORS
# ============================================================================

# Right eye: Red/Magenta tones
RIGHT_EYE_COLORS = {
    "eyeball_wireframe": "#FF0066",
    "pupil_center": "#FF0000",
    "pupil_points": "#CC0044",
    "pupil_ring": "#FF0055",
    "gaze_arrow": "#FF00FF",
    "tear_duct": "#FF6699",
    "outer_eye": "#FF3366",
    "socket_line": "#FF4477",
    "socket_frame": "#FF0088",
    "eyeball_frame": "#FF00CC",
}

# Left eye: Blue/Cyan tones
LEFT_EYE_COLORS = {
    "eyeball_wireframe": "#0088FF",
    "pupil_center": "#0000FF",
    "pupil_points": "#0066CC",
    "pupil_ring": "#0044FF",
    "gaze_arrow": "#00FFFF",
    "tear_duct": "#00CCFF",
    "outer_eye": "#0099FF",
    "socket_line": "#00AAFF",
    "socket_frame": "#0066FF",
    "eyeball_frame": "#00AAFF",
}


# ============================================================================
# LOGGING
# ============================================================================

_LOG_INDENT = 0


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{level}] {'  ' * _LOG_INDENT}{msg}")


def log_enter(name: str) -> None:
    global _LOG_INDENT
    log(f">>> {name}()", "CALL")
    _LOG_INDENT += 1


def log_exit(name: str, result: str = "") -> None:
    global _LOG_INDENT
    _LOG_INDENT = max(0, _LOG_INDENT - 1)
    log(f"<<< {name}(){' -> ' + result if result else ''}", "CALL")


def log_step(s: str) -> None:
    log(f"STEP: {s}", "STEP")


def log_data(k: str, v: object) -> None:
    log(f"DATA: {k} = {v}", "DATA")


def log_warn(s: str) -> None:
    log(f"WARNING: {s}", "WARN")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EyeGeometry:
    """Reference geometry for the eyeball in local coordinates."""
    keypoints: dict[str, np.ndarray]  # name -> [x, y, z] in mm
    display_edges: list[tuple[str, str]]
    rigid_edges: list[tuple[str, str]]
    eye_radius_mm: float

    @classmethod
    def from_json(cls, path: Path) -> "EyeGeometry":
        log_enter("EyeGeometry.from_json")
        with open(path, "r") as f:
            data = json.load(f)

        keypoints: dict[str, np.ndarray] = {}
        for name, coords in data["keypoints"].items():
            keypoints[name] = np.array([coords["x"], coords["y"], coords["z"]])

        pupil_center = keypoints.get("pupil_center", np.array([0.0, 0.0, 3.5]))
        eye_radius_mm = float(np.linalg.norm(pupil_center))

        result = cls(
            keypoints=keypoints,
            display_edges=[tuple(e) for e in data.get("display_edges", [])],
            rigid_edges=[tuple(e) for e in data.get("rigid_edges", [])],
            eye_radius_mm=eye_radius_mm,
        )
        log_data("keypoints", list(keypoints.keys()))
        log_data("eye_radius_mm", eye_radius_mm)
        log_exit("EyeGeometry.from_json")
        return result


@dataclass
class EyeKinematics:
    """Per-frame eye pose and tracked data."""
    num_frames: int
    timestamps: np.ndarray  # [num_frames]
    orientation: np.ndarray  # [num_frames, 4] - quaternion (w, x, y, z)
    tear_duct_mm: np.ndarray  # [num_frames, 3]
    outer_eye_mm: np.ndarray  # [num_frames, 3]
    pupil_center_mm: np.ndarray  # [num_frames, 3]
    pupil_points_mm: np.ndarray  # [num_frames, 8, 3]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_eye_kinematics(csv_path: Path) -> EyeKinematics:
    """Load eye kinematics from tidy CSV format."""
    log_enter("load_eye_kinematics")
    log_data("path", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        frame_idx = header.index("frame")
        traj_idx = header.index("trajectory")
        comp_idx = header.index("component")
        val_idx = header.index("value")
        ts_idx = header.index("timestamp_s")

        line_count = 0
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[frame_idx])
            traj_name = parts[traj_idx]
            component = parts[comp_idx]
            value = float(parts[val_idx])
            timestamp = float(parts[ts_idx])
            data[frame][traj_name][component] = value
            data[frame]["__timestamp__"] = {"value": timestamp}
            line_count += 1
            if line_count % 100000 == 0:
                log_step(f"Read {line_count} lines...")

    log_data("total_lines", line_count)

    num_frames = max(data.keys()) + 1
    log_data("num_frames", num_frames)

    timestamps = np.zeros(num_frames, dtype=np.float64)
    for f in range(num_frames):
        if "__timestamp__" in data[f]:
            timestamps[f] = data[f]["__timestamp__"]["value"]

    orientation = np.zeros((num_frames, 4), dtype=np.float64)
    for f in range(num_frames):
        if "orientation" in data[f]:
            orientation[f, 0] = data[f]["orientation"].get("w", 1.0)
            orientation[f, 1] = data[f]["orientation"].get("x", 0.0)
            orientation[f, 2] = data[f]["orientation"].get("y", 0.0)
            orientation[f, 3] = data[f]["orientation"].get("z", 0.0)

    tear_duct_mm = np.zeros((num_frames, 3), dtype=np.float64)
    outer_eye_mm = np.zeros((num_frames, 3), dtype=np.float64)
    for f in range(num_frames):
        if "socket_landmark__tear_duct" in data[f]:
            tear_duct_mm[f, 0] = data[f]["socket_landmark__tear_duct"].get("x", 0.0)
            tear_duct_mm[f, 1] = data[f]["socket_landmark__tear_duct"].get("y", 0.0)
            tear_duct_mm[f, 2] = data[f]["socket_landmark__tear_duct"].get("z", 0.0)
        if "socket_landmark__outer_eye" in data[f]:
            outer_eye_mm[f, 0] = data[f]["socket_landmark__outer_eye"].get("x", 0.0)
            outer_eye_mm[f, 1] = data[f]["socket_landmark__outer_eye"].get("y", 0.0)
            outer_eye_mm[f, 2] = data[f]["socket_landmark__outer_eye"].get("z", 0.0)

    pupil_center_mm = np.zeros((num_frames, 3), dtype=np.float64)
    for f in range(num_frames):
        if "tracked_pupil__center" in data[f]:
            pupil_center_mm[f, 0] = data[f]["tracked_pupil__center"].get("x", 0.0)
            pupil_center_mm[f, 1] = data[f]["tracked_pupil__center"].get("y", 0.0)
            pupil_center_mm[f, 2] = data[f]["tracked_pupil__center"].get("z", 0.0)

    pupil_points_mm = np.zeros((num_frames, NUM_PUPIL_POINTS, 3), dtype=np.float64)
    for i in range(NUM_PUPIL_POINTS):
        traj_name = f"tracked_pupil__p{i + 1}"
        for f in range(num_frames):
            if traj_name in data[f]:
                pupil_points_mm[f, i, 0] = data[f][traj_name].get("x", 0.0)
                pupil_points_mm[f, i, 1] = data[f][traj_name].get("y", 0.0)
                pupil_points_mm[f, i, 2] = data[f][traj_name].get("z", 0.0)

    result = EyeKinematics(
        num_frames=num_frames,
        timestamps=timestamps,
        orientation=orientation,
        tear_duct_mm=tear_duct_mm,
        outer_eye_mm=outer_eye_mm,
        pupil_center_mm=pupil_center_mm,
        pupil_points_mm=pupil_points_mm,
    )

    log_exit("load_eye_kinematics", f"{num_frames} frames")
    return result


# ============================================================================
# BLENDER UTILITIES
# ============================================================================

def clear_scene() -> None:
    log_enter("clear_scene")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for action in list(bpy.data.actions):
        if action.users == 0:
            bpy.data.actions.remove(action)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    log_exit("clear_scene")


def setup_scene(num_frames: int) -> None:
    log_enter("setup_scene")
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    scene.render.fps = TARGET_FPS
    scene.frame_set(0)
    log_data("frames", num_frames)
    log_data("fps", TARGET_FPS)
    log_exit("setup_scene")


def create_material(name: str, hex_color: str, emission: float = 1.0) -> bpy.types.Material:
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (r, g, b, 1.0)
    emit.inputs["Strength"].default_value = emission

    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def create_empty(
    name: str,
    location: tuple[float, float, float],
    collection: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    display_type: str = "PLAIN_AXES",
    display_size: float = 0.003,
) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = display_type
    empty.empty_display_size = display_size
    empty.location = location
    collection.objects.link(empty)
    if parent:
        empty.parent = parent
    return empty


def create_sphere(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=radius)
    bm.to_mesh(mesh)
    bm.free()

    obj.location = location
    mesh.materials.append(material)
    collection.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


def create_cone(
    name: str,
    length: float,
    radius: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=True,
        segments=12,
        radius1=radius,
        radius2=0.0,
        depth=length,
    )
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=(0, 0, length / 2))
    bm.to_mesh(mesh)
    bm.free()

    mesh.materials.append(material)
    collection.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


def create_cylinder(
    name: str,
    length: float,
    radius: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=True,
        segments=8,
        radius1=radius,
        radius2=radius * 0.5,
        depth=length,
    )
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=(0, 0, length / 2))
    bm.to_mesh(mesh)
    bm.free()

    mesh.materials.append(material)
    collection.objects.link(obj)
    return obj


def create_wireframe_sphere(
    name: str,
    radius: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    n_lat: int = 8,
    n_lon: int = 16,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=n_lon, v_segments=n_lat, radius=radius)
    bm.to_mesh(mesh)
    bm.free()

    mesh.materials.append(material)
    collection.objects.link(obj)

    mod = obj.modifiers.new(name="Wireframe", type="WIREFRAME")
    mod.thickness = 0.0002
    mod.use_replace = True

    if parent:
        obj.parent = parent

    return obj


def create_wireframe_cube(
    name: str,
    size: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Create a wireframe cube (cage) centered at origin."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    half = size / 2
    vertices = [
        (-half, -half, -half),
        (half, -half, -half),
        (half, half, -half),
        (-half, half, -half),
        (-half, -half, half),
        (half, -half, half),
        (half, half, half),
        (-half, half, half),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    mesh.from_pydata(vertices, edges, [])
    mesh.update()

    mesh.materials.append(material)
    collection.objects.link(obj)
    return obj


# ============================================================================
# ANIMATION
# ============================================================================

def animate_position(obj: bpy.types.Object, trajectory: np.ndarray) -> None:
    """Animate object location using fast bulk FCurve API."""
    num_frames = trajectory.shape[0]

    if not obj.animation_data:
        obj.animation_data_create()
    if not obj.animation_data.action:
        action = bpy.data.actions.new(name=f"{obj.name}_action")
        obj.animation_data.action = action

    action = obj.animation_data.action
    anim_data = obj.animation_data

    if anim_data.action_slot is None:
        slot = action.slots.new(id_type='OBJECT', name=obj.name)
        anim_data.action_slot = slot

    channelbag = anim_utils.action_ensure_channelbag_for_slot(action, anim_data.action_slot)

    for i in range(3):
        fc = channelbag.fcurves.ensure(data_path="location", index=i)
        fc.keyframe_points.add(num_frames)
        coords = np.zeros(num_frames * 2, dtype=np.float32)
        coords[0::2] = np.arange(num_frames, dtype=np.float32)
        coords[1::2] = trajectory[:, i].astype(np.float32)
        fc.keyframe_points.foreach_set("co", coords)
        fc.update()


def animate_rotation_quaternion(obj: bpy.types.Object, quaternions: np.ndarray) -> None:
    """Animate object rotation using quaternions (w, x, y, z)."""
    num_frames = quaternions.shape[0]

    obj.rotation_mode = 'QUATERNION'

    if not obj.animation_data:
        obj.animation_data_create()
    if not obj.animation_data.action:
        action = bpy.data.actions.new(name=f"{obj.name}_action")
        obj.animation_data.action = action

    action = obj.animation_data.action
    anim_data = obj.animation_data

    if anim_data.action_slot is None:
        slot = action.slots.new(id_type='OBJECT', name=obj.name)
        anim_data.action_slot = slot

    channelbag = anim_utils.action_ensure_channelbag_for_slot(action, anim_data.action_slot)

    for i in range(4):
        fc = channelbag.fcurves.ensure(data_path="rotation_quaternion", index=i)
        fc.keyframe_points.add(num_frames)
        coords = np.zeros(num_frames * 2, dtype=np.float32)
        coords[0::2] = np.arange(num_frames, dtype=np.float32)
        coords[1::2] = quaternions[:, i].astype(np.float32)
        fc.keyframe_points.foreach_set("co", coords)
        fc.update()


# ============================================================================
# COORDINATE FRAME COMPUTATION
# ============================================================================

def compute_socket_frame_quaternion(
    tear_duct_mean: np.ndarray,
    outer_eye_mean: np.ndarray,
    is_right_eye: bool,
) -> np.ndarray:
    """
    Compute quaternion for socket frame orientation.

    Socket frame definition:
    - Origin: eye center [0, 0, 0]
    - +Z: toward midpoint of tear_duct and outer_eye
    - +X: animal's left (toward nose for right eye, toward ear for left eye)
    - +Y: computed via cross product (roughly superior/up)

    Args:
        tear_duct_mean: Mean tear duct position in eye coords (mm)
        outer_eye_mean: Mean outer eye position in eye coords (mm)
        is_right_eye: True if this is the right eye

    Returns:
        Quaternion [w, x, y, z] for the socket frame orientation
    """
    # +Z direction: toward socket midpoint
    midpoint = (tear_duct_mean + outer_eye_mean) / 2.0
    z_axis = midpoint / np.linalg.norm(midpoint)

    # +X direction: animal's left
    # For right eye: animal's left is toward nose (tear_duct direction)
    # For left eye: animal's left is toward ear (outer_eye direction)
    if is_right_eye:
        x_raw = tear_duct_mean - outer_eye_mean
    else:
        x_raw = outer_eye_mean - tear_duct_mean

    # Orthogonalize x_axis to be perpendicular to z_axis
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-10:
        # Fallback if parallel
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm

    # +Y direction: Z × X (right-handed)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Build rotation matrix (columns are basis vectors)
    R = np.column_stack([x_axis, y_axis, z_axis])

    # Convert rotation matrix to quaternion
    return rotation_matrix_to_quaternion(R)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ============================================================================
# EYE BUILDING
# ============================================================================

def build_single_eye(
    eye_name: str,
    kinematics: EyeKinematics,
    geometry: EyeGeometry,
    parent_collection: bpy.types.Collection,
    parent_object: bpy.types.Object | None = None,
    eye_position_in_parent_mm: np.ndarray | None = None,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """
    Build visualization for a single eye.

    Creates the following hierarchy:
    - eye_socket (Empty ARROWS): socket frame, fixed orientation based on socket landmarks
      - eye_eyeball (Empty ARROWS): eyeball frame, animated rotation
        - wireframe sphere
        - pupil center + boundary points
        - gaze arrow
      - socket landmarks (tear_duct, outer_eye)

    Args:
        eye_name: Name of the eye (e.g., "left_eye" or "right_eye")
        kinematics: Eye kinematics data
        geometry: Eye reference geometry
        parent_collection: Collection to add objects to
        parent_object: Optional parent object (e.g., skull_origin)
        eye_position_in_parent_mm: Position of eye center in parent coords (mm)

    Returns:
        Tuple of (eye_socket, eye_eyeball) Empty objects
    """
    log_enter(f"build_single_eye({eye_name})")

    is_right = "right" in eye_name.lower()
    colors = RIGHT_EYE_COLORS if is_right else LEFT_EYE_COLORS
    eye_radius_m = geometry.eye_radius_mm * MM_TO_M

    # Create collection for this eye
    eye_coll = bpy.data.collections.new(eye_name)
    parent_collection.children.link(eye_coll)

    # Compute socket frame orientation from mean socket landmark positions
    tear_duct_mean = np.mean(kinematics.tear_duct_mm, axis=0)
    outer_eye_mean = np.mean(kinematics.outer_eye_mm, axis=0)
    socket_quaternion = compute_socket_frame_quaternion(
        tear_duct_mean=tear_duct_mean,
        outer_eye_mean=outer_eye_mean,
        is_right_eye=is_right,
    )

    # =========================================================================
    # EYE SOCKET FRAME (fixed orientation, parented to skull if provided)
    # =========================================================================
    log_step(f"Creating {eye_name} socket frame")

    if eye_position_in_parent_mm is not None:
        socket_position = tuple(eye_position_in_parent_mm * MM_TO_M)
    else:
        socket_position = (0.0, 0.0, 0.0)

    eye_socket = create_empty(
        f"{eye_name}_socket",
        socket_position,
        eye_coll,
        parent=parent_object,
        display_type="ARROWS",
        display_size=SOCKET_AXIS_SIZE,
    )
    eye_socket.rotation_mode = 'QUATERNION'
    eye_socket.rotation_quaternion = socket_quaternion

    # =========================================================================
    # EYEBALL FRAME (animated rotation, parented to socket)
    # =========================================================================
    log_step(f"Creating {eye_name} eyeball frame")

    eye_eyeball = create_empty(
        f"{eye_name}_eyeball",
        (0.0, 0.0, 0.0),
        eye_coll,
        parent=eye_socket,
        display_type="ARROWS",
        display_size=EYEBALL_AXIS_SIZE,
    )

    # Animate eyeball rotation
    animate_rotation_quaternion(eye_eyeball, kinematics.orientation)

    # =========================================================================
    # EYEBALL WIREFRAME SPHERE (parented to eyeball)
    # =========================================================================
    log_step(f"Creating {eye_name} wireframe sphere")
    wireframe_mat = create_material(f"{eye_name}_wireframe_mat", colors["eyeball_wireframe"], emission=0.5)
    create_wireframe_sphere(
        f"{eye_name}_wireframe",
        eye_radius_m,
        wireframe_mat,
        eye_coll,
        parent=eye_eyeball,
    )

    # =========================================================================
    # PUPIL CENTER (animated position, parented to eyeball)
    # =========================================================================
    log_step(f"Creating {eye_name} pupil center")
    pupil_center_mat = create_material(f"{eye_name}_pupil_center_mat", colors["pupil_center"], emission=2.0)

    pupil_center_empty = create_empty(
        f"{eye_name}_pupil_center_empty",
        (0, 0, 0),
        eye_coll,
        parent=eye_eyeball,
    )
    pupil_center_traj_m = kinematics.pupil_center_mm * MM_TO_M
    animate_position(pupil_center_empty, pupil_center_traj_m)

    pupil_center_sphere = create_sphere(
        f"{eye_name}_pupil_center_sphere",
        (0, 0, 0),
        EYE_SPHERE_RADIUS,
        pupil_center_mat,
        eye_coll,
    )
    pupil_center_sphere.constraints.new(type="COPY_LOCATION").target = pupil_center_empty

    # =========================================================================
    # PUPIL BOUNDARY POINTS p1-p8 (animated positions, parented to eyeball)
    # =========================================================================
    log_step(f"Creating {eye_name} pupil boundary points")
    pupil_point_mat = create_material(f"{eye_name}_pupil_point_mat", colors["pupil_points"], emission=1.5)
    pupil_ring_mat = create_material(f"{eye_name}_pupil_ring_mat", colors["pupil_ring"], emission=1.0)

    pupil_empties: list[bpy.types.Object] = []
    for i in range(NUM_PUPIL_POINTS):
        point_name = f"{eye_name}_p{i + 1}"

        empty = create_empty(
            f"{point_name}_empty",
            (0, 0, 0),
            eye_coll,
            parent=eye_eyeball,
        )
        pupil_point_traj_m = kinematics.pupil_points_mm[:, i, :] * MM_TO_M
        animate_position(empty, pupil_point_traj_m)
        pupil_empties.append(empty)

        sphere = create_sphere(
            f"{point_name}_sphere",
            (0, 0, 0),
            PUPIL_POINT_RADIUS,
            pupil_point_mat,
            eye_coll,
        )
        sphere.constraints.new(type="COPY_LOCATION").target = empty

    # =========================================================================
    # PUPIL RING BONES (connecting p1->p2->...->p8->p1)
    # =========================================================================
    log_step(f"Creating {eye_name} pupil ring bones")
    for i in range(NUM_PUPIL_POINTS):
        next_i = (i + 1) % NUM_PUPIL_POINTS
        head_empty = pupil_empties[i]
        tail_empty = pupil_empties[next_i]

        lengths = np.linalg.norm(
            kinematics.pupil_points_mm[:, next_i, :] - kinematics.pupil_points_mm[:, i, :],
            axis=1
        )
        median_len = float(np.nanmedian(lengths)) * MM_TO_M

        bone = create_cylinder(
            f"{eye_name}_pupil_bone_{i + 1}",
            median_len,
            EYE_BONE_RADIUS,
            pupil_ring_mat,
            eye_coll,
        )
        bone.constraints.new(type="COPY_LOCATION").target = head_empty
        bone.constraints.new(type="DAMPED_TRACK").target = tail_empty
        bone.constraints["Damped Track"].track_axis = "TRACK_Z"

    # =========================================================================
    # GAZE DIRECTION ARROW (at pupil center, pointing along +Z in eyeball frame)
    # =========================================================================
    log_step(f"Creating {eye_name} gaze arrow")
    gaze_mat = create_material(f"{eye_name}_gaze_mat", colors["gaze_arrow"], emission=3.0)
    gaze_arrow = create_cone(
        f"{eye_name}_gaze_arrow",
        GAZE_ARROW_LENGTH,
        GAZE_ARROW_LENGTH * 0.15,
        gaze_mat,
        eye_coll,
        parent=eye_eyeball,
    )
    gaze_arrow.location = (0, 0, eye_radius_m)

    # =========================================================================
    # SOCKET LANDMARKS (parented to socket - they are fixed relative to socket)
    # =========================================================================
    log_step(f"Creating {eye_name} socket landmarks")

    tear_duct_mat = create_material(f"{eye_name}_tear_duct_mat", colors["tear_duct"], emission=2.0)
    outer_eye_mat = create_material(f"{eye_name}_outer_eye_mat", colors["outer_eye"], emission=2.0)
    socket_line_mat = create_material(f"{eye_name}_socket_line_mat", colors["socket_line"], emission=1.0)

    # Tear duct (animated position in socket frame)
    tear_duct_empty = create_empty(
        f"{eye_name}_tear_duct_empty",
        (0, 0, 0),
        eye_coll,
        parent=eye_socket,
    )
    tear_duct_traj_m = kinematics.tear_duct_mm * MM_TO_M
    animate_position(tear_duct_empty, tear_duct_traj_m)

    tear_duct_sphere = create_sphere(
        f"{eye_name}_tear_duct_sphere",
        (0, 0, 0),
        SOCKET_SPHERE_RADIUS,
        tear_duct_mat,
        eye_coll,
    )
    tear_duct_sphere.constraints.new(type="COPY_LOCATION").target = tear_duct_empty

    # Outer eye (animated position in socket frame)
    outer_eye_empty = create_empty(
        f"{eye_name}_outer_eye_empty",
        (0, 0, 0),
        eye_coll,
        parent=eye_socket,
    )
    outer_eye_traj_m = kinematics.outer_eye_mm * MM_TO_M
    animate_position(outer_eye_empty, outer_eye_traj_m)

    outer_eye_sphere = create_sphere(
        f"{eye_name}_outer_eye_sphere",
        (0, 0, 0),
        SOCKET_SPHERE_RADIUS,
        outer_eye_mat,
        eye_coll,
    )
    outer_eye_sphere.constraints.new(type="COPY_LOCATION").target = outer_eye_empty

    # Socket line (tear_duct to outer_eye)
    socket_lengths = np.linalg.norm(
        kinematics.outer_eye_mm - kinematics.tear_duct_mm,
        axis=1
    )
    median_socket_len = float(np.nanmedian(socket_lengths)) * MM_TO_M

    socket_bone = create_cylinder(
        f"{eye_name}_socket_bone",
        median_socket_len,
        EYE_BONE_RADIUS * 1.5,
        socket_line_mat,
        eye_coll,
    )
    socket_bone.constraints.new(type="COPY_LOCATION").target = tear_duct_empty
    socket_bone.constraints.new(type="DAMPED_TRACK").target = outer_eye_empty
    socket_bone.constraints["Damped Track"].track_axis = "TRACK_Z"

    log_exit(f"build_single_eye({eye_name})")
    return eye_socket, eye_eyeball


def save_blend_file(output_path: Path) -> None:
    """Save the current Blender scene to a .blend file."""
    log_step(f"Saving blend file to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


def setup_viewport() -> None:
    """Configure viewport for visualization."""
    try:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
                        space.clip_end = 2.0
                        space.clip_start = 0.0001
    except Exception as e:
        log_warn(f"Viewport config failed: {e}")
