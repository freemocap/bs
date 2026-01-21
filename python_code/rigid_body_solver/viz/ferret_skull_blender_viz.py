"""
Ferret skull and spine motion capture visualization for Blender 4.4+

Visualizes:
- Rigid-body-enforced trajectories (blue/cyan) - tracked marker positions
- Kinematic skull reconstruction (orange/red) - rigid skull from pose estimation
- Skull origin axes showing pose orientation

Usage: Open in Blender, set DATA_DIR, run with Alt+P
"""

import json
import math
import statistics
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import bmesh
import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Quaternion, Vector


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output")

# Data files
TRAJECTORY_CSV = DATA_DIR / "skull_and_spine_trajectories.csv"
TOPOLOGY_JSON = DATA_DIR / "skull_and_spine_topology.json"
KINEMATICS_CSV = DATA_DIR / "skull_kinematics.csv"
SKULL_GEOMETRY_JSON = DATA_DIR / "skull_reference_geometry.json"

MM_TO_M = 0.001
TARGET_FPS = 90

SPHERE_RADIUS = 0.003  # 3mm
BONE_RADIUS = 0.002    # 2mm
JOINT_RADIUS = 0.003   # 3mm
SKULL_ORIGIN_SIZE = 0.1  # 10cm arrows


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
class BoneDefinition:
    name: str
    head: str
    tail: str
    lengths: list[float] = field(default_factory=list)
    median: float = 0.0


@dataclass
class Topology:
    marker_names: list[str]
    rigid_edges: list[tuple[str, str]]
    display_edges: list[tuple[str, str]]
    name: str

    @classmethod
    def from_json(cls, path: Path) -> "Topology":
        log_enter("Topology.from_json")
        with open(path, "r") as f:
            data = json.load(f)
        result = cls(
            marker_names=data["marker_names"],
            rigid_edges=[tuple(e) for e in data["rigid_edges"]],
            display_edges=[tuple(e) for e in data["display_edges"]],
            name=data["name"],
        )
        log_data("markers", len(result.marker_names))
        log_exit("Topology.from_json")
        return result


@dataclass
class SkullGeometry:
    """Reference geometry for the skull in local coordinates."""
    keypoints: dict[str, np.ndarray]  # name -> [x, y, z] in mm
    display_edges: list[tuple[str, str]]
    rigid_edges: list[tuple[str, str]]

    @classmethod
    def from_json(cls, path: Path) -> "SkullGeometry":
        log_enter("SkullGeometry.from_json")
        with open(path, "r") as f:
            data = json.load(f)
        
        keypoints = {}
        for name, coords in data["keypoints"].items():
            keypoints[name] = np.array([coords["x"], coords["y"], coords["z"]])
        
        result = cls(
            keypoints=keypoints,
            display_edges=[tuple(e) for e in data["display_edges"]],
            rigid_edges=[tuple(e) for e in data["rigid_edges"]],
        )
        log_data("keypoints", list(keypoints.keys()))
        log_exit("SkullGeometry.from_json")
        return result


@dataclass
class SkullKinematics:
    """Per-frame skull pose and reconstructed keypoints."""
    num_frames: int
    position: np.ndarray  # [num_frames, 3] - origin position in meters
    orientation: np.ndarray  # [num_frames, 4] - quaternion (w, x, y, z)
    keypoints: dict[str, np.ndarray]  # name -> [num_frames, 3] global positions in meters


# ============================================================================
# DATA LOADING
# ============================================================================

def load_trajectories(csv_path: Path) -> dict[str, np.ndarray]:
    """Load marker trajectories from CSV. Returns positions in meters."""
    log_enter("load_trajectories")
    log_data("path", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    trajectories: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        frame_idx = header.index("frame")
        traj_idx = header.index("trajectory")
        comp_idx = header.index("component")
        val_idx = header.index("value")

        line_count = 0
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[frame_idx])
            traj_name = parts[traj_idx]
            component = parts[comp_idx]
            value = float(parts[val_idx])
            trajectories[traj_name][frame][component] = value
            line_count += 1
            if line_count % 50000 == 0:
                log_step(f"Read {line_count} lines...")

    log_data("total_lines", line_count)

    result: dict[str, np.ndarray] = {}
    num_frames = 0
    for traj_name, frames_dict in trajectories.items():
        num_frames = max(max(frames_dict.keys()) + 1, num_frames)
        arr = np.zeros((num_frames, 3), dtype=np.float64)
        for frame_num, components in frames_dict.items():
            arr[frame_num, 0] = components.get("x", 0.0) * MM_TO_M
            arr[frame_num, 1] = components.get("y", 0.0) * MM_TO_M
            arr[frame_num, 2] = components.get("z", 0.0) * MM_TO_M
        result[traj_name] = arr

    log_data("trajectories", len(result))
    log_data("frames", num_frames)
    log_exit("load_trajectories")
    return result


def load_kinematics(csv_path: Path) -> SkullKinematics:
    """Load skull kinematics from CSV. Returns positions in meters."""
    log_enter("load_kinematics")
    log_data("path", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Parse into nested dict: data[frame][trajectory][component] = value
    data: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        frame_idx = header.index("frame")
        traj_idx = header.index("trajectory")
        comp_idx = header.index("component")
        val_idx = header.index("value")

        line_count = 0
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[frame_idx])
            traj_name = parts[traj_idx]
            component = parts[comp_idx]
            value = float(parts[val_idx])
            data[frame][traj_name][component] = value
            line_count += 1
            if line_count % 100000 == 0:
                log_step(f"Read {line_count} lines...")

    log_data("total_lines", line_count)

    num_frames = max(data.keys()) + 1
    log_data("num_frames", num_frames)

    # Extract position
    position = np.zeros((num_frames, 3), dtype=np.float64)
    for f in range(num_frames):
        if "position" in data[f]:
            position[f, 0] = data[f]["position"].get("x", 0.0) * MM_TO_M
            position[f, 1] = data[f]["position"].get("y", 0.0) * MM_TO_M
            position[f, 2] = data[f]["position"].get("z", 0.0) * MM_TO_M

    # Extract orientation (quaternion)
    orientation = np.zeros((num_frames, 4), dtype=np.float64)
    for f in range(num_frames):
        if "orientation" in data[f]:
            orientation[f, 0] = data[f]["orientation"].get("w", 1.0)
            orientation[f, 1] = data[f]["orientation"].get("x", 0.0)
            orientation[f, 2] = data[f]["orientation"].get("y", 0.0)
            orientation[f, 3] = data[f]["orientation"].get("z", 0.0)

    # Extract keypoints
    keypoints: dict[str, np.ndarray] = {}
    keypoint_names = [k for k in data[0].keys() if k.startswith("keypoint__")]
    log_data("keypoint_trajectories", keypoint_names)

    for kp_name in keypoint_names:
        short_name = kp_name.replace("keypoint__", "")
        arr = np.zeros((num_frames, 3), dtype=np.float64)
        for f in range(num_frames):
            if kp_name in data[f]:
                arr[f, 0] = data[f][kp_name].get("x", 0.0) * MM_TO_M
                arr[f, 1] = data[f][kp_name].get("y", 0.0) * MM_TO_M
                arr[f, 2] = data[f][kp_name].get("z", 0.0) * MM_TO_M
        keypoints[short_name] = arr

    result = SkullKinematics(
        num_frames=num_frames,
        position=position,
        orientation=orientation,
        keypoints=keypoints,
    )

    log_exit("load_kinematics", f"{num_frames} frames")
    return result


# ============================================================================
# RIGID BODY ENFORCEMENT
# ============================================================================

def enforce_rigid_bodies_spine(
    trajectories: dict[str, np.ndarray],
    spine_bones: dict[str, BoneDefinition],
) -> dict[str, np.ndarray]:
    """Enforce constant bone lengths for spine markers."""
    log_enter("enforce_rigid_bodies_spine")

    num_frames = next(iter(trajectories.values())).shape[0]
    
    # Calculate median lengths
    for bone in spine_bones.values():
        bone.lengths = []
        for f in range(num_frames):
            head = trajectories[bone.head][f, :]
            tail = trajectories[bone.tail][f, :]
            bone.lengths.append(np.linalg.norm(tail - head))
        valid = [l for l in bone.lengths if not math.isnan(l) and l > 0]
        if valid:
            bone.median = statistics.median(valid)
        log_data(f"{bone.name}_median_mm", bone.median * 1000)

    # Enforce lengths
    updated = deepcopy(trajectories)
    hierarchy = {
        "spine_t1": ["sacrum"],
        "sacrum": ["tail_tip"],
        "tail_tip": [],
    }

    for bone in spine_bones.values():
        for f, raw_len in enumerate(bone.lengths):
            if np.isnan(raw_len) or raw_len == 0:
                continue
            head_pos = trajectories[bone.head][f, :]
            tail_pos = updated[bone.tail][f, :]
            vec = tail_pos - head_pos
            cur_len = np.linalg.norm(vec)
            if cur_len < 1e-10:
                continue
            delta = (vec / cur_len) * (bone.median - cur_len)

            def translate(name: str, d: np.ndarray) -> None:
                updated[name][f, :] += d
                for child in hierarchy.get(name, []):
                    translate(child, d)
            translate(bone.tail, delta)

    log_exit("enforce_rigid_bodies_spine")
    return updated


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


# ============================================================================
# OBJECT CREATION (NO bpy.ops!)
# ============================================================================

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


def create_bone(
    name: str,
    length: float,
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
        radius1=BONE_RADIUS,
        radius2=BONE_RADIUS * 0.3,
        depth=length,
    )
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=(0, 0, length / 2))
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=JOINT_RADIUS)

    bm.to_mesh(mesh)
    bm.free()

    mesh.materials.append(material)
    collection.objects.link(obj)
    return obj


# ============================================================================
# ANIMATION (FAST BULK API)
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

    # Quaternion has 4 components: w, x, y, z (indices 0, 1, 2, 3)
    for i in range(4):
        fc = channelbag.fcurves.ensure(data_path="rotation_quaternion", index=i)
        fc.keyframe_points.add(num_frames)
        coords = np.zeros(num_frames * 2, dtype=np.float32)
        coords[0::2] = np.arange(num_frames, dtype=np.float32)
        coords[1::2] = quaternions[:, i].astype(np.float32)
        fc.keyframe_points.foreach_set("co", coords)
        fc.update()


# ============================================================================
# COLORS
# ============================================================================

# Trajectory markers: blue/cyan
TRAJ_COLORS = {
    "nose": "#00FFFF",
    "base": "#00CCCC",
    "left_eye": "#0088FF",
    "right_eye": "#0088FF",
    "left_ear": "#0066CC",
    "right_ear": "#0066CC",
    "left_cam_tip": "#004488",
    "right_cam_tip": "#004488",
    "spine_t1": "#00AAAA",
    "sacrum": "#008888",
    "tail_tip": "#006666",
}

# Kinematic skull: orange/red
KINEMATIC_COLORS = {
    "nose": "#FFAA00",
    "base": "#FF8800",
    "left_eye": "#FF6600",
    "right_eye": "#FF6600",
    "left_ear": "#FF4400",
    "right_ear": "#FF4400",
    "left_cam_tip": "#CC3300",
    "right_cam_tip": "#CC3300",
}

BONE_COLOR_TRAJ = "#00AAFF"
BONE_COLOR_KINEMATIC = "#FF6600"


# ============================================================================
# MAIN BUILD
# ============================================================================

def build_scene(
    trajectories: dict[str, np.ndarray],
    kinematics: SkullKinematics,
    skull_geometry: SkullGeometry,
    topology: Topology,
) -> None:
    log_enter("build_scene")

    num_frames = kinematics.num_frames
    setup_scene(num_frames)

    # Collections
    main_coll = bpy.data.collections.new("FerretMocap")
    bpy.context.scene.collection.children.link(main_coll)

    traj_coll = bpy.data.collections.new("Trajectories_Enforced")
    main_coll.children.link(traj_coll)

    kinematic_coll = bpy.data.collections.new("Kinematic_Skull")
    main_coll.children.link(kinematic_coll)

    bones_coll = bpy.data.collections.new("Bones")
    main_coll.children.link(bones_coll)

    # Root for trajectories
    traj_root = create_empty("trajectory_root", (0, 0, 0), traj_coll)
    traj_empties: dict[str, bpy.types.Object] = {}

    # =========================================================================
    # TRAJECTORY MARKERS (blue/cyan)
    # =========================================================================
    print("\n" + "=" * 50)
    print("CREATING TRAJECTORY MARKERS (enforced)")
    print("=" * 50)

    for marker_name in topology.marker_names:
        if marker_name not in trajectories:
            log_warn(f"No trajectory data for {marker_name}")
            continue

        log_step(f"Creating: {marker_name}")
        traj = trajectories[marker_name]
        color = TRAJ_COLORS.get(marker_name, "#00FFFF")
        mat = create_material(f"{marker_name}_traj_mat", color, emission=1.5)

        empty = create_empty(f"{marker_name}_traj_empty", tuple(traj[0]), traj_coll, traj_root)
        traj_empties[marker_name] = empty
        animate_position(empty, traj)

        sphere = create_sphere(f"{marker_name}_traj_sphere", tuple(traj[0]), SPHERE_RADIUS, mat, traj_coll, traj_root)
        sphere.constraints.new(type="COPY_LOCATION").target = empty

    # =========================================================================
    # SKULL ORIGIN (orange arrows)
    # =========================================================================
    print("\n" + "=" * 50)
    print("CREATING KINEMATIC SKULL")
    print("=" * 50)

    log_step("Creating skull origin axes")
    skull_origin = create_empty(
        "skull_origin",
        tuple(kinematics.position[0]),
        kinematic_coll,
        display_type="ARROWS",
        display_size=SKULL_ORIGIN_SIZE,
    )
    
    # Animate position and rotation
    log_step("Animating skull origin position")
    animate_position(skull_origin, kinematics.position)
    
    log_step("Animating skull origin rotation")
    animate_rotation_quaternion(skull_origin, kinematics.orientation)

    # =========================================================================
    # SKULL KEYPOINTS (parented to origin, local positions from geometry)
    # =========================================================================
    log_step("Creating skull keypoints from reference geometry")
    kinematic_empties: dict[str, bpy.types.Object] = {}

    for kp_name, local_pos in skull_geometry.keypoints.items():
        log_step(f"  Keypoint: {kp_name}")
        
        # Convert local position from mm to m
        local_pos_m = local_pos * MM_TO_M
        
        color = KINEMATIC_COLORS.get(kp_name, "#FF6600")
        mat = create_material(f"{kp_name}_kinematic_mat", color, emission=2.0)

        # Create sphere at local position, parented to skull_origin
        sphere = create_sphere(
            f"{kp_name}_kinematic_sphere",
            tuple(local_pos_m),
            SPHERE_RADIUS,
            mat,
            kinematic_coll,
            parent=skull_origin,
        )
        kinematic_empties[kp_name] = sphere

    # =========================================================================
    # BONES FOR TRAJECTORIES
    # =========================================================================
    print("\n" + "=" * 50)
    print("CREATING BONES")
    print("=" * 50)

    bone_connections = [
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
        ("left_ear", "base"),
        ("right_ear", "base"),
        ("base", "left_cam_tip"),
        ("base", "right_cam_tip"),
        ("left_ear", "spine_t1"),
        ("right_ear", "spine_t1"),
        ("spine_t1", "sacrum"),
        ("sacrum", "tail_tip"),
    ]

    traj_bone_mat = create_material("bone_traj_mat", BONE_COLOR_TRAJ, emission=0.8)

    for head_name, tail_name in bone_connections:
        if head_name not in traj_empties or tail_name not in traj_empties:
            continue

        log_step(f"Bone: {head_name} -> {tail_name}")

        # Calculate median length
        if head_name in trajectories and tail_name in trajectories:
            lengths = np.linalg.norm(trajectories[tail_name] - trajectories[head_name], axis=1)
            median_len = float(np.nanmedian(lengths))
        else:
            median_len = 0.02  # Default 2cm

        log_data("length_mm", median_len * 1000)

        bone = create_bone(f"{head_name}_to_{tail_name}_bone", median_len, traj_bone_mat, bones_coll)
        bone.parent = traj_root
        bone.constraints.new(type="COPY_LOCATION").target = traj_empties[head_name]
        bone.constraints.new(type="DAMPED_TRACK").target = traj_empties[tail_name]
        bone.constraints["Damped Track"].track_axis = "TRACK_Z"

    # =========================================================================
    # KINEMATIC SKULL BONES (connecting the rigid keypoints)
    # =========================================================================
    log_step("Creating kinematic skull bones")
    kinematic_bone_mat = create_material("bone_kinematic_mat", BONE_COLOR_KINEMATIC, emission=1.0)

    for head_name, tail_name in skull_geometry.display_edges:
        if head_name not in kinematic_empties or tail_name not in kinematic_empties:
            continue

        # Calculate length from reference geometry
        head_local = skull_geometry.keypoints[head_name] * MM_TO_M
        tail_local = skull_geometry.keypoints[tail_name] * MM_TO_M
        length = float(np.linalg.norm(tail_local - head_local))

        bone = create_bone(f"{head_name}_to_{tail_name}_kinematic", length, kinematic_bone_mat, kinematic_coll)
        bone.parent = skull_origin
        bone.constraints.new(type="COPY_LOCATION").target = kinematic_empties[head_name]
        bone.constraints.new(type="DAMPED_TRACK").target = kinematic_empties[tail_name]
        bone.constraints["Damped Track"].track_axis = "TRACK_Z"

    log_exit("build_scene")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    start_time = time.time()

    print("\n" + "=" * 70)
    print("   FERRET MOCAP VISUALIZATION")
    print("   Trajectories (blue) + Kinematic Skull (orange)")
    print("=" * 70 + "\n")

    log_enter("main")

    print("PHASE 1: CLEARING SCENE")
    clear_scene()

    print("\nPHASE 2: LOADING TRAJECTORY DATA")
    raw_trajectories = load_trajectories(TRAJECTORY_CSV)
    topology = Topology.from_json(TOPOLOGY_JSON)

    print("\nPHASE 3: ENFORCING RIGID BODIES (spine)")
    spine_bones = {
        "spine_t1_to_sacrum": BoneDefinition(name="spine_t1_to_sacrum", head="spine_t1", tail="sacrum"),
        "sacrum_to_tail_tip": BoneDefinition(name="sacrum_to_tail_tip", head="sacrum", tail="tail_tip"),
    }
    trajectories = enforce_rigid_bodies_spine(raw_trajectories, spine_bones)

    print("\nPHASE 4: LOADING KINEMATICS DATA")
    kinematics = load_kinematics(KINEMATICS_CSV)
    skull_geometry = SkullGeometry.from_json(SKULL_GEOMETRY_JSON)

    print("\nPHASE 5: BUILDING SCENE")
    build_scene(trajectories, kinematics, skull_geometry, topology)

    print("\nPHASE 6: VIEWPORT SETUP")
    try:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
                        space.clip_end = 10.0
    except Exception as e:
        log_warn(f"Viewport config failed: {e}")

    elapsed = time.time() - start_time
    log_exit("main")

    print("\n" + "=" * 70)
    print("   DONE!")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Frames: {bpy.context.scene.frame_end + 1}")
    print(f"   FPS: {bpy.context.scene.render.fps}")
    print("=" * 70)
    print("   Blue/Cyan  = Enforced trajectory markers")
    print("   Orange/Red = Kinematic skull (rigid body)")
    print("   Arrows     = Skull origin pose")
    print("   Press SPACEBAR to play")
    print("=" * 70 + "\n")


if __name__ == "__main__" or __name__ == "<run_path>":
    main()