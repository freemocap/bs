"""
Ferret Full Body + Eye + Gaze Blender Visualization
====================================================

Visualizes skull, spine, both eyes, gaze vectors, and synchronized videos in Blender 4.0+ and 5.0+.

Creates:
- Skull rigid body with animated pose and keypoints (nose, left_eye, right_eye)
- Spine keypoints (spine_t1, sacrum, tail_tip) with connecting bones
- Left/Right eyes with socket frames and eyeball rotations
- Gaze vectors showing where each eye is looking (100mm ray from eye center)
- Pupil geometry in BOTH world coordinates AND socket-local coordinates
- 1m wireframe cage for reference
- Video planes showing synchronized camera footage (if configured)

Data sources:
- Skull kinematics: RigidBodyKinematics (tidy CSV + reference geometry JSON)
- Spine trajectories: Keypoint trajectories from skull_and_spine_trajectories.csv
- Eye kinematics: FerretEyeKinematics (eye-in-socket rotations)
- Eye trajectories: Pupil center and boundary points (p1-p8) for pupil visualization
- Gaze kinematics: RigidBodyKinematics (eye orientation in world coords)
- Videos: Specified explicitly in the VIDEOS configuration list

Usage:
1. Open in Blender 4.0+ or 5.0+
2. Edit the configuration section (DATA_DIR, OUTPUT_PATH, VIDEOS)
3. Run with Alt+P
4. Press Spacebar to play animation

Video Configuration:
- Add video paths to the VIDEOS list in the configuration section
- Each video entry specifies path, position, rotation, and size
- Videos are displayed on emissive planes synchronized with timeline
- Supports: .mp4, .avi, .mov, .mkv, .webm

Color scheme:
- Skull: Yellow/Gold
- Spine: Green
- Right eye/gaze: Red/Magenta
- Left eye/gaze: Blue/Cyan
- Pupil points in socket view: Blue (left) / Red (right)
"""

print("\n" + "=" * 70)
print("   FERRET FULL VIZ - STARTING")
print("=" * 70)

print("[BOOT] Importing standard library...")
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

print(f"[BOOT] Python: {sys.version}")

print("[BOOT] Importing numpy...")
import numpy as np

print(f"[BOOT] numpy: {np.__version__}")

print("[BOOT] Importing bpy...")
import bmesh
import bpy

print(f"[BOOT] Blender: {bpy.app.version_string}")
print(f"[BOOT] Blend file: {bpy.data.filepath or '(unsaved)'}")

# ============================================================================
# CONFIGURATION - EDIT THESE!
# ============================================================================

# Base data directory (parent of skull_kinematics, eye_kinematics, gaze_kinematics)
DATA_DIR = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\analyzable_output"
)


# Output path for saved .blend file
OUTPUT_PATH = DATA_DIR / "full_visualization.blend"

# Video configuration - add your video paths here!
# For RESAMPLED videos (from display_videos/), only "path" and "name" are needed.
# The resampled videos have 1:1 frame correspondence with the scene timeline.
#
# Each entry is a dict with:
#   "path": Path to the video file (required)
#   "name": Display name (optional, defaults to filename)
#   "position": (x, y, z) in meters (optional)
#   "rotation": (rx, ry, rz) in radians (optional)
#   "width": plane width in meters (optional)
#
# Note: timestamps_path is NOT needed for resampled videos!
#
DISPLAY_VIDEOS_DIR = DATA_DIR.parent / "display_videos"

VIDEOS: list[dict] = [
    {
        "path": DISPLAY_VIDEOS_DIR / "top_down_mocap_resampled.mp4",
        "name": "top_down_mocap",
        "position": (0.0, 0.0, 0.0),
        "rotation": (0.0, 0.0, 0.0),
    },
    {
        "path": DISPLAY_VIDEOS_DIR / "left_eye_resampled.mp4",
        "name": "left_eye",
        "position": (-0.5, 0.0, 0.0),
        "rotation": (0.0, 0.0, 0.0),
    },
    {
        "path": DISPLAY_VIDEOS_DIR / "right_eye_resampled.mp4",
        "name": "right_eye",
        "position": (0.5, 0.0, 0.0),
        "rotation": (0.0, 0.0, 0.0),
    },
]


# Cage size in meters
CAGE_SIZE_M = 1.0

# ============================================================================
# CONSTANTS
# ============================================================================

MM_TO_M = 0.001
TARGET_FPS = 90

# Visualization sizes (in meters)
SKULL_AXIS_SIZE = 0.015  # 15mm for skull frame
SPINE_AXIS_SIZE = 0.012  # 12mm for spine frame
KEYPOINT_SPHERE_RADIUS = 0.002  # 2mm for keypoints
GAZE_RAY_RADIUS = 0.0008  # 0.8mm for gaze ray
GAZE_TARGET_RADIUS = 0.003  # 3mm for gaze target sphere
BONE_RADIUS = 0.001  # 1mm for bones

# Eye visualization (smaller, relative to skull)
EYE_SOCKET_AXIS_SIZE = 0.006  # 6mm for socket frame
EYE_BALL_AXIS_SIZE = 0.004  # 4mm for eyeball frame

# Video plane configuration
VIDEO_PLANE_WIDTH_M = 0.3  # 30cm width for video planes
VIDEO_PLANE_HEIGHT_M = 0.225  # 22.5cm height (4:3 aspect ratio by default)
VIDEO_PLANE_OFFSET_Z_M = 0.6  # 60cm above origin
VIDEO_PLANE_OFFSET_Y_M = 0.6  # 60cm in front of origin

# ============================================================================
# COLORS
# ============================================================================

SKULL_COLORS = {
    "frame": "#FFD700",  # Gold
    "keypoints": "#FFAA00",  # Orange-gold
    "edges": "#CC8800",  # Darker gold
}

SPINE_COLORS = {
    "frame": "#00FF00",  # Green
    "keypoints": "#00CC00",  # Darker green
    "edges": "#009900",  # Even darker
}

RIGHT_COLORS = {
    "socket_frame": "#FF0088",
    "eyeball_frame": "#FF00CC",
    "gaze_ray": "#FF00FF",
    "gaze_target": "#FF0066",
}

LEFT_COLORS = {
    "socket_frame": "#0066FF",
    "eyeball_frame": "#00AAFF",
    "gaze_ray": "#00FFFF",
    "gaze_target": "#0088FF",
}

# Toy visualization (big cones to be visible)
TOY_CONE_RADIUS = 0.008  # 8mm base radius
TOY_CONE_HEIGHT = 0.020  # 20mm height
TOY_EDGE_RADIUS = 0.002  # 2mm edge radius

TOY_COLORS = {
    "toy_face": "#FF6B35",  # Bright orange-red
    "toy_top": "#FFD700",   # Gold
    "toy_tail": "#FF4500",  # Orange-red
    "edges": "#FF8C00",     # Dark orange
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


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class RigidBodyKinematicsData:
    """Loaded rigid body kinematics data."""

    name: str
    num_frames: int
    timestamps: np.ndarray  # (N,)
    position_xyz_mm: np.ndarray  # (N, 3)
    quaternion_wxyz: np.ndarray  # (N, 4)
    keypoint_names: list[str]
    keypoint_positions_local_mm: dict[str, np.ndarray]  # name -> (3,)
    keypoint_trajectories_mm: dict[str, np.ndarray]  # name -> (N, 3)
    display_edges: list[tuple[str, str]]


@dataclass
class SpineTrajectoryData:
    """Loaded spine trajectory data (keypoints only, no pose)."""

    name: str
    num_frames: int
    keypoint_names: list[str]
    keypoint_trajectories_mm: dict[str, np.ndarray]  # name -> (N, 3)
    display_edges: list[tuple[str, str]]


@dataclass
class BoneDefinition:
    """Definition for a bone connecting two keypoints."""
    name: str
    head: str
    tail: str
    lengths: list[float] = field(default_factory=list)
    median: float = 0.0


@dataclass
class GazeKinematicsData:
    """Loaded gaze kinematics data."""

    name: str
    num_frames: int
    timestamps: np.ndarray  # (N,)
    position_xyz_mm: np.ndarray  # (N, 3) - eye center in world
    quaternion_wxyz: np.ndarray  # (N, 4) - gaze orientation in world
    gaze_target_mm: np.ndarray  # (N, 3) - gaze target in world (100mm along gaze)


@dataclass
class EyeOrientationData:
    """Eye orientation data (eye-in-socket rotation only)."""

    name: str
    num_frames: int
    timestamps: np.ndarray  # (N,)
    quaternion_wxyz: np.ndarray  # (N, 4) - eye orientation in socket frame


@dataclass
class EyeTrajectoryData:
    """Eye trajectory data with pupil geometry in world coordinates."""

    name: str
    num_frames: int
    pupil_center_mm: np.ndarray  # (N, 3) - pupil center in world coords
    pupil_points_mm: np.ndarray  # (N, 8, 3) - pupil boundary points in world coords
    tear_duct_mm: np.ndarray  # (N, 3) - tear duct landmark
    outer_eye_mm: np.ndarray  # (N, 3) - outer eye landmark


@dataclass
class VideoData:
    """Information about a video file to display in the scene."""

    name: str
    filepath: Path
    position: tuple[float, float, float]  # (x, y, z) in meters
    rotation_euler: tuple[float, float, float]  # (rx, ry, rz) in radians
    width_m: float
    height_m: float
    n_frames: int  # Number of frames in the video
    is_resampled: bool = True  # If True, video has 1:1 frame mapping with scene


@dataclass
class ToyTrajectoryData:
    """Toy trajectory data with keypoint positions in world coordinates."""

    name: str
    num_frames: int
    keypoint_names: list[str]  # ["toy_face", "toy_top", "toy_tail"]
    keypoint_trajectories_mm: dict[str, np.ndarray]  # keypoint_name -> (N, 3)
    display_edges: list[tuple[str, str]]  # [(from, to), ...]


# ============================================================================
# DATA LOADING
# ============================================================================


def load_rigid_body_kinematics(
    kinematics_csv_path: Path,
    reference_geometry_json_path: Path,
) -> RigidBodyKinematicsData:
    """Load RigidBodyKinematics from tidy CSV + reference geometry JSON."""
    log_enter("load_rigid_body_kinematics")
    log_data("csv", kinematics_csv_path.name)
    log_data("json", reference_geometry_json_path.name)

    # Load reference geometry
    with open(reference_geometry_json_path, "r") as f:
        geom_data = json.load(f)

    keypoint_positions_local_mm: dict[str, np.ndarray] = {}
    for name, coords in geom_data["keypoints"].items():
        keypoint_positions_local_mm[name] = np.array(
            [coords["x"], coords["y"], coords["z"]], dtype=np.float64
        )

    display_edges: list[tuple[str, str]] = []
    if "display_edges" in geom_data and geom_data["display_edges"]:
        display_edges = [tuple(e) for e in geom_data["display_edges"]]

    # Load kinematics CSV (tidy format)
    data: dict[int, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    with open(kinematics_csv_path, "r") as f:
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

    log_data("lines", line_count)

    num_frames = max(data.keys()) + 1
    log_data("num_frames", num_frames)

    # Extract timestamps
    timestamps = np.zeros(num_frames, dtype=np.float64)
    for f in range(num_frames):
        if "__timestamp__" in data[f]:
            timestamps[f] = data[f]["__timestamp__"]["value"]

    # Extract position
    position_xyz_mm = np.zeros((num_frames, 3), dtype=np.float64)
    for f in range(num_frames):
        if "position" in data[f]:
            position_xyz_mm[f, 0] = data[f]["position"].get("x", 0.0)
            position_xyz_mm[f, 1] = data[f]["position"].get("y", 0.0)
            position_xyz_mm[f, 2] = data[f]["position"].get("z", 0.0)

    # Extract orientation
    quaternion_wxyz = np.zeros((num_frames, 4), dtype=np.float64)
    quaternion_wxyz[:, 0] = 1.0  # default identity
    for f in range(num_frames):
        if "orientation" in data[f]:
            quaternion_wxyz[f, 0] = data[f]["orientation"].get("w", 1.0)
            quaternion_wxyz[f, 1] = data[f]["orientation"].get("x", 0.0)
            quaternion_wxyz[f, 2] = data[f]["orientation"].get("y", 0.0)
            quaternion_wxyz[f, 3] = data[f]["orientation"].get("z", 0.0)

    # Extract keypoint trajectories from CSV
    keypoint_names = list(keypoint_positions_local_mm.keys())
    keypoint_trajectories_mm: dict[str, np.ndarray] = {}

    for kp_name in keypoint_names:
        traj_name = f"keypoint__{kp_name}"
        kp_traj = np.zeros((num_frames, 3), dtype=np.float64)
        found_any = False
        for f in range(num_frames):
            if traj_name in data[f]:
                found_any = True
                kp_traj[f, 0] = data[f][traj_name].get("x", 0.0)
                kp_traj[f, 1] = data[f][traj_name].get("y", 0.0)
                kp_traj[f, 2] = data[f][traj_name].get("z", 0.0)
        if not found_any:
            raise ValueError(f"Keypoint trajectory '{traj_name}' not found in CSV!")
        keypoint_trajectories_mm[kp_name] = kp_traj

    # Derive name from CSV filename
    name = kinematics_csv_path.stem.replace("_kinematics", "")

    # Debug: Log first frame keypoint positions to verify loading
    log_data("keypoints", keypoint_names)
    for kp_name in keypoint_names[:3]:  # Just first 3 to avoid spam
        kp0 = keypoint_trajectories_mm[kp_name][0]
        log_data(f"  {kp_name}[0]", f"[{kp0[0]:.2f}, {kp0[1]:.2f}, {kp0[2]:.2f}]")

    result = RigidBodyKinematicsData(
        name=name,
        num_frames=num_frames,
        timestamps=timestamps,
        position_xyz_mm=position_xyz_mm,
        quaternion_wxyz=quaternion_wxyz,
        keypoint_names=keypoint_names,
        keypoint_positions_local_mm=keypoint_positions_local_mm,
        keypoint_trajectories_mm=keypoint_trajectories_mm,
        display_edges=display_edges,
    )

    log_data("keypoints", keypoint_names)
    log_exit("load_rigid_body_kinematics", f"{num_frames} frames")
    return result


def load_spine_trajectories(
    trajectories_csv_path: Path,
    topology_json_path: Path,
) -> SpineTrajectoryData:
    """Load spine trajectories from skull_and_spine_trajectories.csv and topology.

    This loads keypoint trajectories for spine keypoints (spine_t1, sacrum, tail_tip)
    from the measured trajectory file, NOT from a kinematics file.
    """
    log_enter("load_spine_trajectories")
    log_data("csv", trajectories_csv_path.name)
    log_data("json", topology_json_path.name)

    if not trajectories_csv_path.exists():
        raise FileNotFoundError(f"Trajectories CSV not found: {trajectories_csv_path}")
    if not topology_json_path.exists():
        raise FileNotFoundError(f"Topology JSON not found: {topology_json_path}")

    # Load topology to get keypoint names
    with open(topology_json_path, "r") as f:
        topology_data = json.load(f)

    # Spine keypoints we care about
    spine_keypoint_names = ["spine_t1", "sacrum", "tail_tip"]

    # Display edges for spine (chain from t1 -> sacrum -> tail_tip)
    display_edges: list[tuple[str, str]] = [
        ("spine_t1", "sacrum"),
        ("sacrum", "tail_tip"),
    ]

    # Load trajectories from CSV (tidy format)
    trajectories: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    with open(trajectories_csv_path, "r") as f:
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

    # Find the number of frames
    num_frames = 0
    for traj_name in spine_keypoint_names:
        if traj_name in trajectories:
            num_frames = max(num_frames, max(trajectories[traj_name].keys()) + 1)

    log_data("num_frames", num_frames)

    # Extract keypoint trajectories (convert mm to mm - they're already in mm)
    keypoint_trajectories_mm: dict[str, np.ndarray] = {}
    found_keypoints: list[str] = []

    for keypoint_name in spine_keypoint_names:
        if keypoint_name not in trajectories:
            raise RuntimeError(f"Spine keypoint '{keypoint_name}' not found in trajectories")

        found_keypoints.append(keypoint_name)
        frames_dict = trajectories[keypoint_name]
        arr = np.zeros((num_frames, 3), dtype=np.float64)

        for frame_num, components in frames_dict.items():
            arr[frame_num, 0] = components.get("x", 0.0)
            arr[frame_num, 1] = components.get("y", 0.0)
            arr[frame_num, 2] = components.get("z", 0.0)

        keypoint_trajectories_mm[keypoint_name] = arr

    result = SpineTrajectoryData(
        name="spine",
        num_frames=num_frames,
        keypoint_names=found_keypoints,
        keypoint_trajectories_mm=keypoint_trajectories_mm,
        display_edges=display_edges,
    )

    log_data("keypoints", found_keypoints)
    log_exit("load_spine_trajectories", f"{num_frames} frames, {len(found_keypoints)} keypoints")
    return result


def enforce_spine_rigid_lengths(spine_data: SpineTrajectoryData) -> SpineTrajectoryData:
    """Enforce constant bone lengths for spine keypoints."""
    log_enter("enforce_spine_rigid_lengths")

    num_frames = spine_data.num_frames
    trajectories = {k: v.copy() for k, v in spine_data.keypoint_trajectories_mm.items()}

    # Define bones
    spine_bones = [
        BoneDefinition(name="spine_t1_to_sacrum", head="spine_t1", tail="sacrum"),
        BoneDefinition(name="sacrum_to_tail_tip", head="sacrum", tail="tail_tip"),
    ]

    # Calculate median lengths
    for bone in spine_bones:
        if bone.head not in trajectories or bone.tail not in trajectories:
            raise RuntimeError(f"Cannot compute length for {bone.name}: missing keypoint")

        bone.lengths = []
        for f in range(num_frames):
            head = trajectories[bone.head][f, :]
            tail = trajectories[bone.tail][f, :]
            bone.lengths.append(np.linalg.norm(tail - head))
        valid = [length for length in bone.lengths if not math.isnan(length) and length > 0]
        if valid:
            bone.median = statistics.median(valid)
        log_data(f"{bone.name}_median_mm", bone.median)

    # Enforce lengths (adjust tail positions to maintain median length)
    hierarchy = {
        "spine_t1": ["sacrum"],
        "sacrum": ["tail_tip"],
        "tail_tip": [],
    }

    for bone in spine_bones:
        if bone.head not in trajectories or bone.tail not in trajectories:
            continue

        for f, raw_len in enumerate(bone.lengths):
            if np.isnan(raw_len) or raw_len == 0:
                continue
            head_pos = trajectories[bone.head][f, :]
            tail_pos = trajectories[bone.tail][f, :]
            vec = tail_pos - head_pos
            cur_len = np.linalg.norm(vec)
            if cur_len < 1e-10:
                continue
            delta = (vec / cur_len) * (bone.median - cur_len)

            def translate(name: str, d: np.ndarray) -> None:
                if name in trajectories:
                    trajectories[name][f, :] += d
                for child in hierarchy.get(name, []):
                    translate(child, d)

            translate(bone.tail, delta)

    result = SpineTrajectoryData(
        name=spine_data.name,
        num_frames=spine_data.num_frames,
        keypoint_names=spine_data.keypoint_names,
        keypoint_trajectories_mm=trajectories,
        display_edges=spine_data.display_edges,
    )

    log_exit("enforce_spine_rigid_lengths")
    return result


def load_gaze_kinematics(
    kinematics_csv_path: Path,
    reference_geometry_json_path: Path,
) -> GazeKinematicsData:
    """Load gaze kinematics and extract gaze_target trajectory."""
    log_enter("load_gaze_kinematics")

    # Load as rigid body first
    rb_data = load_rigid_body_kinematics(kinematics_csv_path, reference_geometry_json_path)

    # Extract gaze_target trajectory if present
    if "gaze_target" in rb_data.keypoint_trajectories_mm:
        gaze_target_mm = rb_data.keypoint_trajectories_mm["gaze_target"]
    else:
        # Compute gaze target as 100mm along +Z in gaze frame
        raise RuntimeError("gaze_target not found in keypoints, computing from orientation")
        gaze_target_mm = np.zeros((rb_data.num_frames, 3), dtype=np.float64)
        for f in range(rb_data.num_frames):
            q = rb_data.quaternion_wxyz[f]
            # Rotate [0, 0, 100] by quaternion
            gaze_dir = _rotate_vector_by_quaternion(np.array([0.0, 0.0, 100.0]), q)
            gaze_target_mm[f] = rb_data.position_xyz_mm[f] + gaze_dir

    result = GazeKinematicsData(
        name=rb_data.name,
        num_frames=rb_data.num_frames,
        timestamps=rb_data.timestamps,
        position_xyz_mm=rb_data.position_xyz_mm,
        quaternion_wxyz=rb_data.quaternion_wxyz,
        gaze_target_mm=gaze_target_mm,
    )

    # Debug: Check gaze target movement
    gaze_target_std = np.std(gaze_target_mm, axis=0)
    eye_pos_std = np.std(rb_data.position_xyz_mm, axis=0)
    log_data(f"{rb_data.name} eye_pos_std", f"[{eye_pos_std[0]:.2f}, {eye_pos_std[1]:.2f}, {eye_pos_std[2]:.2f}]")
    log_data(f"{rb_data.name} gaze_target_std", f"[{gaze_target_std[0]:.2f}, {gaze_target_std[1]:.2f}, {gaze_target_std[2]:.2f}]")

    log_exit("load_gaze_kinematics", f"{rb_data.num_frames} frames")
    return result


def load_eye_orientation(kinematics_csv_path: Path) -> EyeOrientationData:
    """Load eye orientation from eyeball kinematics CSV (tidy format)."""
    log_enter("load_eye_orientation")
    log_data("csv", kinematics_csv_path.name)

    data: dict[int, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    with open(kinematics_csv_path, "r") as f:
        header = f.readline().strip().split(",")
        frame_idx = header.index("frame")
        traj_idx = header.index("trajectory")
        comp_idx = header.index("component")
        val_idx = header.index("value")
        ts_idx = header.index("timestamp_s")

        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[frame_idx])
            traj_name = parts[traj_idx]
            component = parts[comp_idx]
            value = float(parts[val_idx])
            timestamp = float(parts[ts_idx])
            data[frame][traj_name][component] = value
            data[frame]["__timestamp__"] = {"value": timestamp}

    num_frames = max(data.keys()) + 1

    timestamps = np.zeros(num_frames, dtype=np.float64)
    for f in range(num_frames):
        if "__timestamp__" in data[f]:
            timestamps[f] = data[f]["__timestamp__"]["value"]

    quaternion_wxyz = np.zeros((num_frames, 4), dtype=np.float64)
    quaternion_wxyz[:, 0] = 1.0  # default identity
    for f in range(num_frames):
        if "orientation" in data[f]:
            quaternion_wxyz[f, 0] = data[f]["orientation"].get("w", 1.0)
            quaternion_wxyz[f, 1] = data[f]["orientation"].get("x", 0.0)
            quaternion_wxyz[f, 2] = data[f]["orientation"].get("y", 0.0)
            quaternion_wxyz[f, 3] = data[f]["orientation"].get("z", 0.0)

    name = kinematics_csv_path.parent.name.replace("_kinematics", "")

    # Debug: Check if quaternions have any variation (not all identity)
    quat_std = np.std(quaternion_wxyz, axis=0)
    quat_range = np.ptp(quaternion_wxyz, axis=0)  # peak-to-peak (max - min)
    log_data(f"{name} quat_std", f"w={quat_std[0]:.4f}, x={quat_std[1]:.4f}, y={quat_std[2]:.4f}, z={quat_std[3]:.4f}")
    log_data(f"{name} quat_range", f"w={quat_range[0]:.4f}, x={quat_range[1]:.4f}, y={quat_range[2]:.4f}, z={quat_range[3]:.4f}")
    if np.allclose(quat_std[1:], 0.0, atol=1e-5):
        raise RuntimeError(f"{name}: Eye quaternions appear CONSTANT (no eye rotation detected)!")

    result = EyeOrientationData(
        name=name,
        num_frames=num_frames,
        timestamps=timestamps,
        quaternion_wxyz=quaternion_wxyz,
    )

    log_exit("load_eye_orientation", f"{num_frames} frames")
    return result


def _rotate_vector_by_quaternion(
    v: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz format)."""
    w, x, y, z = q
    u = np.array([x, y, z])
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * w * uv + 2.0 * uuv


def _quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate of quaternion q (wxyz format). For unit quaternions, this is the inverse."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _transform_world_to_eyeball_local(
    point_world_mm: np.ndarray,
    eye_center_world_mm: np.ndarray,
    eye_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """
    Transform a point from world coordinates to eyeball-local coordinates.

    Args:
        point_world_mm: Point position in world coords (mm)
        eye_center_world_mm: Eye center position in world coords (mm)
        eye_quat_wxyz: Eye orientation quaternion (world -> eye local)

    Returns:
        Point position in eyeball-local coordinates (mm)
    """
    # Vector from eye center to point in world frame
    offset_world = point_world_mm - eye_center_world_mm

    # Rotate by inverse of eye quaternion to get local coords
    q_inv = _quaternion_conjugate(eye_quat_wxyz)
    offset_local = _rotate_vector_by_quaternion(offset_world, q_inv)

    return offset_local


NUM_PUPIL_POINTS = 8  # Number of pupil boundary points (p1-p8)


def load_eye_trajectories(trajectories_csv_path: Path) -> EyeTrajectoryData:
    """
    Load eye trajectory data from resampled trajectories CSV.

    The CSV contains trajectories for: tear_duct, outer_eye, pupil_center, p1-p8
    in world coordinates (mm).
    """
    log_enter("load_eye_trajectories")
    log_data("csv", trajectories_csv_path.name)

    data: dict[str, dict[int, dict[str, float]]] = {}

    with open(trajectories_csv_path, "r") as f:
        header = f.readline().strip().split(",")
        frame_idx = header.index("frame")
        traj_idx = header.index("trajectory")
        comp_idx = header.index("component")
        val_idx = header.index("value")

        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[frame_idx])
            traj_name = parts[traj_idx]
            component = parts[comp_idx]
            value = float(parts[val_idx])

            if traj_name not in data:
                data[traj_name] = {}
            if frame not in data[traj_name]:
                data[traj_name][frame] = {}
            data[traj_name][frame][component] = value

    # Get number of frames from any trajectory
    first_traj = next(iter(data.values()))
    num_frames = max(first_traj.keys()) + 1

    # Extract pupil center
    pupil_center_mm = np.zeros((num_frames, 3), dtype=np.float64)
    if "pupil_center" in data:
        for f in range(num_frames):
            if f in data["pupil_center"]:
                pupil_center_mm[f, 0] = data["pupil_center"][f].get("x", 0.0)
                pupil_center_mm[f, 1] = data["pupil_center"][f].get("y", 0.0)
                pupil_center_mm[f, 2] = data["pupil_center"][f].get("z", 0.0)

    # Extract pupil boundary points p1-p8
    pupil_points_mm = np.zeros((num_frames, NUM_PUPIL_POINTS, 3), dtype=np.float64)
    for i in range(NUM_PUPIL_POINTS):
        point_name = f"p{i + 1}"
        if point_name in data:
            for f in range(num_frames):
                if f in data[point_name]:
                    pupil_points_mm[f, i, 0] = data[point_name][f].get("x", 0.0)
                    pupil_points_mm[f, i, 1] = data[point_name][f].get("y", 0.0)
                    pupil_points_mm[f, i, 2] = data[point_name][f].get("z", 0.0)

    # Extract socket landmarks
    tear_duct_mm = np.zeros((num_frames, 3), dtype=np.float64)
    if "tear_duct" in data:
        for f in range(num_frames):
            if f in data["tear_duct"]:
                tear_duct_mm[f, 0] = data["tear_duct"][f].get("x", 0.0)
                tear_duct_mm[f, 1] = data["tear_duct"][f].get("y", 0.0)
                tear_duct_mm[f, 2] = data["tear_duct"][f].get("z", 0.0)

    outer_eye_mm = np.zeros((num_frames, 3), dtype=np.float64)
    if "outer_eye" in data:
        for f in range(num_frames):
            if f in data["outer_eye"]:
                outer_eye_mm[f, 0] = data["outer_eye"][f].get("x", 0.0)
                outer_eye_mm[f, 1] = data["outer_eye"][f].get("y", 0.0)
                outer_eye_mm[f, 2] = data["outer_eye"][f].get("z", 0.0)

    name = trajectories_csv_path.stem.replace("_trajectories_resampled", "")

    result = EyeTrajectoryData(
        name=name,
        num_frames=num_frames,
        pupil_center_mm=pupil_center_mm,
        pupil_points_mm=pupil_points_mm,
        tear_duct_mm=tear_duct_mm,
        outer_eye_mm=outer_eye_mm,
    )

    log_data("pupil_center_range", f"{np.nanmin(pupil_center_mm):.1f} to {np.nanmax(pupil_center_mm):.1f} mm")
    log_exit("load_eye_trajectories", f"{num_frames} frames")
    return result


def load_toy_trajectories(trajectories_csv_path: Path, zero_toy_z: bool = True) -> ToyTrajectoryData:
    """
    Load toy trajectory data from resampled trajectories CSV.

    The CSV contains trajectories for: toy_face, toy_top, toy_tail
    in world coordinates (mm).
    """
    log_enter("load_toy_trajectories")
    log_data("csv", trajectories_csv_path.name)

    data: dict[str, dict[int, dict[str, float]]] = {}

    with open(trajectories_csv_path, "r") as f:
        header = f.readline().strip().split(",")
        frame_idx = header.index("frame")
        traj_idx = header.index("trajectory")
        comp_idx = header.index("component")
        val_idx = header.index("value")

        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[frame_idx])
            traj_name = parts[traj_idx]
            component = parts[comp_idx]
            value = float(parts[val_idx])

            if traj_name not in data:
                data[traj_name] = {}
            if frame not in data[traj_name]:
                data[traj_name][frame] = {}
            data[traj_name][frame][component] = value

    # Get number of frames from any trajectory
    first_traj = next(iter(data.values()))
    num_frames = max(first_traj.keys()) + 1

    # Canonical keypoint names
    keypoint_names = ["toy_face", "toy_top", "toy_tail"]

    # Extract trajectories for each keypoint
    # NOTE: Z is intentionally zeroed out - toy is on a flat surface and Z data is unreliable
    keypoint_trajectories_mm: dict[str, np.ndarray] = {}
    for keypoint in keypoint_names:
        traj = np.zeros((num_frames, 3), dtype=np.float64)
        if keypoint in data:
            for f in range(num_frames):
                if f in data[keypoint]:
                    traj[f, 0] = -data[keypoint][f].get("x", 0.0)
                    traj[f, 1] = -data[keypoint][f].get("y", 0.0)
                    if not zero_toy_z:
                        traj[f, 2] = data[keypoint][f].get("z", 0.0)
        keypoint_trajectories_mm[keypoint] = traj

    # Display edges radiate from toy_top to face and tail
    display_edges: list[tuple[str, str]] = [
        ("toy_top", "toy_face"),
        ("toy_top", "toy_tail"),
    ]

    result = ToyTrajectoryData(
        name="toy",
        num_frames=num_frames,
        keypoint_names=keypoint_names,
        keypoint_trajectories_mm=keypoint_trajectories_mm,
        display_edges=display_edges,
    )

    log_data("keypoints", keypoint_names)
    log_exit("load_toy_trajectories", f"{num_frames} frames")
    return result


# ============================================================================
# VIDEO LOADING
# ============================================================================


def get_video_frame_count(filepath: Path) -> int:
    """
    Get the number of frames in a video file using Blender's native image loading.

    Compatible with Blender 4.0+ and 5.0+.
    Raises if the video cannot be loaded or has invalid frame count.
    """
    # Load the video as a movie image to get metadata
    # Use a unique name to avoid conflicts if the same video is loaded later
    temp_name = f"_temp_framecount_{filepath.name}"

    # Check if already loaded (avoid duplicates)
    existing = bpy.data.images.get(temp_name)
    if existing:
        bpy.data.images.remove(existing)

    try:
        video_image = bpy.data.images.load(str(filepath), check_existing=False)
        video_image.name = temp_name
        video_image.source = 'MOVIE'

        frame_count = video_image.frame_duration

        if frame_count <= 0:
            raise RuntimeError(f"Invalid frame count ({frame_count}) for video: {filepath}")

        return frame_count
    finally:
        # Clean up the temporary image to avoid clutter
        temp_img = bpy.data.images.get(temp_name)
        if temp_img:
            bpy.data.images.remove(temp_img)


def load_configured_videos() -> list[VideoData]:
    """
    Load videos from the VIDEOS configuration list.

    For resampled videos (from display_videos/), timestamps_path is not required.
    These videos have 1:1 frame mapping with the scene timeline.
    """
    log_enter("load_configured_videos")

    videos: list[VideoData] = []

    for video_config in VIDEOS:
        # Check required fields
        if "path" not in video_config:
            raise ValueError("Video config missing required 'path' field")

        filepath = Path(video_config["path"])

        if not filepath.exists():
            raise FileNotFoundError(f"Video file not found: {filepath}")

        # Get frame count from the video (raises on failure)
        n_frames = get_video_frame_count(filepath)
        log_data(f"Video frame count for {filepath.name}", n_frames)

        video_data = VideoData(
            name=video_config.get("name", filepath.stem),
            filepath=filepath,
            position=video_config.get("position", (0.0, VIDEO_PLANE_OFFSET_Y_M, VIDEO_PLANE_OFFSET_Z_M)),
            rotation_euler=video_config.get("rotation", (0.0, 0.0, 0.0)),
            width_m=video_config.get("width", VIDEO_PLANE_WIDTH_M if "eye" in video_config.get("name", "").lower() else VIDEO_PLANE_WIDTH_M*2),
            height_m=video_config.get("height", VIDEO_PLANE_HEIGHT_M if "eye" in video_config.get("name", "").lower() else VIDEO_PLANE_HEIGHT_M*2),
            n_frames=n_frames,
            is_resampled=True,  # Assume all videos in display_videos are resampled
        )
        videos.append(video_data)
        log_data("Loaded video config", f"{filepath.name} -> {video_data.name}")

    log_exit("load_configured_videos", f"{len(videos)} videos")
    return videos


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


def create_material(
    name: str, hex_color: str, emission: float = 1.0
) -> bpy.types.Material:
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


def create_cylinder(
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
        segments=8,
        radius1=radius,
        radius2=radius,
        depth=length,
    )
    # Shift so base is at origin, extends along +Z
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=(0, 0, length / 2))
    bm.to_mesh(mesh)
    bm.free()

    mesh.materials.append(material)
    collection.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


def create_cone(
    name: str,
    height: float,
    base_radius: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
) -> bpy.types.Object:
    """
    Create a cone mesh (point at top, base at bottom).

    The cone extends along +Z axis with base at origin and point at z=height.
    """
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=True,
        segments=12,
        radius1=base_radius,  # Bottom radius
        radius2=0.0,          # Top radius (point)
        depth=height,
    )
    # Shift so base is at origin, point extends along +Z
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=(0, 0, height / 2))
    bm.to_mesh(mesh)
    bm.free()

    mesh.materials.append(material)
    collection.objects.link(obj)
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
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    mesh.from_pydata(vertices, edges, [])
    mesh.update()

    mesh.materials.append(material)
    collection.objects.link(obj)
    return obj


# ============================================================================
# ANIMATION (Blender 4.x and 5.x compatible)
# ============================================================================

# Detect Blender version once at module load
BLENDER_VERSION = bpy.app.version
IS_BLENDER_5 = BLENDER_VERSION >= (5, 0, 0)
IS_BLENDER_44_PLUS = BLENDER_VERSION >= (4, 4, 0)

log_data("BLENDER_VERSION", BLENDER_VERSION)
log_data("IS_BLENDER_5", IS_BLENDER_5)
log_data("IS_BLENDER_44_PLUS", IS_BLENDER_44_PLUS)


def _get_or_create_fcurve(
    action: bpy.types.Action,
    anim_data: bpy.types.AnimData,
    data_path: str,
    index: int,
) -> bpy.types.FCurve:
    """Get or create an FCurve, handling Blender 4.x vs 5.x API differences."""
    if IS_BLENDER_5:
        # Blender 5.0+ uses layered animation system
        # Ensure we have a slot
        if not action.slots:
            slot = action.slots.new(id_type='OBJECT', name=anim_data.id_data.name)
        else:
            slot = action.slots[0]

        if anim_data.action_slot is None:
            anim_data.action_slot = slot
        else:
            slot = anim_data.action_slot

        # Ensure we have a layer
        if not action.layers:
            layer = action.layers.new(name="Layer")
        else:
            layer = action.layers[0]

        # Ensure we have a keyframe strip
        if not layer.strips:
            strip = layer.strips.new(type='KEYFRAME')
        else:
            strip = layer.strips[0]

        # Get or create channelbag for this slot
        channelbag = strip.channelbag(slot)
        if channelbag is None:
            channelbag = strip.channelbags.new(slot)

        # Create FCurve in the channelbag
        fc = channelbag.fcurves.new(data_path=data_path, index=index)
        return fc

    elif IS_BLENDER_44_PLUS:
        # Blender 4.4+ with slotted actions (but fcurves still on action)
        if anim_data.action_slot is None:
            slot = action.slots.new(id_type="OBJECT", name=anim_data.id_data.name)
            anim_data.action_slot = slot

        fc = action.fcurves.new(
            data_path=data_path,
            index=index,
            action_slot=anim_data.action_slot
        )
        return fc

    else:
        # Blender 4.0-4.3: simple fcurves directly on action
        fc = action.fcurves.new(data_path=data_path, index=index)
        return fc


def _get_or_create_fcurve_for_node_tree(
    action: bpy.types.Action,
    anim_data: bpy.types.AnimData,
    data_path: str,
    index: int,
) -> bpy.types.FCurve:
    """Get or create an FCurve for a node tree, handling Blender 4.x vs 5.x API differences."""
    if IS_BLENDER_5:
        # Blender 5.0+ uses layered animation system
        # For node trees, use NODETREE id_type
        if not action.slots:
            slot = action.slots.new(id_type='NODETREE', name=anim_data.id_data.name)
        else:
            slot = action.slots[0]

        if anim_data.action_slot is None:
            anim_data.action_slot = slot
        else:
            slot = anim_data.action_slot

        # Ensure we have a layer
        if not action.layers:
            layer = action.layers.new(name="Layer")
        else:
            layer = action.layers[0]

        # Ensure we have a keyframe strip
        if not layer.strips:
            strip = layer.strips.new(type='KEYFRAME')
        else:
            strip = layer.strips[0]

        # Get or create channelbag for this slot
        channelbag = strip.channelbag(slot)
        if channelbag is None:
            channelbag = strip.channelbags.new(slot)

        # Create FCurve in the channelbag
        fc = channelbag.fcurves.new(data_path=data_path, index=index)
        return fc

    elif IS_BLENDER_44_PLUS:
        # Blender 4.4+ with slotted actions
        if anim_data.action_slot is None:
            slot = action.slots.new(id_type="NODETREE", name=anim_data.id_data.name)
            anim_data.action_slot = slot

        fc = action.fcurves.new(
            data_path=data_path,
            index=index,
            action_slot=anim_data.action_slot
        )
        return fc

    else:
        # Blender 4.0-4.3: simple fcurves directly on action
        fc = action.fcurves.new(data_path=data_path, index=index)
        return fc


def animate_position(obj: bpy.types.Object, trajectory_m: np.ndarray) -> None:
    """Animate object location using fast bulk FCurve API."""
    num_frames = trajectory_m.shape[0]

    if not obj.animation_data:
        obj.animation_data_create()
    if not obj.animation_data.action:
        action = bpy.data.actions.new(name=f"{obj.name}_action")
        obj.animation_data.action = action

    action = obj.animation_data.action
    anim_data = obj.animation_data

    for i in range(3):
        fc = _get_or_create_fcurve(action, anim_data, "location", i)
        fc.keyframe_points.add(count=num_frames)

        keyframe_data = np.zeros(num_frames * 2, dtype=np.float32)
        keyframe_data[0::2] = np.arange(num_frames)
        keyframe_data[1::2] = trajectory_m[:, i].astype(np.float32)
        fc.keyframe_points.foreach_set("co", keyframe_data)

        fc.update()


def animate_rotation_quaternion(obj: bpy.types.Object, quaternions_wxyz: np.ndarray) -> None:
    """Animate object rotation using quaternions."""
    num_frames = quaternions_wxyz.shape[0]
    obj.rotation_mode = "QUATERNION"

    if not obj.animation_data:
        obj.animation_data_create()
    if not obj.animation_data.action:
        action = bpy.data.actions.new(name=f"{obj.name}_action")
        obj.animation_data.action = action

    action = obj.animation_data.action
    anim_data = obj.animation_data

    for i in range(4):
        fc = _get_or_create_fcurve(action, anim_data, "rotation_quaternion", i)
        fc.keyframe_points.add(count=num_frames)

        keyframe_data = np.zeros(num_frames * 2, dtype=np.float32)
        keyframe_data[0::2] = np.arange(num_frames)
        keyframe_data[1::2] = quaternions_wxyz[:, i].astype(np.float32)
        fc.keyframe_points.foreach_set("co", keyframe_data)

        fc.update()


# ============================================================================
# BUILDING FUNCTIONS
# ============================================================================


def build_rigid_body(
    rb_data: RigidBodyKinematicsData,
    parent_collection: bpy.types.Collection,
    colors: dict[str, str],
    axis_size: float,
) -> bpy.types.Object:
    """Build a rigid body visualization with animated pose and keypoints."""
    log_enter(f"build_rigid_body({rb_data.name})")

    # Create collection
    rb_coll = bpy.data.collections.new(rb_data.name)
    parent_collection.children.link(rb_coll)

    # Create main frame empty (animated position + rotation)
    frame_empty = create_empty(
        f"{rb_data.name}_frame",
        (0.0, 0.0, 0.0),
        rb_coll,
        display_type="ARROWS",
        display_size=axis_size,
    )
    frame_empty.rotation_mode = "QUATERNION"

    # Animate position and rotation
    animate_position(frame_empty, rb_data.position_xyz_mm * MM_TO_M)
    animate_rotation_quaternion(frame_empty, rb_data.quaternion_wxyz)

    # Create keypoint spheres (animated via constraints to empties)
    kp_mat = create_material(f"{rb_data.name}_kp_mat", colors["keypoints"], emission=2.0)
    keypoint_empties: dict[str, bpy.types.Object] = {}

    for kp_name in rb_data.keypoint_names:
        # Create empty for keypoint trajectory (unparented - world positions)
        kp_empty = create_empty(
            f"{rb_data.name}_{kp_name}_empty",
            (0.0, 0.0, 0.0),
            rb_coll,
        )
        keypoint_empties[kp_name] = kp_empty

        # Animate keypoint position (world coordinates)
        kp_traj_m = rb_data.keypoint_trajectories_mm[kp_name] * MM_TO_M
        animate_position(kp_empty, kp_traj_m)

        # Create sphere constrained to empty
        kp_sphere = create_sphere(
            f"{rb_data.name}_{kp_name}_sphere",
            (0.0, 0.0, 0.0),
            KEYPOINT_SPHERE_RADIUS,
            kp_mat,
            rb_coll,
        )
        kp_sphere.constraints.new(type="COPY_LOCATION").target = kp_empty

    # Create edges between keypoints
    if rb_data.display_edges:
        edge_mat = create_material(f"{rb_data.name}_edge_mat", colors["edges"], emission=1.0)
        for i, (kp1, kp2) in enumerate(rb_data.display_edges):
            if kp1 in keypoint_empties and kp2 in keypoint_empties:
                empty1 = keypoint_empties[kp1]
                empty2 = keypoint_empties[kp2]

                # Compute median edge length
                lengths = np.linalg.norm(
                    rb_data.keypoint_trajectories_mm[kp2]
                    - rb_data.keypoint_trajectories_mm[kp1],
                    axis=1,
                )
                median_len = float(np.nanmedian(lengths)) * MM_TO_M

                bone = create_cylinder(
                    f"{rb_data.name}_edge_{i}",
                    max(median_len, 0.001),
                    0.0005,
                    edge_mat,
                    rb_coll,
                )
                bone.constraints.new(type="COPY_LOCATION").target = empty1
                bone.constraints.new(type="DAMPED_TRACK").target = empty2
                bone.constraints["Damped Track"].track_axis = "TRACK_Z"

    log_exit(f"build_rigid_body({rb_data.name})")
    return frame_empty


def build_spine_trajectories(
    spine_data: SpineTrajectoryData,
    parent_collection: bpy.types.Collection,
    colors: dict[str, str],
) -> None:
    """Build spine visualization from keypoint trajectories (no pose)."""
    log_enter(f"build_spine_trajectories({spine_data.name})")

    # Create collection
    spine_coll = bpy.data.collections.new(spine_data.name)
    parent_collection.children.link(spine_coll)

    # Create keypoint spheres
    kp_mat = create_material(f"{spine_data.name}_kp_mat", colors["keypoints"], emission=2.0)
    keypoint_empties: dict[str, bpy.types.Object] = {}

    for kp_name in spine_data.keypoint_names:
        # Create empty for keypoint trajectory
        kp_empty = create_empty(
            f"{spine_data.name}_{kp_name}_empty",
            (0.0, 0.0, 0.0),
            spine_coll,
        )
        keypoint_empties[kp_name] = kp_empty

        # Animate keypoint position (world coordinates, mm -> m)
        kp_traj_m = spine_data.keypoint_trajectories_mm[kp_name] * MM_TO_M
        animate_position(kp_empty, kp_traj_m)

        # Create sphere constrained to empty
        kp_sphere = create_sphere(
            f"{spine_data.name}_{kp_name}_sphere",
            (0.0, 0.0, 0.0),
            KEYPOINT_SPHERE_RADIUS,
            kp_mat,
            spine_coll,
        )
        kp_sphere.constraints.new(type="COPY_LOCATION").target = kp_empty

    # Create edges between keypoints
    if spine_data.display_edges:
        edge_mat = create_material(f"{spine_data.name}_edge_mat", colors["edges"], emission=1.0)
        for i, (kp1, kp2) in enumerate(spine_data.display_edges):
            if kp1 in keypoint_empties and kp2 in keypoint_empties:
                empty1 = keypoint_empties[kp1]
                empty2 = keypoint_empties[kp2]

                # Compute median edge length
                if kp1 in spine_data.keypoint_trajectories_mm and kp2 in spine_data.keypoint_trajectories_mm:
                    lengths = np.linalg.norm(
                        spine_data.keypoint_trajectories_mm[kp2]
                        - spine_data.keypoint_trajectories_mm[kp1],
                        axis=1,
                    )
                    median_len = float(np.nanmedian(lengths)) * MM_TO_M
                else:
                    median_len = 0.02  # Default 2cm

                bone = create_cylinder(
                    f"{spine_data.name}_edge_{i}",
                    max(median_len, 0.001),
                    BONE_RADIUS,
                    edge_mat,
                    spine_coll,
                )
                bone.constraints.new(type="COPY_LOCATION").target = empty1
                bone.constraints.new(type="DAMPED_TRACK").target = empty2
                bone.constraints["Damped Track"].track_axis = "TRACK_Z"

    log_exit(f"build_spine_trajectories({spine_data.name})")


def build_gaze(
    gaze_data: GazeKinematicsData,
    parent_collection: bpy.types.Collection,
    colors: dict[str, str],
) -> None:
    """Build gaze visualization with ray from eye center to gaze target."""
    log_enter(f"build_gaze({gaze_data.name})")

    # Create collection
    gaze_coll = bpy.data.collections.new(gaze_data.name)
    parent_collection.children.link(gaze_coll)

    # Eye center empty (animated position)
    eye_center_empty = create_empty(
        f"{gaze_data.name}_eye_center",
        (0.0, 0.0, 0.0),
        gaze_coll,
    )
    animate_position(eye_center_empty, gaze_data.position_xyz_mm * MM_TO_M)

    # Gaze target empty (animated position)
    gaze_target_empty = create_empty(
        f"{gaze_data.name}_target_empty",
        (0.0, 0.0, 0.0),
        gaze_coll,
    )
    animate_position(gaze_target_empty, gaze_data.gaze_target_mm * MM_TO_M)

    # Gaze target sphere
    target_mat = create_material(f"{gaze_data.name}_target_mat", colors["gaze_target"], emission=3.0)
    target_sphere = create_sphere(
        f"{gaze_data.name}_target_sphere",
        (0.0, 0.0, 0.0),
        GAZE_TARGET_RADIUS,
        target_mat,
        gaze_coll,
    )
    target_sphere.constraints.new(type="COPY_LOCATION").target = gaze_target_empty

    # Gaze ray (cylinder from eye center to target)
    ray_mat = create_material(f"{gaze_data.name}_ray_mat", colors["gaze_ray"], emission=2.0)

    # Compute median ray length
    ray_lengths = np.linalg.norm(
        gaze_data.gaze_target_mm - gaze_data.position_xyz_mm, axis=1
    )
    median_ray_len = float(np.nanmedian(ray_lengths)) * MM_TO_M

    ray_cyl = create_cylinder(
        f"{gaze_data.name}_ray",
        median_ray_len,
        GAZE_RAY_RADIUS,
        ray_mat,
        gaze_coll,
    )
    ray_cyl.constraints.new(type="COPY_LOCATION").target = eye_center_empty
    ray_cyl.constraints.new(type="DAMPED_TRACK").target = gaze_target_empty
    ray_cyl.constraints["Damped Track"].track_axis = "TRACK_Z"

    log_exit(f"build_gaze({gaze_data.name})")


# Pupil visualization constants
PUPIL_CENTER_RADIUS = 0.0004  # 0.4mm
PUPIL_POINT_RADIUS = 0.0003  # 0.3mm
PUPIL_BOUNDARY_RADIUS = 0.0002  # 0.2mm for boundary cylinders

PUPIL_COLORS = {
    "left": {
        "center": "#000080",  # Dark blue
        "points": "#4682B4",  # Steel blue
        "boundary": "#000066",  # Darker blue
        "face": "#6495ED",  # Cornflower blue
    },
    "right": {
        "center": "#800000",  # Dark red/maroon
        "points": "#B44646",  # Muted red
        "boundary": "#660000",  # Darker red
        "face": "#CD5C5C",  # Indian red
    },
}


def build_pupil_geometry(
    eye_traj_data: EyeTrajectoryData,
    parent_collection: bpy.types.Collection,
    eye_side: str,  # "left" or "right"
) -> None:
    """
    Build animated pupil visualization with center, boundary ring, and filled face.

    Creates:
    - Pupil center sphere (animated position)
    - 8 pupil boundary point spheres (animated positions)
    - Boundary ring connecting p1->p2->...->p8->p1
    - Filled pupil face mesh
    """
    log_enter(f"build_pupil_geometry({eye_traj_data.name})")

    colors = PUPIL_COLORS.get(eye_side, PUPIL_COLORS["left"])

    # Create collection for pupil geometry
    pupil_coll = bpy.data.collections.new(f"{eye_traj_data.name}_pupil")
    parent_collection.children.link(pupil_coll)

    num_frames = eye_traj_data.num_frames

    # =========================================================================
    # PUPIL CENTER
    # =========================================================================
    center_empty = create_empty(
        f"{eye_traj_data.name}_pupil_center_empty",
        (0.0, 0.0, 0.0),
        pupil_coll,
    )
    animate_position(center_empty, eye_traj_data.pupil_center_mm * MM_TO_M)

    center_mat = create_material(
        f"{eye_traj_data.name}_pupil_center_mat",
        colors["center"],
        emission=2.0,
    )
    center_sphere = create_sphere(
        f"{eye_traj_data.name}_pupil_center",
        (0.0, 0.0, 0.0),
        PUPIL_CENTER_RADIUS,
        center_mat,
        pupil_coll,
    )
    center_sphere.constraints.new(type="COPY_LOCATION").target = center_empty

    # =========================================================================
    # PUPIL BOUNDARY POINTS (p1-p8)
    # =========================================================================
    point_empties: list[bpy.types.Object] = []
    point_mat = create_material(
        f"{eye_traj_data.name}_pupil_point_mat",
        colors["points"],
        emission=1.5,
    )

    for i in range(NUM_PUPIL_POINTS):
        point_name = f"p{i + 1}"

        # Create empty and animate
        pt_empty = create_empty(
            f"{eye_traj_data.name}_{point_name}_empty",
            (0.0, 0.0, 0.0),
            pupil_coll,
        )
        animate_position(pt_empty, eye_traj_data.pupil_points_mm[:, i, :] * MM_TO_M)
        point_empties.append(pt_empty)

        # Create sphere at each point
        pt_sphere = create_sphere(
            f"{eye_traj_data.name}_{point_name}",
            (0.0, 0.0, 0.0),
            PUPIL_POINT_RADIUS,
            point_mat,
            pupil_coll,
        )
        pt_sphere.constraints.new(type="COPY_LOCATION").target = pt_empty

    # =========================================================================
    # PUPIL BOUNDARY RING (cylinders connecting consecutive points)
    # =========================================================================
    boundary_mat = create_material(
        f"{eye_traj_data.name}_pupil_boundary_mat",
        colors["boundary"],
        emission=1.0,
    )

    for i in range(NUM_PUPIL_POINTS):
        next_i = (i + 1) % NUM_PUPIL_POINTS

        # Compute median edge length
        p1_traj = eye_traj_data.pupil_points_mm[:, i, :]
        p2_traj = eye_traj_data.pupil_points_mm[:, next_i, :]
        edge_lengths = np.linalg.norm(p2_traj - p1_traj, axis=1)
        median_len = float(np.nanmedian(edge_lengths)) * MM_TO_M

        edge_cyl = create_cylinder(
            f"{eye_traj_data.name}_boundary_{i}_{next_i}",
            max(median_len, 0.0001),
            PUPIL_BOUNDARY_RADIUS,
            boundary_mat,
            pupil_coll,
        )
        edge_cyl.constraints.new(type="COPY_LOCATION").target = point_empties[i]
        edge_cyl.constraints.new(type="DAMPED_TRACK").target = point_empties[next_i]
        edge_cyl.constraints["Damped Track"].track_axis = "TRACK_Z"

    # =========================================================================
    # PUPIL FACE (mesh with hooks for animation)
    # =========================================================================
    # Create mesh with center + 8 boundary vertices forming a fan
    face_mat = create_material(
        f"{eye_traj_data.name}_pupil_face_mat",
        colors["face"],
        emission=0.5,
    )
    # Make face material semi-transparent
    face_mat.blend_method = 'BLEND'
    face_mat.use_nodes = True
    # Find the Principled BSDF node and set alpha
    for node in face_mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs['Alpha'].default_value = 0.7

    # Create face mesh
    face_mesh = bpy.data.meshes.new(f"{eye_traj_data.name}_pupil_face_mesh")
    face_obj = bpy.data.objects.new(f"{eye_traj_data.name}_pupil_face", face_mesh)
    pupil_coll.objects.link(face_obj)

    # Initial vertex positions (will be overridden by hooks)
    # Vertex 0 = center, vertices 1-8 = p1-p8
    initial_center = eye_traj_data.pupil_center_mm[0] * MM_TO_M
    initial_points = eye_traj_data.pupil_points_mm[0] * MM_TO_M

    vertices = [tuple(initial_center)]
    for i in range(NUM_PUPIL_POINTS):
        vertices.append(tuple(initial_points[i]))

    # Create fan triangles: center -> p_i -> p_{i+1}
    faces = []
    for i in range(NUM_PUPIL_POINTS):
        next_i = (i + 1) % NUM_PUPIL_POINTS
        faces.append((0, i + 1, next_i + 1))

    face_mesh.from_pydata(vertices, [], faces)
    face_mesh.update()
    face_mesh.materials.append(face_mat)

    # Add hook modifiers to make vertices follow empties
    # Hook for center vertex (index 0)
    hook_center = face_obj.modifiers.new(name="Hook_Center", type='HOOK')
    hook_center.object = center_empty
    hook_center.vertex_indices_set([0])

    # Hooks for boundary vertices (indices 1-8)
    for i in range(NUM_PUPIL_POINTS):
        hook = face_obj.modifiers.new(name=f"Hook_p{i+1}", type='HOOK')
        hook.object = point_empties[i]
        hook.vertex_indices_set([i + 1])

    log_data("pupil_points", NUM_PUPIL_POINTS)
    log_exit(f"build_pupil_geometry({eye_traj_data.name})")


def build_toy_visualization(
    toy_data: ToyTrajectoryData,
    parent_collection: bpy.types.Collection,
) -> None:
    """
    Build toy visualization with big cones for each keypoint.

    Creates:
    - Big cones for toy_face, toy_top, toy_tail (20mm tall, 8mm radius)
    - Connecting edges between keypoints

    The cones are positioned at the keypoint locations and animated.
    """
    log_enter("build_toy_visualization")

    # Create toy collection
    toy_coll = bpy.data.collections.new("toy")
    parent_collection.children.link(toy_coll)

    # Store empties for edge connections
    keypoint_empties: dict[str, bpy.types.Object] = {}

    # Create cones for each keypoint
    for keypoint_name in toy_data.keypoint_names:
        traj_mm = toy_data.keypoint_trajectories_mm[keypoint_name]
        color = TOY_COLORS.get(keypoint_name, "#FF8C00")

        # Create material with bright emission
        mat = create_material(
            f"toy_{keypoint_name}_mat",
            color,
            emission=2.5,  # Bright!
        )

        # Create empty for position tracking (for edge connections)
        initial_pos_m = tuple(traj_mm[0] * MM_TO_M)
        empty = create_empty(
            f"toy_{keypoint_name}_empty",
            initial_pos_m,
            toy_coll,
            display_type="PLAIN_AXES",
            display_size=0.001,  # Small display
        )
        animate_position(empty, traj_mm * MM_TO_M)
        keypoint_empties[keypoint_name] = empty

        # Create cone
        cone = create_cone(
            f"toy_{keypoint_name}_cone",
            height=TOY_CONE_HEIGHT,
            base_radius=TOY_CONE_RADIUS,
            material=mat,
            collection=toy_coll,
        )

        # Parent cone to empty (so it follows the trajectory)
        cone.parent = empty
        cone.location = (0, 0, 0)  # Relative to parent

        log_data(f"{keypoint_name} initial pos (mm)", f"{traj_mm[0]}")

    # Create connecting edges between keypoints
    edge_mat = create_material(
        "toy_edges_mat",
        TOY_COLORS["edges"],
        emission=1.5,
    )

    for from_name, to_name in toy_data.display_edges:
        from_empty = keypoint_empties[from_name]
        to_empty = keypoint_empties[to_name]

        # Compute median edge length for cylinder
        from_traj = toy_data.keypoint_trajectories_mm[from_name]
        to_traj = toy_data.keypoint_trajectories_mm[to_name]
        edge_lengths = np.linalg.norm(to_traj - from_traj, axis=1)
        median_len_m = float(np.nanmedian(edge_lengths)) * MM_TO_M

        # Create cylinder for edge
        edge_cyl = create_cylinder(
            f"toy_edge_{from_name}_{to_name}",
            max(median_len_m, 0.001),
            TOY_EDGE_RADIUS,
            edge_mat,
            toy_coll,
        )

        # Constrain cylinder to track between points
        edge_cyl.constraints.new(type="COPY_LOCATION").target = from_empty
        track_constraint = edge_cyl.constraints.new(type="DAMPED_TRACK")
        track_constraint.target = to_empty
        track_constraint.track_axis = "TRACK_Z"

    log_data("keypoints", len(toy_data.keypoint_names))
    log_data("edges", len(toy_data.display_edges))
    log_exit("build_toy_visualization")


# Pupil visualization constants for socket view
SOCKET_PUPIL_CENTER_RADIUS = 0.0006  # 0.6mm - slightly larger in socket view
SOCKET_PUPIL_POINT_RADIUS = 0.0004  # 0.4mm
SOCKET_PUPIL_BOUNDARY_RADIUS = 0.00025  # 0.25mm for boundary cylinders

SOCKET_PUPIL_COLORS = {
    "left": {
        "center": "#000080",  # Dark blue
        "points": "#4682B4",  # Steel blue
        "boundary": "#1E90FF",  # Dodger blue
    },
    "right": {
        "center": "#800000",  # Dark red/maroon
        "points": "#B44646",  # Muted red
        "boundary": "#FF4500",  # Orange-red
    },
}


def build_eye_in_socket(
    eye_name: str,
    eye_data: EyeOrientationData,
    eye_traj_data: EyeTrajectoryData,
    eye_center_world_mm: np.ndarray,
    parent_collection: bpy.types.Collection,
    colors: dict[str, str],
    offset_mm: np.ndarray,
) -> None:
    """
    Build eye-in-socket visualization showing eye rotation relative to socket.

    Creates an animated eyeball frame at a fixed offset position with pupil geometry,
    useful for seeing the eye rotation and pupil shape independent of head movement.

    Args:
        eye_name: Name of the eye (e.g., "left_eye", "right_eye")
        eye_data: Eye orientation data with animated quaternions
        eye_traj_data: Eye trajectory data with pupil points in world coordinates
        eye_center_world_mm: Eye center positions in world coords (N, 3) from gaze data
        parent_collection: Parent Blender collection
        colors: Color dictionary for eye visualization
        offset_mm: Offset position for the socket view in mm

    Raises:
        ValueError: If data arrays have mismatched frame counts
        RuntimeError: If pupil transformation produces invalid data
    """
    log_enter(f"build_eye_in_socket({eye_name})")

    # Validate frame counts match
    n_frames_orientation = eye_data.num_frames
    n_frames_trajectory = eye_traj_data.num_frames
    n_frames_center = eye_center_world_mm.shape[0]

    if n_frames_orientation != n_frames_trajectory:
        raise ValueError(
            f"{eye_name}: Frame count mismatch between orientation ({n_frames_orientation}) "
            f"and trajectory ({n_frames_trajectory}) data!"
        )
    if n_frames_orientation != n_frames_center:
        raise ValueError(
            f"{eye_name}: Frame count mismatch between orientation ({n_frames_orientation}) "
            f"and eye center ({n_frames_center}) data!"
        )

    num_frames = n_frames_orientation
    log_data("num_frames", num_frames)

    # Validate eye center data
    if np.any(np.isnan(eye_center_world_mm)):
        nan_count = np.sum(np.isnan(eye_center_world_mm))
        raise ValueError(f"{eye_name}: Eye center contains {nan_count} NaN values!")

    # Create collection
    eye_coll = bpy.data.collections.new(f"{eye_name}_socket_view")
    parent_collection.children.link(eye_coll)

    # Socket frame (fixed position, identity rotation)
    socket_empty = create_empty(
        f"{eye_name}_socket_frame",
        tuple(offset_mm * MM_TO_M),
        eye_coll,
        display_type="ARROWS",
        display_size=EYE_SOCKET_AXIS_SIZE,
    )

    # Eyeball frame (animated rotation, parented to socket)
    eyeball_empty = create_empty(
        f"{eye_name}_eyeball_frame",
        (0.0, 0.0, 0.0),
        eye_coll,
        parent=socket_empty,
        display_type="ARROWS",
        display_size=EYE_BALL_AXIS_SIZE,
    )
    eyeball_empty.rotation_mode = "QUATERNION"
    animate_rotation_quaternion(eyeball_empty, eye_data.quaternion_wxyz)

    # Wireframe sphere for eyeball
    eye_radius_m = 0.0035  # 3.5mm typical ferret eye radius
    wireframe_mat = create_material(
        f"{eye_name}_wireframe_mat", colors["socket_frame"], emission=0.5
    )

    mesh = bpy.data.meshes.new(f"{eye_name}_wireframe_mesh")
    wireframe_obj = bpy.data.objects.new(f"{eye_name}_wireframe", mesh)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=eye_radius_m)
    bm.to_mesh(mesh)
    bm.free()

    mesh.materials.append(wireframe_mat)
    eye_coll.objects.link(wireframe_obj)
    wireframe_obj.parent = eyeball_empty

    mod = wireframe_obj.modifiers.new(name="Wireframe", type="WIREFRAME")
    mod.thickness = 0.0002
    mod.use_replace = True

    # Gaze direction cone (at +Z of eyeball frame)
    gaze_mat = create_material(f"{eye_name}_gaze_cone_mat", colors["gaze_ray"], emission=3.0)
    gaze_cone_length = 0.1  # 10cm
    gaze_cone_radius = 0.0002  # 0.2mm

    cone_mesh = bpy.data.meshes.new(f"{eye_name}_gaze_cone_mesh")
    gaze_cone = bpy.data.objects.new(f"{eye_name}_gaze_cone", cone_mesh)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=True,
        segments=12,
        radius1=gaze_cone_radius,
        radius2=0.0,
        depth=gaze_cone_length,
    )
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=(0, 0, gaze_cone_length / 2))
    bm.to_mesh(cone_mesh)
    bm.free()

    cone_mesh.materials.append(gaze_mat)
    eye_coll.objects.link(gaze_cone)
    gaze_cone.parent = eyeball_empty
    gaze_cone.location = (0, 0, eye_radius_m)

    # Create small sphere at the tip of the gaze cone
    tip_sphere = create_sphere(
        f"{eye_name}_gaze_tip_sphere",
        (0.0, 0.0, eye_radius_m + gaze_cone_length),
        GAZE_TARGET_RADIUS,
        gaze_mat,
        eye_coll,
        parent=eyeball_empty,
    )
    log_data(f"{eye_name} gaze_tip_location", f"(0.0, 0.0, {eye_radius_m + gaze_cone_length:.4f})")

    # =========================================================================
    # PUPIL GEOMETRY IN SOCKET VIEW
    # =========================================================================
    # Transform pupil points from world coordinates to eyeball-local coordinates
    # These will be parented to eyeball_empty and animated

    eye_side = "left" if "left" in eye_name.lower() else "right"
    pupil_colors = SOCKET_PUPIL_COLORS[eye_side]

    log_step(f"Transforming pupil points to eyeball-local coords ({num_frames} frames)...")

    # Pre-allocate arrays for local coordinates
    pupil_center_local_mm = np.zeros((num_frames, 3), dtype=np.float64)
    pupil_points_local_mm = np.zeros((num_frames, NUM_PUPIL_POINTS, 3), dtype=np.float64)

    # Transform each frame
    for f in range(num_frames):
        eye_center = eye_center_world_mm[f]
        eye_quat = eye_data.quaternion_wxyz[f]

        # Validate quaternion is not zero
        quat_norm = np.linalg.norm(eye_quat)
        if quat_norm < 1e-10:
            raise RuntimeError(
                f"{eye_name}: Invalid quaternion at frame {f} (near-zero norm: {quat_norm})"
            )

        # Transform pupil center
        pupil_center_local_mm[f] = _transform_world_to_eyeball_local(
            eye_traj_data.pupil_center_mm[f],
            eye_center,
            eye_quat,
        )

        # Transform pupil boundary points
        for i in range(NUM_PUPIL_POINTS):
            pupil_points_local_mm[f, i] = _transform_world_to_eyeball_local(
                eye_traj_data.pupil_points_mm[f, i],
                eye_center,
                eye_quat,
            )

    # Validate transformed data
    if np.any(np.isnan(pupil_center_local_mm)):
        nan_frames = np.where(np.any(np.isnan(pupil_center_local_mm), axis=1))[0]
        raise RuntimeError(
            f"{eye_name}: Pupil center transformation produced NaN at frames: {nan_frames[:10]}..."
        )
    if np.any(np.isnan(pupil_points_local_mm)):
        raise RuntimeError(f"{eye_name}: Pupil points transformation produced NaN values!")

    # Log stats for debugging
    center_range = np.ptp(pupil_center_local_mm, axis=0)
    log_data(f"{eye_name} pupil_center_local_range_mm", f"x={center_range[0]:.2f}, y={center_range[1]:.2f}, z={center_range[2]:.2f}")

    # Convert to meters for Blender
    pupil_center_local_m = pupil_center_local_mm * MM_TO_M
    pupil_points_local_m = pupil_points_local_mm * MM_TO_M

    # -------------------------------------------------------------------------
    # PUPIL CENTER (in eyeball local coords, parented to eyeball_empty)
    # -------------------------------------------------------------------------
    center_mat = create_material(
        f"{eye_name}_socket_pupil_center_mat",
        pupil_colors["center"],
        emission=2.5,
    )

    # Create empty for pupil center - parented to eyeball_empty
    center_empty = create_empty(
        f"{eye_name}_socket_pupil_center_empty",
        (0.0, 0.0, 0.0),
        eye_coll,
        parent=eyeball_empty,
    )
    # Animate position in local coordinates
    animate_position(center_empty, pupil_center_local_m)

    # Create visible sphere constrained to follow the empty
    center_sphere = create_sphere(
        f"{eye_name}_socket_pupil_center",
        (0.0, 0.0, 0.0),
        SOCKET_PUPIL_CENTER_RADIUS,
        center_mat,
        eye_coll,
    )
    center_sphere.constraints.new(type="COPY_LOCATION").target = center_empty

    # -------------------------------------------------------------------------
    # PUPIL BOUNDARY POINTS (p1-p8, in eyeball local coords)
    # -------------------------------------------------------------------------
    point_empties: list[bpy.types.Object] = []
    point_mat = create_material(
        f"{eye_name}_socket_pupil_point_mat",
        pupil_colors["points"],
        emission=2.0,
    )

    for i in range(NUM_PUPIL_POINTS):
        point_name = f"p{i + 1}"

        # Create empty parented to eyeball_empty
        pt_empty = create_empty(
            f"{eye_name}_socket_{point_name}_empty",
            (0.0, 0.0, 0.0),
            eye_coll,
            parent=eyeball_empty,
        )
        # Animate position in local coordinates
        animate_position(pt_empty, pupil_points_local_m[:, i, :])
        point_empties.append(pt_empty)

        # Create visible sphere
        pt_sphere = create_sphere(
            f"{eye_name}_socket_{point_name}",
            (0.0, 0.0, 0.0),
            SOCKET_PUPIL_POINT_RADIUS,
            point_mat,
            eye_coll,
        )
        pt_sphere.constraints.new(type="COPY_LOCATION").target = pt_empty

    # -------------------------------------------------------------------------
    # PUPIL BOUNDARY RING (cylinders connecting consecutive points)
    # -------------------------------------------------------------------------
    boundary_mat = create_material(
        f"{eye_name}_socket_pupil_boundary_mat",
        pupil_colors["boundary"],
        emission=1.5,
    )

    for i in range(NUM_PUPIL_POINTS):
        next_i = (i + 1) % NUM_PUPIL_POINTS

        # Compute median edge length for cylinder sizing
        p1_traj = pupil_points_local_m[:, i, :]
        p2_traj = pupil_points_local_m[:, next_i, :]
        edge_lengths = np.linalg.norm(p2_traj - p1_traj, axis=1)
        median_len = float(np.nanmedian(edge_lengths))

        if median_len < 1e-6:
            raise RuntimeError(
                f"{eye_name}: Pupil boundary edge {i}->{next_i} has near-zero median length!"
            )

        edge_cyl = create_cylinder(
            f"{eye_name}_socket_boundary_{i}_{next_i}",
            max(median_len, 0.0001),
            SOCKET_PUPIL_BOUNDARY_RADIUS,
            boundary_mat,
            eye_coll,
        )
        edge_cyl.constraints.new(type="COPY_LOCATION").target = point_empties[i]
        edge_cyl.constraints.new(type="DAMPED_TRACK").target = point_empties[next_i]
        edge_cyl.constraints["Damped Track"].track_axis = "TRACK_Z"

    log_data(f"{eye_name} socket_pupil_points", NUM_PUPIL_POINTS)
    log_exit(f"build_eye_in_socket({eye_name})")


def build_video_plane(
    video_data: VideoData,
    parent_collection: bpy.types.Collection,
    num_scene_frames: int,
) -> bpy.types.Object:
    """
    Build a video plane for a resampled video with 1:1 frame mapping.

    Works with Blender 4.0+ and 5.0+.

    For resampled videos, scene frame N corresponds directly to video frame N,
    so no complex timestamp-based keyframing is needed.

    Args:
        video_data: VideoData object containing video info and placement
        parent_collection: Collection to add the video plane to
        num_scene_frames: Number of frames in the scene timeline

    Returns:
        The created plane object
    """
    log_enter(f"build_video_plane({video_data.name})")
    log_data("filepath", video_data.filepath)
    log_data("is_resampled", video_data.is_resampled)
    log_data("scene_frames", num_scene_frames)

    if not video_data.filepath.exists():
        raise FileNotFoundError(f"Video file not found: {video_data.filepath}")

    # Create collection for this video
    video_coll = bpy.data.collections.new(f"{video_data.name}_collection")
    parent_collection.children.link(video_coll)

    # Load the video as an image
    video_image = bpy.data.images.load(str(video_data.filepath), check_existing=True)
    video_image.source = 'MOVIE'

    # Get video dimensions and compute plane size
    vid_width, vid_height = video_image.size[0], video_image.size[1]
    if vid_width <= 0 or vid_height <= 0:
        raise RuntimeError(f"Invalid video dimensions ({vid_width}x{vid_height}) for: {video_data.filepath}")

    aspect_ratio = vid_width / vid_height
    plane_width = video_data.width_m
    plane_height = plane_width / aspect_ratio
    log_data("video_size", f"{vid_width}x{vid_height} (aspect {aspect_ratio:.3f})")

    # Get actual frame count from Blender
    video_frame_count = video_image.frame_duration
    if video_frame_count <= 0:
        raise RuntimeError(f"Invalid frame count ({video_frame_count}) for video: {video_data.filepath}")
    log_data("video_frames", video_frame_count)

    video_image.colorspace_settings.name = 'sRGB'

    # Create the plane mesh
    mesh = bpy.data.meshes.new(f"{video_data.name}_mesh")
    plane_obj = bpy.data.objects.new(video_data.name, mesh)

    # Create plane geometry (centered, in XY plane)
    half_w = plane_width / 2
    half_h = plane_height / 2
    vertices = [
        (-half_w, -half_h, 0),
        (half_w, -half_h, 0),
        (half_w, half_h, 0),
        (-half_w, half_h, 0),
    ]
    faces = [(0, 1, 2, 3)]

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Add UV coordinates for the video texture
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    uv_layer = mesh.uv_layers.active.data
    uv_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i, uv in enumerate(uv_coords):
        uv_layer[i].uv = uv

    # Create material with video texture
    mat = bpy.data.materials.new(name=f"{video_data.name}_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.location = (200, 0)
    emission_node.inputs['Strength'].default_value = 1.0

    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.name = "video_texture"
    tex_node.location = (0, 0)
    tex_node.image = video_image
    tex_node.extension = 'CLIP'

    # =========================================================================
    # SIMPLE 1:1 FRAME MAPPING FOR RESAMPLED VIDEOS
    # =========================================================================
    # Resampled videos have the same number of frames as the scene timeline,
    # so we just set it to play 1:1 with the timeline (no keyframing needed!)

    tex_node.image_user.use_auto_refresh = True
    tex_node.image_user.use_cyclic = False
    tex_node.image_user.frame_start = 0  # Video frame 0 plays at scene frame 0
    tex_node.image_user.frame_offset = 0  # No offset needed
    tex_node.image_user.frame_duration = video_frame_count + 100  # Ensure full video is available

    log_step(f"Configured 1:1 frame mapping for resampled video {video_data.name}")
    log_data("frame_start", 0)
    log_data("frame_offset", 0)
    log_data("frame_duration", video_frame_count)

    links.new(tex_node.outputs['Color'], emission_node.inputs['Color'])
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    mesh.materials.append(mat)

    plane_obj.location = video_data.position
    plane_obj.rotation_euler = video_data.rotation_euler

    video_coll.objects.link(plane_obj)

    log_data("plane_size", f"{plane_width:.3f} x {plane_height:.3f} m")
    log_data("position", video_data.position)
    log_exit(f"build_video_plane({video_data.name})")

    return plane_obj


def build_videos(
    videos: list[VideoData],
    parent_collection: bpy.types.Collection,
    num_scene_frames: int,
) -> list[bpy.types.Object]:
    """Build all video planes with 1:1 frame mapping for resampled videos."""
    log_enter("build_videos")

    created_planes: list[bpy.types.Object] = []

    for video_data in videos:
        plane = build_video_plane(video_data, parent_collection, num_scene_frames)
        created_planes.append(plane)

    log_exit("build_videos", f"{len(created_planes)} planes created")
    return created_planes


def setup_viewport() -> None:
    """Configure viewport for visualization."""
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "MATERIAL"
                    space.clip_end = 10.0
                    space.clip_start = 0.0001



def save_blend_file(output_path: Path) -> None:
    """Save the current Blender scene to a .blend file."""
    log_step(f"Saving blend file to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    start_time = time.time()

    print("\n" + "=" * 70)
    print("   FERRET FULL VISUALIZATION")
    print("   Skull + Spine + Eyes + Gaze")
    print("=" * 70 + "\n")

    log_enter("main")

    print("PHASE 1: CLEARING SCENE")
    clear_scene()

    print("\nPHASE 2: LOADING DATA")

    num_frames = 0


    # Load skull kinematics
    skull_csv = DATA_DIR / "skull_kinematics" / "skull_kinematics.csv"
    skull_json = DATA_DIR / "skull_kinematics" / "skull_reference_geometry.json"
    if skull_csv.exists() and skull_json.exists():
        log_step("Loading skull kinematics...")
        skull_data:RigidBodyKinematicsData = load_rigid_body_kinematics(skull_csv, skull_json)
        num_frames = max(num_frames, skull_data.num_frames)
        print(f"  Skull: {skull_data.num_frames} frames, keypoints: {skull_data.keypoint_names}")
    else:
        raise RuntimeError(f"Skull data not found at {skull_csv}")

    # Load spine trajectories from RESAMPLED data (same directory as skull kinematics)
    spine_traj_csv = DATA_DIR / "skull_and_spine_trajectories_resampled.csv"
    spine_topology_json = DATA_DIR / "skull_kinematics" / "skull_and_spine_topology.json"
    if spine_traj_csv.exists() and spine_topology_json.exists():
        log_step("Loading spine trajectories...")
        spine_data: SpineTrajectoryData  = load_spine_trajectories(spine_traj_csv, spine_topology_json)
        log_step("Enforcing spine rigid lengths...")
        spine_data = enforce_spine_rigid_lengths(spine_data)
        num_frames = max(num_frames, spine_data.num_frames)
        print(f"  Spine: {spine_data.num_frames} frames, keypoints: {spine_data.keypoint_names}")
    else:
        raise RuntimeError(f"Spine trajectory data not found at {spine_traj_csv}")

    # Load gaze kinematics
    gaze_dir = DATA_DIR / "gaze_kinematics"

    left_gaze_csv = gaze_dir / "left_gaze_kinematics.csv"
    left_gaze_json = gaze_dir / "left_gaze_reference_geometry.json"
    if left_gaze_csv.exists() and left_gaze_json.exists():
        log_step("Loading left gaze kinematics...")
        left_gaze_data: GazeKinematicsData  = load_gaze_kinematics(left_gaze_csv, left_gaze_json)
        num_frames = max(num_frames, left_gaze_data.num_frames)
        print(f"  Left gaze: {left_gaze_data.num_frames} frames")
    else:
        raise RuntimeError(f"Left gaze data not found at {left_gaze_csv}")

    right_gaze_csv = gaze_dir / "right_gaze_kinematics.csv"
    right_gaze_json = gaze_dir / "right_gaze_reference_geometry.json"
    if right_gaze_csv.exists() and right_gaze_json.exists():
        log_step("Loading right gaze kinematics...")
        right_gaze_data: GazeKinematicsData  = load_gaze_kinematics(right_gaze_csv, right_gaze_json)
        num_frames = max(num_frames, right_gaze_data.num_frames)
        print(f"  Right gaze: {right_gaze_data.num_frames} frames")
    else:
        raise RuntimeError(f"Right gaze data not found at {right_gaze_csv}")

    # Load raw eye kinematics (for eye-in-socket visualization)

    left_eye_dir = DATA_DIR / "left_eye_kinematics"
    right_eye_dir = DATA_DIR / "right_eye_kinematics"

    # Try eyeball kinematics (newer format)
    left_eye_csv = left_eye_dir / "left_eye_kinematics.csv"
    if left_eye_csv.exists():
        log_step("Loading left eye orientation...")
        left_eye_kin: EyeOrientationData  = load_eye_orientation(left_eye_csv)
        print(f"  Left eye orientation: {left_eye_kin.num_frames} frames")
    else:
        raise RuntimeError(f"Left eye kinematics not found at {left_eye_csv}")

    right_eye_csv = right_eye_dir / "right_eye_kinematics.csv"
    if right_eye_csv.exists():
        log_step("Loading right eye orientation...")
        right_eye_kin: EyeOrientationData  = load_eye_orientation(right_eye_csv)
        print(f"  Right eye orientation: {right_eye_kin.num_frames} frames")
    else:
        raise RuntimeError(f"Right eye kinematics not found at {right_eye_csv}")

    # Load eye trajectories (for pupil geometry visualization)
    left_eye_traj_csv = left_eye_dir / "left_eye_trajectories_resampled.csv"
    if not left_eye_traj_csv.exists():
        raise FileNotFoundError(f"Left eye trajectories not found at {left_eye_traj_csv}")
    log_step("Loading left eye trajectories...")
    left_eye_traj: EyeTrajectoryData = load_eye_trajectories(left_eye_traj_csv)
    print(f"  Left eye trajectories: {left_eye_traj.num_frames} frames")

    right_eye_traj_csv = right_eye_dir / "right_eye_trajectories_resampled.csv"
    if not right_eye_traj_csv.exists():
        raise FileNotFoundError(f"Right eye trajectories not found at {right_eye_traj_csv}")
    log_step("Loading right eye trajectories...")
    right_eye_traj: EyeTrajectoryData = load_eye_trajectories(right_eye_traj_csv)
    print(f"  Right eye trajectories: {right_eye_traj.num_frames} frames")

    # Load toy trajectories
    toy_traj_csv = DATA_DIR / "toy_trajectories_resampled.csv"
    if not toy_traj_csv.exists():
        raise FileNotFoundError(f"Toy trajectories not found at {toy_traj_csv}")
    log_step("Loading toy trajectories...")
    toy_data: ToyTrajectoryData = load_toy_trajectories(toy_traj_csv)
    num_frames = max(num_frames, toy_data.num_frames)
    print(f"  Toy trajectories: {toy_data.num_frames} frames, keypoints: {toy_data.keypoint_names}")

    if num_frames == 0:
        raise FileNotFoundError(f"No data found in {DATA_DIR}")

    print("\nPHASE 3: BUILDING SCENE")
    setup_scene(num_frames)

    # Main collection
    main_coll = bpy.data.collections.new("FerretFull")
    bpy.context.scene.collection.children.link(main_coll)

    # Create wireframe cage
    log_step("Creating 1m wireframe cage")
    cage_mat = create_material("cage_mat", "#404040", emission=0.3)
    create_wireframe_cube("reference_cage", CAGE_SIZE_M, cage_mat, main_coll)

    # Build skull
    if skull_data is not None:
        log_step("Building skull visualization")
        build_rigid_body(skull_data, main_coll, SKULL_COLORS, SKULL_AXIS_SIZE)

    # Build spine (from trajectories, not kinematics)
    if spine_data is not None:
        log_step("Building spine visualization")
        build_spine_trajectories(spine_data, main_coll, SPINE_COLORS)

    # Build gaze visualizations
    if left_gaze_data is not None:
        log_step("Building left gaze visualization")
        build_gaze(left_gaze_data, main_coll, LEFT_COLORS)

    if right_gaze_data is not None:
        log_step("Building right gaze visualization")
        build_gaze(right_gaze_data, main_coll, RIGHT_COLORS)

    # Build eye-in-socket visualizations (offset to side for clarity)
    # These show eye rotation relative to socket, positioned away from main viz
    # NOW INCLUDES PUPIL GEOMETRY IN SOCKET-LOCAL COORDINATES
    EYE_SOCKET_OFFSET_MM = 150.0  # 150mm offset to the side

    # Left eye-in-socket visualization (REQUIRED - will fail if data missing)
    log_step("Building left eye-in-socket visualization with pupil geometry")
    build_eye_in_socket(
        eye_name="left_eye",
        eye_data=left_eye_kin,
        eye_traj_data=left_eye_traj,
        eye_center_world_mm=left_gaze_data.position_xyz_mm,
        parent_collection=main_coll,
        colors=LEFT_COLORS,
        offset_mm=np.array([-EYE_SOCKET_OFFSET_MM, 0.0, 0.0]),
    )

    # Right eye-in-socket visualization (REQUIRED - will fail if data missing)
    log_step("Building right eye-in-socket visualization with pupil geometry")
    build_eye_in_socket(
        eye_name="right_eye",
        eye_data=right_eye_kin,
        eye_traj_data=right_eye_traj,
        eye_center_world_mm=right_gaze_data.position_xyz_mm,
        parent_collection=main_coll,
        colors=RIGHT_COLORS,
        offset_mm=np.array([EYE_SOCKET_OFFSET_MM, 0.0, 0.0]),
    )

    # Build pupil geometry visualizations (world coordinates)
    log_step("Building left pupil geometry visualization")
    build_pupil_geometry(left_eye_traj, main_coll, "left")

    log_step("Building right pupil geometry visualization")
    build_pupil_geometry(right_eye_traj, main_coll, "right")

    # Build toy visualization
    log_step("Building toy visualization")
    build_toy_visualization(toy_data, main_coll)

    # Load and build video planes
    log_step("Loading configured videos...")
    videos = load_configured_videos()
    if videos:
        log_step(f"Building {len(videos)} video planes...")
        # Use num_frames for 1:1 mapping with resampled videos
        build_videos(videos, main_coll, num_frames)
    else:
        log_step("No videos configured (add paths to VIDEOS list to enable)")

    print("\nPHASE 4: VIEWPORT SETUP")
    setup_viewport()

    print("\nPHASE 5: SAVING BLEND FILE")
    save_blend_file(OUTPUT_PATH)

    elapsed = time.time() - start_time
    log_exit("main")

    print("\n" + "=" * 70)
    print("   DONE!")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Frames: {bpy.context.scene.frame_end + 1}")
    print(f"   FPS: {bpy.context.scene.render.fps}")
    print(f"   Saved to: {OUTPUT_PATH}")
    print("=" * 70)
    print("   Color Legend:")
    print("   - Yellow/Gold = Skull")
    print("   - Green = Spine")
    print("   - Red/Magenta = Right eye gaze")
    print("   - Blue/Cyan = Left eye gaze")
    print("   - Orange/Gold cones = Toy (face, top, tail)")
    print("   - Dark blue/steel blue = Left pupil (world + socket view)")
    print("   - Dark red/muted red = Right pupil (world + socket view)")
    print("")
    print("   Layout:")
    print("   - Center: Skull + Spine with gaze rays + Toy + Pupil geometry (world)")
    print("   - Left side (-150mm): Left eye-in-socket view WITH pupil geometry")
    print("   - Right side (+150mm): Right eye-in-socket view WITH pupil geometry")
    print("   - Video planes: Positioned around the scene (if videos found)")
    print("")
    print("   Pupil Visualization:")
    print("   - World view: Shows pupil moving with head/eye in 3D space")
    print("   - Socket view: Shows pupil in eye-local coords (isolates eye rotation)")
    print("")
    print("   Press SPACEBAR to play")
    print("=" * 70 + "\n")


if __name__ == "__main__" or __name__ == "<run_path>":
    main()