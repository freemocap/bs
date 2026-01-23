"""
Ferret Skull + Eyes Combined Blender Visualization
==================================================

Visualizes skull and both eyes with proper parenting in Blender 4.4+.

Hierarchy:
- skull_origin (animated position + rotation)
  - skull keypoints and bones
  - left_eye_socket (at skull's left_eye keypoint)
    - left_eyeball (animated rotation)
  - right_eye_socket (at skull's right_eye keypoint)
    - right_eyeball (animated rotation)

This creates a complete kinematic chain where:
- Skull moves/rotates in world space
- Eye sockets are fixed to skull (move/rotate with skull)
- Eyeballs rotate independently within their sockets

Creates:
- Skull keypoints and bones (orange)
- Left eye (blue tones) attached to skull's left_eye keypoint
- Right eye (red/magenta tones) attached to skull's right_eye keypoint
- 1m wireframe cage for reference

Usage:
1. Open in Blender 4.4+
2. Edit the configuration section (SKULL_DATA_DIR, EYE_DATA_DIR, OUTPUT_PATH)
3. Run with Alt+P
4. Press Spacebar to play animation
"""

print("\n" + "=" * 70)
print("   FERRET SKULL + EYES BLENDER VIZ - STARTING")
print("=" * 70)

print("[BOOT] Importing standard library...")
import sys
print(f"[BOOT] Python: {sys.version}")
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
print("[BOOT] Standard library imported.")

print("[BOOT] Importing numpy...")
import numpy as np
print(f"[BOOT] numpy: {np.__version__}")

print("[BOOT] Importing bpy...")
import bpy
print(f"[BOOT] Blender: {bpy.app.version_string}")
print(f"[BOOT] Blend file: {bpy.data.filepath or '(unsaved)'}")

# ============================================================================
# CONFIGURATION - EDIT THESE!
# ============================================================================

# Skull data directory
SKULL_DATA_DIR = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output")

# Eye kinematics directory
EYE_DATA_DIR = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye_kinematics")

# Output path for saved .blend file
OUTPUT_PATH = EYE_DATA_DIR / "skull_and_eyes_visualization.blend"

# Cage size in meters
CAGE_SIZE_M = 1.0

# Skull visualization sizes
SKULL_SPHERE_RADIUS = 0.002  # 2mm
SKULL_BONE_RADIUS = 0.0015  # 1.5mm
SKULL_ORIGIN_SIZE = 0.05  # 5cm arrows

# ============================================================================
# MANUAL OVERRIDE - Set this if auto-detection fails!
# ============================================================================
# Set to the folder containing ferret_eye_blender_core.py:
MANUAL_SCRIPT_DIR: Path | None = None
# Example: MANUAL_SCRIPT_DIR = Path(r"D:\my\scripts\folder")

print(f"[CONFIG] SKULL_DATA_DIR = {SKULL_DATA_DIR}")
print(f"[CONFIG] EYE_DATA_DIR = {EYE_DATA_DIR}")
print(f"[CONFIG] OUTPUT_PATH = {OUTPUT_PATH}")
print(f"[CONFIG] MANUAL_SCRIPT_DIR = {MANUAL_SCRIPT_DIR}")

# ============================================================================
# SCRIPT DIRECTORY DETECTION
# ============================================================================

def _find_core_module() -> Path:
    """Find the directory containing ferret_eye_blender_core.py"""
    
    core_filename = "ferret_eye_blender_core.py"
    
    def check_dir(d: Path, method: str) -> Path | None:
        if d is None:
            print(f"[FIND]   {method}: None")
            return None
        print(f"[FIND]   {method}: {d}")
        core_path = d / core_filename
        exists = core_path.exists()
        print(f"[FIND]     -> {core_path} exists={exists}")
        return d if exists else None
    
    print(f"[FIND] Looking for {core_filename}...")
    
    # Method 0: Manual override
    print("[FIND] Method 0: MANUAL_SCRIPT_DIR")
    if MANUAL_SCRIPT_DIR is not None:
        result = check_dir(MANUAL_SCRIPT_DIR, "manual")
        if result:
            print("[FIND] SUCCESS via MANUAL_SCRIPT_DIR!")
            return result
        else:
            print("[FIND] WARNING: MANUAL_SCRIPT_DIR set but core not found there!")
    
    # Method 1: __file__ (works when run as external script)
    print("[FIND] Method 1: __file__")
    try:
        script_dir = Path(__file__).parent.resolve()
        result = check_dir(script_dir, "__file__.parent")
        if result:
            print("[FIND] SUCCESS via __file__!")
            return result
    except NameError:
        print("[FIND]   __file__ not defined (in Blender text editor)")
    
    # Method 2: Blender text blocks with filepath
    print("[FIND] Method 2: Blender text blocks")
    print(f"[FIND]   {len(bpy.data.texts)} text block(s):")
    for text in bpy.data.texts:
        print(f"[FIND]   - '{text.name}' filepath='{text.filepath}'")
        if text.filepath:
            script_dir = Path(text.filepath).parent.resolve()
            result = check_dir(script_dir, f"text[{text.name}].parent")
            if result:
                print(f"[FIND] SUCCESS via text block '{text.name}'!")
                return result
    
    # Method 3: Blend file directory
    print("[FIND] Method 3: Blend file directory")
    if bpy.data.filepath:
        blend_dir = Path(bpy.data.filepath).parent
        result = check_dir(blend_dir, "blend_file.parent")
        if result:
            print("[FIND] SUCCESS via blend file!")
            return result
    else:
        print("[FIND]   Blend file not saved")
    
    # Method 4: Check EYE_DATA_DIR and its parents
    print("[FIND] Method 4: EYE_DATA_DIR and parents")
    check_dirs = [
        EYE_DATA_DIR,
        EYE_DATA_DIR.parent,
        EYE_DATA_DIR.parent.parent,
        EYE_DATA_DIR.parent.parent.parent,
        SKULL_DATA_DIR,
        SKULL_DATA_DIR.parent,
    ]
    for d in check_dirs:
        if d.exists():
            result = check_dir(d, f"data_dir_relative")
            if result:
                print("[FIND] SUCCESS via DATA_DIR relative!")
                return result
    
    # Method 5: Check sys.path
    print("[FIND] Method 5: sys.path entries")
    for p in sys.path[:10]:
        if p:
            result = check_dir(Path(p), "sys.path")
            if result:
                print("[FIND] SUCCESS via sys.path!")
                return result
    
    # FAILED
    print("[FIND] " + "!" * 50)
    print("[FIND] FAILED: Could not find ferret_eye_blender_core.py!")
    print("[FIND] " + "!" * 50)
    print("[FIND] Full sys.path:")
    for i, p in enumerate(sys.path):
        print(f"[FIND]   [{i}] {p}")
    print("[FIND]")
    print("[FIND] TO FIX: Set MANUAL_SCRIPT_DIR at the top of this script!")
    print("[FIND] Example: MANUAL_SCRIPT_DIR = Path(r'D:\\my\\scripts')")
    
    raise RuntimeError(
        f"Cannot find {core_filename}!\n\n"
        "FIX: Edit this script and set MANUAL_SCRIPT_DIR to the folder "
        "containing ferret_eye_blender_core.py"
    )

print("\n[IMPORT] Finding core module...")
_SCRIPT_DIR = _find_core_module()
print(f"[IMPORT] Found at: {_SCRIPT_DIR}")

if str(_SCRIPT_DIR) not in sys.path:
    print(f"[IMPORT] Adding to sys.path")
    sys.path.insert(0, str(_SCRIPT_DIR))

print("[IMPORT] Importing ferret_eye_blender_core...")
import ferret_eye_blender_core as core
print(f"[IMPORT] SUCCESS! Loaded: {core.__file__}")
print("=" * 70 + "\n")


# ============================================================================
# SKULL DATA STRUCTURES
# ============================================================================

@dataclass
class SkullGeometry:
    """Reference geometry for the skull in local coordinates."""
    keypoints: dict[str, np.ndarray]  # name -> [x, y, z] in mm
    display_edges: list[tuple[str, str]]
    rigid_edges: list[tuple[str, str]]

    @classmethod
    def from_json(cls, path: Path) -> "SkullGeometry":
        core.log_enter("SkullGeometry.from_json")
        with open(path, "r") as f:
            data = json.load(f)

        keypoints: dict[str, np.ndarray] = {}
        for name, coords in data["keypoints"].items():
            keypoints[name] = np.array([coords["x"], coords["y"], coords["z"]])

        result = cls(
            keypoints=keypoints,
            display_edges=[tuple(e) for e in data.get("display_edges", [])],
            rigid_edges=[tuple(e) for e in data.get("rigid_edges", [])],
        )
        core.log_data("keypoints", list(keypoints.keys()))
        core.log_exit("SkullGeometry.from_json")
        return result


@dataclass
class SkullKinematics:
    """Per-frame skull pose."""
    num_frames: int
    position: np.ndarray  # [num_frames, 3] in meters
    orientation: np.ndarray  # [num_frames, 4] quaternion (w, x, y, z)


# ============================================================================
# SKULL DATA LOADING
# ============================================================================

def load_skull_kinematics(csv_path: Path) -> SkullKinematics:
    """Load skull kinematics from CSV."""
    core.log_enter("load_skull_kinematics")
    core.log_data("path", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

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
                core.log_step(f"Read {line_count} lines...")

    core.log_data("total_lines", line_count)

    num_frames = max(data.keys()) + 1
    core.log_data("num_frames", num_frames)

    position = np.zeros((num_frames, 3), dtype=np.float64)
    for f in range(num_frames):
        if "position" in data[f]:
            position[f, 0] = data[f]["position"].get("x", 0.0) * core.MM_TO_M
            position[f, 1] = data[f]["position"].get("y", 0.0) * core.MM_TO_M
            position[f, 2] = data[f]["position"].get("z", 0.0) * core.MM_TO_M

    orientation = np.zeros((num_frames, 4), dtype=np.float64)
    for f in range(num_frames):
        if "orientation" in data[f]:
            orientation[f, 0] = data[f]["orientation"].get("w", 1.0)
            orientation[f, 1] = data[f]["orientation"].get("x", 0.0)
            orientation[f, 2] = data[f]["orientation"].get("y", 0.0)
            orientation[f, 3] = data[f]["orientation"].get("z", 0.0)

    result = SkullKinematics(
        num_frames=num_frames,
        position=position,
        orientation=orientation,
    )
    core.log_exit("load_skull_kinematics", f"{num_frames} frames")
    return result


# ============================================================================
# SKULL COLORS
# ============================================================================

SKULL_COLORS = {
    "origin": "#FFAA00",
    "keypoint": "#FF6600",
    "bone": "#FF8800",
}


# ============================================================================
# SKULL BUILDING
# ============================================================================

def build_skull(
    kinematics: SkullKinematics,
    geometry: SkullGeometry,
    parent_collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Build skull visualization and return the skull origin empty."""
    core.log_enter("build_skull")

    skull_coll = bpy.data.collections.new("Skull")
    parent_collection.children.link(skull_coll)

    # Skull origin (animated position + rotation)
    core.log_step("Creating skull origin")
    skull_origin = core.create_empty(
        "skull_origin",
        tuple(kinematics.position[0]),
        skull_coll,
        display_type="ARROWS",
        display_size=SKULL_ORIGIN_SIZE,
    )
    core.animate_position(skull_origin, kinematics.position)
    core.animate_rotation_quaternion(skull_origin, kinematics.orientation)

    # Skull keypoints (parented to skull origin)
    core.log_step("Creating skull keypoints")
    keypoint_mat = core.create_material("skull_keypoint_mat", SKULL_COLORS["keypoint"], emission=2.0)
    skull_empties: dict[str, bpy.types.Object] = {}

    for kp_name, local_pos in geometry.keypoints.items():
        local_pos_m = local_pos * core.MM_TO_M
        sphere = core.create_sphere(
            f"skull_{kp_name}_sphere",
            tuple(local_pos_m),
            SKULL_SPHERE_RADIUS,
            keypoint_mat,
            skull_coll,
            parent=skull_origin,
        )
        skull_empties[kp_name] = sphere

    # Skull bones
    core.log_step("Creating skull bones")
    bone_mat = core.create_material("skull_bone_mat", SKULL_COLORS["bone"], emission=1.0)

    for head_name, tail_name in geometry.display_edges:
        if head_name not in skull_empties or tail_name not in skull_empties:
            continue

        head_local = geometry.keypoints[head_name] * core.MM_TO_M
        tail_local = geometry.keypoints[tail_name] * core.MM_TO_M
        length = float(np.linalg.norm(tail_local - head_local))

        bone = core.create_cylinder(
            f"skull_{head_name}_to_{tail_name}_bone",
            length,
            SKULL_BONE_RADIUS,
            bone_mat,
            skull_coll,
        )
        bone.parent = skull_origin
        bone.constraints.new(type="COPY_LOCATION").target = skull_empties[head_name]
        bone.constraints.new(type="DAMPED_TRACK").target = skull_empties[tail_name]
        bone.constraints["Damped Track"].track_axis = "TRACK_Z"

    core.log_exit("build_skull")
    return skull_origin


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    start_time = time.time()

    print("\n" + "=" * 70)
    print("   FERRET SKULL + EYES COMBINED VISUALIZATION")
    print("   Skull (Orange) + Left Eye (Blue) + Right Eye (Red/Magenta)")
    print("=" * 70 + "\n")

    core.log_enter("main")

    print("PHASE 1: CLEARING SCENE")
    core.clear_scene()

    print("\nPHASE 2: LOADING SKULL DATA")
    skull_kin_path = SKULL_DATA_DIR / "skull_kinematics.csv"
    skull_geom_path = SKULL_DATA_DIR / "skull_reference_geometry.json"

    print(f"  Skull kinematics: {skull_kin_path}")
    print(f"  Skull geometry: {skull_geom_path}")

    if not skull_kin_path.exists():
        raise FileNotFoundError(f"Skull kinematics not found: {skull_kin_path}")
    if not skull_geom_path.exists():
        raise FileNotFoundError(f"Skull geometry not found: {skull_geom_path}")

    skull_kinematics = load_skull_kinematics(skull_kin_path)
    skull_geometry = SkullGeometry.from_json(skull_geom_path)
    print(f"  Loaded skull: {skull_kinematics.num_frames} frames")

    print("\nPHASE 3: LOADING EYE DATA")

    left_kinematics: core.EyeKinematics | None = None
    left_geometry: core.EyeGeometry | None = None
    right_kinematics: core.EyeKinematics | None = None
    right_geometry: core.EyeGeometry | None = None

    # Try to load left eye
    left_geom_path = EYE_DATA_DIR / "left_eye_reference_geometry.json"
    left_kin_path = EYE_DATA_DIR / "left_eye_kinematics.csv"
    print(f"  Left eye geometry: {left_geom_path}")
    print(f"  Left eye kinematics: {left_kin_path}")
    if left_geom_path.exists() and left_kin_path.exists():
        core.log_step("Loading left eye...")
        left_geometry = core.EyeGeometry.from_json(left_geom_path)
        left_kinematics = core.load_eye_kinematics(left_kin_path)
        print(f"  Loaded left eye: {left_kinematics.num_frames} frames")
    else:
        core.log_warn("Left eye data not found")

    # Try to load right eye
    right_geom_path = EYE_DATA_DIR / "right_eye_reference_geometry.json"
    right_kin_path = EYE_DATA_DIR / "right_eye_kinematics.csv"
    print(f"  Right eye geometry: {right_geom_path}")
    print(f"  Right eye kinematics: {right_kin_path}")
    if right_geom_path.exists() and right_kin_path.exists():
        core.log_step("Loading right eye...")
        right_geometry = core.EyeGeometry.from_json(right_geom_path)
        right_kinematics = core.load_eye_kinematics(right_kin_path)
        print(f"  Loaded right eye: {right_kinematics.num_frames} frames")
    else:
        core.log_warn("Right eye data not found")

    # Get eye positions from skull geometry
    left_eye_pos_mm = skull_geometry.keypoints.get("left_eye", np.array([0.0, 12.0, 0.0]))
    right_eye_pos_mm = skull_geometry.keypoints.get("right_eye", np.array([0.0, -12.0, 0.0]))

    print(f"  Left eye skull position: {left_eye_pos_mm} mm")
    print(f"  Right eye skull position: {right_eye_pos_mm} mm")

    print("\nPHASE 4: BUILDING SCENE")
    num_frames = skull_kinematics.num_frames
    core.setup_scene(num_frames)

    # Main collection
    main_coll = bpy.data.collections.new("FerretSkullAndEyes")
    bpy.context.scene.collection.children.link(main_coll)

    # Create wireframe cage
    core.log_step("Creating 1m wireframe cage")
    cage_mat = core.create_material("cage_mat", "#404040", emission=0.3)
    core.create_wireframe_cube("reference_cage", CAGE_SIZE_M, cage_mat, main_coll)

    # Build skull
    core.log_step("Building skull")
    skull_origin = build_skull(skull_kinematics, skull_geometry, main_coll)

    # Build left eye (parented to skull, positioned at skull's left_eye keypoint)
    if left_kinematics is not None and left_geometry is not None:
        core.log_step("Building left eye (attached to skull)")
        core.build_single_eye(
            eye_name="left_eye",
            kinematics=left_kinematics,
            geometry=left_geometry,
            parent_collection=main_coll,
            parent_object=skull_origin,
            eye_position_in_parent_mm=left_eye_pos_mm,
        )

    # Build right eye (parented to skull, positioned at skull's right_eye keypoint)
    if right_kinematics is not None and right_geometry is not None:
        core.log_step("Building right eye (attached to skull)")
        core.build_single_eye(
            eye_name="right_eye",
            kinematics=right_kinematics,
            geometry=right_geometry,
            parent_collection=main_coll,
            parent_object=skull_origin,
            eye_position_in_parent_mm=right_eye_pos_mm,
        )

    print("\nPHASE 5: VIEWPORT SETUP")
    core.setup_viewport()

    print("\nPHASE 6: SAVING BLEND FILE")
    core.save_blend_file(OUTPUT_PATH)

    elapsed = time.time() - start_time
    core.log_exit("main")

    print("\n" + "=" * 70)
    print("   DONE!")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Frames: {bpy.context.scene.frame_end + 1}")
    print(f"   FPS: {bpy.context.scene.render.fps}")
    print(f"   Saved to: {OUTPUT_PATH}")
    print("=" * 70)
    print("   Hierarchy:")
    print("     skull_origin (animated pos + rot)")
    print("       └── skull keypoints/bones (orange)")
    print("       └── left_eye_socket → left_eyeball (blue)")
    print("       └── right_eye_socket → right_eyeball (red/magenta)")
    print("")
    print("   Orange = Skull")
    print("   Blue = Left eye")
    print("   Red/Magenta = Right eye")
    print("   Cones = Gaze directions")
    print("   Press SPACEBAR to play")
    print("=" * 70 + "\n")


if __name__ == "__main__" or __name__ == "<run_path>":
    main()
