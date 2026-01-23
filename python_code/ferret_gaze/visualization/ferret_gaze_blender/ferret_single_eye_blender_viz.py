"""
Ferret Single Eye Blender Visualization
========================================

Visualizes a single eye in Blender 4.4+.

Creates:
- Eye socket frame (ARROWS) with +Z toward socket midpoint, +X toward animal's left
- Eyeball frame (ARROWS) with animated rotation
- Wireframe eyeball sphere
- Tracked pupil center and boundary points (p1-p8)
- Socket landmarks (tear_duct, outer_eye)
- 1m wireframe cage for reference

Usage:
1. Open in Blender 4.4+
2. Edit the configuration section (DATA_DIR, EYE_NAME, OUTPUT_PATH)
3. Run with Alt+P
4. Press Spacebar to play animation

Color scheme:
- Right eye: Red/Magenta tones
- Left eye: Blue/Cyan tones
"""

print("\n" + "=" * 70)
print("   FERRET SINGLE EYE BLENDER VIZ - STARTING")
print("=" * 70)

print("[BOOT] Importing standard library...")
import sys
print(f"[BOOT] Python: {sys.version}")
import time
from pathlib import Path
print("[BOOT] Standard library imported.")

print("[BOOT] Importing bpy...")
import bpy
print(f"[BOOT] Blender: {bpy.app.version_string}")
print(f"[BOOT] Blend file: {bpy.data.filepath or '(unsaved)'}")

# ============================================================================
# CONFIGURATION - EDIT THESE!
# ============================================================================

# Data directory containing eye kinematics
DATA_DIR = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye_kinematics")

# Which eye to visualize ("left_eye" or "right_eye")
EYE_NAME = "right_eye"

# Output path for saved .blend file
OUTPUT_PATH = DATA_DIR / "single_eye_visualization.blend"

# Cage size in meters
CAGE_SIZE_M = 1.0

# ============================================================================
# MANUAL OVERRIDE - Set this if auto-detection fails!
# ============================================================================
# Set to the folder containing ferret_eye_blender_core.py:
MANUAL_SCRIPT_DIR: Path | None = None
# Example: MANUAL_SCRIPT_DIR = Path(r"D:\my\scripts\folder")

print(f"[CONFIG] DATA_DIR = {DATA_DIR}")
print(f"[CONFIG] EYE_NAME = {EYE_NAME}")
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
    
    # Method 4: Check DATA_DIR and its parents
    print("[FIND] Method 4: DATA_DIR and parents")
    check_dirs = [
        DATA_DIR,
        DATA_DIR.parent,
        DATA_DIR.parent.parent,
        DATA_DIR.parent.parent.parent,
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
# MAIN
# ============================================================================

def main() -> None:
    start_time = time.time()

    print("\n" + "=" * 70)
    print("   FERRET SINGLE EYE VISUALIZATION")
    print(f"   Eye: {EYE_NAME}")
    print("=" * 70 + "\n")

    core.log_enter("main")

    print("PHASE 1: CLEARING SCENE")
    core.clear_scene()

    print("\nPHASE 2: LOADING DATA")

    # Load eye data
    if EYE_NAME not in ["left_eye", "right_eye"]:
        raise ValueError(f"EYE_NAME must be 'left_eye' or 'right_eye', got '{EYE_NAME}'")

    geom_path = DATA_DIR / f"{EYE_NAME}_reference_geometry.json"
    kin_path = DATA_DIR / f"{EYE_NAME}_kinematics.csv"

    print(f"  Geometry: {geom_path}")
    print(f"  Kinematics: {kin_path}")

    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry file not found: {geom_path}")
    if not kin_path.exists():
        raise FileNotFoundError(f"Kinematics file not found: {kin_path}")

    core.log_step(f"Loading {EYE_NAME}...")
    geometry = core.EyeGeometry.from_json(geom_path)
    kinematics = core.load_eye_kinematics(kin_path)
    print(f"  Loaded {EYE_NAME}: {kinematics.num_frames} frames")

    print("\nPHASE 3: BUILDING SCENE")
    core.setup_scene(kinematics.num_frames)

    # Main collection
    main_coll = bpy.data.collections.new("FerretEye")
    bpy.context.scene.collection.children.link(main_coll)

    # Create wireframe cage
    core.log_step("Creating 1m wireframe cage")
    cage_mat = core.create_material("cage_mat", "#404040", emission=0.3)
    core.create_wireframe_cube("reference_cage", CAGE_SIZE_M, cage_mat, main_coll)

    # Build the eye
    core.build_single_eye(
        eye_name=EYE_NAME,
        kinematics=kinematics,
        geometry=geometry,
        parent_collection=main_coll,
        parent_object=None,
        eye_position_in_parent_mm=None,
    )

    print("\nPHASE 4: VIEWPORT SETUP")
    core.setup_viewport()

    print("\nPHASE 5: SAVING BLEND FILE")
    core.save_blend_file(OUTPUT_PATH)

    elapsed = time.time() - start_time
    core.log_exit("main")

    is_right = "right" in EYE_NAME.lower()
    color_desc = "Red/Magenta" if is_right else "Blue/Cyan"

    print("\n" + "=" * 70)
    print("   DONE!")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Frames: {bpy.context.scene.frame_end + 1}")
    print(f"   FPS: {bpy.context.scene.render.fps}")
    print(f"   Saved to: {OUTPUT_PATH}")
    print("=" * 70)
    print(f"   {EYE_NAME}: {color_desc} tones")
    print("   Large arrows = Socket frame (+Z toward socket midpoint)")
    print("   Small arrows = Eyeball frame (rotates with eye)")
    print("   Cone = Gaze direction")
    print("   Press SPACEBAR to play")
    print("=" * 70 + "\n")


if __name__ == "__main__" or __name__ == "<run_path>":
    main()
