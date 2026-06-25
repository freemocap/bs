from python_code.viz.blender.blender_helpers.blender_recording_model import Simple3dObject
from python_code.viz.blender.blender_helpers.load_simple_object.load_keypoint_trajectories import load_keypoint_trajectories_bpy
from python_code.viz.blender.blender_helpers.load_simple_object.get_or_create_material import get_or_create_material
from python_code.viz.blender.blender_helpers.load_simple_object.create_edge_stick_mesh import create_edge_stick_mesh
import bpy

SKULL_NAMES = ["nose", "eye", "ear", "base", "cam"]
def load_simple_object_bpy(
    simple_object: Simple3dObject,
    drop_skull:bool=True
) -> list[bpy.types.Object]:
    """Pipeline: create keypoint empties + edge stick meshes for a Simple3dObject.

    1. Load keypoint trajectories as animated empties (via load_keypoint_trajectories_bpy).
    2. For each display edge in the topology, create a stick mesh with Hook modifiers
       that track the two endpoint empties (via create_edge_stick_mesh).
    3. Apply a shared material to all stick meshes.

    Returns the list of all created edge stick mesh objects.
    """
    topology = simple_object.topology
    trajectories = simple_object.trajectories

    # ---- Step 1: Create keypoint empties ----
    keypoint_empties: dict[str, bpy.types.Object] = load_keypoint_trajectories_bpy(
        keypoint_trajectories=trajectories,
    )
    if drop_skull:
        keypoints_no_skull = {}
        for key, value in keypoint_empties.items():
            if not any([skull_name in key for skull_name in SKULL_NAMES]):
                keypoints_no_skull[key] = value
        keypoint_empties = keypoints_no_skull
        
    # ---- Step 2: Deduplicate display edges ----
    deduped_edges: list[tuple[str, str]] = list({
        (min(a, b), max(a, b))
        for a, b in topology.display_edges_resolved
    })

    # ---- Step 3: Create edge stick meshes ----
    stick_meshes: list[bpy.types.Object] = []
    for name_a, name_b in deduped_edges:
        empty_a: bpy.types.Object = keypoint_empties[name_a]
        empty_b: bpy.types.Object = keypoint_empties[name_b]
        stick_name: str = f"{topology.name}_{name_a}_{name_b}_stick"
        stick: bpy.types.Object = create_edge_stick_mesh(
            empty_a=empty_a,
            empty_b=empty_b,
            name=stick_name,
        )
        stick_meshes.append(stick)

    # ---- Step 4: Apply shared material to all sticks ----
    mat: bpy.types.Material = get_or_create_material(
        name="simple_object_material",
        base_color=(0.1, 0.6, 0.9, 1.0),  # Light blue
        roughness=0.6,
    )
    for stick in stick_meshes:
        stick.data.materials.append(mat)

    return stick_meshes
