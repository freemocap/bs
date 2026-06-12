
import bpy

def create_edge_stick_mesh(
    empty_a: bpy.types.Object,
    empty_b: bpy.types.Object,
    name: str,
    skin_radius: float = 0.001
) -> bpy.types.Object:
    """Create a 2-vertex edge mesh that tracks two keypoint empties via Hook modifiers.

    The mesh sits at world origin (unparented). Hook modifiers on each vertex
    pull them to the empties' world-space positions, so the stick follows
    the animated keypoints without needing its own keyframes. A Skin modifier
    gives the edge visual thickness.
    """
    # Place initial vertices at the empties' current world positions so the
    # Hook modifier rest-pose offset is correctly computed.
    v0_world: tuple[float, float, float] = tuple(empty_a.matrix_world.translation)
    v1_world: tuple[float, float, float] = tuple(empty_b.matrix_world.translation)

    mesh: bpy.types.Mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([v0_world, v1_world], [(0, 1)], [])
    mesh.update()

    obj: bpy.types.Object = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Hook modifier — vertex 0 follows empty_a
    hook_a: bpy.types.Modifier = obj.modifiers.new(name="Hook_A", type="HOOK")
    hook_a.object = empty_a
    hook_a.vertex_indices_set([0])

    # Hook modifier — vertex 1 follows empty_b
    hook_b: bpy.types.Modifier = obj.modifiers.new(name="Hook_B", type="HOOK")
    hook_b.object = empty_b
    hook_b.vertex_indices_set([1])

    # Skin modifier for thickness
    skin: bpy.types.Modifier = obj.modifiers.new(name="Skin", type="SKIN")
    skin.use_smooth_shade = True
    for vertex_data in mesh.skin_vertices[0].data:
        vertex_data.radius = (skin_radius, skin_radius)

    return obj
