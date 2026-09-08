import bpy


def get_or_create_material(
    name: str,
    base_color: tuple[float, float, float, float] = (0.1, 0.6, 0.9, 1.0),
    roughness: float = 0.6,
) -> bpy.types.Material:
    """Return an existing material by name, or create a new one.

    Uses a Principled BSDF node with the given base_color and roughness.
    Follows the same pattern used by create_arena for material reuse.
    """
    mat: bpy.types.Material | None = bpy.data.materials.get(name)
    if mat is not None:
        return mat

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat
