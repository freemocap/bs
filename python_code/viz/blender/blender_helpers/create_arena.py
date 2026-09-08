import bpy


def create_arena():
    print("Creating arena parent empty 'arena' (PLAIN_AXES, display_size=0.02)")
    arena_empty = bpy.data.objects.new("arena", None)
    arena_empty.empty_display_type = "PLAIN_AXES"
    arena_empty.empty_display_size = 0.02
    bpy.context.scene.collection.objects.link(arena_empty)

    mat_name = "arena_bar_material"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        print(f"Creating new material '{mat_name}'")
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
        print(f"  Material: Base Color = (0.02, 0.02, 0.02, 1.0), Roughness = 0.8")
    else:
        print(f"Reusing existing material '{mat_name}'")

    print("Creating 1.0-meter cube at location (0, 0, 0.5) — spans Z=0 to Z=1.0")
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(0, 0, 0.5), align='WORLD', enter_editmode=False)
    cube = bpy.context.active_object
    cube.name = "arena_cube"
    cube.data.materials.append(mat)
    cube.parent = arena_empty
    print(f"Cube '{cube.name}' created and parented to 'arena'")

    print("Adding Wireframe modifier (thickness=0.01 meters = 1 centimeter, replace faces, include boundary)")
    wire = cube.modifiers.new(name="Wireframe", type='WIREFRAME')
    wire.thickness = 0.01
    wire.use_replace = True
    wire.use_boundary = True
    print(f"Wireframe modifier '{wire.name}' applied")

    print("Arena creation complete.")
    return arena_empty
