"""Operator: Clear the entire Blender scene."""

import bpy


class BS_OT_clear_scene(bpy.types.Operator):
    """Remove all objects, collections, and orphan data from the scene.

    Delegates to ``freemocap_blender_addon``'s ``clear_scene()`` which
    handles mode-set, selection, deletion, and orphan purging.
    """

    bl_idname = "bs.clear_scene"
    bl_label = "Clear Scene"
    bl_description = "Remove all objects / collections / orphan data from the scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        # Late import — the path setup in __init__.py must have run first.
        from freemocap_blender_addon.core_functions.setup_scene.clear_scene import (
            clear_scene,
        )

        clear_scene()
        self.report({"INFO"}, "Scene cleared.")
        return {"FINISHED"}
