"""Main sidebar panel for the BS Recorder addon.

Shows in: 3D Viewport → Sidebar (press N) → "BS" tab
"""

import bpy


class BS_PT_main_panel(bpy.types.Panel):
    """Panel with Clear Scene, directory picker, and Load Recording."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BS"
    bl_label = "BS Recorder"
    bl_idname = "BS_PT_main_panel"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout

        # ── Clear Scene ──────────────────────────────────────────
        layout.operator(
            "bs.clear_scene",
            text="Clear Scene",
            icon="TRASH",
        )

        layout.separator(factor=0.5)

        # ── Recording Path ───────────────────────────────────────
        box = layout.box()
        box.label(text="Recording Folder:")
        box.prop(
            context.scene,
            "bs_recording_path",
            text="",
        )

        # ── Load Recording ───────────────────────────────────────
        layout.operator(
            "bs.load_recording",
            text="Load Recording",
            icon="IMPORT",
        )
