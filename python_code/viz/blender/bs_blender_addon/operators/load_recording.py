"""Operator: Load a BS recording into the Blender scene."""

from pathlib import Path

import bpy


class BS_OT_load_recording(bpy.types.Operator):
    """Run the full BS pipeline on the selected recording folder.

    Reads the path from ``bpy.types.Scene.bs_recording_path`` (set by the
    panel's directory picker) and delegates to
    ``blender_helpers.pipeline_runner.run_pipeline()``.
    """

    bl_idname = "bs.load_recording"
    bl_label = "Load Recording"
    bl_description = (
        "Load keypoint trajectories, rigid-body kinematics, eye/gaze data, "
        "calibrated cameras, and synchronized videos into the scene"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        recording_path: str = context.scene.bs_recording_path

        # ── Validation ───────────────────────────────────────────
        if not recording_path:
            self.report({"ERROR"}, "No recording folder selected.")
            return {"CANCELLED"}

        if not Path(recording_path).exists():
            self.report(
                {"ERROR"},
                f"Recording folder does not exist:\n  {recording_path}",
            )
            return {"CANCELLED"}

        # ── Run pipeline ─────────────────────────────────────────
        # Late import — the path setup in __init__.py must have run first.
        from python_code.viz.blender.blender_helpers.pipeline_runner import (
            run_pipeline,
        )

        try:
            run_pipeline(recording_path)
        except Exception as exc:
            self.report({"ERROR"}, f"Pipeline failed: {exc}")
            import traceback

            traceback.print_exc()
            return {"CANCELLED"}

        self.report({"INFO"}, f"Loaded recording: {recording_path}")
        return {"FINISHED"}
