import bpy
import numpy as np

from python_code.viz.blender.blender_helpers.blender_recording_model import BlenderRecording


def set_scene_parameters(recording: BlenderRecording, start_frame: int = 0, end_frame: int | None = None):
    if end_frame is None:
        end_frame = recording.frame_count - 1

    print(f"start_frame = {start_frame}")
    print(f"end_frame = {end_frame}  (recording.frame_count = {recording.frame_count}, 0-based inclusive)")

    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    print(f"Set scene.frame_start = {bpy.context.scene.frame_start}")
    print(f"Set scene.frame_end   = {bpy.context.scene.frame_end}")

    timestamps = recording.data.timestamps
    time_deltas = np.diff(timestamps)
    print(f"Timestamps: {len(timestamps)} samples, "
          f"range [{timestamps[0]:.6f}, {timestamps[-1]:.6f}] seconds")
    print(f"Time deltas: mean={np.mean(time_deltas):.6f}, "
          f"min={np.min(time_deltas):.6f}, max={np.max(time_deltas):.6f}, std={np.std(time_deltas):.6f} seconds")

    framerate = float(np.mean(time_deltas)) ** -1
    framerate_rounded = int(round(framerate))
    print(f"Computed framerate: {framerate:.6f} frames_per_second")
    print(f"Rounded framerate: {framerate_rounded} frames_per_second")

    bpy.context.scene.render.fps = framerate_rounded
    print(f"Set scene.render.frames_per_second = {bpy.context.scene.render.fps}")
