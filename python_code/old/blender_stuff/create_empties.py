"""Create Blender empty objects from 3D trajectory data."""

import bpy
import numpy as np


def create_trajectories(
        *,
        trajectory_dict: dict[str, np.ndarray],
        parent_object: bpy.types.Object,
        empty_scale: float = 0.01,
        empty_type: str = "SPHERE",
) -> dict[str, bpy.types.Object]:
    """
    Create keyframed empty objects for multiple trajectories.

    Args:
        trajectory_dict: Maps keypoint names to (n_frames, 3) trajectory arrays
        parent_object: Parent object for organization
        empty_scale: Display size of empty objects
        empty_type: Blender empty display type

    Returns:
        Dictionary mapping keypoint names to created empty objects
    """
    print(f"Creating {len(trajectory_dict)} trajectory objects...")

    empties = {}
    for keypoint_name, trajectory_xyz in trajectory_dict.items():
        empties[keypoint_name] = _create_keyframed_empty(
            trajectory_xyz=trajectory_xyz,
            name=keypoint_name,
            parent_object=parent_object,
            empty_scale=empty_scale,
            empty_type=empty_type,
        )

    print(f"✓ Created {len(empties)} trajectory empties")
    return empties


def _create_keyframed_empty(
        *,
        trajectory_xyz: np.ndarray,
        name: str,
        parent_object: bpy.types.Object,
        empty_scale: float,
        empty_type: str,
) -> bpy.types.Object:
    """Create a single keyframed empty object from trajectory data."""

    # Create and configure empty object
    empty = bpy.data.objects.new(name=name, object_data=None)
    empty.empty_display_type = empty_type
    empty.empty_display_size = empty_scale
    empty.parent = parent_object
    bpy.context.collection.objects.link(object=empty)

    # Set up animation
    action = bpy.data.actions.new(name=f"{name}_Action")
    empty.animation_data_create()
    empty.animation_data.action = action

    # Handle Blender 4.4+ action structure
    if bpy.app.version >= (4, 4):
        slot = action.slots.new(id_type='OBJECT', name=name)
        layer = action.layers.new(name="Layer")
        strip = layer.strips.new(type='KEYFRAME')
        channelbag = strip.channelbag(slot=slot, ensure=True)
        empty.animation_data.action_slot = action.slots[0]

    # Add keyframes for each axis
    num_frames = trajectory_xyz.shape[0]
    start_frame = bpy.context.scene.frame_start
    frames = np.arange(start_frame, start_frame + num_frames, dtype=np.float32)

    for axis_idx in range(3):
        # Create fcurve for this axis
        if bpy.app.version >= (4, 4):
            fcurve = channelbag.fcurves.new(data_path="location", index=axis_idx)
        else:
            fcurve = action.fcurves.new(data_path="location", index=axis_idx)

        fcurve.keyframe_points.add(count=num_frames)

        # Efficiently set all keyframes at once
        co = np.empty(shape=(2 * num_frames,), dtype=np.float32)
        co[0::2] = frames
        co[1::2] = trajectory_xyz[:, axis_idx]

        # Note: foreach_set() only accepts positional args in Blender API
        fcurve.keyframe_points.foreach_set("co", co)
        fcurve.update()

    return empty