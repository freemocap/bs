from freemocap_blender_addon.core_functions.empties.creation.create_empty_from_trajectory import \
    create_keyframed_empty_from_3d_trajectory_data

from python_code.kinematics_core.keypoint_trajectories import KeypointTrajectories
import bpy


def load_keypoint_trajectories_bpy(
    keypoint_trajectories: KeypointTrajectories,
    parent_empty: bpy.types.Object,
) -> dict[str, bpy.types.Object]:
    """Create keyframed empties for each keypoint and return a name→empty mapping.

    The first keypoint empty is parented directly under `parent_empty`.
    All subsequent keypoint empties are parented under that first empty,
    forming a chain that moves together as a group.
    """
    keypoint_empties: dict[str, bpy.types.Object] = {}

    for name in keypoint_trajectories.keypoint_names:
        empty: bpy.types.Object = create_keyframed_empty_from_3d_trajectory_data(
            trajectory_name=name,
            trajectory_fr_xyz=keypoint_trajectories[name] * 0.001,
            parent_object=parent_empty,
            empty_scale=0.01,
            empty_type="SPHERE",
        )
        keypoint_empties[name] = empty

    return keypoint_empties
