# Add file to path to import custom modules
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
import mathutils

sys.path.append(str(Path(__file__).parent.parent.parent))

from python_code.blender_stuff.old.blender_helpers.fmc_create_empty_from_trajectory import (
    create_trajectories,
)
from python_code.blender_stuff.old.blender_helpers.make_parent_empties import (
    create_parent_empty,
)
from python_code.blender_stuff.install_blender_dependencies import (
    check_and_install_dependencies,
)

check_and_install_dependencies()

from python_code.blender_stuff.old.blender_helpers.blender_utilities import (
    clear_scene,
)
from python_code.blender_stuff.old.blender_helpers.load_trajectories_from_tidy_csv import (
    read_trajectory_csv,
    trajectories_to_numpy_dict,
)


@dataclass
class RecordingPaths:
    base_recording_path: str
    recording_name: str
    body_data_csv_path: str
    eye_data_csv_path: str
    camera_calibration_toml_path: str
    mocap_videos_path: str
    eye_videos_path: str

    def get_full_path(self, *, path_name: str) -> Path:
        """
        Get the full path for a given path name.
        If the path is relative, it will be combined with the base recording path.
        """
        data_path = getattr(self, path_name)
        if path_name != "base_recording_path":
            full_data_path = Path(self.base_recording_path) / data_path
        else:
            full_data_path = Path(data_path)
        return full_data_path

    def __post_init__(self) -> None:
        # Ensure all paths exist
        for path_name, data_path in self.__dict__.items():
            if "path" not in path_name:
                continue
            full_data_path = self.get_full_path(path_name=path_name)
            if full_data_path and not Path(full_data_path).exists():
                raise FileNotFoundError(
                    f"Path {full_data_path} does not exist for {path_name} in RecordingPaths."
                )


RECORDING_PATHS: dict[str, RecordingPaths] = {
    "2025-07-11_ferret_757_EyeCameras_P43_E15__1": RecordingPaths(
        recording_name="2025-07-11_ferret_757_EyeCameras_P43_E15__1",
        base_recording_path=r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s",
        body_data_csv_path=r"mocap_data\output_data\output_data_head_body_eyecam_retrain_test_v2_model_outputs_iteration_1\dlc\dlc_body_rigid_3d_xyz.csv",
        eye_data_csv_path=r"eye_data\eye_data_with_mean.csv",
        camera_calibration_toml_path=r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\session_2025-7-11_camera_calibration.toml",
        mocap_videos_path=r"mocap_data\annotated_videos\annotated_videos_head_body_eyecam_retrain_test_v2",
        eye_videos_path=r"eye_data\annotated_videos",
    )
}

TARGET_RECORDING: str = "2025-07-11_ferret_757_EyeCameras_P43_E15__1"
STANDARD_PLANE_DISTANCE: float = 1.0  # 1 meter


def main_create_blender_scene(*, recording_paths: RecordingPaths) -> None:
    """
    Create a complete Blender scene from recording data.

    Args:
        recording_paths: RecordingPaths object containing all data paths
    """
    print(f"Creating Blender scene for recording: {recording_paths.recording_name}")

    # Clear scene
    clear_scene()

    # Create parent empty objects for organization
    recording_parent_empty = create_parent_empty(
        name=recording_paths.recording_name,
        display_scale=1.0,
        type="ARROWS"
    )
    raw_data_parent_empty = create_parent_empty(
        name="raw_data",
        display_scale=0.5,
        type="PLAIN_AXES",
        parent_object=recording_parent_empty,
    )

    # Load body trajectory data
    body_csv_path = recording_paths.get_full_path(path_name="body_data_csv_path")

    trajectories = read_trajectory_csv(filepath=body_csv_path)
    trajectory_dict = trajectories_to_numpy_dict(trajectories=trajectories, scale_factor=0.001)

    number_of_frames = len(next(iter(trajectories.values())).frames)
    number_of_markers = len(trajectories)

    print(f"Loaded {number_of_markers} markers with {number_of_frames} frames")

    # Set scene frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = number_of_frames - 1

    # Create trajectory objects in Blender
    trajectory_objects = create_trajectories(
        trajectory_dict=trajectory_dict,
        parent_object=recording_parent_empty,
        empty_scale=0.01,
        empty_type="SPHERE",
    )

    print(f"Created {len(trajectory_objects)} trajectory objects.")


if __name__ == "__main__":
    main_create_blender_scene(
        recording_paths=RECORDING_PATHS[TARGET_RECORDING]
    )
    bpy.context.space_data.shading.type = "MATERIAL"
    print("Done!")