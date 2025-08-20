# Add file to path to import custom modules
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
import mathutils

sys.path.append(str(Path(__file__).parent.parent.parent))

from python_code.blender_stuff.blender_helpers.fmc_create_empty_from_trajectory import (
    create_trajectories,
)
from python_code.blender_stuff.blender_helpers.fmc_make_parent_empties import (
    create_parent_empty,
)


from python_code.blender_stuff.blender_helpers.install_blender_dependencies import (
    check_and_install_dependencies,
)

check_and_install_dependencies()
from python_code.blender_stuff.blender_helpers.blender_utilities import (
    load_calibration_data,
    clear_scene,
    set_render_resolution,
)
from python_code.blender_stuff.blender_helpers.create_mocap_cameras import (
    create_mocap_camera_objects,
    map_cameras_to_videos,
    create_video_planes,
)
from python_code.blender_stuff.blender_helpers.create_trajectory_objects import (
    load_keypoint_csv_as_xyz_arrays,
    load_trajectory_data_npy,
)

# 2025-07-11__1


@dataclass
class RecordingPaths:
    base_recording_path: str
    recording_name: str
    body_data_csv_path: str
    body_raw_data_csv_path: str
    toy_raw_data_csv_path: str
    toy_data_csv_path: str
    right_eye_data_csv_path: str
    left_eye_data_csv_path: str
    camera_calibration_toml_path: str
    mocap_videos_path: str
    eye_videos_path: str

    def get_full_path(self, path_name: str) -> Path:
        """
        Get the full path for a given path name.
        If the path is relative, it will be combined with the base recording path.
        """
        data_path = getattr(self, path_name)
        if not path_name == "base_recording_path":
            full_data_path = Path(self.base_recording_path) / data_path
        else:
            full_data_path = Path(data_path)
        return full_data_path

    def __post_init__(self):
        # Ensure all paths exist
        for path_name, data_path in self.__dict__.items():
            if not "path" in path_name:
                continue
            full_data_path = self.get_full_path(path_name)
            if full_data_path and not Path(full_data_path).exists():
                raise FileNotFoundError(
                    f"Path {full_data_path} does not exist for {path_name} in RecordingPaths."
                )


RECORDING_PATHS = {
    "2025-07-11_ferret_757_EyeCameras_P43_E15__1": RecordingPaths(
        recording_name="2025-07-11_ferret_757_EyeCameras_P43_E15__1",
        base_recording_path=r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1",
        body_data_csv_path=r"output_data\body_data\dlc\freemocap_data_by_frame.csv",
        toy_data_csv_path=r"output_data\toy_data\dlc\freemocap_data_by_frame.csv",
        body_raw_data_csv_path=r"output_data\body_data\raw_data\dlc_3dData_numFrames_numTrackedPoints_spatialXYZ.npy",
        toy_raw_data_csv_path=r"output_data\toy_data\raw_data\dlc_3dData_numFrames_numTrackedPoints_spatialXYZ.npy",

        right_eye_data_csv_path=r"dlc_pupil_tracking\new_model_iteration_2\eye0DLC_Resnet50_pupil_tracking_ferret_757_EyeCameras_P43_E15__1_shuffle1_snapshot_030.csv",
        left_eye_data_csv_path=r"dlc_pupil_tracking\new_model_iteration_2\rotated_eye_1DLC_Resnet50_pupil_tracking_ferret_757_EyeCameras_P43_E15__1_shuffle1_snapshot_030.csv",
        camera_calibration_toml_path="session_2025-7-11_camera_calibration.toml",
        mocap_videos_path=r"basler_pupil_synchronized\synchronized_mocap_videos",
        eye_videos_path=r"basler_pupil_synchronized\synchronized_eye_videos",
    )
}

TARGET_RECORDING = "2025-07-11_ferret_757_EyeCameras_P43_E15__1"

STANDARD_PLANE_DISTANCE = 1.0  # 1 meter


def main_create_blender_scene(recording_paths: RecordingPaths) -> None:
    print(f"Creating Blender scene for recording: {recording_paths.recording_name}")
    # Clear scene
    clear_scene()

    # Create parent empty
    recording_parent_empty = create_parent_empty(
        name=recording_paths.recording_name, display_scale=1.0, type="ARROWS"
    )
    raw_data_parent_empty = create_parent_empty(
        name="raw_data",
        display_scale=0.5,
        type="PLAIN_AXES",
        parent_object=recording_parent_empty,
    )
    # print(f"Created parent empty: {recording_parent_empty.name}")

    # calibration_by_camera = load_calibration_data(
    #     calibration_path=str(
    #         recording_paths.get_full_path("camera_calibration_toml_path")
    #     )
    # )
    # # Create camera objects
    # camera_objects = create_mocap_camera_objects(
    #     calibration_by_camera=calibration_by_camera
    # )
    # print(f"Created {len(camera_objects)} camera objects.")
    # # Map cameras to videos
    # camera_video_map = map_cameras_to_videos(
    #     calibration_by_camera=calibration_by_camera,
    #     videos_path=str(recording_paths.get_full_path("mocap_videos_path")),
    # )

    # Create video planes
    # create_video_planes(
    #     camera_objects=camera_objects,
    #     camera_video_map=camera_video_map,
    #     calibration_by_camera=calibration_by_camera,
    #     plane_distance=STANDARD_PLANE_DISTANCE,
    # )
    print("Created video planes.")

    # # Set render resolution
    # set_render_resolution(width=1024, height=1024)

    # Optional: Add ray visualizations for each camera
    # for camera_id, camera_obj in camera_objects.items():
    #     create_ray_visualization(
    #         camera_obj=camera_obj,
    #         image_points=[
    #             (0, 0),      # Bottom-left
    #             (1, 0),      # Bottom-right
    #             (1, 1),      # Top-right
    #             (0, 1),      # Top-left
    #             (0.5, 0.5)   # Center
    #         ],
    #         length=STANDARD_PLANE_DISTANCE * 1.5
    #     )

    # print("Loaded videos as planes!")

    data_paths = [
        str(recording_paths.get_full_path("body_data_csv_path")),
        # str(recording_paths.get_full_path("toy_data_csv_path")),
    ]
    for data_path in data_paths:
        # Load and process trajectory data
        trajectory_dict, number_of_frames, number_of_markers = (
            load_keypoint_csv_as_xyz_arrays(
                filepath=data_path,
                scale_data=0.001,  # Scale to meters),
            )
        )
        raw_trajectories_dict = {}
        raw_trajectories_array, number_of_frames, number_of_markers = (
            load_trajectory_data_npy(
                trajectories_numpy_path=str(recording_paths.get_full_path('body_raw_data_csv_path')),
            )
        )
        for key_index, key in enumerate(trajectory_dict.keys()):
            key_index += 1
            raw_trajectories_dict[f"{key}_raw"] = raw_trajectories_array[:,key_index-1,:]

        # print(
        #     f"Loaded trajectory data with {number_of_frames} frames and {number_of_markers} markers."
        # )

        # Set scene frame range for trajectories
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = number_of_frames - 1

        trajectory_objects = create_trajectories(
            trajectory_dict=trajectory_dict,
            parent_object=recording_parent_empty,
            empty_scale=0.01,
            empty_type="SPHERE",
        )
        raw_trajectory_objects = create_trajectories(
            trajectory_dict=raw_trajectories_dict,
            parent_object=raw_data_parent_empty,
            empty_scale=0.01,
            empty_type="CUBE",
        )
        print(f"Created {len(trajectory_objects)} trajectory objects.")


main_create_blender_scene(recording_paths=RECORDING_PATHS[TARGET_RECORDING])
bpy.context.space_data.shading.type = "MATERIAL"
print("Done!")
