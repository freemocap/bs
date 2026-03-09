"""Create Blender scene from 3D tracking data."""

import sys
from dataclasses import dataclass
from pathlib import Path

import bpy


# Add project root to path - more robust approach
_script_path = Path(__file__).resolve()
_project_root = _script_path.parent.parent.parent  # Go up to 'bs' directory 

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    print(f"Added to sys.path: {_project_root}")

# Verify path was added
print(f"Python path includes: {_project_root in [Path(p) for p in sys.path]}")
print(f"Looking for module at: {_project_root / 'python_code'}\n")

try:
    from python_code.blender_stuff.blender_helpers import create_parent_empty, clear_scene
    from python_code.blender_stuff.create_empties import create_trajectories
    from python_code.blender_stuff.load_trajectories import load_trajectories_auto
    from python_code.blender_stuff.create_eye_tracking_driver import create_eye_tracking_controller

    print("✓ All imports successful\n")
    
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    print(f"\nMake sure you have __init__.py in blender_helpers/")
    print(f"Run the debug_blender_imports.py script to diagnose\n")
    raise


@dataclass
class RecordingPaths:
    """Paths for a single recording session."""
    recording_name: str
    base_path: Path
    body_csv: Path
    eye_csv: Path
    camera_calibration: Path
    mocap_videos: Path
    eye_videos: Path
    toy_csv: Path | None = None
    
    @classmethod
    def from_relative_paths(
        cls,
        *,
        recording_name: str,
        base_path: str | Path,
        body_csv: str,
        toy_csv: str | None = None,
        eye_csv: str,
        camera_calibration: str | Path,
        mocap_videos: str,
        eye_videos: str,
    ) -> "RecordingPaths":
        """
        Create RecordingPaths with automatic path resolution.
        
        Args:
            recording_name: Name of the recording
            base_path: Base directory for the recording
            body_csv: Relative path to body tracking CSV
            toy_csv: Optional relative path to toy tracking CSV
            eye_csv: Relative path to eye tracking CSV
            camera_calibration: Path to camera calibration file (can be absolute)
            mocap_videos: Relative path to mocap videos
            eye_videos: Relative path to eye videos
            
        Returns:
            RecordingPaths with resolved absolute paths
        """
        base = Path(base_path)
        
        paths = cls(
            recording_name=recording_name,
            base_path=base,
            body_csv=base / body_csv,
            eye_csv=base / eye_csv,
            camera_calibration=Path(camera_calibration),
            mocap_videos=base / mocap_videos,
            eye_videos=base / eye_videos,
            toy_csv=base / toy_csv if toy_csv else None,
        )
        
        # Validate all paths exist
        paths._validate_paths()
        return paths
    
    def _validate_paths(self) -> None:
        """Ensure all paths exist."""
        for field_name, path in self.__dict__.items():
            if not isinstance(path, Path):
                continue
            if path is None:  # Handle optional paths
                continue
            if not path.exists():
                raise FileNotFoundError(f"{field_name}: {path}")


# Configuration
RECORDINGS = {
    "2025-07-11_ferret_757_EyeCameras_P43_E15__1": RecordingPaths.from_relative_paths(
        recording_name="2025-07-11_ferret_757_EyeCameras_P43_E15__1",
        base_path=r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s",
        body_csv=r"mocap_data\head_data_with_mean.csv",
        eye_csv=r"eye_data\eye_data_with_mean.csv",
        camera_calibration=r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\session_2025-7-11_camera_calibration.toml",
        mocap_videos=r"mocap_data\annotated_videos\annotated_videos_head_body_eyecam_retrain_test_v2",
        eye_videos=r"eye_data\annotated_videos",
    ),
    "2025-07-session_2025-07-01_ferret_757_EyeCameras_P33_EO5": RecordingPaths.from_relative_paths(
        recording_name="2025-07-2025-07-session_2025-07-01_ferret_757_EyeCameras_P33_EO5",
        base_path=r"D:\bs\ferret_recordings\session_2025-07-01_ferret_757_EyeCameras_P33_EO5\clips\1m_20s-2m_20s",
        body_csv=r"td_body.csv",
        toy_csv=r"td_toy.csv",
        eye_csv=r"eye_data\dlc_output\eye_model_v1\eye0_clipped_9033_16234DLC_Resnet50_eye_model_v1_shuffle1_snapshot_120_filtered.csv",
        camera_calibration=r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\session_2025-7-11_camera_calibration.toml",
        mocap_videos=r"mocap_data\annotated_videos\annotated_videos_head_body_eyecam_retrain_test_v2",
        eye_videos=r"eye_data\annotated_videos",
    )
}

# TARGET_RECORDING = "2025-07-11_ferret_757_EyeCameras_P43_E15__1"
TARGET_RECORDING = "2025-07-session_2025-07-01_ferret_757_EyeCameras_P33_EO5"


def create_blender_scene(*, recording: RecordingPaths, include_eye_tracking: bool = True) -> None:
    """
    Create a complete Blender scene from recording data.
    
    Args:
        recording: RecordingPaths object with all data locations
        include_eye_tracking: Whether to create eye tracking controller
    """
    print(f"\n{'='*60}")
    print(f"Creating Blender scene: {recording.recording_name}")
    print(f"{'='*60}\n")
    
    # 1. Clear existing scene
    print("Clearing scene...")
    clear_scene()
    print("✓ Scene cleared\n")
    
    # 2. Create organizational hierarchy
    print("Creating parent empties...")
    recording_parent = create_parent_empty(
        name=recording.recording_name,
        display_scale=1.0,
        type="ARROWS"
    )
    print(f"✓ Created root: {recording.recording_name}\n")
    
    # 3. Load body trajectory data
    trajectories_dict = load_trajectories_auto(
        filepath=recording.body_csv,
        scale_factor=0.001,
        z_value=0.0,
    )
    
    # Load toy trajectories if available
    if recording.toy_csv:
        toy_trajectories = load_trajectories_auto(
            filepath=recording.toy_csv,
            scale_factor=0.001,
            z_value=0.0,
        )
        trajectories_dict.update(toy_trajectories)
    
    print()
    
    # 4. Configure scene timeline
    num_frames = next(iter(trajectories_dict.values())).shape[0]
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1
    print(f"Timeline: frames 0-{num_frames - 1}\n")
    
    # 5. Create trajectory empties
    trajectory_objects = create_trajectories(
        trajectory_dict=trajectories_dict,
        parent_object=recording_parent,
        empty_scale=0.01,
        empty_type="SPHERE",
    )
    
    # 6. Create eye tracking controller (if enabled)
    if include_eye_tracking:
        eye_controller, horopter_data = create_eye_tracking_controller(
            eye_csv_path=recording.eye_csv,
            parent_object=recording_parent,
            name="eye_controller",
            scale_factor=0.001,
            horopter_axis=0,  # 0=x (horizontal), 1=y (vertical)
            start_frame=0,
        )
        
        # Example: Create a simple visualization object that responds to eye movement
        # You can modify this to drive constraints on other objects
        create_eye_movement_visualization(
            controller=eye_controller,
            parent=recording_parent,
        )
    
    print(f"\n{'='*60}")
    print(f"✓ Scene creation complete!")
    print(f"{'='*60}\n")



def main() -> None:
    """Entry point for Blender script execution."""
    create_blender_scene(
        recording=RECORDINGS[TARGET_RECORDING],
        include_eye_tracking=True,
    )

    # Set viewport shading
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    print("Ready to animate! 🎬")
    print("\nEye tracking controller created with 'horopter_displacement' property")
    print("You can drive any constraint or property from this value!")


main()