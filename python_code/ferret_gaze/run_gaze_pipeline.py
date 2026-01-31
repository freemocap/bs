"""
Ferret Gaze Pipeline Runner
============================

Complete pipeline to calculate ferret gaze and create Blender visualization.

Pipeline Steps:
1. Calculate eye kinematics from raw eye trajectory CSVs
2. Resample all data (skull, eye, toy trajectories, videos) to common timestamps
3. Calculate gaze kinematics (transform eye rotations to world coordinates)
4. Generate Blender visualization script with configured paths

Each step automatically detects if output data exists and skips processing.
Use reprocess flags to force re-running specific steps.

Usage:
    python run_gaze_pipeline.py

Or import and call:
    from run_gaze_pipeline import run_gaze_pipeline
    run_gaze_pipeline(clip_path=Path("..."))

    # Force reprocess everything:
    run_gaze_pipeline(clip_path=Path("..."), reprocess_all=True)

    # Force reprocess only gaze:
    run_gaze_pipeline(clip_path=Path("..."), reprocess_gaze=True)

Directory Structure Expected:
    clip_path/
    ├── mocap_data/
    │   ├── output_data/
    │   │   ├── solver_output/
    │   │   │   ├── skull_kinematics.csv
    │   │   │   ├── skull_reference_geometry.json
    │   │   │   ├── skull_and_spine_trajectories.csv
    │   │   │   └── skull_and_spine_topology.json
    │   │   └── dlc/
    │   │       └── toy_body_3d_xyz.csv
    │   ├── annotated_videos/
    │   │   └── *_clipped_*.mp4
    │   └── synchronized_videos/
    │       └── *_timestamps_utc_clipped_*.npy
    └── eye_data/
        ├── output_data/
        │   ├── eye0_data.csv
        │   └── eye1_data.csv
        ├── eye_videos/
        │   ├── eye0_timestamps_utc_clipped_*.npy
        │   └── eye1_timestamps_utc_clipped_*.npy
        ├── left_eye_stabilized.mp4
        └── right_eye_stabilized.mp4

Output Structure:
    clip_path/
    ├── eye_data/output_data/eye_kinematics/
    │   ├── left_eye_kinematics.csv
    │   ├── left_eye_reference_geometry.json
    │   ├── right_eye_kinematics.csv
    │   └── right_eye_reference_geometry.json
    ├── analyzable_output/
    │   ├── common_timestamps.npy
    │   ├── skull_kinematics/
    │   ├── left_eye_kinematics/
    │   ├── right_eye_kinematics/
    │   ├── gaze_kinematics/
    │   │   ├── left_gaze_kinematics.csv
    │   │   ├── left_gaze_reference_geometry.json
    │   │   ├── right_gaze_kinematics.csv
    │   │   └── right_gaze_reference_geometry.json
    │   ├── toy_trajectories_resampled.csv
    │   └── full_visualization.blend
    └── display_videos/
        ├── top_down_mocap_resampled.mp4
        ├── left_eye_resampled.mp4
        └── right_eye_resampled.mp4
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from python_code.ferret_gaze.calculate_gaze.calculate_ferret_gaze import calculate_ferret_gaze
from python_code.ferret_gaze.data_resampling.data_resampling_helpers import ResamplingStrategy
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import (
    eye_camera_distance_from_skull_geometry,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.ferret_gaze.data_resampling.ferret_data_resampler import (
    VideoConfig,
    resample_ferret_data,
    create_eye_topology,
)
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ClipPaths:
    """Paths derived from a clip directory."""

    clip_path: Path

    # Mocap data paths
    @property
    def mocap_data_dir(self) -> Path:
        return self.clip_path / "mocap_data"

    @property
    def solver_output_dir(self) -> Path:
        return self.mocap_data_dir / "output_data" / "solver_output"

    @property
    def skull_reference_geometry_json(self) -> Path:
        return self.solver_output_dir / "skull_reference_geometry.json"

    @property
    def dlc_output_dir(self) -> Path:
        return self.mocap_data_dir / "output_data" / "dlc"

    @property
    def toy_trajectories_csv(self) -> Path:
        return self.dlc_output_dir / "toy_body_3d_xyz.csv"

    @property
    def annotated_videos_dir(self) -> Path:
        return self.mocap_data_dir / "annotated_videos"

    @property
    def synchronized_videos_dir(self) -> Path:
        return self.mocap_data_dir / "synchronized_videos"

    # Eye data paths
    @property
    def eye_data_dir(self) -> Path:
        return self.clip_path / "eye_data"

    @property
    def eye_output_dir(self) -> Path:
        return self.eye_data_dir / "output_data"

    @property
    def left_eye_trajectories_csv(self) -> Path:
        return self.eye_output_dir / "eye0_data.csv"

    @property
    def right_eye_trajectories_csv(self) -> Path:
        return self.eye_output_dir / "eye1_data.csv"

    @property
    def eye_kinematics_output_dir(self) -> Path:
        return self.eye_output_dir / "eye_kinematics"

    @property
    def eye_videos_dir(self) -> Path:
        return self.eye_data_dir / "eye_videos"

    @property
    def left_eye_video(self) -> Path:
        return self.eye_data_dir / "right_eye_stabilized.mp4"

    @property
    def right_eye_video(self) -> Path:
        return self.eye_data_dir / "left_eye_stabilized.mp4"

    # Output paths
    @property
    def analyzable_output_dir(self) -> Path:
        return self.clip_path / "analyzable_output"

    @property
    def gaze_kinematics_output_dir(self) -> Path:
        return self.analyzable_output_dir / "gaze_kinematics"

    @property
    def display_videos_dir(self) -> Path:
        return self.clip_path / "display_videos"

    @property
    def blender_output_path(self) -> Path:
        return self.analyzable_output_dir / "full_visualization.blend"

    @property
    def blender_script_path(self) -> Path:
        return self.analyzable_output_dir / "ferret_full_gaze_blender_viz.py"

    def validate_inputs(self) -> None:
        """Validate that required input paths exist."""
        required_paths = [
            (self.solver_output_dir, "Skull solver output directory"),
            (self.skull_reference_geometry_json, "Skull reference geometry JSON"),
            (self.left_eye_trajectories_csv, "Left eye trajectories CSV"),
            (self.right_eye_trajectories_csv, "Right eye trajectories CSV"),
        ]

        missing = []
        for path, description in required_paths:
            if not path.exists():
                missing.append(f"  - {description}: {path}")

        if missing:
            raise FileNotFoundError(
                f"Missing required input files:\n" + "\n".join(missing)
            )

    def eye_kinematics_exists(self) -> bool:
        """Check if eye kinematics output exists."""
        required_files = [
            self.eye_kinematics_output_dir / "left_eye_kinematics.csv",
            self.eye_kinematics_output_dir / "left_eye_reference_geometry.json",
            self.eye_kinematics_output_dir / "right_eye_kinematics.csv",
            self.eye_kinematics_output_dir / "right_eye_reference_geometry.json",
        ]
        return all(f.exists() for f in required_files)

    def resampled_data_exists(self) -> bool:
        """Check if resampled data output exists."""
        required_files = [
            self.analyzable_output_dir / "common_timestamps.npy",
            self.analyzable_output_dir / "skull_kinematics" / "skull_kinematics.csv",
            self.analyzable_output_dir / "left_eye_kinematics" / "left_eye_kinematics.csv",
            self.analyzable_output_dir / "right_eye_kinematics" / "right_eye_kinematics.csv",
        ]
        return all(f.exists() for f in required_files)

    def gaze_kinematics_exists(self) -> bool:
        """Check if gaze kinematics output exists."""
        required_files = [
            self.gaze_kinematics_output_dir / "left_gaze_kinematics.csv",
            self.gaze_kinematics_output_dir / "left_gaze_reference_geometry.json",
            self.gaze_kinematics_output_dir / "right_gaze_kinematics.csv",
            self.gaze_kinematics_output_dir / "right_gaze_reference_geometry.json",
        ]
        return all(f.exists() for f in required_files)

    def blender_script_exists(self) -> bool:
        """Check if Blender script exists."""
        return self.blender_script_path.exists()


def find_video_file(directory: Path, pattern: str) -> Path | None:
    """Find a video file matching a pattern in a directory."""
    if not directory.exists():
        return None
    matches = list(directory.glob(pattern))
    if matches:
        return matches[0]
    return None


def find_timestamps_file(directory: Path, pattern: str) -> Path | None:
    """Find a timestamps file matching a pattern in a directory."""
    if not directory.exists():
        return None
    matches = list(directory.glob(pattern))
    if matches:
        return matches[0]
    return None


def calculate_eye_kinematics(paths: ClipPaths) -> dict[str, FerretEyeKinematics]:
    """
    Calculate eye kinematics from raw eye trajectory CSVs.

    Args:
        paths: ClipPaths object with all path information

    Returns:
        Dictionary mapping eye names to FerretEyeKinematics objects
    """
    logger.info("=" * 80)
    logger.info("STEP 1: CALCULATING EYE KINEMATICS")
    logger.info("=" * 80)

    skull_reference_geometry = ReferenceGeometry.from_json_file(
        paths.skull_reference_geometry_json
    )

    paths.eye_kinematics_output_dir.mkdir(parents=True, exist_ok=True)

    kinematics_by_eye: dict[str, FerretEyeKinematics] = {}

    for eye_name, csv_path in [
        ("left_eye", paths.left_eye_trajectories_csv),
        ("right_eye", paths.right_eye_trajectories_csv),
    ]:
        eye_side: Literal["left", "right"] = "left" if eye_name == "left_eye" else "right"

        logger.info(f"\nProcessing {eye_name}...")
        logger.info(f"  Input CSV: {csv_path}")

        eye_camera_distance_mm = eye_camera_distance_from_skull_geometry(
            skull_reference_geometry=skull_reference_geometry,
            eye_side=eye_side,
        )
        logger.info(f"  Eye-camera distance: {eye_camera_distance_mm:.2f} mm")

        eye_kinematics = FerretEyeKinematics.calculate_from_trajectories(
            eye_trajectories_csv_path=csv_path,
            eye_camera_distance_mm=eye_camera_distance_mm,
            eye_name=eye_name,
        )

        logger.info(f"  Frames: {eye_kinematics.n_frames}")
        logger.info(f"  Framerate: {eye_kinematics.framerate_hz:.2f} Hz")

        eye_kinematics.save_to_disk(output_directory=paths.eye_kinematics_output_dir)
        logger.info(f"  Saved to: {paths.eye_kinematics_output_dir}")

        eye_topology = create_eye_topology(eye_name)
        topology_path = paths.eye_kinematics_output_dir / f"{eye_name}_topology.json"
        eye_topology.save_json(topology_path)
        logger.info(f"  Saved topology: {topology_path.name}")

        kinematics_by_eye[eye_name] = eye_kinematics

    return kinematics_by_eye


def build_video_configs(paths: ClipPaths) -> list[VideoConfig]:
    """
    Build video configurations for resampling.

    Automatically discovers videos and timestamps files.

    Args:
        paths: ClipPaths object

    Returns:
        List of VideoConfig objects
    """
    configs: list[VideoConfig] = []

    mocap_video = find_video_file(paths.annotated_videos_dir, "*_clipped_*.mp4")
    mocap_timestamps = find_timestamps_file(
        paths.synchronized_videos_dir, "*_timestamps_utc_clipped_*.npy"
    )

    if not mocap_video:
        raise FileNotFoundError(
            f"Mocap video not found in {paths.annotated_videos_dir} matching pattern '*_clipped_*.mp4'"
        )
    if not mocap_timestamps:
        raise FileNotFoundError(
            f"Mocap timestamps not found in {paths.synchronized_videos_dir} matching pattern '*_timestamps_utc_clipped_*.npy'"
        )
    configs.append(
        VideoConfig(
            path=mocap_video,
            timestamps_path=mocap_timestamps,
            name="top_down_mocap",
        )
    )
    logger.info(f"  Found mocap video: {mocap_video.name}")

    left_eye_timestamps = find_timestamps_file(
        paths.eye_videos_dir, "eye0_timestamps_utc_clipped_*.npy"
    )

    if not left_eye_timestamps:
        raise FileNotFoundError(
            f"Left eye timestamps not found in {paths.eye_videos_dir} matching pattern 'eye0_timestamps_utc_clipped_*.npy'"
        )
    if not paths.left_eye_video.exists():
        raise FileNotFoundError(f"Left eye video not found at {paths.left_eye_video}")
    configs.append(
        VideoConfig(
            path=paths.left_eye_video,
            timestamps_path=left_eye_timestamps,
            name="left_eye",
        )
    )
    logger.info(f"  Found left eye video: {paths.left_eye_video.name}")

    right_eye_timestamps = find_timestamps_file(
        paths.eye_videos_dir, "eye1_timestamps_utc_clipped_*.npy"
    )

    if not right_eye_timestamps:
        raise FileNotFoundError(
            f"Right eye timestamps not found in {paths.eye_videos_dir} matching pattern 'eye1_timestamps_utc_clipped_*.npy'"
        )
    if not paths.right_eye_video.exists():
        raise FileNotFoundError(f"Right eye video not found at {paths.right_eye_video}")
    configs.append(
        VideoConfig(
            path=paths.right_eye_video,
            timestamps_path=right_eye_timestamps,
            name="right_eye",
        )
    )
    logger.info(f"  Found right eye video: {paths.right_eye_video.name}")

    return configs


def resample_all_data(
    paths: ClipPaths,
    video_configs: list[VideoConfig],
    resampling_strategy: ResamplingStrategy,
    reprocess_videos: bool,
) -> None:
    """
    Resample all data to common timestamps.

    Args:
        paths: ClipPaths object
        video_configs: List of VideoConfig for videos to resample
        resampling_strategy: Strategy for selecting target framerate
        toy_trajectory_type: Which trajectory type to load from toy CSV
        reprocess_videos: If True, recreate videos even if they exist
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: RESAMPLING ALL DATA TO COMMON TIMESTAMPS")
    logger.info("=" * 80)



    resample_ferret_data(
        skull_solver_output_dir=paths.solver_output_dir,
        eye_kinematics_dir=paths.eye_kinematics_output_dir,
        resampled_data_output_dir=paths.analyzable_output_dir,
        toy_trajectories_csv=paths.toy_trajectories_csv,
        resampling_strategy=resampling_strategy,
        video_configs=video_configs if video_configs else None,
        recreate_videos=reprocess_videos,
    )


def calculate_gaze(paths: ClipPaths) -> None:
    """
    Calculate gaze kinematics from resampled skull and eye data.

    Args:
        paths: ClipPaths object
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CALCULATING GAZE KINEMATICS")
    logger.info("=" * 80)

    calculate_ferret_gaze(
        resampled_data_dir=paths.analyzable_output_dir,
        output_dir=paths.gaze_kinematics_output_dir,
    )


def generate_blender_script(paths: ClipPaths) -> Path:
    """
    Generate a Blender visualization script with configured paths.

    Args:
        paths: ClipPaths object

    Returns:
        Path to the generated script
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: GENERATING BLENDER VISUALIZATION SCRIPT")
    logger.info("=" * 80)

    template_script_path = Path(__file__).parent /"visualization"/"ferret_gaze_blender"/"ferret_full_gaze_blender_viz.py"
    if not template_script_path.exists():
        raise FileNotFoundError(
            f"Blender visualization template not found at {template_script_path}"
        )

    with open(template_script_path, "r") as f:
        script_content = f.read()

    data_dir_pattern = r'DATA_DIR\s*=\s*Path\([^)]+\)'
    new_data_dir = f'DATA_DIR = Path(r"{paths.analyzable_output_dir}")'
    script_content = re.sub(data_dir_pattern, lambda _: new_data_dir, script_content)

    output_pattern = r'OUTPUT_PATH\s*=\s*[^\n]+'
    new_output = f'OUTPUT_PATH = Path(r"{paths.blender_output_path}")'
    script_content = re.sub(output_pattern, lambda _: new_output, script_content)

    with open(paths.blender_script_path, "w") as f:
        f.write(script_content)

    logger.info(f"Generated Blender script: {paths.blender_script_path}")
    logger.info(f"  DATA_DIR: {paths.analyzable_output_dir}")
    logger.info(f"  OUTPUT_PATH: {paths.blender_output_path}")

    return paths.blender_script_path


def run_gaze_pipeline(
    clip_path: Path,
    resampling_strategy: ResamplingStrategy = ResamplingStrategy.FASTEST,
    reprocess_all: bool = False,
    reprocess_eye_kinematics: bool = False,
    reprocess_resampling: bool = False,
    reprocess_gaze: bool = False,
    reprocess_blender_script: bool = False,
    reprocess_videos: bool = False,
) -> Path:
    """
    Run the complete ferret gaze pipeline.

    Each step automatically detects if output data exists and skips processing
    unless a reprocess flag is set.

    Args:
        clip_path: Path to the clip directory containing all input data
        resampling_strategy: Strategy for selecting target framerate
        toy_trajectory_type: Which trajectory type to load from toy CSV
        reprocess_all: If True, reprocess all steps regardless of existing data
        reprocess_eye_kinematics: If True, recalculate eye kinematics
        reprocess_resampling: If True, redo resampling
        reprocess_gaze: If True, recalculate gaze kinematics
        reprocess_blender_script: If True, regenerate Blender script
        reprocess_videos: If True, recreate resampled videos

    Returns:
        Path to the analyzable_output directory containing all results
    """
    logger.info("\n" + "=" * 80)
    logger.info("FERRET GAZE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Clip path: {clip_path}")

    if reprocess_all:
        reprocess_eye_kinematics = True
        reprocess_resampling = True
        reprocess_gaze = True
        reprocess_blender_script = True
        reprocess_videos = True
        logger.info("Reprocess all: ENABLED - will reprocess all steps")

    paths = ClipPaths(clip_path=clip_path)
    paths.validate_inputs()

    # Step 1: Calculate eye kinematics
    if reprocess_eye_kinematics or not paths.eye_kinematics_exists():
        if not reprocess_eye_kinematics:
            logger.info("\nEye kinematics not found - calculating...")
        calculate_eye_kinematics(paths)
    else:
        logger.info("\n[SKIP] Eye kinematics already exists")
        logger.info(f"       Location: {paths.eye_kinematics_output_dir}")

    # Step 2: Build video configs and resample
    if reprocess_resampling or not paths.resampled_data_exists():
        if not reprocess_resampling:
            logger.info("\nResampled data not found - resampling...")
        logger.info("\nDiscovering video files...")
        video_configs = build_video_configs(paths)
        logger.info(f"Found {len(video_configs)} videos to resample")

        resample_all_data(
            paths=paths,
            video_configs=video_configs,
            resampling_strategy=resampling_strategy,
            reprocess_videos=reprocess_videos,
        )
    else:
        logger.info("\n[SKIP] Resampled data already exists")
        logger.info(f"       Location: {paths.analyzable_output_dir}")

    # Step 3: Calculate gaze
    if reprocess_gaze or not paths.gaze_kinematics_exists():
        if not reprocess_gaze:
            logger.info("\nGaze kinematics not found - calculating...")
        calculate_gaze(paths)
    else:
        logger.info("\n[SKIP] Gaze kinematics already exists")
        logger.info(f"       Location: {paths.gaze_kinematics_output_dir}")

    # Step 4: Generate Blender script
    if reprocess_blender_script or not paths.blender_script_exists():
        if not reprocess_blender_script:
            logger.info("\nBlender script not found - generating...")
        generate_blender_script(paths)
    else:
        logger.info("\n[SKIP] Blender script already exists")
        logger.info(f"       Location: {paths.blender_script_path}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {paths.analyzable_output_dir}")
    logger.info("\nOutput locations:")
    logger.info(f"  - Eye kinematics:  {paths.eye_kinematics_output_dir}")
    logger.info(f"  - Resampled data:  {paths.analyzable_output_dir}")
    logger.info(f"  - Gaze kinematics: {paths.gaze_kinematics_output_dir}")
    logger.info(f"  - Display videos:  {paths.display_videos_dir}")
    logger.info(f"  - Blender script:  {paths.blender_script_path}")
    logger.info("\nTo create visualization:")
    logger.info("  1. Open Blender 4.0+ or 5.0+")
    logger.info(f"  2. Open {paths.blender_script_path}")
    logger.info("  3. Run with Alt+P")
    logger.info("  4. Press Spacebar to play animation")

    return paths.analyzable_output_dir


if __name__ == "__main__":
    _clip_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s"
    )

    run_gaze_pipeline(
        clip_path=_clip_path,
        resampling_strategy=ResamplingStrategy.FASTEST,
        reprocess_all=True,
    )