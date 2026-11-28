"""Re-triangulate head DLC 2D data to 3D WITHOUT temporal filtering.

Production triangulation (external dlc_to_3d repo) applies a 7 Hz Butterworth via
skellyforge and the raw 3D .npy it saves gets overwritten by the toy run, so no
unfiltered head 3D exists on disk. This runs the triangulation half of dlc_to_3d's
`process_recording_session` (same confidence threshold, DLC CSVs and calibration as
production) into a separate folder, leaving production files untouched.
"""

import logging
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DLC_TO_3D_REPO: Path = Path.home() / "Documents" / "GitHub" / "dlc_to_3d"
HEAD_DLC_MODEL_FOLDER: str = "head_body_eyecam_retrain_test_v2"
DLC_CONFIDENCE_THRESHOLD: float = 0.4  # matches production (dlc_to_3d.py __main__)

# dlc_to_3d imports `freemocap_anipose` from freemocap v1. If the freemocap checkout installed
# in the dlc_to_3d venv has moved on to v2 (module removed), we shadow it with the v1 module,
# read straight from the freemocap git history into a temp dir. Nothing on disk is modified.
DEFAULT_FREEMOCAP_REPO: Path = Path.home() / "Documents" / "GitHub" / "freemocap"
FREEMOCAP_V1_REF: str = "v1.8.2"
_ANIPOSE_MODULE = "freemocap/core_processes/capture_volume_calibration/anipose_camera_calibration/freemocap_anipose.py"


def _venv_can_import_freemocap_anipose(python_exe: Path, cwd: Path) -> bool:
    check = "from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration import freemocap_anipose"
    return subprocess.run([str(python_exe), "-c", check], cwd=cwd, capture_output=True).returncode == 0


def _write_freemocap_v1_shim(shim_root: Path, freemocap_repo: Path) -> None:
    source = subprocess.run(
        ["git", "-C", str(freemocap_repo), "show", f"{FREEMOCAP_V1_REF}:{_ANIPOSE_MODULE}"],
        check=True, capture_output=True, text=True,
    ).stdout
    module_path = shim_root / _ANIPOSE_MODULE
    module_path.parent.mkdir(parents=True)
    for package_dir in [module_path.parent, *module_path.parent.parents]:
        if package_dir == shim_root:
            break
        (package_dir / "__init__.py").touch()
    module_path.write_text(source)


def find_calibration_toml(recording_path: Path) -> Path:
    session_folder = recording_path.parent if "session" in recording_path.parent.name else recording_path.parent.parent
    tomls = sorted((session_folder / "calibration").glob("*calibration.toml"))
    if not tomls:
        raise FileNotFoundError(f"No *calibration.toml in {session_folder / 'calibration'}")
    return tomls[0]


HEAD_KEYPOINT_NAMES: list[str] = [  # order of tracked_points in dlc_to_3d ferret_head_and_spine.yaml
    "nose", "left_cam_tip", "right_cam_tip", "base", "left_eye", "right_eye",
    "left_ear", "right_ear", "spine_t1", "sacrum", "tail_tip",
]


def unfiltered_head_npy_path(triangulation_dir: Path) -> Path:
    return triangulation_dir / "head_3d_unfiltered_numFrames_numTrackedPoints_xyz.npy"


def triangulate_unfiltered_head(
    *,
    recording_path: Path,
    triangulation_dir: Path,
    dlc_to_3d_repo: Path = DEFAULT_DLC_TO_3D_REPO,
    freemocap_repo: Path = DEFAULT_FREEMOCAP_REPO,
    overwrite: bool = False,
) -> Path:
    """Triangulate head DLC 2D -> 3D with no gap filling or filtering.

    Calls dlc_to_3d's `compile_dlc_csvs` (same confidence threshold as production) and
    `reconstruct_3d` directly, i.e. the part of `process_recording_session` before the
    skellyforge filter. Returns path to an (n_frames, n_keypoints, 3) .npy in mm, with
    keypoints in HEAD_KEYPOINT_NAMES order.
    """
    output_npy = unfiltered_head_npy_path(triangulation_dir)
    if output_npy.exists() and not overwrite:
        logger.info(f"Unfiltered triangulation exists, skipping: {output_npy}")
        return output_npy

    python_exe = dlc_to_3d_repo / ".venv" / "bin" / "python"
    if not python_exe.exists():
        raise FileNotFoundError(f"dlc_to_3d venv python not found: {python_exe}")

    dlc_csv_folder = recording_path / "mocap_data" / "dlc_output" / HEAD_DLC_MODEL_FOLDER
    calibration_toml = find_calibration_toml(recording_path)
    triangulation_dir.mkdir(parents=True, exist_ok=True)

    script = textwrap.dedent(
        f"""
        from pathlib import Path
        import numpy as np
        from dlc_reconstruction.reconstruction_to_3d.compile_dlc_csv_to_2d_data import compile_dlc_csvs
        from dlc_reconstruction.reconstruction_to_3d.reconstruction import reconstruct_3d
        data_2d = compile_dlc_csvs(Path({str(dlc_csv_folder)!r}), confidence_threshold={DLC_CONFIDENCE_THRESHOLD!r})
        data_3d = reconstruct_3d(data_2d, Path({str(calibration_toml)!r}))
        assert data_3d.shape[1] == {len(HEAD_KEYPOINT_NAMES)}, data_3d.shape
        np.save({str(output_npy)!r}, data_3d)
        """
    )
    logger.info(f"Triangulating unfiltered head data with {python_exe}")
    with tempfile.TemporaryDirectory(prefix="freemocap_v1_shim_") as shim_dir:
        env = os.environ.copy()
        if not _venv_can_import_freemocap_anipose(python_exe, dlc_to_3d_repo):
            logger.info(f"freemocap_anipose not importable in dlc_to_3d venv; using {FREEMOCAP_V1_REF} shim")
            _write_freemocap_v1_shim(Path(shim_dir), freemocap_repo)
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [shim_dir, env.get("PYTHONPATH")]))
        subprocess.run([str(python_exe), "-c", script], cwd=dlc_to_3d_repo, check=True, env=env)
    if not output_npy.exists():
        raise FileNotFoundError(f"Triangulation finished but {output_npy} was not written")
    return output_npy


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    rec = Path(sys.argv[1])
    print(triangulate_unfiltered_head(recording_path=rec, triangulation_dir=rec / "filter_comparison" / "_triangulation"))
