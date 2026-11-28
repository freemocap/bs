"""Build one filter variant: filtered head 3D -> skull solver, filtered eye 2D -> alignment,
then eye kinematics -> resampling -> gaze, all inside `<comparison_dir>/<variant>/`.

The variant folder mirrors a clip folder's layout so `ClipPaths` from the gaze pipeline
works unchanged. Production files are only read, never written.
"""

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from python_code.eye_analysis.data_processing.align_data.eye_anatomical_alignment import eye_alignment_main
from python_code.eye_analysis.process_eye_session import EYE_DLC_MODEL_FOLDER, find_eye_dlc_csv
from python_code.ferret_gaze.data_resampling.data_resampling_helpers import ResamplingStrategy
from python_code.ferret_gaze.run_gaze_pipeline import (
    ClipPaths,
    calculate_eye_kinematics,
    calculate_gaze,
    resample_all_data,
)
from python_code.filter_comparison.triangulate_unfiltered import HEAD_KEYPOINT_NAMES
from python_code.filter_comparison.variants import HEAD_FS_HZ, VARIANTS_BY_NAME, FilterVariant
from python_code.rigid_body_solver.core.optimization import OptimizationConfig
from python_code.rigid_body_solver.ferret_skull_solver import run_ferret_skull_solver
from python_code.utilities.temporal_filters import apply_temporal_filter

logger = logging.getLogger(__name__)

REFERENCE_MOCAP_VIDEO: str = "24676894"  # same reference camera as production skull solving

# Production weights from ferret_skull_solver.run_ferret_skull_solver
_SOLVER_DEFAULT = OptimizationConfig(max_iter=100, lambda_data=100.0, lambda_trans_smooth=300.0, lambda_rot_smooth=500.0)
_SOLVER_NO_SMOOTH = OptimizationConfig(max_iter=100, lambda_data=100.0, lambda_trans_smooth=0.0, lambda_rot_smooth=0.0)


def gap_fill_3d(data: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN gaps; leading/trailing NaNs take the nearest value."""
    flat = pd.DataFrame(data.reshape(data.shape[0], -1))
    filled = flat.interpolate(method="linear", axis=0, limit_area="inside").ffill().bfill()
    return filled.to_numpy().reshape(data.shape)


def write_head_csv(data_3d: np.ndarray, output_csv: Path) -> None:
    """Write (n_frames, n_kp, 3) in the head_freemocap_data_by_frame.csv format the solver reads."""
    n_frames, n_kp, _ = data_3d.shape
    frames = np.repeat(np.arange(n_frames), n_kp)
    df = pd.DataFrame(
        {
            "frame": frames,
            "keypoint": np.tile(HEAD_KEYPOINT_NAMES, n_frames),
            "x": data_3d[:, :, 0].ravel(),
            "y": data_3d[:, :, 1].ravel(),
            "z": data_3d[:, :, 2].ravel(),
            "model": "dlc.body",
            "trajectory": "3d_xyz",
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def find_dlc_median_eye_csv(recording_path: Path, eye_name: str) -> Path:
    """DLC's `_filtered.csv` (filterpredictions: 5-frame median on x/y, likelihood unchanged)."""
    folder = recording_path / "eye_data" / "dlc_output" / EYE_DLC_MODEL_FOLDER
    candidates = sorted(folder.glob(f"{eye_name}*snapshot*_filtered.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {eye_name} DLC _filtered.csv in {folder}, found {candidates}")
    return candidates[0]


def variant_is_complete(variant_dir: Path) -> bool:
    return ClipPaths(clip_path=variant_dir).gaze_kinematics_exists()


def build_variant(
    *,
    variant: FilterVariant,
    recording_path: Path,
    unfiltered_head_npy: Path,
    comparison_dir: Path,
    resampling_strategy: ResamplingStrategy = ResamplingStrategy.FASTEST,
) -> Path:
    variant_dir = comparison_dir / variant.name
    paths = ClipPaths(clip_path=variant_dir)
    logger.info(f"\n{'#' * 80}\n# VARIANT {variant.name}: {variant.describe()}\n{'#' * 80}")
    t0 = time.perf_counter()

    if variant.head_from is not None:
        # ---- head: reuse another variant's skull solve (same head filter + solver) ----
        source = VARIANTS_BY_NAME[variant.head_from]
        if (source.head_filter, source.solver_smoothing) != (variant.head_filter, variant.solver_smoothing):
            raise ValueError(f"{variant.name} cannot reuse the head of {source.name}: head settings differ")
        source_paths = ClipPaths(clip_path=comparison_dir / source.name)
        if not source_paths.solver_output_dir.exists():
            raise FileNotFoundError(f"Build {source.name} first: {source_paths.solver_output_dir} is missing")
        logger.info(f"Reusing head output of {source.name}")
        for src, dst in [(source_paths.solver_output_dir, paths.solver_output_dir),
                         (source_paths.dlc_output_dir, paths.dlc_output_dir)]:
            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
    else:
        # ---- head: filter 3D markers, solve skull --------------------------------
        head_raw = np.load(unfiltered_head_npy)
        head_filtered = apply_temporal_filter(gap_fill_3d(head_raw), fs=HEAD_FS_HZ, spec=variant.head_filter)
        head_csv = paths.dlc_output_dir / "head_freemocap_data_by_frame.csv"
        write_head_csv(head_filtered, head_csv)

        mocap_timestamps = next(
            (recording_path / "mocap_data" / "synchronized_corrected_videos").glob(f"{REFERENCE_MOCAP_VIDEO}*_utc*.npy")
        )
        measured_fs = 1e9 / np.median(np.diff(np.load(mocap_timestamps)))
        if abs(measured_fs - HEAD_FS_HZ) > 1.0:
            raise ValueError(f"Mocap runs at {measured_fs:.2f} Hz, but head filters assume {HEAD_FS_HZ} Hz")

        paths.solver_output_dir.mkdir(parents=True, exist_ok=True)
        run_ferret_skull_solver(
            input_csv=head_csv,
            timestamps_path=mocap_timestamps,
            output_dir=paths.solver_output_dir,
            optimization_config=_SOLVER_DEFAULT if variant.solver_smoothing else _SOLVER_NO_SMOOTH,
        )

    # toy trajectories are needed by the resampler; unaffected by head/eye filters
    toy_link = paths.toy_trajectories_csv
    if not toy_link.exists():
        toy_link.symlink_to(recording_path / "mocap_data" / "output_data" / "dlc" / "toy_body_3d_xyz.csv")

    # ---- eyes: filter 2D pixels, anatomical alignment -------------------------
    paths.eye_output_dir.mkdir(parents=True, exist_ok=True)
    for eye_name in ("left_eye", "right_eye"):
        eye_alignment_main(
            recording_name=f"{recording_path.parent.name}_{variant.name}_{eye_name}",
            csv_path=(find_dlc_median_eye_csv if variant.eye_dlc_input == "dlc_median" else find_eye_dlc_csv)(
                recording_path, eye_name
            ),
            timestamps_path=recording_path / "eye_data" / "eye_videos" / f"{eye_name}_timestamps_utc.npy",
            output_path=paths.eye_output_dir,
            filter_spec=variant.eye_filter,
        )

    # ---- gaze pipeline stages (no videos / blender / dropbox copy) -----------
    calculate_eye_kinematics(paths)
    resample_all_data(
        paths=paths, video_configs=[], resampling_strategy=resampling_strategy, reprocess_videos=False
    )
    calculate_gaze(paths)

    (variant_dir / "variant.json").write_text(
        json.dumps(
            {
                "name": variant.name,
                "description": variant.describe(),
                "variant": variant.model_dump(),
                "unfiltered_head_npy": str(unfiltered_head_npy),
                "build_seconds": round(time.perf_counter() - t0, 1),
            },
            indent=2,
        )
    )
    logger.info(f"Variant {variant.name} done in {time.perf_counter() - t0:.0f} s")
    return variant_dir
