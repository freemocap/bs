"""
Keypoint MoSeq Pipeline Runner
================================

Wrapper functions for each stage of the Keypoint MoSeq pipeline, plus a
`run_pipeline` function that chains them and a `main` convenience entry point
that handles data loading.

Supported loaders
-----------------
- kpms built-ins: "deeplabcut", "sleap", "anipose", "sleap-anipose", "nwb",
  "facemap", "freipose", "dannce"
- "3d_data": custom loader for triangulated 3D keypoint CSVs
  (see kpms_loader.load_3d_data_kpms)
- "solver_output": custom loader for mocap solver tidy output
  (see kpms_loader.load_solver_output_kpms)
- "eye_3d": custom loader for resampled 3D eye trajectory CSVs
  (see kpms_loader.load_3d_eye_kpms)
- "head_with_pupil_points": custom loader combining head/skull keypoints with both
  eyes' tracked pupil boundary points, read from parquet
  (see kpms_loader.load_head_with_pupil_points_kpms)

Usage
-----
    from python_code.moseq.run_moseq_pipeline import main, run_configured

    # Simple entry point for RecordingFolder-based loaders (DATA_3D, SOLVER_OUTPUT, EYE_3D):
    main(
        project_dir="/path/to/project",
        recording_folder=recording_folder,
        loader=KPMS_Loader.EYE_3D,
    )

    # Pass a list to train one model across multiple sessions:
    main(
        project_dir="/path/to/project",
        recording_folder=[recording_folder_1, recording_folder_2],
        loader=KPMS_Loader.EYE_3D,
    )

    # Full explicit config for other loaders:
    run_configured(
        project_dir="/path/to/project",
        loader=KPMS_Loader.DEEPLABCUT,
        source="/path/to/file.csv",
        video_dir="/path/to/videos",
        use_bodyparts=["nose", "left_ear", ...],
        anterior_bodyparts=["nose"],
        posterior_bodyparts=["tail_base"],
        fps=90,
    )
"""

import multiprocessing
import os
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

# Must be set before JAX (imported by keypoint_moseq) starts its thread pool.
# On Linux, subprocess.Popen uses os.fork(), which deadlocks in multi-threaded
# processes. "forkserver" avoids this by spawning a clean server process.
multiprocessing.set_start_method("forkserver", force=True)

import matplotlib

# Use a non-interactive backend so kpms's plotting calls (which call plt.show()
# after already saving figures to project_dir) don't block waiting for a window
# to be closed. Must be set before keypoint_moseq/matplotlib.pyplot are imported.
matplotlib.use("Agg")

import keypoint_moseq as kpms
import matplotlib.pyplot as plt

from python_code.moseq.block_pca import fit_pca_by_keypoint_group
from python_code.moseq.kpms_loader import (
    load_3d_data_kpms,
    load_3d_eye_kpms,
    load_both_eyes_kpms,
    load_head_with_pupil_points_kpms,
    load_skull_and_gaze_kpms,
    load_solver_output_kpms,
)
from python_code.moseq.visualization.variance_explained_viz import (
    plot_variance_explained,
)
from python_code.moseq.visualization.eye_pc_viz import run_eye_pc_diagnostics
from python_code.moseq.utils.bodyparts import (
    BOTH_EYES_ANTERIOR_BODYPARTS,
    BOTH_EYES_BODYPARTS,
    BOTH_EYES_POSTERIOR_BODYPARTS,
    EYE_ANTERIOR_BODYPARTS,
    EYE_BODYPARTS,
    EYE_POSTERIOR_BODYPARTS,
    FERRET_ANTERIOR_BODYPARTS,
    FERRET_BODYPARTS,
    FERRET_POSTERIOR_BODYPARTS,
    HEAD_BODYPARTS,
    HEAD_WITH_PUPIL_POINTS_ANTERIOR_BODYPARTS,
    HEAD_WITH_PUPIL_POINTS_BODYPARTS,
    HEAD_WITH_PUPIL_POINTS_LEFT_EYE_BODYPARTS,
    HEAD_WITH_PUPIL_POINTS_POSTERIOR_BODYPARTS,
    HEAD_WITH_PUPIL_POINTS_RIGHT_EYE_BODYPARTS,
    SKULL_AND_GAZE_ANTERIOR_BODYPARTS,
    SKULL_AND_GAZE_BODYPARTS,
    SKULL_AND_GAZE_POSTERIOR_BODYPARTS,
)
from python_code.moseq.utils.skeletons import (
    BOTH_EYES_SKELETON,
    EYE_SKELETON_3D,
    FERRET_SKELETON,
    HEAD_WITH_PUPIL_POINTS_SKELETON,
    SKULL_AND_GAZE_SKELETON,
)
from python_code.batch_processing.session_manager import SessionManager
from python_code.moseq.training_config import save_training_config
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


_KPMS_BUILTIN_LOADERS = {
    "deeplabcut", "sleap", "anipose", "sleap-anipose",
    "nwb", "facemap", "freipose", "dannce",
}


def _format_duration(seconds: float) -> str:
    """Render a duration as e.g. '1h 12m 03s', '4m 07s', or '12.3s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


@contextmanager
def _timed_step(name: str):
    """Print how long a pipeline step took once it finishes."""
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"[done] {name}: {_format_duration(time.perf_counter() - start)}")


class KPMS_Loader(Enum):
    DEEPLABCUT = "deeplabcut"
    SLEAP = "sleap"
    ANIPOSE = "anipose"
    SLEAP_ANIPOSE = "sleap-anipose"
    NWB = "nwb"
    FACEMAP = "facemap"
    FREIPOSE = "freipose"
    DANNCE = "dannce"
    DATA_3D = "3d_data"
    SOLVER_OUTPUT = "solver_output"
    EYE_3D = "eye_3d"
    SKULL_AND_GAZE = "skull_and_gaze"
    BOTH_EYES = "both_eyes"
    HEAD_WITH_PUPIL_POINTS = "head_with_pupil_points"


def setup_project(
    project_dir: str | Path,
    dlc_config: str | Path | None = None,
) -> None:
    """
    Initialise a kpms project directory.

    Parameters
    ----------
    project_dir:
        Directory where kpms config and outputs will be stored.
    dlc_config:
        Optional path to a DeepLabCut config.yaml.  Only needed when loading
        data with the "deeplabcut" loader.
    """
    kpms.setup_project(str(project_dir), deeplabcut_config=str(dlc_config) if dlc_config else None)

def configure_project(
    project_dir: str | Path,
    video_dir: str | Path,
    bodyparts: list[str],
    use_bodyparts: list[str],
    anterior_bodyparts: list[str],
    posterior_bodyparts: list[str],
    fps: int,
    skeleton: list[list[str]] | None = None,
    outlier_scale_factor: float = 6.0,
) -> None:
    """Update kpms config with project-specific parameters."""
    kpms.update_config(
        str(project_dir),
        video_dir=str(video_dir),
        bodyparts=bodyparts,
        use_bodyparts=use_bodyparts,
        anterior_bodyparts=anterior_bodyparts,
        posterior_bodyparts=posterior_bodyparts,
        fps=fps,
        outlier_scale_factor=outlier_scale_factor,
    )
    if skeleton is not None:
        kpms.update_config(str(project_dir), skeleton=skeleton)

def load_keypoints(
    loader: KPMS_Loader,
    source,
) -> tuple[dict, dict, list[str]]:
    """
    Load keypoint data using the specified loader.

    Parameters
    ----------
    loader:
        One of the kpms built-in format strings ("deeplabcut", "sleap", etc.)
        or a custom loader name ("3d_data", "solver_output").
    source:
        For kpms built-ins: a file path, directory, or glob pattern
        (str or Path, or list thereof).
        For "3d_data" and "solver_output": a RecordingFolder or list of
        RecordingFolders.

    Returns
    -------
    coordinates, confidences, bodyparts
    """
    if loader.value in _KPMS_BUILTIN_LOADERS:
        return kpms.load_keypoints(source, loader.value)
    elif loader == KPMS_Loader.DATA_3D:
        return load_3d_data_kpms(source)
    elif loader == KPMS_Loader.SOLVER_OUTPUT:
        return load_solver_output_kpms(source)
    elif loader == KPMS_Loader.EYE_3D:
        return load_3d_eye_kpms(source)
    elif loader == KPMS_Loader.SKULL_AND_GAZE:
        return load_skull_and_gaze_kpms(source)
    elif loader == KPMS_Loader.BOTH_EYES:
        return load_both_eyes_kpms(source)
    elif loader == KPMS_Loader.HEAD_WITH_PUPIL_POINTS:
        return load_head_with_pupil_points_kpms(source)
    else:
        raise ValueError(
            f"Unknown loader '{loader.value}'. Must be one of: "
            f"{sorted(_KPMS_BUILTIN_LOADERS | {'3d_data', 'solver_output', 'eye_3d', 'skull_and_gaze', 'both_eyes', 'head_with_pupil_points'})}"
        )


def prepare_data(
    coordinates: dict,
    confidences: dict,
    project_dir: str | Path,
) -> tuple[dict, dict]:
    """
    Remove outliers, format data, and run noise calibration.

    Returns
    -------
    data, metadata
        As returned by kpms.format_data.
    """
    config = lambda: kpms.load_config(str(project_dir))

    coordinates, confidences = kpms.outlier_removal(
        coordinates,
        confidences,
        str(project_dir),
        overwrite=False,
        **config(),
    )

    data, metadata = kpms.format_data(coordinates, confidences, **config())
    num_dims = coordinates[next(iter(coordinates))].shape[2]
    if num_dims == 2:  # Noise calibration does not support 3D data
        kpms.noise_calibration(str(project_dir), coordinates, confidences, **config())

    return data, metadata

def fit_pca(
    data: dict,
    project_dir: str | Path,
    keypoint_groups: dict[str, list[str]] | None = None,
    block_latent_dims: dict[str, int] | None = None,
) -> object:
    """
    Fit a PCA model, save it, generate diagnostic plots, and update sigmasq config.

    Parameters
    ----------
    keypoint_groups, block_latent_dims:
        If both given, fit PCA using the separate-per-keypoint-group block
        method instead of kpms's standard joint PCA -- see
        `block_pca.fit_pca_by_keypoint_group` and `eye_movement_in_pcs.md`'s
        Option B. `keypoint_groups` partitions `use_bodyparts` (e.g. head
        vs. left/right eye); `block_latent_dims` gives each group's share of
        the latent space (`{group_name: n_dims}`) -- config.yml's
        `latent_dim` is overwritten to their sum before fitting, so the
        AR/full model pick up exactly these components. Leave both `None`
        (the default) to use kpms's standard joint PCA.

    Returns
    -------
    pca
        The fitted PCA model.
    """
    config = lambda: kpms.load_config(str(project_dir))

    plt.close("all")
    if keypoint_groups is not None and block_latent_dims is not None:
        cfg = config()
        kpms.update_config(str(project_dir), latent_dim=sum(block_latent_dims.values()))
        pca = fit_pca_by_keypoint_group(
            data["Y"],
            data["mask"],
            cfg["anterior_idxs"],
            cfg["posterior_idxs"],
            use_bodyparts=cfg["use_bodyparts"],
            keypoint_groups=keypoint_groups,
            latent_dims=block_latent_dims,
            conf=data.get("conf"),
        )
    else:
        pca = kpms.fit_pca(**data, **config())
    kpms.save_pca(pca, str(project_dir))

    kpms.print_dims_to_explain_variance(pca, 0.9)
    kpms.plot_scree(pca, project_dir=str(project_dir))
    kpms.plot_pcs(pca, project_dir=str(project_dir), **config())
    plt.close("all")

    kpms.update_config(
        str(project_dir),
        sigmasq_loc=kpms.estimate_sigmasq_loc(
            data["Y"], data["mask"], filter_size=config()["fps"]
        ),
    )

    return pca

def fit_ar_model(
    data: dict,
    metadata: dict,
    pca,
    project_dir: str | Path,
    num_ar_iters: int = 50,
) -> tuple[object, str]:
    """
    Initialise and fit an AR-only model.

    Returns
    -------
    model, model_name
    """
    config = lambda: kpms.load_config(str(project_dir))

    model = kpms.init_model(data, pca=pca, **config())

    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        str(project_dir),
        ar_only=True,
        num_iters=num_ar_iters,
        generate_progress_plots=False,
    )

    return model, model_name


def fit_full_model(
    data: dict,
    metadata: dict,
    project_dir: str | Path,
    model_name: str,
    num_ar_iters: int = 50,
    num_full_iters: int = 500,
    kappa: float = 1e4,
) -> object:
    """
    Load the AR checkpoint, apply kappa, and fit the full model.

    Returns
    -------
    model
        The fitted full model.
    """
    model, data, metadata, current_iter = kpms.load_checkpoint(
        str(project_dir),
        model_name,
        iteration=num_ar_iters,
    )

    model = kpms.update_hypparams(model, kappa=kappa)

    model = kpms.fit_model(
        model,
        data,
        metadata,
        str(project_dir),
        model_name,
        ar_only=False,
        start_iter=current_iter,
        num_iters=current_iter + num_full_iters,
        generate_progress_plots=False,
    )

    return model


def extract_results(
    project_dir: str | Path,
    model_name: str,
) -> dict:
    """
    Reindex syllables, extract results, and save as CSV.

    Returns
    -------
    results dict
    """
    kpms.reindex_syllables_in_checkpoint(str(project_dir), model_name)

    model, data, metadata, current_iter = kpms.load_checkpoint(
        str(project_dir),
        model_name,
    )

    results = kpms.extract_results(model, metadata, str(project_dir), model_name)
    kpms.save_results_as_csv(results, str(project_dir), model_name)

    return results

def _load_config_with_overrides(
    project_dir: str | Path,
    min_duration: float | None = None,
    min_frequency: float | None = None,
) -> dict:
    """
    Load the project's config.yml and, if given, override `min_duration`
    and/or `min_frequency` in the returned dict only. The overrides are
    never written back to config.yml.
    """
    config = kpms.load_config(str(project_dir))
    if min_duration is not None:
        config["min_duration"] = min_duration
    if min_frequency is not None:
        config["min_frequency"] = min_frequency
    return config

def trajectory_plots(
        project_dir: str | Path,
        model_name: str,
        coordinates: dict,
        results: dict,
        min_duration: float | None = None,
        min_frequency: float | None = None,
        n_neighbors: int | None = None,
):
    """
    Generate plots showing the median trajectory of poses associated with each syllable.

    `min_duration`/`min_frequency` override the corresponding config.yml
    values for this call only, if provided; otherwise the config's values
    are used unchanged.

    `n_neighbors` overrides kpms's default density-sampling neighborhood
    size (50), which is also the *minimum number of instances* a syllable
    must have to get a trajectory plot at all (kpms.generate_trajectory_plots
    requires >= n_neighbors instances when density_sample=True, its default).
    That floor is much stricter than grid movies' (rows*cols=24 by default),
    which is why a syllable can show up in grid movies but not trajectory
    plots. Lower this (e.g. to 24, matching grid movies) to include more
    syllables; leave unset to use kpms's default of 50.
    """
    kwargs = {}
    if n_neighbors is not None:
        kwargs["sampling_options"] = {"n_neighbors": n_neighbors}
    kpms.generate_trajectory_plots(
        coordinates,
        results,
        str(project_dir),
        model_name,
        **_load_config_with_overrides(project_dir, min_duration, min_frequency),
        **kwargs,
    )

def generate_grid_movies(
        project_dir: str | Path,
        model_name: str,
        coordinates: dict,
        results: dict,
        min_duration: float | None = None,
        min_frequency: float | None = None,
):
    """
    `min_duration`/`min_frequency` override the corresponding config.yml
    values for this call only, if provided; otherwise the config's values
    are used unchanged.
    """
    keypoints_only = True if coordinates[next(iter(coordinates))].shape[2] != 2 else False
    kpms.generate_grid_movies(
        results,
        str(project_dir),
        model_name,
        coordinates=coordinates,
        keypoints_only=keypoints_only,
        keypoints_scale=5.0,
        use_dims=[0, 1],
        **_load_config_with_overrides(project_dir, min_duration, min_frequency),
    )

def plot_syllable_dendrogram(
    project_dir: str | Path,
    model_name: str,
    coordinates: dict,
    results: dict,
):
    config = lambda: kpms.load_config(str(project_dir))
    kpms.plot_similarity_dendrogram(
        coordinates,
        results,
        project_dir,
        model_name,
        **config(),
    )

def variance_explained(
    project_dir: str | Path,
    model_name: str,
) -> dict:
    """
    Plot how much of the continuous pose trajectory's variance is explained
    by the fitted syllables/AR dynamics (see
    `visualization/variance_explained_viz.py`), and save
    `variance_explained.pdf` alongside the model's other diagnostic plots.

    Returns
    -------
    stats dict: {"ar_r2": {...}, "eta_squared": {...}}
    """
    stats, fig = plot_variance_explained(project_dir, model_name)
    plt.close(fig)
    return stats


def eye_pc_diagnostics(
    project_dir: str | Path,
    model_name: str,
    recording_folder,
    use_bodyparts: list[str],
    keypoint_groups: dict[str, list[str]],
    pca,
    results: dict,
) -> dict:
    """
    Check whether the fitted PCA/AR-HMM is actually capturing eye movement
    (see `visualization/eye_pc_viz.py`): per-PC loading energy by keypoint
    group, and per-PC correlation against independently-measured
    eye-in-head angles. Saves plots alongside the model's other diagnostics.

    Only meaningful for a mixed head+eye keypoint set (i.e.
    `KPMS_Loader.HEAD_WITH_PUPIL_POINTS`) -- see `main()`, which is the only
    caller that wires this up.

    Returns
    -------
    stats dict: {"loading_by_keypoint_group": {...}, "eye_angle_correlation": {...}}
    """
    return run_eye_pc_diagnostics(
        project_dir, model_name, recording_folder, use_bodyparts, keypoint_groups,
        pca=pca, results=results,
    )


def run_pipeline(
    project_dir: str | Path,
    coordinates: dict,
    confidences: dict,
    bodyparts: list[str],
    video_dir: str | Path,
    anterior_bodyparts: list[str],
    posterior_bodyparts: list[str],
    fps: int,
    use_bodyparts: list[str] | None = None,
    skeleton: list[list[str]] | None = None,
    num_ar_iters: int = 50,
    num_full_iters: int = 500,
    kappa: float = 1e4,
    outlier_scale_factor: float = 6.0,
    dlc_config: str | Path | None = None,
    eye_diagnostics_recording_folder=None,
    eye_diagnostics_keypoint_groups: dict[str, list[str]] | None = None,
    use_block_pca: bool = False,
    block_pca_latent_dims: dict[str, int] | None = None,
) -> dict:
    """
    Run the complete kpms pipeline from pre-loaded keypoint data.

    Parameters
    ----------
    project_dir:
        kpms project directory (created if it does not exist).
    coordinates, confidences, bodyparts:
        Pre-loaded keypoint data, e.g. from `load_keypoints`.
    video_dir:
        Directory containing source videos (used by kpms for visualisation).
    use_bodyparts:
        Subset of bodyparts to model.
    anterior_bodyparts:
        Bodyparts defining the anterior direction (e.g. ["nose"]).
    posterior_bodyparts:
        Bodyparts defining the posterior direction (e.g. ["tail_base"]).
    fps:
        Frames per second of the recordings.
    num_ar_iters:
        Number of AR-only fitting iterations.
    num_full_iters:
        Additional iterations for the full model after AR warmup.
    kappa:
        Concentration parameter applied before full-model fitting.
    outlier_scale_factor:
        Scale factor used by outlier removal.
    dlc_config:
        Optional DLC config path for `setup_project`.
    eye_diagnostics_recording_folder, eye_diagnostics_keypoint_groups:
        If both are given, run `eye_pc_diagnostics` after fitting (see
        `visualization/eye_pc_viz.py`) -- the `RecordingFolder`(s) the model
        was trained on (for ground-truth eye-in-head angles) and a
        `{group_name: [bodypart names]}` partition of `use_bodyparts`.
        Leave both `None` to skip (the default; only meaningful for a mixed
        head+eye keypoint set). Wired up automatically in `main()` for
        `KPMS_Loader.HEAD_WITH_PUPIL_POINTS`.
    use_block_pca, block_pca_latent_dims:
        Optional, off by default. When `use_block_pca=True`, fits PCA using
        the separate-per-keypoint-group block method
        (`eye_movement_in_pcs.md`'s Option B, see
        `block_pca.fit_pca_by_keypoint_group`) instead of kpms's standard
        joint PCA, so head displacement can't crowd eye movement out of the
        latent space. Requires `eye_diagnostics_keypoint_groups` (reused as
        the block-PCA keypoint groups) and `block_pca_latent_dims`
        (`{group_name: n_dims}`, each group's share of the latent space) to
        both be given.

    Returns
    -------
    results dict from kpms.extract_results.
    """
    if use_bodyparts is None:
        use_bodyparts = bodyparts
    if use_block_pca and (eye_diagnostics_keypoint_groups is None or block_pca_latent_dims is None):
        raise ValueError(
            "use_block_pca=True requires both eye_diagnostics_keypoint_groups "
            "(reused as the block-PCA keypoint groups) and block_pca_latent_dims."
        )
    run_start = time.perf_counter()

    setup_project(project_dir, dlc_config=dlc_config)
    configure_project(
        project_dir,
        video_dir=video_dir,
        bodyparts=bodyparts,
        use_bodyparts=use_bodyparts,
        anterior_bodyparts=anterior_bodyparts,
        posterior_bodyparts=posterior_bodyparts,
        fps=fps,
        outlier_scale_factor=outlier_scale_factor,
        skeleton=skeleton,
    )
    print(f"Config: {kpms.load_config(str(project_dir))}")

    with _timed_step("prepare_data"):
        data, metadata = prepare_data(coordinates=coordinates, confidences=confidences, project_dir=project_dir)
    with _timed_step("fit_pca"):
        pca = fit_pca(
            data,
            project_dir,
            keypoint_groups=eye_diagnostics_keypoint_groups if use_block_pca else None,
            block_latent_dims=block_pca_latent_dims if use_block_pca else None,
        )
    with _timed_step(f"fit_ar_model ({num_ar_iters} iters)"):
        _, model_name = fit_ar_model(data, metadata, pca, project_dir, num_ar_iters=num_ar_iters)
    with _timed_step(f"fit_full_model ({num_full_iters} iters)"):
        fit_full_model(
            data, metadata, project_dir, model_name,
            num_ar_iters=num_ar_iters,
            num_full_iters=num_full_iters,
            kappa=kappa,
        )
    with _timed_step("extract_results"):
        results = extract_results(project_dir, model_name)
    with _timed_step("variance_explained"):
        variance_explained(project_dir, model_name)
    if eye_diagnostics_recording_folder is not None and eye_diagnostics_keypoint_groups is not None:
        with _timed_step("eye_pc_diagnostics"):
            eye_pc_diagnostics(
                project_dir, model_name, eye_diagnostics_recording_folder,
                use_bodyparts, eye_diagnostics_keypoint_groups,
                pca=pca, results=results,
            )

    with _timed_step("trajectory_plots"):
        trajectory_plots(project_dir, model_name, coordinates, results)
    with _timed_step("generate_grid_movies"):
        generate_grid_movies(project_dir, model_name, coordinates, results)
    with _timed_step("plot_syllable_dendrogram"):
        plot_syllable_dendrogram(project_dir, model_name, coordinates, results)

    print(f"[done] run_pipeline total: {_format_duration(time.perf_counter() - run_start)}")

    return results


def run_configured(
    project_dir: str | Path,
    loader: KPMS_Loader,
    source,
    video_dir: str | Path,
    use_bodyparts: list[str],
    anterior_bodyparts: list[str],
    posterior_bodyparts: list[str],
    fps: int,
    num_ar_iters: int = 50,
    num_full_iters: int = 500,
    kappa: float = 1e4,
    outlier_scale_factor: float = 6.0,
    dlc_config: str | Path | None = None,
    skeleton: list[list[str]] | None = None,
    eye_diagnostics_keypoint_groups: dict[str, list[str]] | None = None,
    use_block_pca: bool = False,
    block_pca_latent_dims: dict[str, int] | None = None,
) -> dict:
    """
    Load keypoints and run the full kpms pipeline with explicit configuration.

    Parameters
    ----------
    loader:
        Loader to use.  One of the kpms built-in format strings
        ("deeplabcut", "sleap", "anipose", "sleap-anipose", "nwb", "facemap",
        "freipose", "dannce") or a custom loader ("3d_data", "solver_output",
        "eye_3d").
    source:
        Passed directly to `load_keypoints` — a filepath pattern for most
        loaders, or a RecordingFolder / list of RecordingFolders for the
        custom loaders. Also passed as `eye_diagnostics_recording_folder` to
        `run_pipeline` when `eye_diagnostics_keypoint_groups` is given, so
        `source` must be a RecordingFolder / list of RecordingFolders in
        that case (i.e. one of the custom, non-path-based loaders).
    eye_diagnostics_keypoint_groups:
        See `run_pipeline`. Leave `None` to skip.
    use_block_pca, block_pca_latent_dims:
        See `run_pipeline`. Off by default.

    All other parameters are forwarded to `run_pipeline`.
    """
    if skeleton is None and loader in {KPMS_Loader.SOLVER_OUTPUT, KPMS_Loader.DATA_3D}:
        raise ValueError(f"Loader '{loader.value}' requires a skeleton definition.")
    with _timed_step(f"load_keypoints ({loader.value})"):
        coordinates, confidences, bodyparts = load_keypoints(loader, source)
    print(f"Loaded keypoints with bodyparts: {bodyparts}")

    return run_pipeline(
        project_dir=project_dir,
        coordinates=coordinates,
        confidences=confidences,
        bodyparts=bodyparts,
        video_dir=video_dir,
        use_bodyparts=use_bodyparts,
        anterior_bodyparts=anterior_bodyparts,
        posterior_bodyparts=posterior_bodyparts,
        fps=fps,
        num_ar_iters=num_ar_iters,
        num_full_iters=num_full_iters,
        kappa=kappa,
        outlier_scale_factor=outlier_scale_factor,
        dlc_config=dlc_config,
        skeleton=skeleton,
        eye_diagnostics_recording_folder=source if eye_diagnostics_keypoint_groups else None,
        eye_diagnostics_keypoint_groups=eye_diagnostics_keypoint_groups,
        use_block_pca=use_block_pca,
        block_pca_latent_dims=block_pca_latent_dims,
    )


_LOADER_CONFIG = {
    KPMS_Loader.DATA_3D: {
        "use_bodyparts": FERRET_BODYPARTS,
        "anterior_bodyparts": FERRET_ANTERIOR_BODYPARTS,
        "posterior_bodyparts": FERRET_POSTERIOR_BODYPARTS,
        "skeleton": FERRET_SKELETON,
        "video_dir_attr": "mocap_synchronized_videos",
        "fps": 90,
    },
    KPMS_Loader.SOLVER_OUTPUT: {
        "use_bodyparts": FERRET_BODYPARTS,
        "anterior_bodyparts": FERRET_ANTERIOR_BODYPARTS,
        "posterior_bodyparts": FERRET_POSTERIOR_BODYPARTS,
        "skeleton": FERRET_SKELETON,
        "video_dir_attr": "mocap_synchronized_videos",
        "fps": 90,
    },
    KPMS_Loader.EYE_3D: {
        "use_bodyparts": EYE_BODYPARTS,
        "anterior_bodyparts": EYE_ANTERIOR_BODYPARTS,
        "posterior_bodyparts": EYE_POSTERIOR_BODYPARTS,
        "skeleton": EYE_SKELETON_3D,
        "video_dir_attr": "eye_videos",
        "fps": 120,
    },
    KPMS_Loader.SKULL_AND_GAZE: {
        "use_bodyparts": SKULL_AND_GAZE_BODYPARTS,
        "anterior_bodyparts": SKULL_AND_GAZE_ANTERIOR_BODYPARTS,
        "posterior_bodyparts": SKULL_AND_GAZE_POSTERIOR_BODYPARTS,
        "skeleton": SKULL_AND_GAZE_SKELETON,
        "video_dir_attr": "display_videos",
        "fps": 120,
    },
    KPMS_Loader.BOTH_EYES: {
        "use_bodyparts": BOTH_EYES_BODYPARTS,
        "anterior_bodyparts": BOTH_EYES_ANTERIOR_BODYPARTS,
        "posterior_bodyparts": BOTH_EYES_POSTERIOR_BODYPARTS,
        "skeleton": BOTH_EYES_SKELETON,
        "video_dir_attr": "eye_videos",
        "fps": 120,
    },
    KPMS_Loader.HEAD_WITH_PUPIL_POINTS: {
        "use_bodyparts": HEAD_WITH_PUPIL_POINTS_BODYPARTS,
        "anterior_bodyparts": HEAD_WITH_PUPIL_POINTS_ANTERIOR_BODYPARTS,
        "posterior_bodyparts": HEAD_WITH_PUPIL_POINTS_POSTERIOR_BODYPARTS,
        "skeleton": HEAD_WITH_PUPIL_POINTS_SKELETON,
        "video_dir_attr": "display_videos",
        "fps": 120,
        # Mixed head+eye keypoint set -- run eye_pc_diagnostics (see
        # visualization/eye_pc_viz.py) automatically after fitting, to check
        # whether eye movement is actually being captured by the PCA/AR-HMM.
        "eye_diagnostics_keypoint_groups": {
            "head": HEAD_BODYPARTS,
            "left_eye": HEAD_WITH_PUPIL_POINTS_LEFT_EYE_BODYPARTS,
            "right_eye": HEAD_WITH_PUPIL_POINTS_RIGHT_EYE_BODYPARTS,
        },
    },
}


def main(
    project_dir: str | Path,
    recording_folder: RecordingFolder | list[RecordingFolder],
    loader: KPMS_Loader,
    num_ar_iters: int = 50,
    num_full_iters: int = 500,
    kappa: float = 1e4,
    outlier_scale_factor: float = 6.0,
    use_block_pca: bool = False,
    block_pca_latent_dims: dict[str, int] | None = None,
) -> dict:
    """
    Load keypoints and run the full kpms pipeline for one or more
    RecordingFolders.

    Bodyparts, skeleton, and video directory are resolved automatically from
    the loader type, for any loader registered in `_LOADER_CONFIG` (DATA_3D,
    SOLVER_OUTPUT, EYE_3D, SKULL_AND_GAZE, BOTH_EYES, HEAD_WITH_PUPIL_POINTS).

    Writes the loader and exact recording folder paths to
    `{project_dir}/training_recordings.yaml` (see `training_config.py`)
    before running, so downstream scripts (`replot_session.py`,
    `visualization/eye_syllable_viz.py`) can look up which recordings this
    project's model was trained on instead of having that list duplicated
    (and potentially drifting) across scripts.

    Parameters
    ----------
    project_dir:
        kpms project directory (created if it does not exist).
    recording_folder:
        The recording(s) to process. Pass a list to train a single model
        across multiple sessions.
    loader:
        One of the loaders registered in `_LOADER_CONFIG`.  FPS is resolved
        automatically per loader (90 for body loaders, 120 for eye/gaze
        loaders).
    use_block_pca, block_pca_latent_dims:
        Optional, off by default (see `run_pipeline`). Only meaningful for
        `KPMS_Loader.HEAD_WITH_PUPIL_POINTS`, whose `head`/`left_eye`/
        `right_eye` keypoint groups (registered in `_LOADER_CONFIG` as
        `eye_diagnostics_keypoint_groups`) are reused as the block-PCA
        groups. `block_pca_latent_dims` is required when `use_block_pca` is
        set, e.g. `{"head": 6, "left_eye": 2, "right_eye": 2}`.
    """
    if loader not in _LOADER_CONFIG:
        raise ValueError(
            f"main() only supports loaders registered in _LOADER_CONFIG: "
            f"{sorted(l.value for l in _LOADER_CONFIG)}. "
            f"Got '{loader.value}'. Use run_configured() for other loaders."
        )
    cfg = _LOADER_CONFIG[loader]
    if use_block_pca and cfg.get("eye_diagnostics_keypoint_groups") is None:
        raise ValueError(
            f"use_block_pca=True is only meaningful for loaders with registered "
            f"keypoint groups (currently just KPMS_Loader.HEAD_WITH_PUPIL_POINTS), "
            f"got '{loader.value}'."
        )

    recording_folders = (
        recording_folder if isinstance(recording_folder, list) else [recording_folder]
    )
    if not recording_folders:
        raise ValueError("recording_folder must not be an empty list.")

    video_dirs = [getattr(rf, cfg["video_dir_attr"]) for rf in recording_folders]
    missing = [rf.recording_name for rf, vd in zip(recording_folders, video_dirs) if vd is None]
    if missing:
        raise ValueError(f"No '{cfg['video_dir_attr']}' found for recordings: {missing}")

    if len(video_dirs) == 1:
        video_dir = video_dirs[0]
    else:
        # kpms searches video_dir recursively, so a shared parent directory
        # covering every session's videos works as a single video_dir.
        video_dir = Path(os.path.commonpath([str(vd) for vd in video_dirs]))

    # setup_project must run before save_training_config: kpms.setup_project
    # silently no-ops (doesn't raise) if project_dir already exists, so if
    # save_training_config created the directory first (writing only
    # training_recordings.yaml), config.yml would never get created on a
    # fresh project.
    setup_project(project_dir)
    save_training_config(project_dir, loader.value, recording_folders)

    return run_configured(
        project_dir=project_dir,
        loader=loader,
        source=recording_folder,
        video_dir=video_dir,
        use_bodyparts=cfg["use_bodyparts"],
        anterior_bodyparts=cfg["anterior_bodyparts"],
        posterior_bodyparts=cfg["posterior_bodyparts"],
        skeleton=cfg["skeleton"],
        fps=cfg["fps"],
        eye_diagnostics_keypoint_groups=cfg.get("eye_diagnostics_keypoint_groups"),
        num_ar_iters=num_ar_iters,
        num_full_iters=num_full_iters,
        kappa=kappa,
        outlier_scale_factor=outlier_scale_factor,
        use_block_pca=use_block_pca,
        block_pca_latent_dims=block_pca_latent_dims,
    )


if __name__ == "__main__":
    #######################################################################################
    # Uncomment below to train 2d data from DLC project
    #######################################################################################
    # main(
    #     project_dir="/home/scholab/moseq/2d_dlc_behavior_test",
    #     loader=KPMS_Loader.DEEPLABCUT,
    #     source=(
    #         "/mnt/data/ferret_recordings"
    #         "/session_2025-07-09_ferret_757_EyeCameras_P41_E13"
    #         "/full_recording/mocap_data/dlc_output"
    #         "/head_body_eyecam_retrain_test_v2"
    #         "/24676894_synchronized_correctedDLC_Resnet50"
    #         "_head_body_eyecam_retrain_test_v2_shuffle1_snapshot_best-90.csv"
    #     ),
    #     video_dir=(
    #         "/mnt/data/ferret_recordings"
    #         "/session_2025-07-09_ferret_757_EyeCameras_P41_E13"
    #         "/full_recording/mocap_data/synchronized_corrected_videos"
    #     ),
    #     use_bodyparts=[
    #         "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    #         "left_cam_tip", "right_cam_tip", "base", "spine_t1", "tail_base",
    #     ],
    #     anterior_bodyparts=["nose"],
    #     posterior_bodyparts=["tail_base"],
    #     fps=90,
    #     outlier_scale_factor=6.0,
    #     dlc_config="/home/scholab/moseq/config.yaml",
    # )

    #######################################################################################
    # Uncomment below to train 3d data from RecordingFolder
    #######################################################################################

    session_manager = SessionManager(base_recordings_root=Path("/mnt/data/ferret_recordings"))
    session_entry = next(
        entry for entry in session_manager.all()
        if entry.name == "session_2025-10-17_ferret_420_E08"
    )
    recording_folder = RecordingFolder.from_folder_path(
        session_manager.recording_folder_path(session_entry)
    )

    # main(
    #     project_dir="/home/scholab/moseq/3d_behavior_gaze_test",
    #     recording_folder=recording_folder,
    #     loader=KPMS_Loader.SKULL_AND_GAZE,
    # )

    # main(
    #     project_dir="/home/scholab/moseq/head_with_pupil_points_test",
    #     recording_folder=recording_folder,
    #     loader=KPMS_Loader.HEAD_WITH_PUPIL_POINTS,
    # )

    #######################################################################################
    # head_with_pupil_points test across multiple sessions
    # (ferret_405_EO1 is not yet registered in sessions.yaml, so recording
    # folders are built directly from base_recordings_root rather than via
    # SessionManager for consistency across all four sessions.)
    #######################################################################################
    multi_session_names = [
        "session_2026-02-28_ferret_405_EO0",
        "session_2026-02-28_ferret_407_EO0",
        "session_2026-03-01_ferret_405_EO1",
        "session_2026-03-01_ferret_407_EO1",
    ]
    multi_recording_folders = [
        RecordingFolder.from_folder_path(
            Path("/mnt/data/ferret_recordings") / name / "full_recording"
        )
        for name in multi_session_names
    ]

    # main(
    #     project_dir="/home/scholab/moseq/head_with_pupil_points_405_407_test",
    #     recording_folder=multi_recording_folders,
    #     loader=KPMS_Loader.HEAD_WITH_PUPIL_POINTS,
    # )

    #######################################################################################
    # Same 4-session test, re-run with the block PCA method (Option B in
    # eye_movement_in_pcs.md) instead of kpms's default joint PCA -- separate
    # PCA per keypoint group (head / left_eye / right_eye), each normalized
    # to unit variance before assembling into one block-diagonal loading
    # matrix, so head displacement can't crowd eye movement out of the top
    # latent_dim components. Off by default elsewhere (use_block_pca=False);
    # enabled here to compare against the joint-PCA run above.
    #######################################################################################
    main(
        project_dir="/home/scholab/moseq/head_with_pupil_points_405_407_block_pca_test",
        recording_folder=multi_recording_folders,
        loader=KPMS_Loader.HEAD_WITH_PUPIL_POINTS,
        use_block_pca=True,
        block_pca_latent_dims={"head": 6, "left_eye": 2, "right_eye": 2},
    )