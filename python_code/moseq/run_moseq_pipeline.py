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

Usage
-----
    from python_code.moseq.run_moseq_pipeline import main, run_configured

    # Simple entry point for RecordingFolder-based loaders (DATA_3D, SOLVER_OUTPUT, EYE_3D):
    main(
        project_dir="/path/to/project",
        recording_folder=recording_folder,
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
from enum import Enum
from pathlib import Path

# Must be set before JAX (imported by keypoint_moseq) starts its thread pool.
# On Linux, subprocess.Popen uses os.fork(), which deadlocks in multi-threaded
# processes. "forkserver" avoids this by spawning a clean server process.
multiprocessing.set_start_method("forkserver", force=True)

import keypoint_moseq as kpms
import matplotlib.pyplot as plt

from python_code.moseq.kpms_loader import (
    load_3d_data_kpms,
    load_3d_eye_kpms,
    load_both_eyes_kpms,
    load_skull_and_gaze_kpms,
    load_solver_output_kpms,
)
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
    SKULL_AND_GAZE_ANTERIOR_BODYPARTS,
    SKULL_AND_GAZE_BODYPARTS,
    SKULL_AND_GAZE_POSTERIOR_BODYPARTS,
)
from python_code.moseq.utils.skeletons import BOTH_EYES_SKELETON, EYE_SKELETON_3D, FERRET_SKELETON, SKULL_AND_GAZE_SKELETON
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


_KPMS_BUILTIN_LOADERS = {
    "deeplabcut", "sleap", "anipose", "sleap-anipose",
    "nwb", "facemap", "freipose", "dannce",
}


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
    else:
        raise ValueError(
            f"Unknown loader '{loader.value}'. Must be one of: "
            f"{sorted(_KPMS_BUILTIN_LOADERS | {'3d_data', 'solver_output', 'eye_3d', 'skull_and_gaze'})}"
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
) -> object:
    """
    Fit a PCA model, save it, generate diagnostic plots, and update sigmasq config.

    Returns
    -------
    pca
        The fitted PCA model.
    """
    config = lambda: kpms.load_config(str(project_dir))

    plt.close("all")
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

def trajectory_plots(
        project_dir: str | Path,
        model_name: str,
        coordinates: dict,
        results: dict,
):
    """
    Generate plots showing the median trajectory of poses associated with each syllable.
    """
    config = lambda: kpms.load_config(str(project_dir))
    kpms.generate_trajectory_plots(
        coordinates,
        results,
        str(project_dir),
        model_name,
        **config(),
    )

def generate_grid_movies(
        project_dir: str | Path,
        model_name: str,
        coordinates: dict,
        results: dict,
):
    config = lambda: kpms.load_config(str(project_dir))
    keypoints_only = True if coordinates[next(iter(coordinates))].shape[2] != 2 else False
    kpms.generate_grid_movies(
        results,
        str(project_dir),
        model_name,
        coordinates=coordinates,
        keypoints_only=keypoints_only,
        keypoints_scale=5.0,
        use_dims=[0, 1],
        **config(),
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

    Returns
    -------
    results dict from kpms.extract_results.
    """
    if use_bodyparts is None:
        use_bodyparts = bodyparts
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

    data, metadata = prepare_data(coordinates=coordinates, confidences=confidences, project_dir=project_dir)
    pca = fit_pca(data, project_dir)
    _, model_name = fit_ar_model(data, metadata, pca, project_dir, num_ar_iters=num_ar_iters)
    fit_full_model(
        data, metadata, project_dir, model_name,
        num_ar_iters=num_ar_iters,
        num_full_iters=num_full_iters,
        kappa=kappa,
    )
    results = extract_results(project_dir, model_name)

    trajectory_plots(project_dir, model_name, coordinates, results)
    generate_grid_movies(project_dir, model_name, coordinates, results)
    plot_syllable_dendrogram(project_dir, model_name, coordinates, results)

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
        custom loaders.

    All other parameters are forwarded to `run_pipeline`.
    """
    if skeleton is None and loader in {KPMS_Loader.SOLVER_OUTPUT, KPMS_Loader.DATA_3D}:
        raise ValueError(f"Loader '{loader.value}' requires a skeleton definition.")
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
}


def main(
    project_dir: str | Path,
    recording_folder: RecordingFolder,
    loader: KPMS_Loader,
    num_ar_iters: int = 50,
    num_full_iters: int = 500,
    kappa: float = 1e4,
    outlier_scale_factor: float = 6.0,
) -> dict:
    """
    Load keypoints and run the full kpms pipeline for a RecordingFolder.

    Bodyparts, skeleton, and video directory are resolved automatically from
    the loader type.  Only DATA_3D, SOLVER_OUTPUT, and EYE_3D are supported.

    Parameters
    ----------
    project_dir:
        kpms project directory (created if it does not exist).
    recording_folder:
        The recording to process.
    loader:
        One of KPMS_Loader.DATA_3D, KPMS_Loader.SOLVER_OUTPUT, or
        KPMS_Loader.EYE_3D.  FPS is resolved automatically (90 for body
        loaders, 120 for EYE_3D).
    """
    if loader not in _LOADER_CONFIG:
        raise ValueError(
            f"main() only supports DATA_3D, SOLVER_OUTPUT, and EYE_3D. "
            f"Got '{loader.value}'. Use run_configured() for other loaders."
        )
    cfg = _LOADER_CONFIG[loader]
    video_dir = getattr(recording_folder, cfg["video_dir_attr"])

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
        num_ar_iters=num_ar_iters,
        num_full_iters=num_full_iters,
        kappa=kappa,
        outlier_scale_factor=outlier_scale_factor,
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

    recording_folder = RecordingFolder.from_folder_path(
       "/mnt/data/ferret_recordings/session_2025-07-09_ferret_757_EyeCameras_P41_E13/full_recording"
    )

    # main(
    #     project_dir="/home/scholab/moseq/3d_behavior_gaze_test",
    #     recording_folder=recording_folder,
    #     loader=KPMS_Loader.SKULL_AND_GAZE,
    # )

    main(
        project_dir="/home/scholab/moseq/both_eyes_test",
        recording_folder=recording_folder,
        loader=KPMS_Loader.BOTH_EYES,
    )