"""
Regenerate Keypoint MoSeq visualizations (trajectory plots, grid movies) for a
session that has already been fit, without re-running PCA/AR/full-model
fitting.

Reuses `load_keypoints`, `trajectory_plots`, and `generate_grid_movies` from
`run_moseq_pipeline.py` so plot-related config (loaded from the project's
`config.yml` via `kpms.load_config`) is honored exactly as it is during a
normal `run_pipeline` call.

Usage
-----
    from python_code.moseq.replot_session import replot

    replot(
        project_dir="/home/scholab/moseq/head_with_pupil_points_test",
        model_name="2025_10_20-12_00_00",
        loader=KPMS_Loader.HEAD_WITH_PUPIL_POINTS,
        source=recording_folder,
    )
"""

import shutil
from pathlib import Path

import keypoint_moseq as kpms

from python_code.moseq.run_moseq_pipeline import (
    KPMS_Loader,
    SessionManager,
    generate_grid_movies,
    load_keypoints,
    trajectory_plots,
)
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


def _clear_dir(path: Path) -> None:
    """Remove a plot output directory if it exists, so stale files from a
    previous run (e.g. syllables that no longer pass min_duration/
    min_frequency) can't linger alongside freshly generated ones."""
    if path.exists():
        shutil.rmtree(path)


def replot(
    project_dir: str | Path,
    model_name: str,
    loader: KPMS_Loader,
    source,
    reindex_syllables: bool = False,
    skip_trajectory_plots: bool = False,
    skip_grid_movies: bool = False,
    min_duration: float | None = None,
    min_frequency: float | None = None,
    n_neighbors: int | None = None,
    fresh: bool = True,
) -> dict:
    """
    Reload keypoints and a fitted model's results, then regenerate trajectory
    plots and/or grid movies for an already-run session.

    Parameters
    ----------
    project_dir:
        Existing kpms project directory (must already contain the model's
        checkpoint and a `config.yml`).
    model_name:
        Name of the fitted model's checkpoint subfolder under `project_dir`.
    loader, source:
        Same arguments as `load_keypoints` — must match what was used for the
        original run so coordinates line up with the fitted model
        (e.g. `loader=KPMS_Loader.HEAD_WITH_PUPIL_POINTS, source=recording_folder`).
    reindex_syllables:
        If True, re-run `kpms.reindex_syllables_in_checkpoint` before
        extracting results (matches `extract_results` in
        `run_moseq_pipeline.py`). Leave False to reuse the ordering already
        baked into the checkpoint from the original run.
    skip_trajectory_plots, skip_grid_movies:
        Set either to True to only regenerate the other.
    min_duration, min_frequency:
        Optional overrides for the corresponding config.yml values, applied
        only to this call (config.yml on disk is left untouched). Leave
        unset to use whatever is already in the project's config.
    n_neighbors:
        Optional override for trajectory plots' density-sampling neighborhood
        size (kpms default: 50), which also sets the minimum number of
        instances a syllable needs to get a trajectory plot at all. This is
        much stricter than grid movies' instance floor (rows*cols=24 by
        default), so syllables commonly appear in grid movies but not
        trajectory plots. Pass e.g. 24 to match grid movies' threshold and
        get more syllables plotted. Ignored if skip_trajectory_plots=True.
    fresh:
        If True (default), delete the existing `trajectory_plots`/
        `grid_movies` output directories before regenerating, so a replot
        never leaves stale files behind from syllables that no longer pass
        the current min_duration/min_frequency (kpms writes plots keyed by
        syllable index and never removes files on its own, so without this
        a changed filter can leave old and new syllable plots mixed
        together). Set False to let kpms overwrite in place instead.

    Returns
    -------
    results dict, as returned by `kpms.extract_results`.
    """
    project_dir = str(project_dir)

    if reindex_syllables:
        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
    # overwrite=True: replotting always re-extracts results for a session
    # that was already extracted by the original run_pipeline call, unlike
    # extract_results() in run_moseq_pipeline.py which only ever runs once
    # against a fresh results.h5.
    results = kpms.extract_results(model, metadata, project_dir, model_name, overwrite=True)

    coordinates, confidences, bodyparts = load_keypoints(loader, source)

    model_dir = Path(project_dir) / model_name
    if fresh:
        if not skip_trajectory_plots:
            _clear_dir(model_dir / "trajectory_plots")
        if not skip_grid_movies:
            _clear_dir(model_dir / "grid_movies")

    if not skip_trajectory_plots:
        trajectory_plots(
            project_dir, model_name, coordinates, results,
            min_duration=min_duration, min_frequency=min_frequency,
            n_neighbors=n_neighbors,
        )
    if not skip_grid_movies:
        generate_grid_movies(
            project_dir, model_name, coordinates, results,
            min_duration=min_duration, min_frequency=min_frequency,
        )

    return results

if __name__ == "__main__":
    session_manager = SessionManager(base_recordings_root=Path("/mnt/data/ferret_recordings"))
    session_entry = next(
        entry for entry in session_manager.all()
        if entry.name == "session_2025-10-17_ferret_420_E08"
    )
    recording_folder = RecordingFolder.from_folder_path(
        session_manager.recording_folder_path(session_entry)
    )

    replot(
        project_dir="/home/scholab/moseq/head_with_pupil_points_test",
        model_name="2026_09_22-12_49_23",
        loader=KPMS_Loader.HEAD_WITH_PUPIL_POINTS,
        source=recording_folder,
        # Match grid movies' instance floor (rows*cols=24) instead of kpms's
        # default of 50, so trajectory plots aren't stricter than grid
        # movies about how many instances a syllable needs.
        n_neighbors=24,
    )