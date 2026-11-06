"""
Kappa sweeps over an 18-session ferret dataset, for four models (plus
`head_recentered` / `scaled_eyes_recentered`, defined after `SWEEPS`):

- `head`: head/skull keypoints only (`KPMS_Loader.HEAD`)
- `eyes`: both eyes' keypoints in the skull frame (`KPMS_Loader.BOTH_EYES_SKULL_FRAME`)
- `eyes_recentered`: `eyes` with each session's static eye shape replaced by
  the across-session mean (`recenter_static_shape`), removing animal identity
- `scaled_eyes`: head + pupil points with eye-in-head motion amplified 10x
  (`KPMS_Loader.HEAD_WITH_SCALED_EYES`), 16 latent dims

Run each model as its own process (one per GPU) in the `keypoint_moseq` env:

    CUDA_VISIBLE_DEVICES=0 python -m python_code.moseq.run_multi_session_kappa_sweeps head
    CUDA_VISIBLE_DEVICES=1 python -m python_code.moseq.run_multi_session_kappa_sweeps eyes

`run_kappa_sweep` skips kappas whose `results.h5` already exists, so to resume
an interrupted sweep rerun with `--ar-model-name <name printed after warmup>`
(reuses the PCA/AR warmup instead of refitting it). `--check` only loads the
keypoints and prints per-session frame counts.
"""
import argparse
from pathlib import Path

from python_code.moseq.run_moseq_pipeline import KPMS_Loader, load_keypoints, run_kappa_sweep
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

RECORDINGS_ROOT = Path("/mnt/data/ferret_recordings")

SESSION_NAMES = [
    "session_2025-07-03_ferret_753_EyeCameras_P35_EO7",
    "session_2025-07-03_ferret_757_EyeCameras_P35_EO7",
    "session_2025-07-07_ferret_753_EyeCameras_P39_E11",
    "session_2025-07-07_ferret_757_EyeCameras_P39_E11",
    "session_2025-10-11_ferret_402_E02",
    "session_2025-10-11_ferret_420_E02",
    "session_2025-10-12_ferret_402_E03",
    "session_2025-10-12_ferret_420_E03",
    "session_2025-10-13_ferret_402_E04",
    "session_2025-10-13_ferret_420_E04",
    "session_2025-10-14_ferret_402_E05",
    "session_2025-10-14_ferret_420_E05",
    "session_2025-10-15_ferret_420_E06",
    "session_2025-10-17_ferret_420_E08",
    "session_2026-02-28_ferret_405_EO0",
    "session_2026-02-28_ferret_407_EO0",
    "session_2026-03-01_ferret_405_EO1",
    "session_2026-03-01_ferret_407_EO1",
]

# Frame ranges [start, stop) to keep, per session. From ~frame 22460 on,
# 2025-10-15_ferret_420_E06's head is upside down / ~0.5 m above the arena
# (handled or mis-tracked), not natural behavior.
KEEP_FRAMES = {
    "session_2025-10-15_ferret_420_E06": (0, 22400),
}

SWEEPS = {
    "head": {
        "loader": KPMS_Loader.HEAD,
        "project_dir": "/mnt/data/moseq/head_18_sessions_kappa_sweep",
        # Head syllables are slower than eye syllables, so this sweep sits
        # higher than the eye sweep's range.
        "kappas": [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7],
        "exclude_sessions": [],
    },
    "eyes": {
        "loader": KPMS_Loader.BOTH_EYES_SKULL_FRAME,
        "project_dir": "/mnt/data/moseq/both_eyes_skull_frame_18_sessions_kappa_sweep",
        # Same range as the 4-session test sweep (1e1-1e5), extended up a
        # decade because median durations were still short at 1e5 there.
        "kappas": [1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6],
        # Left eye passes the medium eye-data-quality threshold on only 0.1%
        # of frames.
        "exclude_sessions": ["session_2025-07-03_ferret_753_EyeCameras_P35_EO7"],
    },
    "eyes_recentered": {
        "loader": KPMS_Loader.BOTH_EYES_SKULL_FRAME,
        "project_dir": "/mnt/data/moseq/both_eyes_skull_frame_recentered_17_sessions_kappa_sweep",
        # Each session's static eye shape (headmount/skull-geometry offsets)
        # replaced by the across-session mean: in the `eyes` model the top
        # two PCs (85% of variance) were that offset and made animal identity
        # decodable at ~86% (animal_identity_decoding, static_shape_check).
        "loader_options": {"recenter_static_shape": True},
        # Bracket the `eyes` sweep's short end; same latent_dim (10, 92% of
        # variance after recentering) for a direct comparison.
        "kappas": [1e2, 1e3, 1e4],
        "exclude_sessions": ["session_2025-07-03_ferret_753_EyeCameras_P35_EO7"],
    },
    "scaled_eyes": {
        "loader": KPMS_Loader.HEAD_WITH_SCALED_EYES,
        "project_dir": "/mnt/data/moseq/head_with_scaled_eyes_a10_ld16_16_sessions_kappa_sweep",
        # Short end only: eye movements are the focus, and on 405/407 the
        # gaze-shift syllables held at 1e3-1e6 with only duration changing.
        "kappas": [1e3, 3e3, 1e4],
        # 753 2025-07-03: no usable left eye. 420 E06: last ~45 s the head is
        # upside down / ~0.5 m up (see KEEP_FRAMES); dropped whole for now.
        "exclude_sessions": [
            "session_2025-07-03_ferret_753_EyeCameras_P35_EO7",
            "session_2025-10-15_ferret_420_E06",
        ],
        # 10 PCs lose much of head roll and eye elevation across animals
        # (eye_scale_sweep 18_sessions); 16 recovered them on 405/407.
        "latent_dim": 16,
        # Parallel message passing in 32 sequential chunks: 9.5 GB peak and
        # ~41 s/iter on 16 sessions, vs 9.0 GB / ~51 s serial (4 chunks).
        # Unchunked, either mode runs out of memory on a 24 GB card. Run with
        # XLA_PYTHON_CLIENT_MEM_FRACTION=.95 (default .75 is too tight).
        "parallel_message_passing": True,
        "mixed_map_iters": 32,
    },
}

# Same as `head` / `scaled_eyes`, but with each session's skull geometry (and
# resting pupil positions) replaced by the across-session mean in the skull
# frame (`replace_static_shape_in_skull_frame`). Headmount markers vary by
# ~4 mm SD across sessions, which made recording cohort decodable at ~90%
# (static_shape_check). With a shared rigid skull, the head part carries only
# pitch and roll after kpms's translation + yaw alignment.
SWEEPS["head_recentered"] = {
    **SWEEPS["head"],
    "project_dir": "/mnt/data/moseq/head_recentered_18_sessions_kappa_sweep",
    "loader_options": {"recenter_static_shape": True},
    "kappas": [1e3, 1e4, 1e5],
}
SWEEPS["scaled_eyes_recentered"] = {
    **SWEEPS["scaled_eyes"],
    "project_dir": "/mnt/data/moseq/head_with_scaled_eyes_recentered_a10_ld16_16_sessions_kappa_sweep",
    "loader_options": {"recenter_static_shape": True},
}


def recording_folders(exclude_sessions: list[str]) -> list[RecordingFolder]:
    return [
        RecordingFolder.from_folder_path(RECORDINGS_ROOT / name / "full_recording")
        for name in SESSION_NAMES
        if name not in exclude_sessions
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(SWEEPS))
    parser.add_argument("--check", action="store_true", help="only load keypoints and report frame counts")
    parser.add_argument("--ar-model-name", default=None, help="reuse this AR warmup in the project dir")
    args = parser.parse_args()

    sweep = SWEEPS[args.model]
    folders = recording_folders(sweep["exclude_sessions"])

    if args.check:
        coordinates, _, bodyparts = load_keypoints(sweep["loader"], folders, **sweep.get("loader_options", {}))
        total = 0
        for name, coords in coordinates.items():
            total += len(coords)
            print(f"{name}: {coords.shape}")
        print(f"{len(coordinates)} recordings, {len(bodyparts)} bodyparts, {total} frames total")
    else:
        run_kappa_sweep(
            project_dir=sweep["project_dir"],
            recording_folder=folders,
            loader=sweep["loader"],
            kappas=sweep["kappas"],
            ar_model_name=args.ar_model_name,
            # The default GPU-parallel Kalman scan needs ~46 GB on ~680k
            # frames, more than a 24 GB card has.
            parallel_message_passing=sweep.get("parallel_message_passing", False),
            latent_dim=sweep.get("latent_dim"),
            mixed_map_iters=sweep.get("mixed_map_iters"),
            loader_options=sweep.get("loader_options"),
        )
