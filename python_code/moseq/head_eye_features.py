"""
Head + eye kinematic features for a joint AR-HMM
================================================

Builds the 9-channel feature trajectory fit by `run_head_eye_arhmm.py`: head
pose, both eyes' head-relative gaze angles, and pupil size, on the shared
common-timestamp grid. See `eye_movement_in_pcs.md` (Option C) for why this
skips keypoint PCA.

Channels
--------
- `head_pitch`, `head_roll`: gravity-referenced Euler angles (ZYX) of the
  skull orientation. Positive pitch is nose down. World axes aren't guaranteed
  by calibration (`analyzable_output_csvs.md`), so `_skull_angles` checks that
  world +Z is up for each session. Head channels are invalid where the skull
  is rolled past `_MAX_ROLL_DEG` or jumps by more than
  `_MAX_ROTATION_DEG_PER_FRAME` (see those constants). Roll stays wrapped:
  unwrapping turns each brief flip through +-180 into a permanent 360 offset.
- `head_yaw_rate`: `angular_velocity_local` yaw, the skull-frame turning rate.
  Absolute yaw isn't used: its zero is arbitrary and it runs through several
  full turns per session.
- `{left,right}_adduction`, `{left,right}_elevation`: `eye_in_head` angles.
- `{left,right}_pupil`: `kpms_loader.load_pupil_radius_kpms`.

Normalization
-------------
Each session's median is subtracted per channel (not `head_yaw_rate`, whose
zero is meaningful), removing the static headmount/eye-camera offsets that
otherwise make animal identity decodable (`animal_identity_decoding`). Each
channel is then divided by one scale shared by all sessions, not a per-session
one, so real differences in movement amplitude across ages survive for
`development_stats`.

Frames stay full length so frame indices match every other tool: bad frames
are linearly interpolated in `features` and flagged in `valid`, and it's up to
the caller to mask them (see `training_mask`).

Noise floor
-----------
`add_noise_floor` must be applied before fitting. These signals are so smooth
at 120 Hz that one AR state extrapolates any movement (AR R^2 0.9999), so with
no added noise one state takes ~96% of frames and the rest are brief
high-noise bursts: states differ by noise level, not by what's moving. On the
405/407 sessions, `NOISE_FLOOR_SD` gave 15 used states with noise levels within
2x of each other, and is what kpms's `added_noise_level` does for keypoints.
Lowering the frame rate also helps, but less (10 states at 30 Hz, with noise
levels 15x apart) and at the cost of saccade resolution.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from python_code.kinematics_core.tidy_dataframe_io import read_parquet_or_csv
from python_code.moseq.kpms_loader import (
    _SKULL_KINEMATICS_CSV_FILENAME,
    _SKULL_KINEMATICS_FILENAME,
    _polars_tidy_to_pandas,
    load_eye_in_head_kpms,
    load_pupil_radius_kpms,
)
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

HEAD_CHANNELS = ["head_pitch", "head_roll", "head_yaw_rate"]
EYE_CHANNELS = [
    "left_adduction",
    "left_elevation",
    "right_adduction",
    "right_elevation",
    "left_pupil",
    "right_pupil",
]
CHANNELS = HEAD_CHANNELS + EYE_CHANNELS

CHANNEL_UNITS = {
    "head_pitch": "deg",
    "head_roll": "deg",
    "head_yaw_rate": "deg/s",
    "left_adduction": "deg",
    "left_elevation": "deg",
    "right_adduction": "deg",
    "right_elevation": "deg",
    "left_pupil": "mm",
    "right_pupil": "mm",
}

# Normalized units, so about 0.09 deg of eye angle or 1.7 deg/s of yaw rate.
NOISE_FLOOR_SD = 0.02

# Minimum mean skull +Z along world +Z for a session's Euler angles to count as
# gravity-referenced. Every session in the dataset is >= 0.98.
_MIN_WORLD_UP_ALIGNMENT = 0.9

# Most frames beyond these are ~0.1 s bouts entered at 15-100 deg/frame, which
# are tracking flips or shakes too fast to model. The rest are slow roll-overs
# in the youngest kits (405/407 EO0). Roll past 60 deg is <= 3% of frames in
# every session once KEEP_FRAMES is applied.
_MAX_ROLL_DEG = 60.0
_MAX_ROTATION_DEG_PER_FRAME = 20.0

_ORIENTATION_TRAJECTORY = "orientation"
_ANGULAR_VELOCITY_TRAJECTORY = "angular_velocity_local"


@dataclass(frozen=True)
class HeadEyeFeatures:
    """
    Attributes
    ----------
    features:
        `{recording_name: (n_frames, 9)}`, normalized, no NaNs.
    valid:
        `{recording_name: (n_frames, 9)}` bool, False where a channel was
        interpolated or the frame is outside `keep_frames`.
    channels:
        Channel names, `features`' column order.
    scales:
        (9,) divisor per channel, in `CHANNEL_UNITS`.
    medians:
        `{recording_name: (9,)}` subtracted per session, in `CHANNEL_UNITS`.
    """

    features: dict[str, NDArray[np.float64]]
    valid: dict[str, NDArray[np.bool_]]
    channels: list[str]
    scales: NDArray[np.float64]
    medians: dict[str, NDArray[np.float64]]


def _skull_kinematics(rf: RecordingFolder) -> pd.DataFrame:
    return _polars_tidy_to_pandas(
        read_parquet_or_csv(
            parquet_path=rf.skull_kinematics / _SKULL_KINEMATICS_FILENAME,
            csv_path=rf.skull_kinematics / _SKULL_KINEMATICS_CSV_FILENAME,
        )
    )


def _skull_angles(rf: RecordingFolder) -> NDArray[np.float64]:
    """(n_frames, 3) [pitch, roll, yaw rate], degrees and degrees/s, NaN where
    the head pose is implausible (see `_MAX_ROLL_DEG`)."""
    df = _skull_kinematics(rf)
    quaternions = (
        df[df["trajectory"] == _ORIENTATION_TRAJECTORY]
        .pivot(index="frame", columns="component", values="value")
        .sort_index()[["x", "y", "z", "w"]]
        .to_numpy()
    )
    angular_velocity = (
        df[df["trajectory"] == _ANGULAR_VELOCITY_TRAJECTORY]
        .pivot(index="frame", columns="component", values="value")
        .sort_index()["yaw"]
        .to_numpy()
    )
    if len(quaternions) != len(angular_velocity):
        raise ValueError(
            f"'{rf.recording_name}': {len(quaternions)} orientation frames but "
            f"{len(angular_velocity)} angular velocity frames"
        )

    rotations = Rotation.from_quat(quaternions)
    skull_up = rotations.as_matrix()[:, :, 2].mean(axis=0)
    up_alignment = skull_up[2] / np.linalg.norm(skull_up)
    if up_alignment < _MIN_WORLD_UP_ALIGNMENT:
        raise ValueError(
            f"'{rf.recording_name}': mean skull up has z = {up_alignment:.2f} in world "
            "coordinates, so world +Z isn't up and pitch/roll aren't gravity-referenced."
        )

    _, pitch, roll = rotations.as_euler("ZYX").T
    angles = np.degrees(np.column_stack([pitch, roll, angular_velocity]))

    rotation_step = np.degrees((rotations[1:] * rotations[:-1].inv()).magnitude())
    jumped = np.zeros(len(angles), dtype=bool)
    jumped[1:] = rotation_step > _MAX_ROTATION_DEG_PER_FRAME
    angles[jumped | (np.abs(angles[:, 1]) > _MAX_ROLL_DEG)] = np.nan
    return angles


def _raw_channels(rf: RecordingFolder, quality_level: str | None) -> NDArray[np.float64]:
    """(n_frames, 9) unnormalized channels, NaN where an eye fails quality."""
    name = rf.recording_name
    left_angles, right_angles = load_eye_in_head_kpms(rf, quality_level=quality_level)
    left_radius, right_radius = load_pupil_radius_kpms(rf)
    skull = _skull_angles(rf)

    # The pupil channels share their eye's quality flags, which are the only
    # thing that makes `eye_in_head` NaN here.
    left_good = ~np.isnan(left_angles[name]).any(axis=1)
    right_good = ~np.isnan(right_angles[name]).any(axis=1)

    lengths = {
        "skull": len(skull),
        "left eye": len(left_good),
        "right eye": len(right_good),
        "left pupil": len(left_radius[name]),
        "right pupil": len(right_radius[name]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"'{name}': frame counts don't match: {lengths}")

    return np.column_stack(
        [
            skull,
            left_angles[name],
            right_angles[name],
            np.where(left_good, left_radius[name], np.nan),
            np.where(right_good, right_radius[name], np.nan),
        ]
    )


def _interpolate_nans(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fill each column's NaNs linearly, holding the ends constant."""
    filled = values.copy()
    frames = np.arange(len(values))
    for column in range(values.shape[1]):
        ok = np.isfinite(values[:, column])
        filled[:, column] = np.interp(frames, frames[ok], values[ok, column])
    return filled


def load_head_eye_features(
    recording_folders: list[RecordingFolder],
    quality_level: str | None = "medium",
    keep_frames: dict[str, tuple[int, int]] | None = None,
) -> HeadEyeFeatures:
    """
    Load and normalize the 9 channels for each recording.

    Parameters
    ----------
    quality_level:
        `load_eye_data_quality` level; an eye's channels are invalid on frames
        where it fails. None keeps every frame.
    keep_frames:
        `{recording_name: (start, stop)}`; frames outside are invalid in every
        channel, but stay in the arrays. As
        `run_multi_session_kappa_sweeps.KEEP_FRAMES`.
    """
    keep_frames = keep_frames or {}
    centered: dict[str, NDArray] = {}
    medians: dict[str, NDArray] = {}

    for rf in recording_folders:
        name = rf.recording_name
        if name in centered:
            raise ValueError(
                f"Duplicate recording name '{name}'. Each RecordingFolder must "
                "have a unique recording_name."
            )
        raw = _raw_channels(rf, quality_level)
        if name in keep_frames:
            start, stop = keep_frames[name]
            outside = np.ones(len(raw), dtype=bool)
            outside[start:stop] = False
            raw[outside] = np.nan

        median = np.nanmedian(raw, axis=0)
        median[CHANNELS.index("head_yaw_rate")] = 0.0
        medians[name] = median
        centered[name] = raw - median

    scales = np.nanstd(np.concatenate(list(centered.values())), axis=0)

    return HeadEyeFeatures(
        features={name: _interpolate_nans(c) / scales for name, c in centered.items()},
        valid={name: np.isfinite(c) for name, c in centered.items()},
        channels=CHANNELS,
        scales=scales,
        medians=medians,
    )


def add_noise_floor(features: HeadEyeFeatures, noise_sd: float, seed: int = 0) -> HeadEyeFeatures:
    """Add white noise of `noise_sd` normalized units to every channel. See the
    module docstring for why."""
    rng = np.random.default_rng(seed)
    return replace(
        features,
        features={
            name: x + rng.normal(scale=noise_sd, size=x.shape) for name, x in features.features.items()
        },
    )


def use_head_rates(
    features: HeadEyeFeatures, recording_folders: list[RecordingFolder]
) -> HeadEyeFeatures:
    """
    Replace `head_pitch` and `head_roll` with their `angular_velocity_local`
    rates, normalized like `head_yaw_rate`, so all three head channels are
    velocities. `valid` is unchanged because the head channels are always
    invalid together.
    """
    pitch, roll = CHANNELS.index("head_pitch"), CHANNELS.index("head_roll")
    centered: dict[str, NDArray] = {}
    for rf in recording_folders:
        name = rf.recording_name
        df = _skull_kinematics(rf)
        rates = np.degrees(
            df[df["trajectory"] == _ANGULAR_VELOCITY_TRAJECTORY]
            .pivot(index="frame", columns="component", values="value")
            .sort_index()[["pitch", "roll"]]
            .to_numpy()
        )
        rates[~features.valid[name][:, [pitch, roll]]] = np.nan
        centered[name] = rates

    rate_scales = np.nanstd(np.concatenate(list(centered.values())), axis=0)

    def with_rates(x: NDArray, rates: NDArray) -> NDArray:
        replaced = x.copy()
        replaced[:, [pitch, roll]] = rates
        return replaced

    scales = features.scales.copy()
    scales[[pitch, roll]] = rate_scales
    channels = list(features.channels)
    channels[pitch], channels[roll] = "head_pitch_rate", "head_roll_rate"
    return replace(
        features,
        features={
            name: with_rates(x, _interpolate_nans(centered[name]) / rate_scales)
            for name, x in features.features.items()
        },
        channels=channels,
        scales=scales,
        medians={name: with_rates(m[None], 0.0)[0] for name, m in features.medians.items()},
    )


def drop_pupil(features: HeadEyeFeatures) -> HeadEyeFeatures:
    """
    Remove the pupil channels, to check whether syllables depend on them.
    `valid` keeps all 9 columns so `training_mask` selects the same frames as
    the 9-channel model.
    """
    keep = [i for i, name in enumerate(features.channels) if not name.endswith("_pupil")]
    return replace(
        features,
        features={name: x[:, keep] for name, x in features.features.items()},
        channels=[features.channels[i] for i in keep],
        scales=features.scales[keep],
        medians={name: m[keep] for name, m in features.medians.items()},
    )


def keep_channels(features: HeadEyeFeatures, channels: list[str]) -> HeadEyeFeatures:
    """
    Keep only `channels`, e.g. to fit the head or eye block alone. As with
    `drop_pupil`, `valid` keeps all 9 columns, so `training_mask` selects the
    same frames as the joint model and syllable sequences and held-out
    likelihood frames line up with it frame for frame.
    """
    missing = [c for c in channels if c not in features.channels]
    if missing:
        raise ValueError(f"Channels {missing} not in {features.channels}")
    keep = [features.channels.index(c) for c in channels]
    return replace(
        features,
        features={name: x[:, keep] for name, x in features.features.items()},
        channels=list(channels),
        scales=features.scales[keep],
        medians={name: m[keep] for name, m in features.medians.items()},
    )


def training_mask(valid: NDArray[np.bool_], nlags: int) -> NDArray[np.bool_]:
    """
    (n_frames,) True where a frame and its `nlags` frames of AR history are
    valid in every channel. jax_moseq's mask only covers the frame itself, so
    without this the first frames after a blink would regress on interpolated
    history.
    """
    bad = ~valid.all(axis=1)
    has_bad_history = np.convolve(bad, np.ones(nlags + 1), mode="full")[: len(bad)] > 0
    return ~has_bad_history


def shift_eye_block(
    features: HeadEyeFeatures,
    seed: int,
    splits: dict[str, int],
    min_shift_seconds: float = 60.0,
    fps: int = 120,
) -> HeadEyeFeatures:
    """
    Circularly shift the eye and pupil channels against the head channels by a
    random offset per session, within the training window only. This destroys
    head-eye coupling in training while keeping each block's own dynamics, as
    the control for the joint model.

    `splits` is each recording's first held-out frame
    (`run_head_eye_arhmm.train_splits`). Frames from there on stay untouched,
    so a real fit and its control train on the same frames and are scored on identical held-out
    data. Rolling the whole session instead would put different eye frames in
    the held-out window, and leak the real held-out eye frames into training.
    """
    rng = np.random.default_rng(seed)
    min_shift = int(min_shift_seconds * fps)
    n_head = len(HEAD_CHANNELS)

    def roll_eyes_in_training_window(x: NDArray, split: int, offset: int) -> NDArray:
        eyes = x[:, n_head:].copy()
        eyes[:split] = np.roll(eyes[:split], offset, axis=0)
        return np.column_stack([x[:, :n_head], eyes])

    shifted_features: dict[str, NDArray] = {}
    shifted_valid: dict[str, NDArray] = {}
    for name in features.features:
        split = splits[name]
        if split < 2 * min_shift:
            raise ValueError(
                f"'{name}' has {split} training frames, too short for a "
                f"{min_shift}-frame shift in both directions."
            )
        offset = int(rng.integers(min_shift, split - min_shift + 1))
        shifted_features[name] = roll_eyes_in_training_window(features.features[name], split, offset)
        shifted_valid[name] = roll_eyes_in_training_window(features.valid[name], split, offset)

    return replace(features, features=shifted_features, valid=shifted_valid)
