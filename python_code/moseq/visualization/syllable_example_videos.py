"""
Per-syllable example videos (both eyes + top-down)
==================================================

One MP4 per syllable, stitching together example instances from the
recordings' `display_videos/` (`left_eye_resampled.mp4`,
`right_eye_resampled.mp4`, `top_down_mocap_resampled.mp4`). Those videos are
resampled onto the same common-timestamp grid as the model's results, so a
results frame index is a video frame index.

Layout: left eye top-left, right eye top-right, top-down underneath. The
stabilized eye canvases have nasal on the page left for both eyes, so the left
eye is mirrored to put both noses toward the middle ("nose in"). Its burned-in
annotation text reads mirrored as a result.

Instances are sampled as in `eye_syllable_gif` (kpms density sampling on the
egocentric head keypoints, so they match the GIFs/PNGs), then kept only if
both eyes pass the eye-data-quality threshold over the syllable bout. Each
clip shows `pre` frames before onset through `post` frames after the bout
ends, in slow motion; a coloured border and "SYLLABLE" tag mark the bout.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.syllable_example_videos \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil kappa_1e+03_nlags8_seed0 \\
        --syllables 15 19 10 14 12 6 9
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import keypoint_moseq as kpms
import numpy as np
from numpy.typing import NDArray

from python_code.moseq.kpms_loader import load_eye_in_head_kpms
from python_code.moseq.run_moseq_pipeline import KPMS_Loader, load_keypoints
from python_code.moseq.training_config import (
    load_training_config,
    load_training_loader_options,
    load_training_recording_folders,
)
from python_code.moseq.utils.bodyparts import HEAD_BODYPARTS

FPS = 120  # recording frame rate

# Output geometry (even numbers for yuv420p).
EYE_SIZE = 480
TOPDOWN_W, TOPDOWN_H = 2 * EYE_SIZE, int(2 * EYE_SIZE * 1024 / 1280)
CAPTION_H = 80  # text band between the eyes and the top-down view
# The 800x800 stabilized eye canvas is mostly black; the eye sits in the middle.
EYE_CROP = (120, 120, 560, 560)  # x, y, w, h in canvas pixels

BOUT_COLOR = (0, 0, 255)  # BGR red
CONTEXT_COLOR = (90, 90, 90)
BORDER = 8
TITLE_CARD_SECONDS = 0.8

# Sampling, as in `eye_syllable_gif.make_eye_syllable_gifs`.
SAMPLE_PRE, SAMPLE_POST = 5, 15
N_SAMPLED = 24


class _VideoReader:
    """Random-access frames from one video, reading sequentially when the
    next request follows the last one (cheap) and seeking otherwise."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.next_index = 0

    def read(self, index: int) -> NDArray:
        if index != self.next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise IOError(f"could not read frame {index} of {self.path}")
        self.next_index = index + 1
        return frame

    def close(self):
        self.cap.release()


def _text(img: NDArray, text: str, org: tuple[int, int], scale: float = 0.8, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _eye_tile(frame: NDArray, mirror: bool) -> NDArray:
    x, y, w, h = EYE_CROP
    tile = cv2.resize(frame[y : y + h, x : x + w], (EYE_SIZE, EYE_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.flip(tile, 1) if mirror else tile


def _composite(
    left: NDArray, right: NDArray, topdown: NDArray, in_bout: bool, header: str, time_ms: float
) -> NDArray:
    eyes = np.hstack([_eye_tile(left, mirror=True), _eye_tile(right, mirror=False)])
    _text(eyes, "left eye", (12, EYE_SIZE - 14))
    _text(eyes, "right eye", (2 * EYE_SIZE - 140, EYE_SIZE - 14))
    cv2.line(eyes, (EYE_SIZE, 0), (EYE_SIZE, EYE_SIZE), (60, 60, 60), 2)
    td = cv2.resize(topdown, (TOPDOWN_W, TOPDOWN_H), interpolation=cv2.INTER_AREA)
    caption = np.zeros((CAPTION_H, TOPDOWN_W, 3), dtype=np.uint8)
    _text(caption, header, (20, 32), 0.6)
    _text(caption, f"t = {time_ms:+.0f} ms from onset", (20, 66), 0.7)
    if in_bout:
        _text(caption, "SYLLABLE", (TOPDOWN_W - 190, 66), 1.0, BOUT_COLOR)
    frame = np.vstack([eyes, caption, td])

    color = BOUT_COLOR if in_bout else CONTEXT_COLOR
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, BORDER)
    return frame


def _title_card(shape: tuple[int, ...], lines: list[str]) -> NDArray:
    card = np.zeros(shape, dtype=np.uint8)
    y = shape[0] // 2 - 30 * (len(lines) - 1)
    for i, line in enumerate(lines):
        scale = 1.6 if i == 0 else 0.9
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        _text(card, line, ((shape[1] - w) // 2, y), scale)
        y += 70 if i == 0 else 45
    return card


class _Encoder:
    """Pipe BGR frames into ffmpeg/libx264 (plays in PowerPoint/Keynote)."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: NDArray):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def _eye_ok(eye_left: dict, eye_right: dict, key: str, start: int, end: int) -> bool:
    """Both eyes pass the quality threshold on every frame of the bout."""
    return all(
        np.isfinite(eye[key][start:end]).all() for eye in (eye_left, eye_right)
    )


def make_syllable_example_videos(
    project_dir: str | Path,
    model_name: str,
    syllables: list[int],
    n_examples: int = 8,
    pre: int = 30,
    post: int = 30,
    playback_fps: float = 30,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """
    Save `{output_dir}/syllable_XX_examples.mp4` for each syllable, with up to
    `n_examples` instances. `playback_fps` of 30 plays the 120 Hz recording at
    quarter speed.
    """
    project_dir = Path(project_dir)
    output_dir = Path(output_dir or project_dir / model_name / "syllable_example_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_folders = load_training_recording_folders(project_dir)
    folders_by_name = {rf.recording_name: rf for rf in recording_folders}
    coordinates, _, bodyparts = load_keypoints(
        KPMS_Loader(load_training_config(project_dir)["loader"]),
        recording_folders,
        **load_training_loader_options(project_dir),
    )
    eye_left, eye_right = load_eye_in_head_kpms(recording_folders, quality_level="medium")
    results = kpms.load_results(str(project_dir), model_name)

    head_coordinates = kpms.reindex_by_bodyparts(coordinates, bodyparts, HEAD_BODYPARTS)
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}
    instances = kpms.get_syllable_instances(
        {k: v["syllable"] for k, v in results.items()},
        pre=max(pre, SAMPLE_PRE), post=max(post, SAMPLE_POST),
        min_duration=3, min_frequency=0.005, min_instances=N_SAMPLED,
    )
    instances = {s: v for s, v in instances.items() if s in syllables}
    missing = set(syllables) - set(instances)
    if missing:
        print(f"[warn] not enough instances for syllables {sorted(missing)}; skipping them")
    sampled = kpms.sample_instances(
        instances, N_SAMPLED, mode="density", coordinates=head_coordinates,
        centroids=centroids, headings=headings, pre=SAMPLE_PRE, post=SAMPLE_POST, n_neighbors=N_SAMPLED,
    )

    frame_h, frame_w = EYE_SIZE + CAPTION_H + TOPDOWN_H, TOPDOWN_W
    paths = []
    for syllable in syllables:
        if syllable not in sampled:
            continue
        picks = [(k, s, e) for k, s, e in sampled[syllable] if _eye_ok(eye_left, eye_right, k, s, e)]
        if len(picks) < n_examples:
            print(f"[warn] syllable {syllable}: only {len(picks)} sampled instances have good eye data")
        picks = sorted(picks[:n_examples])  # group by session so reads stay local

        path = output_dir / f"syllable_{syllable:02d}_examples.mp4"
        encoder = _Encoder(path, frame_w, frame_h, playback_fps)
        for frame in [_title_card((frame_h, frame_w, 3), [
            f"Syllable {syllable}",
            f"{len(picks)} examples, {playback_fps / FPS:.2g}x speed",
            "red border = syllable active",
        ])] * int(TITLE_CARD_SECONDS * 2 * playback_fps):
            encoder.write(frame)

        for i, (key, start, end) in enumerate(picks, 1):
            rf = folders_by_name[key]
            readers = [
                _VideoReader(rf.left_eye_display_video),
                _VideoReader(rf.right_eye_display_video),
                _VideoReader(rf.topdown_mocap_display_video),
            ]
            n_results = len(results[key]["syllable"])
            if any(r.n_frames != n_results for r in readers):
                raise ValueError(
                    f"{key}: video frame counts {[r.n_frames for r in readers]} != {n_results} results frames"
                )
            duration_ms = (end - start) / FPS * 1000
            header = f"syl {syllable} | ex {i}/{len(picks)} | {key} | bout {duration_ms:.0f} ms"
            for t in range(max(0, start - pre), min(n_results, end + post)):
                left, right, topdown = (r.read(t) for r in readers)
                encoder.write(_composite(left, right, topdown, start <= t < end, header, (t - start) / FPS * 1000))
            for r in readers:
                r.close()
            print(f"  syllable {syllable} example {i}: {key} frames {start}-{end}")
        encoder.close()
        print(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("--syllables", type=int, nargs="+", required=True)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--pre", type=int, default=30, help="frames shown before onset")
    parser.add_argument("--post", type=int, default=30, help="frames shown after the bout ends")
    parser.add_argument("--fps", type=float, default=30, help="playback rate (120 = real time)")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    make_syllable_example_videos(
        args.project_dir, args.model_name, args.syllables,
        n_examples=args.n_examples, pre=args.pre, post=args.post,
        playback_fps=args.fps, output_dir=args.output_dir,
    )
