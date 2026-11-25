"""Compare filter variants: one self-contained Plotly HTML report.

Signals (all deg/s, on the common 120 Hz clock, positive = counter-clockwise yaw in world):
    head_yaw    skull angular_velocity_global yaw
    eye_h_L/R   eye-in-head horizontal velocity, in yaw sense:
                left = -d(adduction)/dt, right = +d(adduction)/dt
                (empirically: gaze_yaw ~= head_yaw + eye_h, R^2 0.97 / 0.997 on production data)
    gaze_L/R    gaze angular_velocity_global yaw

With VOR, eye_h ~= -head_yaw, so the head/eye cross-correlation peak is negative; its lag
tells whether filtering moved eye relative to head (positive lag = eye later than head).

Artifact mask: eye/gaze samples where the production eye_data_quality.csv (DLC-confidence
based, independent of filtering, on the same common clock) flags bad data, widened by
+/-50 ms, are set to NaN in EVERY variant, so all variants are scored on the same frames.
Without it, blink/occlusion artifacts (up to ~8500 deg/s) dominate every metric.
Head mask: skull-solver orientation flips (frame-to-frame rotation > HEAD_FLIP_DEG on the
common clock, up to ~180 deg) in ANY variant, widened by +/-50 ms, are set to NaN in head
yaw and gaze for every variant. The per-variant flip count is reported as a metric, since
less head filtering produces more flips.

    uv run python -m python_code.filter_comparison.compare_filter_variants <recording>/full_recording
"""

import html
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, welch

from python_code.filter_comparison.variants import EYE_FS_HZ, HEAD_FS_HZ, VARIANTS_BY_NAME, FilterVariant
from python_code.utilities.temporal_filters import frequency_response

logger = logging.getLogger(__name__)

# Reference categorical palette (dataviz skill), fixed order -> fixed colour per variant.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Colour follows the variant (never its position in a chart). The head cutoff sweep reuses the slots of
# no_eye_filter / unfiltered, which never share a report with it (and merge into other conditions in the
# head Rerun viewer), so no chart has more than 8 categorical hues.
VARIANT_COLORS: dict[str, str] = {
    "current": "#2a78d6", "no_head_filter": "#eb6834", "no_eye_filter": "#1baf7a", "unfiltered": "#eda100",
    "unfiltered_no_solver_smooth": "#e87ba4", "sg_55ms_p3": "#008300", "sg_100ms_p3": "#6250d6", "sg_100ms_p4": "#e34948",
    "head_bw10hz": "#1baf7a", "head_bw15hz": "#eda100",
}
SIGNALS: dict[str, str] = {
    "head_yaw": "Head yaw velocity (world)",
    "eye_h_L": "Left eye horizontal velocity (in head)",
    "eye_h_R": "Right eye horizontal velocity (in head)",
    "gaze_L": "Left gaze yaw velocity (world)",
    "gaze_R": "Right gaze yaw velocity (world)",
}
XCORR_MAX_LAG_S = 0.15
HF_BAND_HZ = 20.0
SACCADE_REFERENCE_VARIANT = "unfiltered"
SACCADE_WINDOW_S = 0.3  # +/- window for triggered averages
SACCADE_PEAK_SEARCH_S = 0.03
SACCADE_MIN_SEPARATION_S = 0.2
SACCADE_MIN_AMPLITUDE_DEG = 5.0  # ~3.5x the unfiltered eye position noise SD (~1.4 deg)
QUALITY_COMPONENT = "low_threshold"
QUALITY_DILATION_S = 0.05
HEAD_FLIP_DEG = 10.0
MIN_VALID_FRACTION = 0.2  # an eye with less usable (unmasked) data than this is left out of the metrics
EYE_SIGN = {"L": -1.0, "R": 1.0}  # adduction -> yaw-sense velocity
EYE_NAME = {"L": "left", "R": "right"}


@dataclass
class VariantSignals:
    variant: FilterVariant
    t: NDArray[np.float64]
    signals: dict[str, NDArray[np.float64]]
    adduction: dict[str, NDArray[np.float64]]  # deg, for saccade amplitude

    @property
    def fs(self) -> float:
        return float(1.0 / np.median(np.diff(self.t)))


def _series(csv_path: Path, trajectory: str, component: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    df = (
        pl.scan_csv(csv_path)
        .filter((pl.col("trajectory") == trajectory) & (pl.col("component") == component))
        .select("frame", "timestamp_s", "value")
        .sort("frame")
        .collect()
    )
    return df["timestamp_s"].to_numpy(), df["value"].to_numpy().astype(np.float64)


def load_bad_frame_mask(quality_csv: Path, n_frames: int) -> dict[str, NDArray[np.bool_]] | None:
    """Per-eye boolean mask of frames to exclude (bad DLC confidence, dilated)."""
    if not quality_csv.exists():
        logger.warning(f"No eye data quality CSV at {quality_csv}; metrics will include tracking artifacts")
        return None
    q = pl.read_csv(quality_csv).filter(pl.col("component") == QUALITY_COMPONENT)
    masks = {}
    for side, eye in (("L", "left"), ("R", "right")):
        good = q.filter(pl.col("trajectory") == f"{eye}_eye_data_quality").sort("frame")["value"].to_numpy().astype(bool)
        if len(good) != n_frames:
            logger.warning(f"Quality mask length {len(good)} != {n_frames} frames; not masking")
            return None
        k = int(round(QUALITY_DILATION_S * 120.0))
        masks[side] = np.convolve(~good, np.ones(2 * k + 1), mode="same") > 0
    return masks


def apply_mask(vs: VariantSignals, masks: dict[str, NDArray[np.bool_]] | None) -> VariantSignals:
    if masks is None:
        return vs
    for side in ("L", "R"):
        for key in (f"eye_h_{side}", f"gaze_{side}"):
            vs.signals[key] = np.where(masks[side], np.nan, vs.signals[key])
        vs.adduction[side] = np.where(masks[side], np.nan, vs.adduction[side])
    return vs


def head_flip_frames(variant_dir: Path) -> NDArray[np.bool_]:
    """Frames (common clock) where the solved skull orientation jumps by > HEAD_FLIP_DEG."""
    q = (
        pl.scan_csv(variant_dir / "analyzable_output" / "skull_kinematics" / "skull_kinematics.csv")
        .filter(pl.col("trajectory") == "orientation")
        .collect()
        .pivot(on="component", index="frame", values="value")
        .sort("frame")
        .select("w", "x", "y", "z")
        .to_numpy()
    )
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0.0, 1.0)
    return np.r_[False, np.rad2deg(2.0 * np.arccos(dots)) > HEAD_FLIP_DEG]


def apply_head_mask(vs: VariantSignals, mask: NDArray[np.bool_]) -> VariantSignals:
    for key in ("head_yaw", "gaze_L", "gaze_R"):
        vs.signals[key] = np.where(mask, np.nan, vs.signals[key])
    return vs


def load_variant_signals(variant_dir: Path) -> VariantSignals:
    out = variant_dir / "analyzable_output"
    t, head_yaw = _series(out / "skull_kinematics" / "skull_kinematics.csv", "angular_velocity_global", "yaw")
    signals = {"head_yaw": np.rad2deg(head_yaw)}
    adduction = {}
    for side, eye, sign in (("L", "left", -1.0), ("R", "right", 1.0)):
        t_eye, add = _series(out / f"{eye}_eye_kinematics" / f"{eye}_eye_kinematics.csv", "eye_in_head", "adduction")
        _, gaze = _series(out / "gaze_kinematics" / f"{eye}_gaze_kinematics.csv", "angular_velocity_global", "yaw")
        if len(t_eye) != len(t) or len(gaze) != len(t):
            raise ValueError(f"{variant_dir.name}: {eye} streams not on the common clock")
        adduction[side] = np.rad2deg(add)
        signals[f"eye_h_{side}"] = sign * np.gradient(np.rad2deg(add), t)
        signals[f"gaze_{side}"] = np.rad2deg(gaze)
    return VariantSignals(variant=VARIANTS_BY_NAME[variant_dir.name], t=t, signals=signals, adduction=adduction)


# ----------------------------------------------------------------------------- metrics
def _clean(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean-removed, with NaN gaps linearly interpolated (zero-filling would add broadband power)."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        raise ValueError("signal has no valid samples (fully masked)")
    if not ok.all():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[ok], x[ok])
    return x - x.mean()


def head_eye_xcorr(head: NDArray, eye: NDArray, fs: float) -> tuple[NDArray, NDArray, float, float]:
    """Normalised cross-correlation r(lag) = corr(head(t), eye(t + lag)).

    Returns lags_s, r, lag of the most negative peak (sub-sample, parabolic) and its value.
    """
    h, e = _clean(head), _clean(eye)
    n = len(h)
    max_lag = int(round(XCORR_MAX_LAG_S * fs))
    lags = np.arange(-max_lag, max_lag + 1)
    denom = np.sqrt(np.sum(h * h) * np.sum(e * e))
    r = np.array([np.sum(h[max(0, -k): n - max(0, k)] * e[max(0, k): n - max(0, -k)]) for k in lags]) / denom
    i = int(np.argmin(r))
    offset = 0.0
    if 0 < i < len(r) - 1:
        a, b, c = r[i - 1], r[i], r[i + 1]
        if (a - 2 * b + c) != 0:
            offset = 0.5 * (a - c) / (a - 2 * b + c)
    return lags / fs, r, float((lags[i] + offset) / fs), float(r[i])


def psd(x: NDArray, fs: float) -> tuple[NDArray, NDArray]:
    return welch(_clean(x), fs=fs, nperseg=1024)


def hf_rms(x: NDArray, fs: float) -> float:
    f, p = psd(x, fs)
    band = f >= HF_BAND_HZ
    return float(np.sqrt(np.trapezoid(p[band], f[band])))


def detect_saccades(adduction: NDArray, eye_sign: float, fs: float) -> tuple[NDArray[np.int64], NDArray]:
    """Saccade peak indices + direction (+1/-1), detected on the reference variant.

    Detection uses a Savitzky-Golay-smoothed copy of the reference eye position (for
    detection only), so tracking noise in an unfiltered reference can't create events:
    an event needs a speed above a robust threshold AND a position change of at least
    SACCADE_MIN_AMPLITUDE_DEG. Events whose window touches masked (bad) frames are dropped.
    """
    bad = ~np.isfinite(adduction)
    pos = savgol_filter(_clean(adduction), window_length=13, polyorder=3)
    vel = eye_sign * np.gradient(pos) * fs
    speed = np.abs(vel)
    mad = np.median(np.abs(speed - np.median(speed)))
    threshold = max(100.0, float(np.median(speed) + 6.0 * 1.4826 * mad))
    half = int(round(SACCADE_WINDOW_S * fs))
    near, far = int(round(0.05 * fs)), int(round(0.10 * fs))
    min_sep = int(SACCADE_MIN_SEPARATION_S * fs)
    idx = np.flatnonzero(speed > threshold)
    peaks: list[int] = []
    if idx.size:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            p = int(run[np.argmax(speed[run])])
            if not (half <= p < len(pos) - half) or bad[p - half: p + half + 1].any():
                continue
            amplitude = abs(pos[p + near: p + far].mean() - pos[p - far: p - near].mean())
            if amplitude < SACCADE_MIN_AMPLITUDE_DEG:
                continue
            if peaks and p - peaks[-1] < min_sep:
                continue
            peaks.append(p)
    peaks_arr = np.array(peaks, dtype=np.int64)
    logger.info(f"Saccade threshold {threshold:.0f} deg/s, amplitude >= {SACCADE_MIN_AMPLITUDE_DEG} deg -> {len(peaks)} events")
    return peaks_arr, np.sign(vel[peaks_arr]) if peaks else np.array([])


def saccade_metrics(vs: VariantSignals, side: str, peaks: NDArray) -> tuple[NDArray, NDArray]:
    """Peak |velocity| (deg/s) near each reference peak and amplitude (deg) of the movement."""
    v = vs.signals[f"eye_h_{side}"]
    add = vs.adduction[side]
    search = int(round(SACCADE_PEAK_SEARCH_S * vs.fs))
    half = int(round(0.06 * vs.fs))
    peak_v, amp = [], []
    for p in peaks:
        seg = v[p - search: p + search + 1]
        peak_v.append(np.nanmax(np.abs(seg)))
        amp.append(abs(add[p + half] - add[p - half]))
    return np.array(peak_v), np.array(amp)


def triggered_average(x: NDArray, peaks: NDArray, sign: NDArray, fs: float) -> tuple[NDArray, NDArray]:
    half = int(round(SACCADE_WINDOW_S * fs))
    segs = np.stack([x[p - half: p + half + 1] * s for p, s in zip(peaks, sign)])
    return np.arange(-half, half + 1) / fs * 1000.0, np.nanmean(segs, axis=0)


# ----------------------------------------------------------------------------- report
def _color(name: str) -> str:
    return VARIANT_COLORS[name]


def _dash(name: str) -> str:
    return "solid"


def _layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    # title on the first line, legend on its own line below it, then the plots
    fig.update_layout(
        title=dict(text=title, x=0, xref="container", y=1 - 12 / height, yref="container", yanchor="top", font=dict(size=16)),
        template="plotly_white",
        height=height,
        margin=dict(l=70, r=20, t=110, b=50),
        legend=dict(orientation="h", x=0, xref="container", y=1 - 44 / height, yref="container", yanchor="top", font=dict(size=11)),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color="#0b0b0b"),
    )
    fig.update_xaxes(gridcolor="#eceae5", zeroline=False)
    fig.update_yaxes(gridcolor="#eceae5", zerolinecolor="#c9c7c0")
    return fig


def _default_window_start(vs: VariantSignals, duration_s: float) -> float:
    """Window with the most head-yaw movement (rolling mean |head yaw|)."""
    x = np.abs(np.nan_to_num(vs.signals["head_yaw"]))
    n = int(duration_s * vs.fs)
    if n >= len(x):
        return float(vs.t[0])
    rolling = np.convolve(x, np.ones(n) / n, mode="valid")
    return float(vs.t[int(np.argmax(rolling))])


def fig_frequency_response(variants: list[FilterVariant]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Head 3D markers @ {HEAD_FS_HZ:g} Hz", f"Eye 2D pixels @ {EYE_FS_HZ:g} Hz"))
    for i, v in enumerate(variants):
        for col, (spec, fs) in enumerate(((v.head_filter, HEAD_FS_HZ), (v.eye_filter, EYE_FS_HZ)), start=1):
            f, g = frequency_response(spec=spec, fs=fs)
            fig.add_trace(
                go.Scatter(x=f, y=20 * np.log10(np.maximum(g, 1e-4)), name=v.name, legendgroup=v.name,
                           showlegend=col == 1, line=dict(color=_color(v.name), width=2, dash=_dash(v.name)),
                           hovertemplate=f"{v.name}<br>{spec.label(fs)}<br>%{{x:.1f}} Hz: %{{y:.1f}} dB<extra></extra>"),
                row=1, col=col,
            )
    for col in (1, 2):
        for db, dash in ((-3, "dot"), (-6, "dash")):
            fig.add_hline(y=db, line=dict(color="#8a8880", width=1, dash=dash), row=1, col=col)
    fig.update_yaxes(title_text="gain (dB)", range=[-30, 3])
    fig.update_xaxes(title_text="frequency (Hz)")
    return _layout(fig, "1. Filter magnitude response (as applied; all zero-phase)", 420)


def fig_timeseries(data: list[VariantSignals], start_s: float, duration_s: float) -> go.Figure:
    keys = list(SIGNALS)
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=[SIGNALS[k] for k in keys])
    for i, vs in enumerate(data):
        m = (vs.t >= start_s) & (vs.t <= start_s + duration_s)
        for row, k in enumerate(keys, start=1):
            fig.add_trace(
                go.Scattergl(x=vs.t[m], y=vs.signals[k][m], name=vs.variant.name, legendgroup=vs.variant.name,
                             showlegend=row == 1, mode="lines", line=dict(color=_color(vs.variant.name), width=1.5, dash=_dash(vs.variant.name)),
                             hovertemplate=f"{vs.variant.name}: %{{y:.0f}} deg/s<extra></extra>"),
                row=row, col=1,
            )
            fig.update_yaxes(title_text="deg/s", row=row, col=1)
    fig.update_xaxes(title_text="time (s)", row=len(keys), col=1)
    fig.update_layout(hovermode="x unified")
    return _layout(fig, f"2. Time series, {start_s:.1f}-{start_s + duration_s:.1f} s (drag to zoom; click legend to hide variants)", 1300)


def fig_psd(data: list[VariantSignals], eye: str) -> go.Figure:
    keys = ["head_yaw", f"eye_h_{eye}", f"gaze_{eye}"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[SIGNALS[k] for k in keys])
    for i, vs in enumerate(data):
        for col, k in enumerate(keys, start=1):
            f, p = psd(vs.signals[k], vs.fs)
            fig.add_trace(
                go.Scatter(x=f[1:], y=p[1:], name=vs.variant.name, legendgroup=vs.variant.name, showlegend=col == 1,
                           line=dict(color=_color(vs.variant.name), width=1.5, dash=_dash(vs.variant.name)),
                           hovertemplate=f"{vs.variant.name}<br>%{{x:.1f}} Hz: %{{y:.3g}}<extra></extra>"),
                row=1, col=col,
            )
    fig.update_xaxes(type="log", title_text="frequency (Hz)")
    fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="PSD ((deg/s)²/Hz)", row=1, col=1)
    return _layout(fig, "3. Power spectral density (Welch, full recording)", 450)


def fig_xcorr(data: list[VariantSignals], results: dict[str, dict], eyes: list[str]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(eyes), subplot_titles=[f"Head yaw vs {EYE_NAME[s]} eye (in head)" for s in eyes])
    for i, vs in enumerate(data):
        for col, side in enumerate(eyes, start=1):
            lags, r, lag, peak = head_eye_xcorr(vs.signals["head_yaw"], vs.signals[f"eye_h_{side}"], vs.fs)
            results[vs.variant.name][f"xcorr_lag_ms_{side}"] = lag * 1000
            results[vs.variant.name][f"xcorr_peak_{side}"] = peak
            fig.add_trace(
                go.Scatter(x=lags * 1000, y=r, name=vs.variant.name, legendgroup=vs.variant.name, showlegend=col == 1,
                           line=dict(color=_color(vs.variant.name), width=2, dash=_dash(vs.variant.name)),
                           hovertemplate=f"{vs.variant.name}<br>lag %{{x:.0f}} ms: r=%{{y:.3f}}<extra></extra>"),
                row=1, col=col,
            )
    fig.add_vline(x=0, line=dict(color="#8a8880", width=1, dash="dot"))
    fig.update_xaxes(title_text="lag (ms, + = eye later than head)")
    fig.update_yaxes(title_text="r", row=1, col=1)
    return _layout(fig, "4. Head-eye cross-correlation (VOR: negative peak; peak lag should not move between variants)", 420)


def _profile_peak_and_fwhm(lag_ms: NDArray, mean: NDArray) -> tuple[float, float]:
    centre = np.abs(lag_ms) <= SACCADE_PEAK_SEARCH_S * 1000
    i = int(np.flatnonzero(centre)[np.argmax(mean[centre])])
    peak = float(mean[i])
    above = mean >= peak / 2
    lo = i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i
    while hi < len(mean) - 1 and above[hi + 1]:
        hi += 1
    return peak, float(lag_ms[hi] - lag_ms[lo])


def fig_saccades(data: list[VariantSignals], results: dict[str, dict], eye: str) -> tuple[go.Figure, go.Figure, int]:
    """Saccade attenuation measured on saccade-triggered averages.

    Tracking noise is zero-mean, so it averages out across events: the unfiltered
    variant's triggered average is an unbiased estimate of the true mean saccade profile,
    and each variant's average peak relative to it is the filter's attenuation.
    Per-event peaks (main sequence) are shown too, but in noisy variants they are
    inflated by noise.
    """
    names = [vs.variant.name for vs in data]
    ref = data[names.index(SACCADE_REFERENCE_VARIANT)] if SACCADE_REFERENCE_VARIANT in names else data[0]
    peaks, direction = detect_saccades(ref.adduction[eye], EYE_SIGN[eye], ref.fs)
    if peaks.size == 0:
        return go.Figure(), go.Figure(), 0

    profiles = {vs.variant.name: triggered_average(vs.signals[f"eye_h_{eye}"], peaks, direction, vs.fs) for vs in data}
    ref_peak, _ = _profile_peak_and_fwhm(*profiles[ref.variant.name])

    main_seq = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], subplot_titles=(
        f"Per-event main sequence, {EYE_NAME[eye]} eye (noise inflates peaks in unfiltered eye variants)",
        f"Mean saccade peak velocity relative to '{ref.variant.name}'"))
    trig = make_subplots(rows=1, cols=3, shared_xaxes=True, subplot_titles=(
        f"{EYE_NAME[eye].capitalize()} eye horizontal (in head)", "Head yaw", f"{EYE_NAME[eye].capitalize()} gaze yaw"))
    for i, vs in enumerate(data):
        name = vs.variant.name
        peak_v, amp = saccade_metrics(vs, eye, peaks)
        prof_peak, fwhm = _profile_peak_and_fwhm(*profiles[name])
        results[name]["saccade_mean_peak_ratio"] = prof_peak / ref_peak
        results[name]["saccade_mean_peak_deg_s"] = prof_peak
        results[name]["saccade_fwhm_ms"] = fwhm
        results[name]["saccade_event_peak_median_deg_s"] = float(np.nanmedian(peak_v))
        main_seq.add_trace(go.Scattergl(x=amp, y=peak_v, mode="markers", name=name, legendgroup=name,
                                        marker=dict(color=_color(vs.variant.name), size=8, opacity=0.6, line=dict(color="#fcfcfb", width=1)),
                                        hovertemplate=f"{name}<br>%{{x:.1f}} deg, %{{y:.0f}} deg/s<extra></extra>"), row=1, col=1)
        main_seq.add_trace(go.Bar(x=[name], y=[prof_peak / ref_peak], name=name, legendgroup=name, showlegend=False,
                                  marker=dict(color=_color(vs.variant.name), line=dict(color="#fcfcfb", width=2)),
                                  hovertemplate=f"{name}: %{{y:.2f}} (FWHM {fwhm:.0f} ms)<extra></extra>"), row=1, col=2)
        for col, k in enumerate((f"eye_h_{eye}", "head_yaw", f"gaze_{eye}"), start=1):
            lag_ms, mean = profiles[name] if k.startswith("eye_h_") else triggered_average(vs.signals[k], peaks, direction, vs.fs)
            trig.add_trace(go.Scatter(x=lag_ms, y=mean, name=name, legendgroup=name, showlegend=col == 1,
                                      line=dict(color=_color(vs.variant.name), width=2, dash=_dash(vs.variant.name)),
                                      hovertemplate=f"{name}<br>%{{x:.0f}} ms: %{{y:.0f}} deg/s<extra></extra>"), row=1, col=col)
    main_seq.add_hline(y=1.0, line=dict(color="#8a8880", width=1, dash="dot"), row=1, col=2)
    main_seq.update_xaxes(title_text="amplitude (deg)", row=1, col=1)
    main_seq.update_yaxes(title_text="peak velocity (deg/s)", row=1, col=1)
    main_seq.update_yaxes(title_text="ratio", row=1, col=2)
    trig.add_vline(x=0, line=dict(color="#8a8880", width=1, dash="dot"))
    trig.update_xaxes(title_text="time from saccade peak (ms)")
    trig.update_yaxes(title_text="deg/s (sign-aligned to saccade)", row=1, col=1)
    n = int(peaks.size)
    _layout(main_seq, f"5a. Saccade attenuation: {n} {EYE_NAME[eye]}-eye saccades (≥{SACCADE_MIN_AMPLITUDE_DEG:g}°) detected once, measured in every variant", 460)
    _layout(trig, "5b. Saccade-triggered average (same events in every variant; noise averages out)", 420)
    return main_seq, trig, n


def summary_table(data: list[VariantSignals], results: dict[str, dict], eye: str) -> str:
    cols = [
        ("variant", "Variant"), ("head", "Head filter"), ("eye", "Eye filter"), ("solver", "Solver smoothing"),
        ("saccade_mean_peak_ratio", "Saccade peak ratio (mean profile)"), ("saccade_fwhm_ms", "Saccade FWHM (ms)"),
        ("xcorr_lag_ms_L", "Head-eye lag L (ms)"), ("xcorr_lag_ms_R", "Head-eye lag R (ms)"),
        ("xcorr_peak_L", "Head-eye r L"), ("xcorr_peak_R", "Head-eye r R"),
        ("hf_head_yaw", f"Head yaw RMS >{HF_BAND_HZ:g} Hz"), (f"hf_eye_h_{eye}", f"Eye {eye} RMS >{HF_BAND_HZ:g} Hz"),
        (f"hf_gaze_{eye}", f"Gaze {eye} RMS >{HF_BAND_HZ:g} Hz"), ("solver_flip_frames", "Solver flip frames"),
    ]
    rows = []
    for i, vs in enumerate(data):
        r = results[vs.variant.name]
        cells = {
            "variant": f'<span class="swatch" style="background:{_color(vs.variant.name)}"></span>{html.escape(vs.variant.name)}',
            "head": vs.variant.head_filter.label(HEAD_FS_HZ), "eye": vs.variant.eye_filter.label(EYE_FS_HZ),
            "solver": "on" if vs.variant.solver_smoothing else "off",
        }
        for key, _ in cols[4:]:
            val = r.get(key)
            cells[key] = "–" if val is None else (
                str(val) if isinstance(val, int) else f"{val:.2f}" if "ratio" in key or "xcorr_peak" in key else f"{val:.1f}"
            )
        rows.append("<tr>" + "".join(f"<td>{cells[k]}</td>" for k, _ in cols) + "</tr>")
    head = "<tr>" + "".join(f"<th>{html.escape(label)}</th>" for _, label in cols) + "</tr>"
    return f'<table class="summary"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def write_comparison_report(
    *,
    comparison_dir: Path,
    variant_names: list[str] | None = None,
    window_start_s: float | None = None,
    window_duration_s: float = 20.0,
    report_name: str = "filter_comparison",
    title: str = "Head / eye filtering comparison",
) -> Path:
    def _complete(n: str) -> bool:
        gaze_dir = comparison_dir / n / "analyzable_output" / "gaze_kinematics"
        return all((gaze_dir / f"{eye}_gaze_kinematics.csv").exists() for eye in ("left", "right"))

    requested = variant_names or list(VARIANTS_BY_NAME)
    names = [n for n in requested if _complete(n)]
    if missing := [n for n in requested if n not in names]:
        logger.warning(f"Skipping variants that are not fully built: {missing}")
    if not names:
        raise FileNotFoundError(f"No built variants in {comparison_dir}")
    order = list(VARIANTS_BY_NAME)
    names.sort(key=order.index)  # fixed order -> fixed colour per variant
    data = [load_variant_signals(comparison_dir / n) for n in names]
    results: dict[str, dict] = {n: {} for n in names}
    masks = load_bad_frame_mask(comparison_dir.parent / "analyzable_output" / "eye_data_quality.csv", len(data[0].t))
    data = [apply_mask(vs, masks) for vs in data]
    k = int(round(QUALITY_DILATION_S * data[0].fs))
    head_mask = np.zeros(len(data[0].t), dtype=bool)
    for vs in data:
        flips = head_flip_frames(comparison_dir / vs.variant.name)
        results[vs.variant.name]["solver_flip_frames"] = int(flips.sum())
        head_mask |= np.convolve(flips, np.ones(2 * k + 1), mode="same") > 0
    data = [apply_head_mask(vs, head_mask) for vs in data]
    mask_note = (
        f"Eye/gaze frames flagged by eye_data_quality ({QUALITY_COMPONENT}, ±{QUALITY_DILATION_S * 1000:.0f} ms) are excluded "
        f"in every variant: left {masks['L'].mean():.1%}, right {masks['R'].mean():.1%} of frames."
        if masks else "No eye quality mask applied: metrics include blink/occlusion artifacts."
    ) + (
        f" Head yaw and gaze frames within ±{QUALITY_DILATION_S * 1000:.0f} ms of a skull-solver orientation flip "
        f"(>{HEAD_FLIP_DEG:g}° per frame) in any variant are excluded: {head_mask.mean():.1%} of frames."
    )
    valid = {s: float(np.isfinite(data[0].signals[f"eye_h_{s}"]).mean()) for s in ("L", "R")}
    eyes = [s for s in ("L", "R") if valid[s] >= MIN_VALID_FRACTION]
    if not eyes:
        raise ValueError(f"No eye has >= {MIN_VALID_FRACTION:.0%} usable frames after masking ({valid}); nothing to compare")
    if dropped := [s for s in ("L", "R") if s not in eyes]:
        logger.warning(f"Leaving out {[EYE_NAME[s] for s in dropped]} eye: usable fraction {valid} < {MIN_VALID_FRACTION:.0%}")
        mask_note += " " + " ".join(
            f"The {EYE_NAME[s]} eye is left out of the metrics: only {valid[s]:.1%} of its frames are usable." for s in dropped)
    eye = "L" if "L" in eyes else eyes[0]  # eye used for spectra, HF RMS and saccades
    for vs in data:
        for k in ("head_yaw", f"eye_h_{eye}", f"gaze_{eye}"):
            results[vs.variant.name][f"hf_{k}"] = hf_rms(vs.signals[k], vs.fs)

    start = window_start_s if window_start_s is not None else _default_window_start(data[0], window_duration_s)
    figs = [fig_frequency_response([vs.variant for vs in data]), fig_timeseries(data, start, window_duration_s), fig_psd(data, eye)]
    figs.append(fig_xcorr(data, results, eyes))
    main_seq, trig, n_saccades = fig_saccades(data, results, eye)
    if n_saccades:
        figs += [main_seq, trig]

    parts = [f.to_html(full_html=False, include_plotlyjs=(i == 0), config={"displaylogo": False}) for i, f in enumerate(figs)]
    (comparison_dir / f"{report_name}_metrics.json").write_text(json.dumps(results, indent=2))
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Filter comparison – {html.escape(comparison_dir.parent.parent.name)}</title>
<style>
 body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; color:#0b0b0b; background:#fcfcfb; margin:24px auto; max-width:1500px; padding:0 16px; }}
 h1 {{ font-size:22px; margin-bottom:4px; }} p.note {{ color:#52514e; font-size:13px; max-width:1000px; line-height:1.5; }}
 table.summary {{ border-collapse:collapse; font-size:13px; margin:12px 0 28px; }}
 table.summary th, table.summary td {{ border-bottom:1px solid #e4e2dc; padding:6px 10px; text-align:right; white-space:nowrap; }}
 table.summary th {{ color:#52514e; font-weight:600; }} table.summary td:first-child, table.summary th:first-child {{ text-align:left; }}
 .swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:8px; vertical-align:middle; }}
 .fig {{ margin-bottom:28px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="note">{html.escape(str(comparison_dir.parent))}. All variants start from the same unfiltered head 3D triangulation and eye 2D DLC output;
only the temporal filter differs. Velocities are deg/s on the common clock, positive = counter-clockwise yaw in world.
Eye horizontal velocity is in yaw sense (VOR makes it the mirror of head yaw). Saccade peak ratio is the peak of the saccade-triggered average relative to the unfiltered one (noise averages out): &lt; 1 means the filter attenuates saccades, and a larger FWHM means it spreads them in time;
RMS above {HF_BAND_HZ:g} Hz is the high-frequency content left in (mostly tracking noise). {html.escape(mask_note)}</p>
{summary_table(data, results, eye)}
{''.join(f'<div class="fig">{p}</div>' for p in parts)}
</body></html>"""
    report = comparison_dir / f"{report_name}_report.html"
    report.write_text(page)
    return report


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", type=Path)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--duration", type=float, default=20.0)
    a = parser.parse_args()
    print(write_comparison_report(comparison_dir=a.recording_path / "filter_comparison", window_start_s=a.start, window_duration_s=a.duration))
