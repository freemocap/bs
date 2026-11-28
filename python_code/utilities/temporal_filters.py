"""Zero-phase temporal filters shared by the head (3D marker) and eye (2D pixel) pipelines.

Filters are described by a `FilterSpec` so the same filter can be applied to
streams recorded at different sampling rates. Savitzky-Golay windows are given
in seconds and converted to an odd number of samples per stream, so a 90 Hz head
stream and a 120 Hz eye stream get (nearly) the same response in time.

All filters here are zero-phase (filtfilt / centred polynomial fit): they never
introduce a time shift, only amplitude attenuation and temporal spreading.

A spec can also run a median filter first (`median_frames`), which reproduces
DeepLabCut's `filterpredictions` (scipy.signal.medfilt, zero-padded edges). It is
given in frames, like DLC's, and as a nonlinear filter it has no frequency response.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.signal import butter, filtfilt, freqz, medfilt, savgol_coeffs, savgol_filter

FilterKind = Literal["none", "butterworth", "savgol"]


class FilterSpec(BaseModel):
    """Description of a zero-phase temporal filter."""

    model_config = ConfigDict(frozen=True)

    kind: FilterKind
    # Butterworth (applied with filtfilt, so effective order is 2 * order)
    cutoff_hz: float | None = None
    order: int | None = None
    # Savitzky-Golay
    window_s: float | None = None
    polyorder: int | None = None
    # Median filter applied before the filter above (odd number of frames), as in DLC
    median_frames: int | None = None

    @model_validator(mode="after")
    def _check_params(self) -> "FilterSpec":
        if self.kind == "butterworth" and (self.cutoff_hz is None or self.order is None):
            raise ValueError("butterworth filter needs cutoff_hz and order")
        if self.kind == "savgol" and (self.window_s is None or self.polyorder is None):
            raise ValueError("savgol filter needs window_s and polyorder")
        if self.median_frames is not None and (self.median_frames < 3 or self.median_frames % 2 == 0):
            raise ValueError(f"median_frames must be odd and >= 3, got {self.median_frames}")
        return self

    @classmethod
    def none(cls, *, median_frames: int | None = None) -> "FilterSpec":
        return cls(kind="none", median_frames=median_frames)

    @classmethod
    def butterworth(cls, *, cutoff_hz: float, order: int = 4, median_frames: int | None = None) -> "FilterSpec":
        return cls(kind="butterworth", cutoff_hz=cutoff_hz, order=order, median_frames=median_frames)

    @classmethod
    def savgol(cls, *, window_s: float, polyorder: int, median_frames: int | None = None) -> "FilterSpec":
        return cls(kind="savgol", window_s=window_s, polyorder=polyorder, median_frames=median_frames)

    def label(self, fs: float | None = None) -> str:
        main = self._main_label(fs)
        if self.median_frames is None:
            return main
        median = f"median {self.median_frames} frames"
        return median if self.kind == "none" else f"{median} -> {main}"

    def _main_label(self, fs: float | None) -> str:
        if self.kind == "none":
            return "none"
        if self.kind == "butterworth":
            return f"BW{self.order} {self.cutoff_hz:g} Hz"
        window_text = f"{self.window_s * 1000:.0f} ms"
        if fs is not None:
            window_text += f" ({savgol_window_samples(fs=fs, spec=self)} samp)"
        return f"SG {window_text} p{self.polyorder}"


def savgol_window_samples(*, fs: float, spec: FilterSpec) -> int:
    """Nearest odd window length (in samples) to spec.window_s at rate fs."""
    if spec.kind != "savgol":
        raise ValueError("savgol_window_samples needs a savgol spec")
    n = int(round(spec.window_s * fs))
    if n % 2 == 0:
        # pick whichever odd neighbour is closer to the requested duration
        n = n + 1 if (n + 1) / fs - spec.window_s <= spec.window_s - (n - 1) / fs else n - 1
    return max(n, spec.polyorder + 2 + (spec.polyorder % 2 == 0))


def apply_temporal_filter(data: NDArray[np.float64], *, fs: float, spec: FilterSpec) -> NDArray[np.float64]:
    """Filter `data` along axis 0. Data must not contain NaNs (gap-fill first)."""
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any() and (spec.kind != "none" or spec.median_frames is not None):
        raise ValueError("apply_temporal_filter requires NaN-free data; interpolate gaps first")
    if spec.median_frames is not None:
        data = medfilt(data, kernel_size=[spec.median_frames] + [1] * (data.ndim - 1))
    if spec.kind == "none":
        return data.copy()
    if spec.kind == "butterworth":
        b, a = butter(N=spec.order, Wn=spec.cutoff_hz / (0.5 * fs), btype="low")
        return filtfilt(b, a, data, axis=0)
    window = savgol_window_samples(fs=fs, spec=spec)
    return savgol_filter(data, window_length=window, polyorder=spec.polyorder, axis=0, mode="interp")


def frequency_response(
    *, spec: FilterSpec, fs: float, n_points: int = 512
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Magnitude response (linear gain) of the filter as applied, 0..Nyquist.

    Ignores `median_frames`: a median filter is nonlinear and has no frequency response.
    """
    freqs = np.linspace(0.0, fs / 2.0, n_points)
    if spec.kind == "none":
        return freqs, np.ones_like(freqs)
    if spec.kind == "butterworth":
        b, a = butter(N=spec.order, Wn=spec.cutoff_hz / (0.5 * fs), btype="low")
        _, h = freqz(b, a, worN=freqs, fs=fs)
        return freqs, np.abs(h) ** 2  # filtfilt squares the magnitude
    window = savgol_window_samples(fs=fs, spec=spec)
    coeffs = savgol_coeffs(window_length=window, polyorder=spec.polyorder)
    _, h = freqz(coeffs, 1, worN=freqs, fs=fs)
    return freqs, np.abs(h)  # centred symmetric FIR: zero phase


if __name__ == "__main__":
    for s in [FilterSpec.butterworth(cutoff_hz=7.0), FilterSpec.savgol(window_s=0.055, polyorder=3),
              FilterSpec.savgol(window_s=0.1, polyorder=3), FilterSpec.savgol(window_s=0.1, polyorder=2)]:
        for rate in (90.0, 120.0):
            f, g = frequency_response(spec=s, fs=rate)
            f3 = f[np.argmax(g < 10 ** (-3 / 20))]
            print(f"{s.label(rate):28s} @ {rate:5.0f} Hz: -3 dB at {f3:5.1f} Hz")
