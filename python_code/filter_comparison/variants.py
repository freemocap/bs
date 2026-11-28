"""Filter variants compared by the filter comparison pipeline.

Every variant starts from the same unfiltered head 3D markers and eye 2D pixel
positions (gaps linearly interpolated); only the temporal filter differs.

Note: Savitzky-Golay smoothing with polyorder 2 and 3 (or 4 and 5) produces
identical output for a centred window, so polyorder 2 is not a separate variant.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from python_code.utilities.temporal_filters import FilterSpec

HEAD_FS_HZ: float = 90.0
EYE_FS_HZ: float = 120.0


class FilterVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    head_filter: FilterSpec
    eye_filter: FilterSpec
    solver_smoothing: bool = True  # False -> skull solver lambda_*_smooth = 0
    # "dlc_median": start from DLC's `_filtered.csv` (5-frame median) instead of the raw DLC csv
    eye_dlc_input: Literal["raw", "dlc_median"] = "raw"
    # Reuse the head (skull solver) output of this variant instead of solving again;
    # its head filter and solver settings must match this variant's.
    head_from: str | None = None

    def describe(self) -> str:
        smoothing = "solver smoothing on" if self.solver_smoothing else "solver smoothing OFF"
        eye_input = "DLC median5 -> " if self.eye_dlc_input == "dlc_median" else ""
        return (
            f"head: {self.head_filter.label(HEAD_FS_HZ)} | "
            f"eye: {eye_input}{self.eye_filter.label(EYE_FS_HZ)} | {smoothing}"
        )


_BW_HEAD = FilterSpec.butterworth(cutoff_hz=7.0, order=4)  # dlc_to_3d / skellyforge before Nov 2026 (now 15 Hz)
_BW_EYE = FilterSpec.butterworth(cutoff_hz=6.0, order=4)  # production eye pipeline (pre-savgol)
_NONE = FilterSpec.none()
_SG55 = FilterSpec.savgol(window_s=0.055, polyorder=3)

FILTER_VARIANTS: list[FilterVariant] = [
    FilterVariant(name="current", head_filter=_BW_HEAD, eye_filter=_BW_EYE),
    FilterVariant(name="no_head_filter", head_filter=_NONE, eye_filter=_BW_EYE),
    FilterVariant(name="no_eye_filter", head_filter=_BW_HEAD, eye_filter=_NONE),
    FilterVariant(name="unfiltered", head_filter=_NONE, eye_filter=_NONE),
    FilterVariant(
        name="unfiltered_no_solver_smooth", head_filter=_NONE, eye_filter=_NONE, solver_smoothing=False
    ),
    *[
        FilterVariant(name=f"sg_{int(w * 1000)}ms_p{p}", head_filter=sg, eye_filter=sg)
        for w, p in [(0.055, 3), (0.100, 3), (0.100, 4)]
        for sg in [FilterSpec.savgol(window_s=w, polyorder=p)]
    ],
    # Head-only Butterworth cutoff sweep: production eye filter and solver, only the head cutoff changes
    *[
        FilterVariant(name=f"head_bw{c}hz", head_filter=FilterSpec.butterworth(cutoff_hz=float(c), order=4), eye_filter=_BW_EYE)
        for c in (10, 15)
    ],
    # Eye input: DLC's median-filtered csv instead of the raw one (head reused from sg_55ms_p3)
    FilterVariant(name="dlc_median", head_filter=_SG55, eye_filter=_NONE, eye_dlc_input="dlc_median", head_from="sg_55ms_p3"),
    FilterVariant(
        name="dlc_median_sg_55ms_p3", head_filter=_SG55, eye_filter=_SG55, eye_dlc_input="dlc_median", head_from="sg_55ms_p3"
    ),
]

VARIANTS_BY_NAME: dict[str, FilterVariant] = {v.name: v for v in FILTER_VARIANTS}

# Reports: the main comparison, the head Butterworth cutoff sweep against its endpoints, and the
# eye input (raw vs DLC median-filtered csv).
# Kept separate so neither chart needs more than 8 categorical colours.
EYE_INPUT_REPORT_VARIANTS: list[str] = ["unfiltered", "sg_55ms_p3", "dlc_median", "dlc_median_sg_55ms_p3"]
HEAD_CUTOFF_REPORT_VARIANTS: list[str] = ["current", "no_head_filter", "head_bw10hz", "head_bw15hz"]
MAIN_REPORT_VARIANTS: list[str] = [
    v.name for v in FILTER_VARIANTS
    if v.name not in HEAD_CUTOFF_REPORT_VARIANTS[2:] and v.name not in EYE_INPUT_REPORT_VARIANTS[2:]
]
