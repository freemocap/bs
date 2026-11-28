"""Run the head/eye filter comparison on one recording.

    uv run python -m python_code.filter_comparison.run_filter_comparison <recording>/full_recording \
        [--variants current unfiltered ...] [--jobs 4] [--overwrite] [--report-only]

Each variant's skull solve is single-threaded (~35 min on 32k frames), so --jobs N builds
N variants in parallel, each in its own subprocess: a crash (e.g. a SIGBUS from Dropbox
syncing a file while it is memory-mapped) only affects that variant, which is retried once.
Finished variants are skipped on re-run, so an interrupted run can simply be restarted.
Per-variant logs go to filter_comparison/_logs/<variant>.log.

Writes everything to <recording>/filter_comparison/ (production outputs are only read):
    _triangulation/   unfiltered head 3D (shared by all variants)
    <variant>/        clip-shaped folder per variant (solver output, eye data, analyzable_output)
    filter_comparison_report.html
"""

import argparse
import logging
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from python_code.filter_comparison.build_variant import build_variant, variant_is_complete
from python_code.filter_comparison.compare_filter_variants import write_comparison_report
from python_code.filter_comparison.triangulate_unfiltered import (
    DEFAULT_DLC_TO_3D_REPO,
    DEFAULT_FREEMOCAP_REPO,
    triangulate_unfiltered_head,
)
from python_code.filter_comparison.variants import (
    EYE_INPUT_REPORT_VARIANTS,
    FILTER_VARIANTS,
    HEAD_CUTOFF_REPORT_VARIANTS,
    MAIN_REPORT_VARIANTS,
    VARIANTS_BY_NAME,
)

logger = logging.getLogger(__name__)


def _build_one(name: str, recording_path: Path, head_npy: Path, comparison_dir: Path) -> str:
    """Worker: build one variant, logging to its own file."""
    log_dir = comparison_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / f"{name}.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    variant_dir = comparison_dir / name
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    build_variant(
        variant=VARIANTS_BY_NAME[name],
        recording_path=recording_path,
        unfiltered_head_npy=head_npy,
        comparison_dir=comparison_dir,
    )
    return name


def _build_in_subprocess(name: str, recording_path: Path, comparison_dir: Path, retries: int = 1) -> str:
    """Build one variant in a fresh interpreter; retry on failure (crashes can be transient)."""
    cmd = [sys.executable, "-m", "python_code.filter_comparison.run_filter_comparison", str(recording_path),
           "--variants", name, "--jobs", "1", "--no-report", "--overwrite"]
    for attempt in range(retries + 1):
        log_path = comparison_dir / "_logs" / f"{name}.stdout{'' if attempt == 0 else f'.retry{attempt}'}.log"
        with open(log_path, "w") as out:
            result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
        if result.returncode == 0 and variant_is_complete(comparison_dir / name):
            return name
        logger.warning(f"{name} attempt {attempt + 1} failed (exit {result.returncode}); see {log_path}")
    raise RuntimeError(f"{name} failed after {retries + 1} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording_path", type=Path, help="Clip folder, e.g. <session>/full_recording")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS_BY_NAME), default=[v.name for v in FILTER_VARIANTS])
    parser.add_argument("--overwrite", action="store_true", help="Rebuild variants that already exist")
    parser.add_argument("--jobs", type=int, default=1, help="Variants to build in parallel")
    parser.add_argument("--report-only", action="store_true", help="Only (re)write the HTML report")
    parser.add_argument("--no-report", action="store_true", help="Build variants but skip the report")
    parser.add_argument("--dlc-to-3d-repo", type=Path, default=DEFAULT_DLC_TO_3D_REPO)
    parser.add_argument("--freemocap-repo", type=Path, default=DEFAULT_FREEMOCAP_REPO)
    parser.add_argument("--start", type=float, default=None, help="Report time-series window start (s)")
    parser.add_argument("--duration", type=float, default=20.0, help="Report time-series window length (s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    recording_path: Path = args.recording_path.resolve()
    comparison_dir = recording_path / "filter_comparison"

    if not args.report_only:
        head_npy = triangulate_unfiltered_head(
            recording_path=recording_path,
            triangulation_dir=comparison_dir / "_triangulation",
            dlc_to_3d_repo=args.dlc_to_3d_repo,
            freemocap_repo=args.freemocap_repo,
        )
        todo = []
        for name in args.variants:
            if variant_is_complete(comparison_dir / name) and not args.overwrite:
                logger.info(f"Variant {name} already built, skipping (use --overwrite to rebuild)")
            else:
                todo.append(name)
        failed: list[str] = []
        (comparison_dir / "_logs").mkdir(parents=True, exist_ok=True)
        if args.jobs <= 1:
            for name in todo:
                try:
                    _build_one(name, recording_path, head_npy, comparison_dir)
                    logger.info(f"✓ {name} built")
                except Exception:
                    logger.exception(f"✗ {name} failed (see {comparison_dir / '_logs' / (name + '.log')})")
                    failed.append(name)
        else:
            logger.info(f"Building {len(todo)} variant(s) with {args.jobs} job(s); logs in {comparison_dir / '_logs'}")
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(_build_in_subprocess, n, recording_path, comparison_dir): n for n in todo}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                        logger.info(f"✓ {name} built")
                    except Exception as e:
                        logger.error(f"✗ {name}: {e}")
                        failed.append(name)
        if failed and args.no_report:  # worker subprocess: signal failure to the parent
            sys.exit(1)
        if failed:
            logger.error(f"Failed variants: {failed}; report will include only the built ones")

    if args.no_report:
        return
    reports = [("filter_comparison", "Head / eye filtering comparison", MAIN_REPORT_VARIANTS)]
    if any(variant_is_complete(comparison_dir / n) for n in HEAD_CUTOFF_REPORT_VARIANTS if n.startswith("head_bw")):
        reports.append(("head_cutoff", "Head Butterworth cutoff comparison", HEAD_CUTOFF_REPORT_VARIANTS))
    if any(variant_is_complete(comparison_dir / n) for n in EYE_INPUT_REPORT_VARIANTS if n.startswith("dlc_median")):
        reports.append(("eye_input", "Eye input: raw vs DLC median-filtered csv", EYE_INPUT_REPORT_VARIANTS))
    for report_name, title, names in reports:
        report = write_comparison_report(
            comparison_dir=comparison_dir,
            variant_names=names,
            window_start_s=args.start,
            window_duration_s=args.duration,
            report_name=report_name,
            title=title,
        )
        logger.info(f"Report: {report}")


if __name__ == "__main__":
    main()
