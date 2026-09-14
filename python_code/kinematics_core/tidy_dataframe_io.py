"""
Shared read helper for tidy-format kinematics dataframes.

Parquet is the canonical, full-fidelity round-trip format; CSV is a secondary,
human/analysis-oriented export that is free to diverge in shape from parquet.
Loaders should always prefer parquet when it exists, falling back to CSV only
for recordings saved before parquet output existed.
"""
from pathlib import Path

import polars as pl


def read_parquet_or_csv(parquet_path: Path, csv_path: Path) -> pl.DataFrame:
    """Read parquet_path if it exists, else fall back to csv_path."""
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)
    return pl.read_csv(csv_path)
