#!/usr/bin/env python3
"""Combine per-sna prediction parquet files into one per-dataset intermediate parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine per-sna predictions into one dataset parquet.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames = [pl.read_parquet(path) for path in args.inputs]
    combined = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.lit(args.dataset).alias("dataset_name")
    )

    leading = [c for c in ["method", "dataset_name", "sna"] if c in combined.columns]
    trailing = [c for c in combined.columns if c not in leading]
    combined = combined.select(leading + trailing)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()
