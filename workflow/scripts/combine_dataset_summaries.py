#!/usr/bin/env python3
"""Combine per-dataset comparison parquet files into one summary parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine per-dataset parquet summaries.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames = []
    for path in args.inputs:
        dataset_name = Path(path).stem
        frame = pl.read_parquet(path).with_columns(pl.lit(dataset_name).alias("dataset_name"))
        frames.append(frame)

    combined = pl.concat(frames, how="diagonal_relaxed").sort(["dataset_name", "method"])

    leading = [c for c in ["method", "dataset_name", "sna"] if c in combined.columns]
    trailing = [c for c in combined.columns if c not in leading]
    combined = combined.select(leading + trailing)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()
