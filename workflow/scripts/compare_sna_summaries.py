#!/usr/bin/env python3
"""Combine per-method summary metrics across SNA settings into a tidy parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", nargs="+", required=True, help="Paths to val_metrics_summary_<split>.parquet")
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels aligned with --summaries (e.g., false true).",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def _split_name_from_path(path: str) -> str:
    stem = Path(path).stem
    prefix = "val_metrics_summary_sna_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def _collect_metric_cols(df: pl.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.endswith("_median")
    ]


def main() -> None:
    args = parse_args()

    if args.labels is not None and len(args.labels) != len(args.summaries):
        raise ValueError("When provided, --labels must have the same length as --summaries.")

    frames = []
    for idx, p in enumerate(args.summaries):
        split_name = args.labels[idx] if args.labels is not None else _split_name_from_path(p)
        df = pl.read_parquet(p)
        if "method" not in df.columns:
            raise ValueError(f"Missing 'method' column in summary file: {p}")
        frames.append(df.with_columns(pl.lit(split_name).alias("sna")))

    combined = pl.concat(frames, how="vertical_relaxed")

    metric_cols = _collect_metric_cols(combined)
    if not metric_cols:
        raise ValueError("No *_median metric columns found in summaries.")

    cols = combined.columns
    non_metric_cols = [c for c in cols if c not in metric_cols and c not in {"method", "sna"}]

    out_df = (
        combined
        .select(["method", "sna"] + non_metric_cols + metric_cols)
        .sort(["method", "sna"])
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()
