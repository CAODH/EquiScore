#!/usr/bin/env python3
"""Compare per-method metrics across sna=false vs sna=true summary parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", nargs="+", required=True, help="Paths to val_metrics_summary_<split>.parquet")
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
        if c.endswith("_mean")
        and c not in {"n_queries_mean", "total_ligands_mean", "total_actives_mean", "total_inactives_mean"}
    ]


def main() -> None:
    args = parse_args()

    frames = []
    for p in args.summaries:
        split_name = _split_name_from_path(p)
        df = pl.read_parquet(p)
        if "method" not in df.columns:
            raise ValueError(f"Missing 'method' column in summary file: {p}")
        frames.append(df.with_columns(pl.lit(split_name).alias("split_name")))

    combined = pl.concat(frames, how="vertical_relaxed")

    metric_cols = _collect_metric_cols(combined)
    if not metric_cols:
        raise ValueError("No *_mean metric columns found in summaries.")

    # Wide format: one row per method, columns per split/metric.
    wide = (
        combined.select(["method", "split_name"] + metric_cols)
        .unpivot(index=["method", "split_name"], variable_name="metric", value_name="value")
        .with_columns(pl.concat_str(["split_name", pl.lit("::"), "metric"]).alias("split_metric"))
        .select(["method", "split_metric", "value"])
        .pivot(on="split_metric", index="method", values="value")
        .sort("method")
    )

    # Optional deltas if both true and false exist.
    split_names = set(combined.get_column("split_name").unique().to_list())
    if {"true", "false"}.issubset(split_names):
        for metric in metric_cols:
            false_col = f"false::{metric}"
            true_col = f"true::{metric}"
            if false_col in wide.columns and true_col in wide.columns:
                wide = wide.with_columns((pl.col(true_col) - pl.col(false_col)).alias(f"delta_true_minus_false::{metric}"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.write_parquet(out, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()
