#!/usr/bin/env python3
"""Validate a PDBscreen manifest parquet produced by generate_manifest.py."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional

import polars as pl


DEFAULT_MANIFEST = "data/manifests/pdbscreen_manifest.parquet"
DEFAULT_SUMMARY = "data/manifests/pdbscreen_manifest_validation.parquet"
DEFAULT_SPLIT_COUNTS = "data/manifests/pdbscreen_manifest_split_counts.parquet"
DEFAULT_EXPECTED_COUNTS = "data/manifests/pdbscreen_manifest_expected_vs_actual.parquet"


def _read_key_len(path: Optional[str]) -> Optional[int]:
    if path is None:
        return None
    with open(path, "rb") as f:
        keys = pickle.load(f)
    if not isinstance(keys, list):
        raise TypeError(f"Expected list in {path}, got {type(keys)}")
    return len(keys)


def _validation_summary_lf(manifest_path: str) -> pl.LazyFrame:
    required_cols = [
        "sample_id",
        "split",
        "path",
        "filename",
        "source_folder",
        "data_quality",
        "sample_family",
        "protein_id",
        "label",
    ]

    lf = pl.scan_parquet(manifest_path)
    null_checks = [pl.col(c).is_null().sum().alias(f"null_{c}") for c in required_cols]

    return lf.select(
        [
            pl.len().alias("rows_total"),
            pl.col("sample_id").n_unique().alias("sample_id_unique"),
            (pl.col("sample_id").is_duplicated().sum()).alias("duplicated_sample_id_rows"),
            (~pl.col("label").is_in([0, 1])).sum().alias("bad_label_rows"),
            (~pl.col("split").is_in(["train", "val", "test"]))
            .sum()
            .alias("bad_split_rows"),
            (~pl.col("data_quality").is_in(["HQ", "LQ", "unknown"]))
            .sum()
            .alias("bad_data_quality_rows"),
            (
                ~pl.col("sample_family").is_in(
                    [
                        "active",
                        "cross_decoy",
                        "generated_decoy",
                        "pose_augmented_active",
                        "unknown",
                    ]
                )
            )
            .sum()
            .alias("bad_sample_family_rows"),
            (~pl.col("decoy_type").is_in(["cross_decoy", "generated_decoy", None]))
            .sum()
            .alias("bad_decoy_type_rows"),
        ]
        + null_checks
    )


def _split_counts_lf(manifest_path: str) -> pl.LazyFrame:
    return (
        pl.scan_parquet(manifest_path)
        .group_by("split")
        .len()
        .rename({"len": "rows"})
        .sort("split")
    )


def _expected_vs_actual_df(manifest_path: str, expected_counts: Dict[str, int]) -> pl.DataFrame:
    expected_df = pl.DataFrame(
        {
            "split": list(expected_counts.keys()),
            "expected_rows": list(expected_counts.values()),
        },
        schema={"split": pl.Utf8, "expected_rows": pl.Int64},
    )

    actual_lf = _split_counts_lf(manifest_path).lazy().rename({"rows": "actual_rows"})

    return (
        expected_df.lazy()
        .join(actual_lf, on="split", how="left")
        .with_columns(
            [
                pl.col("actual_rows").fill_null(0),
                (pl.col("actual_rows") - pl.col("expected_rows")).alias("row_delta"),
            ]
        )
        .sort("split")
        .collect(streaming=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PDBscreen manifest parquet using Polars.")
    parser.add_argument("--manifest-parquet", default=DEFAULT_MANIFEST)
    parser.add_argument("--summary-parquet", default=DEFAULT_SUMMARY)
    parser.add_argument("--split-counts-parquet", default=DEFAULT_SPLIT_COUNTS)
    parser.add_argument("--expected-vs-actual-parquet", default=DEFAULT_EXPECTED_COUNTS)
    parser.add_argument("--train-keys", default=None)
    parser.add_argument("--val-keys", default=None)
    parser.add_argument("--test-keys", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_summary = Path(args.summary_parquet)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    _validation_summary_lf(args.manifest_parquet).collect(streaming=True).write_parquet(
        args.summary_parquet,
        compression="zstd",
        statistics=True,
    )

    _split_counts_lf(args.manifest_parquet).collect(streaming=True).write_parquet(
        args.split_counts_parquet,
        compression="zstd",
        statistics=True,
    )

    expected = {
        "train": _read_key_len(args.train_keys),
        "val": _read_key_len(args.val_keys),
        "test": _read_key_len(args.test_keys),
    }
    expected = {k: v for k, v in expected.items() if v is not None}

    if expected:
        _expected_vs_actual_df(args.manifest_parquet, expected).write_parquet(
            args.expected_vs_actual_parquet,
            compression="zstd",
            statistics=True,
        )


if __name__ == "__main__":
    main()
