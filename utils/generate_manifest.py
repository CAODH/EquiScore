#!/usr/bin/env python3
"""Generate a single PDBscreen manifest parquet with train/val/test split column."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import polars as pl


EXPECTED_FOLDERS = [
    "PDBscreen_HQ_active_augment_2a_pose_pocket",
    "PDBscreen_HQ_active_pocket",
    "PDBscreen_HQ_cross_decoy_pocket",
    "PDBscreen_HQ_generated_decoy_pocket",
    "PDBscreen_LQ_active_augment_2a_pose_pocket",
    "PDBscreen_LQ_active_pocket",
    "PDBscreen_LQ_cross_decoy_pocket",
    "PDBscreen_LQ_generated_decoy_pocket",
]


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PDBscreen manifest parquet.")
    parser.add_argument("--dataset-root", default=str(repo_root / "PDBscreen"), help="Path to PDBscreen directory")
    parser.add_argument(
        "--out-parquet",
        default=str(repo_root / "dataset" / "pdbscreen_manifest.parquet"),
        help="Output parquet path (default: <repo-root>/dataset/pdbscreen_manifest.parquet)",
    )
    parser.add_argument(
        "--split-dir",
        default=str(repo_root / "data" / "data_splits" / "screen_model"),
        help="Directory containing train_keys.pkl, val_keys.pkl, test_keys.pkl",
    )
    return parser.parse_args()


def collect_paths(dataset_root: Path, repo_root: Path) -> pl.DataFrame:
    rows = []
    seen_folders = set()

    for folder in EXPECTED_FOLDERS:
        folder_path = dataset_root / folder
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        seen_folders.add(folder)

        for child in sorted(folder_path.iterdir()):
            if child.is_file():
                rows.append({"path": str(child.resolve().relative_to(repo_root)), "source_folder": folder})

    missing = [f for f in EXPECTED_FOLDERS if f not in seen_folders]
    if missing:
        raise FileNotFoundError(f"Missing expected folders under dataset root: {missing}")

    if not rows:
        raise ValueError("No files found under expected PDBscreen folders.")

    return pl.DataFrame(rows, schema={"path": pl.Utf8, "source_folder": pl.Utf8})


def build_manifest(df: pl.DataFrame) -> pl.DataFrame:
    manifest = (
        df.lazy()
        .with_columns(
            [
                pl.col("path").str.replace_all(r"\\", "/").alias("path"),
                pl.col("path").str.extract(r"([^/]+)$", group_index=1).alias("filename"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("source_folder").str.contains("_HQ_"))
                .then(pl.lit("HQ"))
                .otherwise(pl.lit("LQ"))
                .alias("data_quality"),
                pl.when(pl.col("source_folder").str.contains("active_augment"))
                .then(pl.lit("pose_augmented_active"))
                .when(pl.col("source_folder").str.contains("generated_decoy"))
                .then(pl.lit("generated_decoy"))
                .when(pl.col("source_folder").str.contains("cross_decoy"))
                .then(pl.lit("cross_decoy"))
                .otherwise(pl.lit("active"))
                .alias("sample_family"),
                pl.col("filename").str.contains("_active").cast(pl.Int8).alias("label"),
                pl.col("filename")
                .str.extract(r"(?:-|_)(\d+)(?:\.[A-Za-z0-9]+)?$", group_index=1)
                .cast(pl.Int32, strict=False)
                .alias("pose_id"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("source_folder") == "PDBscreen_LQ_cross_decoy_pocket")
                .then(pl.col("filename").str.split("-").list.first())
                .when(pl.col("source_folder") == "PDBscreen_HQ_generated_decoy_pocket")
                .then(pl.col("filename").str.split("_").list.first())
                .when(pl.col("source_folder") == "PDBscreen_LQ_generated_decoy_pocket")
                .then(pl.col("filename").str.split("_align").list.first())
                .when(pl.col("source_folder").str.contains("active_augment"))
                .then(pl.col("filename").str.split("_active").list.first())
                .when(pl.col("source_folder") == "PDBscreen_LQ_active_pocket")
                .then(
                    pl.col("filename")
                    .str.replace(r"_active_\d+$", "")
                    .str.replace(r"\.[A-Za-z0-9]+$", "")
                )
                .otherwise(pl.col("filename").str.split("_").list.first())
                .alias("protein_id")
            ]
        )
        .select(
            [
                "path",
                "filename",
                "source_folder",
                "data_quality",
                "sample_family",
                "protein_id",
                "label",
                "pose_id",
            ]
        )
        .unique(subset=["path"], keep="first")
        .collect(engine="streaming")
    )

    if manifest.height == 0:
        raise ValueError("Manifest is empty.")

    required_cols = [
        "path",
        "filename",
        "source_folder",
        "data_quality",
        "sample_family",
        "protein_id",
        "label",
    ]
    null_totals = manifest.select([pl.col(c).is_null().sum().alias(c) for c in required_cols]).row(0)
    if any(v > 0 for v in null_totals):
        raise ValueError("Required columns contain null values.")

    label_values = set(manifest.select(pl.col("label").unique()).to_series().to_list())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"Invalid label values found: {sorted(label_values)}")

    return manifest


def split_df(split_file: Path, split_name: str, repo_root: Path) -> pl.DataFrame:
    with open(split_file, "rb") as f:
        keys = pickle.load(f)
    norm = [str(Path(k).resolve().relative_to(repo_root)).replace("\\", "/") for k in keys]
    return pl.DataFrame({"path": norm, "split": [split_name] * len(norm)}, schema={"path": pl.Utf8, "split": pl.Utf8})


def add_split_column(manifest_df: pl.DataFrame, split_dir: Path, repo_root: Path) -> pl.DataFrame:
    splits = pl.concat(
        [
            split_df(split_dir / "train_keys.pkl", "train", repo_root),
            split_df(split_dir / "val_keys.pkl", "val", repo_root),
            split_df(split_dir / "test_keys.pkl", "test", repo_root),
        ]
    ).unique(subset=["path"], keep="first")
    return manifest_df.join(splits, on="path", how="left")


def ensure_rankable_eval_splits(manifest_df: pl.DataFrame) -> pl.DataFrame:
    eval_stats = (
        manifest_df.filter(pl.col("split").is_in(["val", "test"]))
        .group_by(["split", "protein_id"])
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("label").sum().alias("n_pos"),
            ]
        )
        .with_columns((pl.col("n_rows") - pl.col("n_pos")).alias("n_neg"))
    )

    invalid_eval = (
        eval_stats
        .filter((pl.col("n_rows") < 2) | (pl.col("n_pos") == 0) | (pl.col("n_neg") == 0))
        .select(["split", "protein_id"])
        .with_columns(pl.lit(True).alias("invalid_eval_query"))
    )

    return (
        manifest_df.join(invalid_eval, on=["split", "protein_id"], how="left")
        .with_columns(
            pl.when(pl.col("invalid_eval_query") == True)
            .then(None)
            .otherwise(pl.col("split"))
            .alias("split")
        )
        .drop("invalid_eval_query")
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(repo_root)

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    out_path = Path(args.out_parquet).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_df = build_manifest(collect_paths(dataset_root, repo_root))
    manifest_df = add_split_column(manifest_df, Path(args.split_dir).resolve(), repo_root)
    manifest_df = ensure_rankable_eval_splits(manifest_df)
    manifest_df.write_parquet(out_path, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()