#!/usr/bin/env python3
"""Generate BigBind manifest parquet plus summary metadata parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


TEST_ROWS = 1000
REQUIRED_SOURCE_COLS = ["lig_smiles", "lig_file", "active", "pocket"]
REQUIRED_MANIFEST_COLS = ["split", "protein_id", "ligand_smiles", "label", "path", "sample_id"]
SPLITS = ["train", "val", "test"]


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BigBind manifest and summary parquet files.")
    parser.add_argument("--dataset-root", default=str(repo_root / "BigBindV1.5"))
    parser.add_argument("--out-parquet", default=str(repo_root / "dataset" / "bigbind_manifest.parquet"))
    parser.add_argument("--summary-parquet", default=str(repo_root / "dataset" / "bigbind_manifest_summary.parquet"))
    parser.add_argument(
        "--include-sna",
        action="store_true",
        default=True,
        help="Deprecated: SNA is always included in the manifest.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    return parser.parse_args()


def _with_test_suffix(path: Path) -> Path:
    return path.with_name(f"{path.stem}_test{path.suffix}")


def _load_split_csv(
    dataset_root: Path,
    split: str,
    sna: bool,
    n_rows: int | None = None,
) -> pl.DataFrame:
    if sna is False:
        csv_path = dataset_root / f"activities_{split}.csv"
    elif sna is True:
        csv_path = dataset_root / f"activities_sna_1_{split}.csv"
    else:
        raise ValueError(f"Unsupported sna value: {sna}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing BigBind file: {csv_path}")

    df = pl.read_csv(csv_path, n_rows=n_rows)
    missing = sorted(set(REQUIRED_SOURCE_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    return df.with_columns(
        [
            pl.lit(split).alias("split"),
            pl.lit(sna).cast(pl.Boolean).alias("sna"),
        ]
    )


def _build_manifest(df: pl.DataFrame, dataset_root_name: str) -> pl.DataFrame:
    manifest = (
        df.with_row_index("row_idx")
        .with_columns(
            [
                pl.lit("BigBind").alias("dataset_source"),
                pl.col("pocket").cast(pl.Utf8).alias("protein_id"),
                pl.col("lig_smiles")
                .cast(pl.Utf8)
                .str.strip_chars()
                .alias("ligand_smiles"),
                pl.col("lig_file").cast(pl.Utf8).alias("lig_file"),
                pl.concat_str([pl.lit(f"{dataset_root_name}/"), pl.col("lig_file")]).alias("path"),
                pl.lit("measured").alias("source_set"),
                pl.when(pl.col("active").cast(pl.Utf8).str.to_lowercase().is_in(["true", "1", "t", "yes"]))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("label"),
                pl.concat_str(
                    [
                        pl.lit("bigbind::"),
                        pl.col("sna").cast(pl.Utf8),
                        pl.lit("::"),
                        pl.col("split"),
                        pl.lit("::"),
                        pl.col("row_idx").cast(pl.Utf8),
                    ]
                ).alias("sample_id"),
            ]
        )
        .drop("row_idx")
    )

    # Keep only known splits and sna labels even if source files include anything unexpected.
    manifest = manifest.filter(pl.col("split").is_in(SPLITS) & pl.col("sna").is_in([False, True]))

    if manifest.height == 0:
        raise ValueError("Manifest is empty after filtering to train/val/test.")

    null_totals = manifest.select([pl.col(c).is_null().sum().alias(c) for c in REQUIRED_MANIFEST_COLS]).row(0)
    if any(v > 0 for v in null_totals):
        raise ValueError("Required manifest columns contain null values.")

    label_values = set(manifest.select(pl.col("label").unique()).to_series().to_list())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"Invalid label values found: {sorted(label_values)}")

    dup_count = int(manifest.select(pl.col("sample_id").is_duplicated().sum()).item())
    if dup_count > 0:
        raise ValueError(f"Duplicate sample_id rows found: {dup_count}")

    return manifest


def _build_summary(manifest: pl.DataFrame) -> pl.DataFrame:
    group_cols = ["sna", "split"]

    base = manifest.group_by(group_cols).agg(
        [
            pl.len().alias("rows"),
            pl.col("label").sum().cast(pl.Int64).alias("n_pos"),
            (pl.len() - pl.col("label").sum()).cast(pl.Int64).alias("n_neg"),
            pl.col("label").mean().alias("pos_rate"),
            pl.col("protein_id").n_unique().alias("n_proteins"),
            pl.col("ligand_smiles").n_unique().alias("n_unique_ligands"),
        ]
    )

    per_protein = manifest.group_by(group_cols + ["protein_id"]).agg(
        [pl.len().alias("n_rows"), pl.col("label").sum().alias("n_pos")]
    )

    evaluable = (
        per_protein.with_columns((pl.col("n_rows") - pl.col("n_pos")).alias("n_neg"))
        .filter((pl.col("n_pos") > 0) & (pl.col("n_neg") > 0))
        .group_by(group_cols)
        .agg(pl.len().alias("n_evaluable_proteins"))
    )

    ligands_per_protein = per_protein.group_by(group_cols).agg(
        [
            pl.col("n_rows").mean().alias("ligands_per_protein_mean"),
            pl.col("n_rows").median().alias("ligands_per_protein_median"),
        ]
    )

    nulls = manifest.group_by(group_cols).agg(
        [
            pl.col("split").is_null().sum().alias("null_split"),
            pl.col("protein_id").is_null().sum().alias("null_protein_id"),
            pl.col("ligand_smiles").is_null().sum().alias("null_ligand_smiles"),
            pl.col("label").is_null().sum().alias("null_label"),
            pl.col("path").is_null().sum().alias("null_path"),
        ]
    )

    dup = manifest.group_by(group_cols).agg(
        pl.col("sample_id").is_duplicated().sum().alias("duplicate_sample_id_rows")
    )

    return (
        base.join(evaluable, on=group_cols, how="left")
        .join(ligands_per_protein, on=group_cols, how="left")
        .join(nulls, on=group_cols, how="left")
        .join(dup, on=group_cols, how="left")
        .with_columns(pl.col("n_evaluable_proteins") .fill_null(0).cast(pl.Int64))
        .sort(group_cols)
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(repo_root)

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    out_manifest = Path(args.out_parquet).resolve()
    out_summary = Path(args.summary_parquet).resolve()
    if args.test:
        out_manifest = _with_test_suffix(out_manifest)
        out_summary = _with_test_suffix(out_summary)

    sna_values = [False, True]
    n_rows = TEST_ROWS if args.test else None

    frames: list[pl.DataFrame] = []
    for sna in sna_values:
        for split in SPLITS:
            frames.append(_load_split_csv(dataset_root, split=split, sna=sna, n_rows=n_rows))

    manifest = _build_manifest(pl.concat(frames, how="vertical_relaxed"), dataset_root.name)
    summary = _build_summary(manifest)

    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"Dry run: {args.dry_run}")
    print(f"Manifest output: {out_manifest}")
    print(f"Summary output: {out_summary}")
    print(f"Manifest rows: {manifest.height}")
    print(summary)

    if args.dry_run:
        return

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    manifest.write_parquet(out_manifest, compression="zstd", statistics=True)
    summary.write_parquet(out_summary, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()
