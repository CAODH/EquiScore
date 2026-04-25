#!/usr/bin/env python3
"""Build dataset manifests. Supports BigBind and PDBscreen."""

from __future__ import annotations

import argparse
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
from features import COMPUTERS


# --- Shared constants ---

SPLITS = ["train", "val", "test"]

REQUIRED_COLS = ["protein_id", "label", "path"]

PDBSCREEN_EXPECTED_FOLDERS = [
    "PDBscreen_HQ_active_augment_2a_pose_pocket",
    "PDBscreen_HQ_active_pocket",
    "PDBscreen_HQ_cross_decoy_pocket",
    "PDBscreen_HQ_generated_decoy_pocket",
    "PDBscreen_LQ_active_augment_2a_pose_pocket",
    "PDBscreen_LQ_active_pocket",
    "PDBscreen_LQ_cross_decoy_pocket",
    "PDBscreen_LQ_generated_decoy_pocket",
]

BIGBIND_REQUIRED_SOURCE_COLS = ["lig_smiles", "lig_file", "ex_rec_file", "active", "pocket"]
BIGBIND_TEST_ROWS = 1000


# --- Abstract base ---

class ManifestBuilder(ABC):
    default_dataset_root: str
    default_out: str

    @abstractmethod
    def collect_and_normalize(self) -> pl.DataFrame:
        pass

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        null_totals = df.select([pl.col(c).is_null().sum().alias(c) for c in REQUIRED_COLS]).row(0)
        if any(v > 0 for v in null_totals):
            raise ValueError(f"Required columns contain null values: {dict(zip(REQUIRED_COLS, null_totals))}")

        split_nulls = df["split"].null_count() if "split" in df.columns else 0
        if split_nulls > 0:
            print(f"  [warn] {split_nulls} rows have no split assignment and will be excluded from train/val/test")

        label_values = set(df.select(pl.col("label").unique()).to_series().to_list())
        if not label_values.issubset({0, 1}):
            raise ValueError(f"Invalid label values: {sorted(label_values)}")

        key_col = "sample_id" if "sample_id" in df.columns else "path"
        dup_count = int(df.select(pl.col(key_col).is_duplicated().sum()).item())
        if dup_count > 0:
            raise ValueError(f"Duplicate {key_col} rows: {dup_count}")

        df = self._ensure_rankable_eval_splits(df)
        return df

    def build(self) -> pl.DataFrame:
        return self.validate(self.collect_and_normalize())

    def write(self, df: pl.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd", statistics=True)

    @staticmethod
    def _ensure_rankable_eval_splits(df: pl.DataFrame) -> pl.DataFrame:
        eval_stats = (
            df.filter(pl.col("split").is_in(["val", "test"]))
            .group_by(["split", "protein_id"])
            .agg(
                pl.len().alias("n_rows"),
                pl.col("label").sum().alias("n_pos"),
            )
            .with_columns((pl.col("n_rows") - pl.col("n_pos")).alias("n_neg"))
        )
        invalid = (
            eval_stats
            .filter((pl.col("n_rows") < 2) | (pl.col("n_pos") == 0) | (pl.col("n_neg") == 0))
            .select(["split", "protein_id"])
            .with_columns(pl.lit(True).alias("_invalid"))
        )
        return (
            df.join(invalid, on=["split", "protein_id"], how="left")
            .with_columns(
                pl.when(pl.col("_invalid"))
                .then(None)
                .otherwise(pl.col("split"))
                .alias("split")
            )
            .drop("_invalid")
        )


# --- BigBind ---

class BigBindManifestBuilder(ManifestBuilder):
    default_dataset_root = "BigBindV1.5"
    default_out = "dataset/bigbind_manifest.parquet"
    default_summary_out = "dataset/bigbind_manifest_summary.parquet"

    def __init__(self, dataset_root: Path, test: bool = False) -> None:
        self.dataset_root = dataset_root
        self.test = test

    def collect_and_normalize(self) -> pl.DataFrame:
        n_rows = BIGBIND_TEST_ROWS if self.test else None
        frames = [
            self._load_split_csv(split, sna, n_rows)
            for sna in [False, True]
            for split in SPLITS
        ]
        df = pl.concat(frames, how="vertical_relaxed")
        return self._normalize(df)

    def _load_split_csv(self, split: str, sna: bool, n_rows: int | None) -> pl.DataFrame:
        csv_path = (
            self.dataset_root / f"activities_sna_1_{split}.csv"
            if sna
            else self.dataset_root / f"activities_{split}.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing BigBind file: {csv_path}")

        df = pl.read_csv(csv_path, n_rows=n_rows)
        missing = sorted(set(BIGBIND_REQUIRED_SOURCE_COLS) - set(df.columns))
        if missing:
            raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

        return df.with_columns(
            pl.lit(split).alias("split"),
            pl.lit(sna).cast(pl.Boolean).alias("sna"),
        )

    def _normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        dataset_root_name = self.dataset_root.name
        manifest = (
            df.with_row_index("row_idx")
            .with_columns(
                pl.lit("BigBind").alias("dataset_source"),
                pl.col("pocket").cast(pl.Utf8).alias("protein_id"),
                pl.col("lig_smiles").cast(pl.Utf8).str.strip_chars().alias("ligand_smiles"),
                pl.col("lig_file").cast(pl.Utf8).alias("lig_file"),
                pl.concat_str([pl.lit(f"{dataset_root_name}/"), pl.col("lig_file")]).alias("path"),
                pl.concat_str([pl.lit(f"{dataset_root_name}/"), pl.col("ex_rec_file").cast(pl.Utf8)]).alias("ex_rec_file"),
                pl.lit("measured").alias("source_set"),
                pl.when(
                    pl.col("active").cast(pl.Utf8).str.to_lowercase().is_in(["true", "1", "t", "yes"])
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("label"),
                pl.concat_str([
                    pl.lit("bigbind::"),
                    pl.col("sna").cast(pl.Utf8),
                    pl.lit("::"),
                    pl.col("split"),
                    pl.lit("::"),
                    pl.col("row_idx").cast(pl.Utf8),
                ]).alias("sample_id"),
            )
            .drop("row_idx")
            .filter(pl.col("split").is_in(SPLITS) & pl.col("sna").is_in([False, True]))
        )

        if manifest.height == 0:
            raise ValueError("Manifest is empty after filtering.")

        return manifest

    def build_summary(self, manifest: pl.DataFrame) -> pl.DataFrame:
        group_cols = ["sna", "split"]

        base = manifest.group_by(group_cols).agg(
            pl.len().alias("rows"),
            pl.col("label").sum().cast(pl.Int64).alias("n_pos"),
            (pl.len() - pl.col("label").sum()).cast(pl.Int64).alias("n_neg"),
            pl.col("label").mean().alias("pos_rate"),
            pl.col("protein_id").n_unique().alias("n_proteins"),
            pl.col("ligand_smiles").n_unique().alias("n_unique_ligands"),
        )

        per_protein = manifest.group_by(group_cols + ["protein_id"]).agg(
            pl.len().alias("n_rows"),
            pl.col("label").sum().alias("n_pos"),
        )

        evaluable = (
            per_protein
            .with_columns((pl.col("n_rows") - pl.col("n_pos")).alias("n_neg"))
            .filter((pl.col("n_pos") > 0) & (pl.col("n_neg") > 0))
            .group_by(group_cols)
            .agg(pl.len().alias("n_evaluable_proteins"))
        )

        ligands_per_protein = per_protein.group_by(group_cols).agg(
            pl.col("n_rows").mean().alias("ligands_per_protein_mean"),
            pl.col("n_rows").median().alias("ligands_per_protein_median"),
        )

        nulls = manifest.group_by(group_cols).agg(
            pl.col("split").is_null().sum().alias("null_split"),
            pl.col("protein_id").is_null().sum().alias("null_protein_id"),
            pl.col("ligand_smiles").is_null().sum().alias("null_ligand_smiles"),
            pl.col("label").is_null().sum().alias("null_label"),
            pl.col("path").is_null().sum().alias("null_path"),
        )

        dup = manifest.group_by(group_cols).agg(
            pl.col("sample_id").is_duplicated().sum().alias("duplicate_sample_id_rows")
        )

        return (
            base
            .join(evaluable, on=group_cols, how="left")
            .join(ligands_per_protein, on=group_cols, how="left")
            .join(nulls, on=group_cols, how="left")
            .join(dup, on=group_cols, how="left")
            .with_columns(pl.col("n_evaluable_proteins").fill_null(0).cast(pl.Int64))
            .sort(group_cols)
        )


# --- PDBscreen ---

class PDBscreenManifestBuilder(ManifestBuilder):
    default_dataset_root = "PDBscreen"
    default_out = "dataset/pdbscreen_manifest.parquet"

    def __init__(self, dataset_root: Path, split_dir: Path, repo_root: Path) -> None:
        self.dataset_root = dataset_root
        self.split_dir = split_dir
        self.repo_root = repo_root

    def collect_and_normalize(self) -> pl.DataFrame:
        df = self._collect_paths()
        df = self._normalize(df)
        df = self._add_ex_rec_file(df)
        df = self._add_splits(df)
        return df

    def _collect_paths(self) -> pl.DataFrame:
        rows = []
        seen_folders: set[str] = set()

        for folder in PDBSCREEN_EXPECTED_FOLDERS:
            folder_path = self.dataset_root / folder
            if not folder_path.exists() or not folder_path.is_dir():
                continue
            seen_folders.add(folder)
            for child in sorted(folder_path.iterdir()):
                if child.is_file():
                    rows.append({
                        "path": str(child.resolve().relative_to(self.repo_root)),
                        "source_folder": folder,
                    })

        missing = [f for f in PDBSCREEN_EXPECTED_FOLDERS if f not in seen_folders]
        if missing:
            raise FileNotFoundError(f"Missing PDBscreen folders: {missing}")
        if not rows:
            raise ValueError("No files found under PDBscreen folders.")

        return pl.DataFrame(rows, schema={"path": pl.Utf8, "source_folder": pl.Utf8})

    def _normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.lazy()
            .with_columns(
                pl.col("path").str.replace_all(r"\\", "/").alias("path"),
                pl.col("path").str.extract(r"([^/]+)$", group_index=1).alias("filename"),
            )
            .with_columns(
                pl.when(pl.col("source_folder").str.contains("_HQ_"))
                .then(pl.lit("HQ")).otherwise(pl.lit("LQ")).alias("data_quality"),
                pl.when(pl.col("source_folder").str.contains("active_augment"))
                .then(pl.lit("pose_augmented_active"))
                .when(pl.col("source_folder").str.contains("generated_decoy"))
                .then(pl.lit("generated_decoy"))
                .when(pl.col("source_folder").str.contains("cross_decoy"))
                .then(pl.lit("cross_decoy"))
                .otherwise(pl.lit("active")).alias("sample_family"),
                pl.col("filename").str.contains("_active").cast(pl.Int8).alias("label"),
                pl.col("filename")
                .str.extract(r"(?:-|_)(\d+)(?:\.[A-Za-z0-9]+)?$", group_index=1)
                .cast(pl.Int32, strict=False).alias("pose_id"),
            )
            .with_columns(
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
            )
            .select(["path", "filename", "source_folder", "data_quality", "sample_family",
                     "protein_id", "label", "pose_id"])
            .unique(subset=["path"], keep="first")
            .collect(engine="streaming")
        )

    def _add_ex_rec_file(self, df: pl.DataFrame) -> pl.DataFrame:
        ex_rec = (
            df.filter(pl.col("source_folder").str.ends_with("_active_pocket"))
            .group_by("protein_id")
            .agg(pl.col("path").first().alias("ex_rec_file"))
        )
        return df.join(ex_rec, on="protein_id", how="left")

    def _add_splits(self, df: pl.DataFrame) -> pl.DataFrame:
        def _load_pkl(name: str, split_name: str) -> pl.DataFrame:
            with open(self.split_dir / name, "rb") as f:
                keys = pickle.load(f)
            norm = [
                str(Path(k).resolve().relative_to(self.repo_root)).replace("\\", "/")
                for k in keys
            ]
            return pl.DataFrame(
                {"path": norm, "split": [split_name] * len(norm)},
                schema={"path": pl.Utf8, "split": pl.Utf8},
            )

        path_splits = pl.concat([
            _load_pkl("train_keys.pkl", "train"),
            _load_pkl("val_keys.pkl", "val"),
            _load_pkl("test_keys.pkl", "test"),
        ]).unique(subset=["path"], keep="first")

        # Split files only cover LQ active pocket paths; propagate split to all rows
        # for the same protein_id so decoys and HQ actives are included.
        protein_splits = (
            df.join(path_splits, on="path", how="inner")
            .select(["protein_id", "split"])
            .unique(subset=["protein_id"], keep="first")
        )
        return df.join(protein_splits, on="protein_id", how="left")


# --- Registry & CLI ---

BUILDERS: dict[str, type[ManifestBuilder]] = {
    "bigbind": BigBindManifestBuilder,
    "pdbscreen": PDBscreenManifestBuilder,
}


def parse_args(repo_root: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dataset manifest parquets.")
    p.add_argument("--dataset", default="all", choices=list(BUILDERS.keys()) + ["all"])
    p.add_argument("--dataset-root", default=None, help="Override dataset root (single-dataset mode only).")
    p.add_argument("--out", default=None, help="Override output path (single-dataset mode only).")
    p.add_argument("--summary-out", default=None)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--test", action="store_true", default=False)
    p.add_argument(
        "--split-dir",
        default=str(repo_root / "data" / "data_splits" / "screen_model"),
    )
    p.add_argument("--features", default="ligand,protein", help="Comma-separated feature computers to run after building.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunksize", type=int, default=200)
    p.add_argument("--skip-features", action="store_true", default=False)
    return p.parse_args()


def _with_test_suffix(path: Path) -> Path:
    return path.with_name(f"{path.stem}_test{path.suffix}")


def _run_features(manifest_path: Path, args: argparse.Namespace) -> None:
    if args.skip_features or args.dry_run:
        return
    names = [n.strip() for n in args.features.split(",") if n.strip()]
    if not names:
        return
    df = pl.read_parquet(manifest_path)
    for name in names:
        if name not in COMPUTERS:
            raise ValueError(f"Unknown feature computer: {name!r}. Available: {list(COMPUTERS)}")
        print(f"  [{name}] computing features...")
        computer = COMPUTERS[name]()
        df = computer.augment(df, workers=args.workers, chunksize=args.chunksize)
    df.write_parquet(manifest_path, compression="zstd", statistics=True)
    print(f"  features written: {manifest_path}")


def _run_bigbind(args: argparse.Namespace, repo_root: Path) -> None:
    dataset_root = Path(args.dataset_root or (repo_root / BigBindManifestBuilder.default_dataset_root)).resolve()
    out_path = Path(args.out or (repo_root / BigBindManifestBuilder.default_out)).resolve()
    summary_path = Path(
        args.summary_out or (repo_root / BigBindManifestBuilder.default_summary_out)
    ).resolve()

    if args.test:
        out_path = _with_test_suffix(out_path)
        summary_path = _with_test_suffix(summary_path)

    if not dataset_root.exists():
        raise FileNotFoundError(f"BigBind dataset root not found: {dataset_root}")

    builder = BigBindManifestBuilder(dataset_root, test=args.test)
    manifest = builder.build()
    summary = builder.build_summary(manifest)

    print(f"[bigbind] rows={manifest.height}  out={out_path}")
    print(summary)

    if not args.dry_run:
        builder.write(manifest, out_path)
        builder.write(summary, summary_path)
        print(f"[bigbind] summary={summary_path}")
        _run_features(out_path, args)


def _run_pdbscreen(args: argparse.Namespace, repo_root: Path) -> None:
    dataset_root = Path(args.dataset_root or (repo_root / PDBscreenManifestBuilder.default_dataset_root)).resolve()
    out_path = Path(args.out or (repo_root / PDBscreenManifestBuilder.default_out)).resolve()

    if args.test:
        out_path = _with_test_suffix(out_path)

    if not dataset_root.exists():
        raise FileNotFoundError(f"PDBscreen dataset root not found: {dataset_root}")

    builder = PDBscreenManifestBuilder(
        dataset_root,
        split_dir=Path(args.split_dir).resolve(),
        repo_root=repo_root,
    )
    manifest = builder.build()

    print(f"[pdbscreen] rows={manifest.height}  out={out_path}")

    if not args.dry_run:
        builder.write(manifest, out_path)
        # PDBscreen ligand files store the molecule as a pickled tuple — extract SMILES first
        pdbscreen_args = argparse.Namespace(**vars(args))
        if args.features == "ligand,protein":
            pdbscreen_args.features = "ligand_smiles_from_file,ligand,protein"
        _run_features(out_path, pdbscreen_args)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(repo_root)

    targets = list(BUILDERS.keys()) if args.dataset == "all" else [args.dataset]

    for dataset in targets:
        if dataset == "bigbind":
            _run_bigbind(args, repo_root)
        elif dataset == "pdbscreen":
            _run_pdbscreen(args, repo_root)


if __name__ == "__main__":
    main()
