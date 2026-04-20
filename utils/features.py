#!/usr/bin/env python3
"""Compute fixed chemistry features for manifest rows in parallel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import polars as pl
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.contrib.concurrent import process_map

ECFP_COL = "ecfp_morgan_r2_2048_bits"

FEATURE_DTYPES = {
    ECFP_COL: pl.Binary,
    "ligand_smiles": pl.Utf8,
    "hbd": pl.Int16,
    "hba": pl.Int16,
    "tpsa": pl.Float64,
    "molecular_weight": pl.Float64,
    "rotatable_bonds": pl.Int16,
    "ring_count": pl.Int16,
    "aromatic_ring_count": pl.Int16,
    "lipinski_compliant": pl.Boolean,
    "verber_compliant": pl.Boolean,
    "scaffold": pl.Utf8,
}

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=False,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize chemistry features into manifest parquet.")
    parser.add_argument("--manifest", default="dataset/pdbscreen_manifest.parquet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=200)
    parser.add_argument("--force-recompute", action="store_true", default=False)
    return parser.parse_args()


def ensure_feature_columns(manifest_df: pl.DataFrame) -> pl.DataFrame:
    missing = [name for name in FEATURE_DTYPES if name not in manifest_df.columns]
    if missing:
        additions = [pl.lit(None, dtype=FEATURE_DTYPES[name]).alias(name) for name in missing]
        manifest_df = manifest_df.with_columns(additions)

    # Normalize empty strings so they are treated as missing features.
    return manifest_df.with_columns(
        [
            pl.when(pl.col("scaffold") == "").then(None).otherwise(pl.col("scaffold")).alias("scaffold"),
            pl.when(pl.col("ligand_smiles") == "").then(None).otherwise(pl.col("ligand_smiles")).alias("ligand_smiles"),
        ]
    )


def rows_to_compute(manifest_df: pl.DataFrame, force_recompute: bool) -> pl.DataFrame:
    if force_recompute:
        return manifest_df.select(["path", "ligand_smiles"]).unique(subset=["path"], keep="first")

    checks = [pl.col(name).is_null() for name in FEATURE_DTYPES]
    checks.append(pl.col("scaffold") == "")
    checks.append(pl.col("ligand_smiles") == "")
    cond = checks[0]
    for check in checks[1:]:
        cond = cond | check
    return (
        manifest_df.filter(cond)
        .select(["path", "ligand_smiles"])
        .unique(subset=["path"], keep="first")
    )


def _fill_features_from_mol(result: dict, mol) -> dict:
    if mol is None:
        return result

    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    mw = float(Descriptors.MolWt(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    ring_count = int(rdMolDescriptors.CalcNumRings(mol))
    aromatic_ring_count = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    logp = float(Crippen.MolLogP(mol))
    ligand_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    result[ECFP_COL] = MORGAN_GENERATOR.GetFingerprint(mol).ToBinary()
    result["ligand_smiles"] = ligand_smiles if ligand_smiles else None
    result["hbd"] = hbd
    result["hba"] = hba
    result["tpsa"] = tpsa
    result["molecular_weight"] = mw
    result["rotatable_bonds"] = rot
    result["ring_count"] = ring_count
    result["aromatic_ring_count"] = aromatic_ring_count
    result["lipinski_compliant"] = bool((mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10))
    result["verber_compliant"] = bool((rot <= 10) and (tpsa <= 140))
    result["scaffold"] = scaffold if scaffold else None
    return result


def _compute_one(work_item: tuple[str, str | None]) -> dict:
    rel_path, ligand_smiles_in = work_item

    result = {
        "path": rel_path,
        ECFP_COL: None,
        "ligand_smiles": None,
        "hbd": None,
        "hba": None,
        "tpsa": None,
        "molecular_weight": None,
        "rotatable_bonds": None,
        "ring_count": None,
        "aromatic_ring_count": None,
        "lipinski_compliant": None,
        "verber_compliant": None,
        "scaffold": None,
    }

    if not ligand_smiles_in:
        return result

    mol = Chem.MolFromSmiles(ligand_smiles_in)
    if mol is None:
        return result

    try:
        return _fill_features_from_mol(result, mol)
    except Exception:
        return result


def merge_updates(manifest_df: pl.DataFrame, updates_df: pl.DataFrame) -> pl.DataFrame:
    merged = manifest_df.join(updates_df, on="path", how="left", suffix="_new")

    coalesced = []
    for col in FEATURE_DTYPES:
        new_col = f"{col}_new"
        coalesced.append(pl.coalesce([pl.col(new_col), pl.col(col)]).alias(col))

    merged = merged.with_columns(coalesced)
    drop_cols = [f"{col}_new" for col in FEATURE_DTYPES if f"{col}_new" in merged.columns]
    return merged.drop(drop_cols)


def to_work_items(rows: Iterable[dict]) -> list[tuple[str, str | None]]:
    return [(r["path"], r.get("ligand_smiles")) for r in rows]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()

    manifest_df = ensure_feature_columns(pl.read_parquet(manifest_path))
    todo_df = rows_to_compute(manifest_df, args.force_recompute)

    if todo_df.height == 0:
        print("No rows require feature computation.")
        return

    work_items = to_work_items(todo_df.to_dicts())
    updates = process_map(
        _compute_one,
        work_items,
        max_workers=max(1, args.workers),
        chunksize=max(1, args.chunksize),
        desc="Computing chemistry features",
    )
    updates_df = pl.DataFrame(updates).unique(subset=["path"], keep="last")

    merged = merge_updates(manifest_df, updates_df)
    merged.write_parquet(manifest_path, compression="zstd", statistics=True)

    print(f"Updated manifest: {manifest_path}")
    print(f"Rows processed: {todo_df.height}")


if __name__ == "__main__":
    main()
