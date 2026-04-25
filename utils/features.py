#!/usr/bin/env python3
"""Augment a manifest parquet with precomputed features."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
from tqdm.contrib.concurrent import process_map


# --- Abstract base ---

class FeatureComputer(ABC):
    name: str
    key_col: str
    output_dtypes: dict[str, pl.DataType]

    @staticmethod
    @abstractmethod
    def compute_one(key: str) -> dict:
        pass

    def augment(
        self,
        manifest_df: pl.DataFrame,
        workers: int = 8,
        chunksize: int = 200,
        force_recompute: bool = False,
    ) -> pl.DataFrame:
        manifest_df = self._ensure_columns(manifest_df)
        keys = self._todo(manifest_df, force_recompute)
        if not keys:
            return manifest_df

        updates = process_map(
            self.__class__.compute_one,
            keys,
            max_workers=max(1, workers),
            chunksize=max(1, chunksize),
            desc=f"Computing {self.name} features",
        )
        updates_df = pl.DataFrame(updates).unique(subset=[self.key_col], keep="last")
        return self._merge(manifest_df, updates_df)

    def _ensure_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [c for c in self.output_dtypes if c not in df.columns]
        if missing:
            df = df.with_columns([
                pl.lit(None, dtype=self.output_dtypes[c]).alias(c) for c in missing
            ])
        return df

    def _todo(self, df: pl.DataFrame, force_recompute: bool) -> list[str]:
        if force_recompute:
            return df.select(self.key_col).unique().drop_nulls().to_series().to_list()

        null_checks = [pl.col(c).is_null() for c in self.output_dtypes]
        cond = null_checks[0]
        for check in null_checks[1:]:
            cond = cond | check

        return (
            df.filter(cond)
            .select(self.key_col)
            .unique()
            .drop_nulls()
            .to_series()
            .to_list()
        )

    def _merge(self, manifest_df: pl.DataFrame, updates_df: pl.DataFrame) -> pl.DataFrame:
        merged = manifest_df.join(updates_df, on=self.key_col, how="left", suffix="_new")
        coalesced = [
            pl.coalesce([pl.col(f"{c}_new"), pl.col(c)]).alias(c)
            for c in self.output_dtypes
            if f"{c}_new" in merged.columns
        ]
        if coalesced:
            merged = merged.with_columns(coalesced)
        drop_cols = [f"{c}_new" for c in self.output_dtypes if f"{c}_new" in merged.columns]
        return merged.drop(drop_cols)


# --- Concrete computers ---

class LigandFeatures(FeatureComputer):
    name = "ligand"
    key_col = "ligand_smiles"
    output_dtypes = {
        "ligand_fingerprint": pl.Binary,
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

    _MORGAN_GENERATOR = None

    @classmethod
    def _get_generator(cls):
        if cls._MORGAN_GENERATOR is None:
            from rdkit.Chem import rdFingerprintGenerator
            cls._MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=2048, includeChirality=False
            )
        return cls._MORGAN_GENERATOR

    @staticmethod
    def compute_one(smiles: str) -> dict:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold

        result: dict = {
            "ligand_smiles": smiles,
            "ligand_fingerprint": None,
            "hbd": None, "hba": None,
            "tpsa": None, "molecular_weight": None,
            "rotatable_bonds": None, "ring_count": None,
            "aromatic_ring_count": None,
            "lipinski_compliant": None, "verber_compliant": None,
            "scaffold": None,
        }

        if not smiles:
            return result

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result

        try:
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=2048, includeChirality=False
            )
            hbd = int(Lipinski.NumHDonors(mol))
            hba = int(Lipinski.NumHAcceptors(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            mw = float(Descriptors.MolWt(mol))
            rot = int(Lipinski.NumRotatableBonds(mol))
            logp = float(Crippen.MolLogP(mol))
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

            result["ligand_fingerprint"] = gen.GetFingerprint(mol).ToBinary()
            result["ligand_smiles"] = canonical if canonical else smiles
            result["hbd"] = hbd
            result["hba"] = hba
            result["tpsa"] = tpsa
            result["molecular_weight"] = mw
            result["rotatable_bonds"] = rot
            result["ring_count"] = int(rdMolDescriptors.CalcNumRings(mol))
            result["aromatic_ring_count"] = int(rdMolDescriptors.CalcNumAromaticRings(mol))
            result["lipinski_compliant"] = bool(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
            result["verber_compliant"] = bool(rot <= 10 and tpsa <= 140)
            result["scaffold"] = scaffold if scaffold else None
        except Exception:
            pass

        return result


class LigandSmilesFromFile(FeatureComputer):
    """Extract ligand SMILES from pickled (mol, protein, ...) tuple files (PDBscreen format)."""
    name = "ligand_smiles_from_file"
    key_col = "path"
    output_dtypes = {"ligand_smiles": pl.Utf8}

    @staticmethod
    def compute_one(path: str) -> dict:
        import pickle
        from rdkit import Chem

        result: dict = {"path": path, "ligand_smiles": None}
        if not path:
            return result
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            mol = obj[0] if isinstance(obj, (tuple, list)) else obj
            if mol is None:
                return result
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            result["ligand_smiles"] = smiles if smiles else None
        except Exception:
            pass
        return result


class ProteinFingerprint(FeatureComputer):
    name = "protein"
    key_col = "ex_rec_file"
    output_dtypes = {"protein_fingerprint": pl.Binary}

    @staticmethod
    def compute_one(ex_rec_file: str) -> dict:
        import pickle
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        result: dict = {"ex_rec_file": ex_rec_file, "protein_fingerprint": None}

        if not ex_rec_file:
            return result

        try:
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=2048, includeChirality=False
            )
            if ex_rec_file.endswith(".pdb"):
                mol = Chem.MolFromPDBFile(ex_rec_file, removeHs=True, sanitize=True)
                if mol is None:
                    mol = Chem.MolFromPDBFile(ex_rec_file, removeHs=True, sanitize=False)
                    if mol is not None:
                        Chem.FastFindRings(mol)
            else:
                with open(ex_rec_file, "rb") as f:
                    obj = pickle.load(f)
                mol = obj[1] if isinstance(obj, (tuple, list)) and len(obj) > 1 else None
            if mol is None:
                return result
            result["protein_fingerprint"] = gen.GetFingerprint(mol).ToBinary()
        except Exception:
            pass

        return result


# --- Registry & CLI ---

COMPUTERS: dict[str, type[FeatureComputer]] = {
    "ligand_smiles_from_file": LigandSmilesFromFile,
    "ligand": LigandFeatures,
    "protein": ProteinFingerprint,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment a manifest parquet with precomputed features.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--features", default="ligand", help="Comma-separated list of feature computers to run.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunksize", type=int, default=200)
    p.add_argument("--force-recompute", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest_df = pl.read_parquet(manifest_path)

    for name in [n.strip() for n in args.features.split(",") if n.strip()]:
        if name not in COMPUTERS:
            raise ValueError(f"Unknown feature computer: {name!r}. Available: {list(COMPUTERS)}")
        computer = COMPUTERS[name]()
        manifest_df = computer.augment(
            manifest_df,
            workers=args.workers,
            chunksize=args.chunksize,
            force_recompute=args.force_recompute,
        )

    manifest_df.write_parquet(manifest_path, compression="zstd", statistics=True)
    print(f"Updated: {manifest_path}")


if __name__ == "__main__":
    main()
