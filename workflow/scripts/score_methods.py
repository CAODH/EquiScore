#!/usr/bin/env python3
"""OOP ranking pipeline: train/score ranking baselines and emit validation predictions."""

from __future__ import annotations
import argparse
import random
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import polars as pl
from lightgbm import LGBMRanker
from rdkit import DataStructs
from sklearn.linear_model import LogisticRegression


# LightGBM lambdarank hard limit per query group
_LGBM_MAX_QUERY_ROWS = 10_000


# --- Abstract Base Class ---

class RankingMethod(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def features(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def fit(self, train_df: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, val_df: pl.DataFrame) -> np.ndarray:
        pass


# --- Concrete Implementations ---

class LambdaMARTLigand(RankingMethod):
    name = "lambdamart_ligand"
    features = {
        "tpsa": "Topological polar surface area",
        "molecular_weight": "Molecular weight",
        "rotatable_bonds": "Number of rotatable bonds",
        "ring_count": "Number of rings",
        "aromatic_ring_count": "Number of aromatic rings",
        "hbd": "Hydrogen bond donors",
        "hba": "Hydrogen bond acceptors",
        "lipinski_compliant": "Lipinski rule-of-5 compliance (bool)",
        "verber_compliant": "Verber rule compliance (bool)",
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None

    def fit(self, train_df: pl.DataFrame) -> None:
        train_df = (
            train_df
            .sort(["protein_id", "label"], descending=[False, True])
            .filter(pl.int_range(pl.len()).over("protein_id") < _LGBM_MAX_QUERY_ROWS)
        )
        X = train_df.select(list(self.features.keys())).to_numpy()
        y = train_df.get_column("label").to_numpy()
        group = train_df.get_column("protein_id").to_numpy()
        _, group_sizes = np.unique(group, return_counts=True)
        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=400,
            random_state=self.seed,
        )
        self.model.fit(X, y, group=group_sizes)

    def predict(self, val_df: pl.DataFrame) -> np.ndarray:
        X = val_df.select(list(self.features.keys())).to_numpy()
        return self.model.predict(X)


class LambdaMARTProteo(RankingMethod):
    name = "lambdamart_proteo"
    features = {
        **LambdaMARTLigand.features,
        "protein_fingerprint": "Protein-level fingerprint (e.g., ESM embedding)",
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None

    def fit(self, train_df: pl.DataFrame) -> None:
        train_df = (
            train_df
            .sort(["protein_id", "label"], descending=[False, True])
            .filter(pl.int_range(pl.len()).over("protein_id") < _LGBM_MAX_QUERY_ROWS)
        )
        X = train_df.select(list(self.features.keys())).to_numpy()
        y = train_df.get_column("label").to_numpy()
        group = train_df.get_column("protein_id").to_numpy()
        _, group_sizes = np.unique(group, return_counts=True)
        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=400,
            random_state=self.seed,
        )
        self.model.fit(X, y, group=group_sizes)

    def predict(self, val_df: pl.DataFrame) -> np.ndarray:
        X = val_df.select(list(self.features.keys())).to_numpy()
        return self.model.predict(X)


class TanimotoRandomRef(RankingMethod):
    name = "tanimoto_random_ref"
    features = {
        "ligand_fingerprint": "Ligand Morgan fingerprint, radius 2, 2048 bits",
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, train_df: pl.DataFrame) -> None:
        self.active_refs = train_df.filter(pl.col("label") == 1).get_column("ligand_fingerprint").to_list()

    def predict(self, val_df: pl.DataFrame) -> np.ndarray:
        rng = random.Random(self.seed)
        ref_fp = rng.choice(self.active_refs)
        return np.array([
            DataStructs.TanimotoSimilarity(fp, ref_fp)
            for fp in val_df.get_column("ligand_fingerprint").to_list()
        ], dtype=float)


class LogisticLigand(RankingMethod):
    name = "logistic_ligand"
    features = LambdaMARTLigand.features

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None

    def fit(self, train_df: pl.DataFrame) -> None:
        X = train_df.select(list(self.features.keys())).to_numpy()
        y = train_df.get_column("label").to_numpy()
        self.model = LogisticRegression(max_iter=1000, random_state=self.seed)
        self.model.fit(X, y)

    def predict(self, val_df: pl.DataFrame) -> np.ndarray:
        X = val_df.select(list(self.features.keys())).to_numpy()
        return self.model.predict_proba(X)[:, 1]


class LogisticProteo(RankingMethod):
    name = "logistic_proteo"
    features = LambdaMARTProteo.features

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None

    def fit(self, train_df: pl.DataFrame) -> None:
        X = train_df.select(list(self.features.keys())).to_numpy()
        y = train_df.get_column("label").to_numpy()
        self.model = LogisticRegression(max_iter=1000, random_state=self.seed)
        self.model.fit(X, y)

    def predict(self, val_df: pl.DataFrame) -> np.ndarray:
        X = val_df.select(list(self.features.keys())).to_numpy()
        return self.model.predict_proba(X)[:, 1]


# --- Method Registry ---

METHOD_REGISTRY = {
    "lambdamart_ligand": LambdaMARTLigand,
    "lambdamart_proteo": LambdaMARTProteo,
    "tanimoto_random_ref": TanimotoRandomRef,
    "logistic_ligand": LogisticLigand,
    "logistic_proteo": LogisticProteo,
}


# --- CLI Entrypoint ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument(
        "--sna",
        default="all",
        choices=["all", "true", "false"],
        help="Filter rows by sna value (true/false). Use 'all' for no filtering.",
    )
    p.add_argument("--methods", default="lambdamart_ligand,tanimoto_random_ref")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(args.manifest)

    if "sna" in df.columns and args.sna != "all":
        df = df.filter(pl.col("sna") == (args.sna == "true"))

    train_df = df.filter(pl.col("split") == args.train_split)
    val_df = df.filter(pl.col("split") == args.val_split)

    if "sna" in val_df.columns:
        val_df = val_df.drop("sna")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_results = []

    for method_name in methods:
        if method_name not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method: {method_name}")
        method_cls = METHOD_REGISTRY[method_name]
        method = method_cls(seed=args.seed)

        missing = [f for f in method.features if f not in train_df.columns]
        if missing:
            raise ValueError(f"Missing required features for {method_name}: {missing}")

        method.fit(train_df)
        scores = method.predict(val_df)

        ligand_id_expr = (
            pl.col("ligand_id") if "ligand_id" in val_df.columns
            else pl.col("ligand_smiles").alias("ligand_id")
        )
        out_df = val_df.select([
            pl.lit(method_name).alias("method"),
            pl.col("protein_id"),
            ligand_id_expr,
            pl.col("label").alias("true_label"),
            pl.Series("score", scores),
        ])
        all_results.append(out_df)

    result = pl.concat(all_results, how="vertical")

    result = result.with_columns(
        pl.col("score").rank("dense", descending=True).over(["method", "protein_id"]).alias("rank")
    )
    result = result.with_columns(pl.lit(args.val_split).alias("split"))
    result = result.select([
        "protein_id",
        "ligand_id",
        "true_label",
        "method",
        "score",
        "rank",
        "split",
    ])

    result.write_parquet(args.out, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()
