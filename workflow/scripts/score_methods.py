#!/usr/bin/env python3
"""Train/score ranking baselines on train split and emit validation predictions."""

from __future__ import annotations

import argparse
import random
from typing import Dict, List

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


ECFP_COL = "ecfp_morgan_r2_2048_bits"


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


def _feature_cols(df: pl.DataFrame) -> List[str]:
    preferred = [
        "tpsa",
        "molecular_weight",
        "rotatable_bonds",
        "ring_count",
        "aromatic_ring_count",
        "hbd",
        "hba",
        "lipinski_compliant",
        "verber_compliant",
    ]
    return [c for c in preferred if c in df.columns]


def _build_feature_tables(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    *,
    include_assay: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, List[str]]:
    feature_cols = _feature_cols(train_df)
    if not feature_cols:
        raise ValueError("No ligand descriptor feature columns found in manifest.")

    if not include_assay:
        return train_df, val_df, feature_cols

    assay_cols: List[str] = []
    if "standard_value" in train_df.columns:
        train_val = train_df.get_column("standard_value").cast(pl.Float64, strict=False)
        non_null = train_val.drop_nulls()
        standard_value_fill = float(non_null.median()) if non_null.len() > 0 else 0.0

        train_df = train_df.with_columns(
            [
                pl.col("standard_value")
                .cast(pl.Float64, strict=False)
                .fill_null(standard_value_fill)
                .alias("standard_value_filled"),
            ]
        )
        val_df = val_df.with_columns(
            [
                pl.col("standard_value")
                .cast(pl.Float64, strict=False)
                .fill_null(standard_value_fill)
                .alias("standard_value_filled"),
            ]
        )
        assay_cols.append("standard_value_filled")

    if assay_cols:
        feature_cols = feature_cols + assay_cols
    else:
        print("[assay_features] No standard_value column found; falling back to ligand-only features.")

    return train_df, val_df, feature_cols


def _build_ligand_id_expr() -> pl.Expr:
    label_token = pl.concat_str([pl.lit("y"), pl.col("label").cast(pl.Utf8)])
    return (
        pl.when(pl.col("ligand_smiles").is_not_null())
        .then(
            pl.concat_str(
                [
                    pl.col("protein_id"),
                    pl.lit("::"),
                    label_token,
                    pl.lit("::"),
                    pl.col("ligand_smiles"),
                ]
            )
        )
        .otherwise(
            pl.concat_str(
                [
                    pl.col("protein_id"),
                    pl.lit("::"),
                    label_token,
                    pl.lit("::path::"),
                    pl.col("path"),
                ]
            )
        )
        .alias("ligand_id")
    )


def _group_sizes(df: pl.DataFrame, group_col: str) -> List[int]:
    return (
        df.group_by(group_col)
        .len()
        .sort(group_col)
        .get_column("len")
        .to_list()
    )


def _mol_fp(smiles: str | None):
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _tanimoto(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(a, b))


def _ecfp_bitvect_from_bytes(payload: bytes | None):
    if payload is None:
        return None
    try:
        return DataStructs.ExplicitBitVect(payload)
    except Exception:
        return None


def _ecfp_csr_matrix(payloads: List[bytes | None], method_name: str):
    try:
        from scipy import sparse
    except Exception as exc:
        raise RuntimeError(
            f"scipy is required for method '{method_name}'. Install with: pip install scipy"
        ) from exc

    row_idx: List[int] = []
    col_idx: List[int] = []
    n_bits = 2048

    for i, payload in enumerate(payloads):
        fp = _ecfp_bitvect_from_bytes(payload)
        if fp is None:
            continue
        if i == 0:
            n_bits = int(fp.GetNumBits())
        on_bits = list(fp.GetOnBits())
        if not on_bits:
            continue
        row_idx.extend([i] * len(on_bits))
        col_idx.extend(on_bits)

    data = np.ones(len(row_idx), dtype=np.float32)
    return sparse.csr_matrix((data, (row_idx, col_idx)), shape=(len(payloads), n_bits), dtype=np.float32)


def _lambda_mart_scores(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    seed: int,
    *,
    include_assay: bool,
    method_name: str,
) -> np.ndarray:
    try:
        from lightgbm import LGBMRanker
    except Exception as exc:
        raise RuntimeError(
            "lightgbm is required for method 'lambdamart_ligand'. Install with: pip install lightgbm"
        ) from exc

    train_df, val_df, feature_cols = _build_feature_tables(
        train_df,
        val_df,
        include_assay=include_assay,
    )

    # Cast booleans once so all models see numeric tensors.
    cast_expr = []
    for c in feature_cols:
        if train_df.schema[c] == pl.Boolean:
            cast_expr.append(pl.col(c).cast(pl.Int8).alias(c))
    if cast_expr:
        train_df = train_df.with_columns(cast_expr)
        val_df = val_df.with_columns(cast_expr)

    train_rank = train_df.drop_nulls(feature_cols + ["label", "protein_id"]).sort("protein_id")
    val_with_idx = val_df.with_row_index("__row_idx")
    val_rank = val_with_idx.drop_nulls(feature_cols + ["label", "protein_id"]).sort("protein_id")

    # LightGBM ranking enforces an upper bound of 10,000 rows per query.
    oversized = (
        train_rank.group_by("protein_id")
        .len()
        .filter(pl.col("len") > 10_000)
        .get_column("protein_id")
        .to_list()
    )
    if oversized:
        print(
            f"[{method_name}] Dropping {len(oversized)} training queries with >10000 ligands "
            "(LightGBM limitation)."
        )
        train_rank = train_rank.filter(~pl.col("protein_id").is_in(oversized))
        if train_rank.height == 0:
            raise ValueError("No train rows remain for LambdaMART after removing oversized queries.")

    X_train = train_rank.select(feature_cols).to_numpy()
    y_train = train_rank.get_column("label").to_numpy()
    g_train = _group_sizes(train_rank, "protein_id")

    X_val = val_rank.select(feature_cols).to_numpy()

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=400,
        random_state=seed,
    )
    model.fit(X_train, y_train, group=g_train)

    val_scores = model.predict(X_val)
    score_df = val_rank.select(["__row_idx"]).with_columns(pl.Series("score", val_scores))

    # Merge back so output order exactly follows val_df rows.
    merged = (
        val_with_idx.join(score_df, on=["__row_idx"], how="left")
        .sort("__row_idx")
        .drop("__row_idx")
    )
    score_col = merged.get_column("score")
    if score_col.null_count() > 0:
        n_null = score_col.null_count()
        print(f"[{method_name}] Assigning score=0.0 to {n_null} validation rows with missing descriptors.")
        merged = merged.with_columns(pl.col("score").fill_null(0.0))
        score_col = merged.get_column("score")
    return score_col.to_numpy()


def _logistic_regression_scores(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    seed: int,
    *,
    include_assay: bool,
    method_name: str,
) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for method 'logistic_regression_ligand'. Install with: pip install scikit-learn"
        ) from exc

    train_df, val_df, feature_cols = _build_feature_tables(
        train_df,
        val_df,
        include_assay=include_assay,
    )

    cast_expr = []
    for c in feature_cols:
        if train_df.schema[c] == pl.Boolean:
            cast_expr.append(pl.col(c).cast(pl.Int8).alias(c))
    if cast_expr:
        train_df = train_df.with_columns(cast_expr)
        val_df = val_df.with_columns(cast_expr)

    train_cls = train_df.drop_nulls(feature_cols + ["label"])
    val_with_idx = val_df.with_row_index("__row_idx")
    val_cls = val_with_idx.drop_nulls(feature_cols + ["label"])

    if train_cls.height == 0:
        raise ValueError("No training rows with non-null descriptor features for logistic regression.")
    if val_cls.height == 0:
        raise ValueError("No validation rows with non-null descriptor features for logistic regression.")

    X_train = train_cls.select(feature_cols).to_numpy()
    y_train = train_cls.get_column("label").to_numpy()
    X_val = val_cls.select(feature_cols).to_numpy()

    if len(np.unique(y_train)) < 2:
        raise ValueError("Logistic regression requires at least two classes in training labels.")

    model = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    score_df = val_cls.select(["__row_idx"]).with_columns(pl.Series("score", val_scores))

    merged = (
        val_with_idx.join(score_df, on=["__row_idx"], how="left")
        .sort("__row_idx")
        .drop("__row_idx")
    )
    score_col = merged.get_column("score")
    if score_col.null_count() > 0:
        n_null = score_col.null_count()
        print(
            f"[{method_name}] Assigning score=0.0 to {n_null} validation rows with missing descriptors."
        )
        merged = merged.with_columns(pl.col("score").fill_null(0.0))
        score_col = merged.get_column("score")
    return score_col.to_numpy()


def _tanimoto_random_ref_scores(train_df: pl.DataFrame, val_df: pl.DataFrame, seed: int) -> np.ndarray:
    rng = random.Random(seed)

    train_active = train_df.filter((pl.col("label") == 1) & pl.col("ligand_smiles").is_not_null())
    global_refs = train_active.get_column("ligand_smiles").to_list()
    if not global_refs:
        return np.zeros(val_df.height, dtype=float)

    refs_by_query: Dict[str, List[str]] = {}
    for row in train_active.select(["protein_id", "ligand_smiles"]).iter_rows(named=True):
        refs_by_query.setdefault(row["protein_id"], []).append(row["ligand_smiles"])

    chosen_ref = {}
    for q in val_df.get_column("protein_id").unique(maintain_order=True).to_list():
        pool = refs_by_query.get(q, global_refs)
        chosen_ref[q] = pool[rng.randrange(len(pool))]

    ref_fp = {q: _mol_fp(smi) for q, smi in chosen_ref.items()}
    ligand_fps = [_mol_fp(smi) for smi in val_df.get_column("ligand_smiles").to_list()]
    query_vals = val_df.get_column("protein_id").to_list()

    return np.array([_tanimoto(fp, ref_fp[q]) for fp, q in zip(ligand_fps, query_vals)], dtype=float)


def _logistic_regression_fingerprint_scores(train_df: pl.DataFrame, val_df: pl.DataFrame, seed: int) -> np.ndarray:
    try:
        from sklearn.linear_model import SGDClassifier
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for method 'logistic_regression_fingerprint'. "
            "Install with: pip install scikit-learn"
        ) from exc

    if ECFP_COL not in train_df.columns:
        raise ValueError(f"Manifest missing required fingerprint column: {ECFP_COL}")

    train_cls = train_df.drop_nulls([ECFP_COL, "label"])
    val_with_idx = val_df.with_row_index("__row_idx")
    val_cls = val_with_idx.drop_nulls([ECFP_COL, "label"])

    if train_cls.height == 0:
        raise ValueError("No training rows with non-null fingerprint features.")
    if val_cls.height == 0:
        raise ValueError("No validation rows with non-null fingerprint features.")

    max_train_rows = 300_000
    if train_cls.height > max_train_rows:
        print(
            f"[logistic_regression_fingerprint] Downsampling training rows from {train_cls.height} "
            f"to {max_train_rows} for faster ablation."
        )
        train_cls = train_cls.sample(n=max_train_rows, seed=seed, shuffle=True)

    X_train = _ecfp_csr_matrix(train_cls.get_column(ECFP_COL).to_list(), "logistic_regression_fingerprint")
    y_train = train_cls.get_column("label").to_numpy()
    X_val = _ecfp_csr_matrix(val_cls.get_column(ECFP_COL).to_list(), "logistic_regression_fingerprint")

    if len(np.unique(y_train)) < 2:
        raise ValueError("Fingerprint logistic regression requires at least two classes in training labels.")

    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    score_df = val_cls.select(["__row_idx"]).with_columns(pl.Series("score", val_scores))

    merged = (
        val_with_idx.join(score_df, on=["__row_idx"], how="left")
        .sort("__row_idx")
        .drop("__row_idx")
    )
    score_col = merged.get_column("score")
    if score_col.null_count() > 0:
        n_null = score_col.null_count()
        print(
            f"[logistic_regression_fingerprint] Assigning score=0.0 to {n_null} validation rows "
            "with missing fingerprints."
        )
        merged = merged.with_columns(pl.col("score").fill_null(0.0))
        score_col = merged.get_column("score")
    return score_col.to_numpy()


def _lambda_mart_fingerprint_scores(train_df: pl.DataFrame, val_df: pl.DataFrame, seed: int) -> np.ndarray:
    try:
        from lightgbm import LGBMRanker
    except Exception as exc:
        raise RuntimeError(
            "lightgbm is required for method 'lambdamart_fingerprint'. Install with: pip install lightgbm"
        ) from exc

    if ECFP_COL not in train_df.columns:
        raise ValueError(f"Manifest missing required fingerprint column: {ECFP_COL}")

    train_rank = train_df.drop_nulls([ECFP_COL, "label", "protein_id"]).sort("protein_id")
    val_with_idx = val_df.with_row_index("__row_idx")
    val_rank = val_with_idx.drop_nulls([ECFP_COL, "label", "protein_id"]).sort("protein_id")

    if train_rank.height == 0:
        raise ValueError("No training rows with non-null fingerprint features for LambdaMART.")
    if val_rank.height == 0:
        raise ValueError("No validation rows with non-null fingerprint features for LambdaMART.")

    oversized = (
        train_rank.group_by("protein_id")
        .len()
        .filter(pl.col("len") > 10_000)
        .get_column("protein_id")
        .to_list()
    )
    if oversized:
        print(
            f"[lambdamart_fingerprint] Dropping {len(oversized)} training queries with >10000 ligands "
            "(LightGBM limitation)."
        )
        train_rank = train_rank.filter(~pl.col("protein_id").is_in(oversized))
        if train_rank.height == 0:
            raise ValueError("No train rows remain for lambdamart_fingerprint after removing oversized queries.")

    X_train = _ecfp_csr_matrix(train_rank.get_column(ECFP_COL).to_list(), "lambdamart_fingerprint")
    y_train = train_rank.get_column("label").to_numpy()
    g_train = _group_sizes(train_rank, "protein_id")

    X_val = _ecfp_csr_matrix(val_rank.get_column(ECFP_COL).to_list(), "lambdamart_fingerprint")

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=400,
        random_state=seed,
    )
    model.fit(X_train, y_train, group=g_train)

    val_scores = model.predict(X_val)
    score_df = val_rank.select(["__row_idx"]).with_columns(pl.Series("score", val_scores))

    merged = (
        val_with_idx.join(score_df, on=["__row_idx"], how="left")
        .sort("__row_idx")
        .drop("__row_idx")
    )
    score_col = merged.get_column("score")
    if score_col.null_count() > 0:
        n_null = score_col.null_count()
        print(
            f"[lambdamart_fingerprint] Assigning score=0.0 to {n_null} validation rows "
            "with missing fingerprints."
        )
        merged = merged.with_columns(pl.col("score").fill_null(0.0))
        score_col = merged.get_column("score")
    return score_col.to_numpy()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    df = pl.read_parquet(args.manifest)
    required = {"path", "split", "protein_id", "label", "ligand_smiles"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    if "pose_id" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Int32).alias("pose_id"))

    if "sna" not in df.columns:
        if args.sna == "true":
            raise ValueError(
                "Manifest has no 'sna' column, so '--sna true' is not valid for this dataset."
            )
        df = df.with_columns(pl.lit(False).alias("sna"))

    if df.schema["sna"] != pl.Boolean:
        raise ValueError(
            f"Manifest column 'sna' must be Boolean, got {df.schema['sna']}"
        )

    if args.sna != "all":
        wanted_sna = args.sna == "true"
        df = df.filter(pl.col("sna") == wanted_sna)
        if df.height == 0:
            raise ValueError(
                f"No rows found for sna='{args.sna}' in manifest '{args.manifest}'."
            )

    df = df.with_columns(
        [
            pl.col("label").cast(pl.Int8),
            pl.when(pl.col("ligand_smiles") == "")
            .then(None)
            .otherwise(pl.col("ligand_smiles"))
            .alias("ligand_smiles"),
            pl.col("pose_id").cast(pl.Int32, strict=False),
        ]
    ).with_columns(_build_ligand_id_expr())

    train_df = df.filter(pl.col("split") == args.train_split)
    val_df = df.filter(pl.col("split") == args.val_split)

    if train_df.height == 0:
        raise ValueError(f"No training rows found for split='{args.train_split}'")
    if val_df.height == 0:
        raise ValueError(f"No validation rows found for split='{args.val_split}'")

    preds = []
    if "lambdamart_ligand" in methods:
        lm_scores = _lambda_mart_scores(
            train_df,
            val_df,
            seed=args.seed,
            include_assay=False,
            method_name="lambdamart_ligand",
        )
        preds.append(
            val_df.select(["path", "protein_id", "ligand_id", "ligand_smiles", "pose_id", "label", "sna"])
            .rename({"label": "true_label"})
            .with_columns(
                pl.lit("lambdamart_ligand").alias("method"),
                pl.Series("score", lm_scores),
            )
        )

    if "tanimoto_random_ref" in methods:
        t_scores = _tanimoto_random_ref_scores(train_df, val_df, seed=args.seed)
        preds.append(
            val_df.select(["path", "protein_id", "ligand_id", "ligand_smiles", "pose_id", "label", "sna"])
            .rename({"label": "true_label"})
            .with_columns(
                pl.lit("tanimoto_random_ref").alias("method"),
                pl.Series("score", t_scores),
            )
        )

    if "logistic_regression_ligand" in methods:
        lr_scores = _logistic_regression_scores(
            train_df,
            val_df,
            seed=args.seed,
            include_assay=False,
            method_name="logistic_regression_ligand",
        )
        preds.append(
            val_df.select(["path", "protein_id", "ligand_id", "ligand_smiles", "pose_id", "label", "sna"])
            .rename({"label": "true_label"})
            .with_columns(
                pl.lit("logistic_regression_ligand").alias("method"),
                pl.Series("score", lr_scores),
            )
        )

    if "lambdamart_ligand_assay" in methods:
        lma_scores = _lambda_mart_scores(
            train_df,
            val_df,
            seed=args.seed,
            include_assay=True,
            method_name="lambdamart_ligand_assay",
        )
        preds.append(
            val_df.select(["path", "protein_id", "ligand_id", "ligand_smiles", "pose_id", "label", "sna"])
            .rename({"label": "true_label"})
            .with_columns(
                pl.lit("lambdamart_ligand_assay").alias("method"),
                pl.Series("score", lma_scores),
            )
        )

    if "logistic_regression_ligand_assay" in methods:
        lra_scores = _logistic_regression_scores(
            train_df,
            val_df,
            seed=args.seed,
            include_assay=True,
            method_name="logistic_regression_ligand_assay",
        )
        preds.append(
            val_df.select(["path", "protein_id", "ligand_id", "ligand_smiles", "pose_id", "label", "sna"])
            .rename({"label": "true_label"})
            .with_columns(
                pl.lit("logistic_regression_ligand_assay").alias("method"),
                pl.Series("score", lra_scores),
            )
        )

    if "logistic_regression_fingerprint" in methods:
        lrfp_scores = _logistic_regression_fingerprint_scores(train_df, val_df, seed=args.seed)
        preds.append(
            val_df.select(["path", "protein_id", "ligand_id", "ligand_smiles", "pose_id", "label", "sna"])
            .rename({"label": "true_label"})
            .with_columns(
                pl.lit("logistic_regression_fingerprint").alias("method"),
                pl.Series("score", lrfp_scores),
            )
        )

    if "lambdamart_fingerprint" in methods:
        lmfp_scores = _lambda_mart_fingerprint_scores(train_df, val_df, seed=args.seed)
        preds.append(
            val_df.select(["path", "protein_id", "ligand_id", "ligand_smiles", "pose_id", "label", "sna"])
            .rename({"label": "true_label"})
            .with_columns(
                pl.lit("lambdamart_fingerprint").alias("method"),
                pl.Series("score", lmfp_scores),
            )
        )

    if not preds:
        raise ValueError("No enabled methods produced predictions.")

    out_df = (
        pl.concat(preds, how="vertical")
        .with_columns(
            [
                pl.lit(args.val_split).alias("split"),
                pl.col("true_label").cast(pl.Int8),
            ]
        )
        .select(
            [
                "path",
                "protein_id",
                "ligand_id",
                "ligand_smiles",
                "pose_id",
                "true_label",
                "method",
                "score",
                "sna",
            ]
        )
    )

    rank_df = (
        out_df.select(["method", "protein_id", "path", "score"])
        .sort(
            ["method", "protein_id", "score", "path"],
            descending=[False, False, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["method", "protein_id"]).alias("rank")
        )
        .select(["method", "protein_id", "path", "rank"])
    )

    out_df = (
        out_df.join(rank_df, on=["method", "protein_id", "path"], how="left")
        .with_columns(pl.lit(args.val_split).alias("split"))
        .select(
            [
                "path",
                "protein_id",
                "ligand_id",
                "ligand_smiles",
                "pose_id",
                "true_label",
                "method",
                "score",
                "rank",
                "split",
                "sna",
            ]
        )
    )

    out_df.write_parquet(args.out, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()
