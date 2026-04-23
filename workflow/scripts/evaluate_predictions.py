#!/usr/bin/env python3
"""Evaluate ranking predictions on a validation split."""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


FRACTIONS = [0.01, 0.05, 0.10]
ALPHA = 20.0
METRIC_COLS = [
    "ef@1%",
    "ef@5%",
    "ef@10%",
    "efb@1%",
    "efb_max",
    "nef@1%",
    "nef@5%",
    "nef@10%",
    "ndcg@1%",
    "ndcg@5%",
    "ndcg@10%",
    "map",
    "err",
    "roc_auc",
    "bedroc_a20",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--summary-out", required=True)
    p.add_argument("--by-query-out", required=True)
    return p.parse_args()


def _top_k(n: int, frac: float) -> int:
    return max(1, int(np.ceil(frac * n)))


def _sorted_indices_by_rank(y_rank: np.ndarray) -> np.ndarray:
    return np.argsort(y_rank, kind="mergesort")


def _expected_reciprocal_rank(y_true_sorted: np.ndarray) -> float:
    rel = np.clip(y_true_sorted.astype(np.float64), 0.0, 1.0)

    err = 0.0
    continue_prob = 1.0
    for rank_idx, gain in enumerate(rel, start=1):
        stop_prob = continue_prob * gain
        err += stop_prob / rank_idx
        continue_prob *= 1.0 - gain
        if continue_prob <= 0.0:
            break

    return float(err)


def _calc_efb(act_preds: np.ndarray, rand_preds: np.ndarray, select_frac: float, is_sorted: bool = False) -> float:
    """Compute Bayes enrichment factor for a target select fraction."""
    if act_preds.size == 0 or rand_preds.size == 0:
        return 0.0

    n_rand = len(rand_preds)
    select_num = round((1 - select_frac) * n_rand)
    select_num = max(1, min(select_num, n_rand))

    if is_sorted:
        select = rand_preds[select_num - 1]
    else:
        select = np.sort(rand_preds)[select_num - 1]

    p_rand = float((rand_preds >= select).sum() / n_rand)
    if p_rand == 0:
        return 0.0

    p_act = float((act_preds >= select).sum() / len(act_preds))
    if p_act == 0:
        return 0.0

    return float(p_act / p_rand)


def _calc_best_efb(act_preds: np.ndarray, rand_preds: np.ndarray) -> float:
    """Compute best achievable Bayes enrichment factor (EFB_max)."""
    if act_preds.size == 0 or rand_preds.size == 0:
        return 0.0

    rand_sorted = np.sort(rand_preds)
    cutoffs = np.unique(act_preds.astype(np.float16))

    cur_best = None
    seen_fracs = set()

    for act in cutoffs:
        cur_frac = float((rand_sorted >= act).sum() / len(rand_sorted))
        if cur_frac in seen_fracs:
            continue
        if cur_frac == 0:
            cur_frac = 1 / len(rand_sorted)

        seen_fracs.add(cur_frac)
        efb = _calc_efb(act_preds, rand_sorted, cur_frac, is_sorted=True)

        if cur_best is None or efb > cur_best:
            cur_best = efb

    return float(cur_best) if cur_best is not None else 0.0


def evaluate_group(y_true: np.ndarray, y_score: np.ndarray, y_rank: np.ndarray) -> dict[str, float] | None:
    n = len(y_true)
    n_pos = int(y_true.sum())

    if n < 2 or n_pos == 0 or n_pos == n:
        return None

    order = _sorted_indices_by_rank(y_rank)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    rank_score = -y_rank.astype(np.float64)

    # RDKit expects the list to already be in descending ranking order.
    # The active/inactive flag is stored in column 0 here.
    rdkit_rows = [[int(y_true_sorted[i]), float(y_score_sorted[i])] for i in range(n)]

    out: dict[str, float] = {}

    # Raw enrichment from RDKit, plus normalized enrichment.
    ef_values = CalcEnrichment(rdkit_rows, 0, FRACTIONS)

    for frac, ef in zip(FRACTIONS, ef_values):
        k = _top_k(n, frac)
        hits_top_k = int(y_true_sorted[:k].sum())

        out[f"ef@{int(frac * 100)}%"] = float(ef)
        out[f"nef@{int(frac * 100)}%"] = float(hits_top_k / min(n_pos, k))
        out[f"ndcg@{int(frac * 100)}%"] = float(ndcg_score([y_true], [rank_score], k=k))

    out["map"] = float(average_precision_score(y_true, y_score))
    out["err"] = _expected_reciprocal_rank(y_true_sorted)
    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["bedroc_a20"] = float(CalcBEDROC(rdkit_rows, 0, ALPHA))

    act_preds = y_score[y_true == 1]
    rand_preds = y_score[y_true == 0]
    out["efb@1%"] = _calc_efb(act_preds, rand_preds, 0.01)
    out["efb_max"] = _calc_best_efb(act_preds, rand_preds)

    return out


def _metrics_for_group(g: pl.DataFrame) -> pl.DataFrame:
    base = {col: None for col in METRIC_COLS}
    y_true = g.get_column("true_label").to_numpy()
    y_score = g.get_column("score").to_numpy()
    y_rank = g.get_column("rank").to_numpy()

    metrics = evaluate_group(y_true, y_score, y_rank)
    if metrics is not None:
        base.update(metrics)

    return pl.DataFrame(
        {
            "method": [g.item(0, "method")],
            "protein_id": [g.item(0, "protein_id")],
            **{k: [base[k]] for k in METRIC_COLS},
        }
    )


def summarize_by_method(by_query: pl.DataFrame) -> pl.DataFrame:
    agg_exprs = []
    for col in METRIC_COLS:
        agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))
    agg_exprs.extend(
        [
            pl.len().alias("n_queries"),
            pl.col("n_ligands").sum().alias("total_ligands"),
            pl.col("n_actives").sum().alias("total_actives"),
            pl.col("n_inactives").sum().alias("total_inactives"),
        ]
    )

    return (
        by_query.group_by("method")
        .agg(agg_exprs)
        .sort("method")
    )


def main() -> None:
    args = parse_args()

    pred = pl.read_parquet(args.predictions)
    required = {"method", "protein_id", "ligand_id", "true_label", "score", "rank"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    pred = pred.with_columns(
        [
            pl.col("true_label").cast(pl.Int8),
            pl.col("score").cast(pl.Float64),
            pl.col("rank").cast(pl.Int64),
        ]
    )

    label_check = (
        pred.group_by(["method", "protein_id", "ligand_id"])
        .agg(pl.col("true_label").n_unique().alias("n_labels"))
        .filter(pl.col("n_labels") > 1)
    )
    if label_check.height > 0:
        raise ValueError(
            "Found ligand groups with inconsistent labels within method/protein/ligand_id."
        )

    # Collapse multiple poses for each ligand into a single ligand-level score.
    ligand_pred = (
        pred.group_by(["method", "protein_id", "ligand_id"], maintain_order=True)
        .agg(
            [
                pl.col("true_label").first().alias("true_label"),
                pl.col("score").max().alias("score"),
                pl.col("rank").min().alias("rank"),
                pl.len().alias("n_poses"),
            ]
        )
    )

    counts_by_query = (
        ligand_pred.group_by(["method", "protein_id"], maintain_order=True)
        .agg(
            [
                pl.len().cast(pl.Int64).alias("n_ligands"),
                pl.col("true_label").sum().cast(pl.Int64).alias("n_actives"),
                (pl.len() - pl.col("true_label").sum()).cast(pl.Int64).alias("n_inactives"),
                pl.col("n_poses").sum().cast(pl.Int64).alias("n_poses_total"),
            ]
        )
    )

    metrics_by_query = (
        ligand_pred.group_by(["method", "protein_id"], maintain_order=True)
        .map_groups(_metrics_for_group)
    )

    by_query = (
        counts_by_query.join(
            metrics_by_query,
            on=["method", "protein_id"],
            how="inner",
        )
        .filter(pl.col("roc_auc").is_not_null())
    )

    if by_query.height == 0:
        raise ValueError("No evaluable method/protein_id groups were found in predictions.")

    summary = summarize_by_method(by_query)

    by_query.write_parquet(args.by_query_out, compression="zstd", statistics=True)
    summary.write_parquet(args.summary_out, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()