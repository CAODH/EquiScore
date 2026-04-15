#!/usr/bin/env python3
"""Evaluate ranking predictions on a validation split."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import polars as pl
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment
from sklearn.metrics import ndcg_score, roc_auc_score


FRACTIONS = [0.01, 0.05, 0.10]
ALPHA = 20.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--summary-out", required=True)
    p.add_argument("--by-query-out", required=True)
    return p.parse_args()


def _top_k(n: int, frac: float) -> int:
    return max(1, int(np.ceil(frac * n)))


def _sorted_indices_desc(y_score: np.ndarray) -> np.ndarray:
    return np.argsort(-y_score, kind="mergesort")


def evaluate_group(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float] | None:
    n = len(y_true)
    n_pos = int(y_true.sum())

    if n < 2 or n_pos == 0 or n_pos == n:
        return None

    order = _sorted_indices_desc(y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

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
        out[f"ndcg@{int(frac * 100)}%"] = float(ndcg_score([y_true], [y_score], k=k))

    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["bedroc_a20"] = float(CalcBEDROC(rdkit_rows, 0, ALPHA))

    return out


def summarize_by_method(by_query: pl.DataFrame) -> pl.DataFrame:
    metric_cols = [
        "ef@1%",
        "ef@5%",
        "ef@10%",
        "nef@1%",
        "nef@5%",
        "nef@10%",
        "ndcg@1%",
        "ndcg@5%",
        "ndcg@10%",
        "roc_auc",
        "bedroc_a20",
    ]

    agg_exprs = []
    for col in metric_cols:
        agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))
    agg_exprs.append(pl.len().alias("n_queries"))

    return (
        by_query.group_by("method")
        .agg(agg_exprs)
        .sort("method")
    )


def main() -> None:
    args = parse_args()

    pred = pl.read_parquet(args.predictions)
    required = {"method", "protein_id", "true_label", "score"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    rows: list[dict[str, Any]] = []

    for (method, protein_id), g in pred.group_by(["method", "protein_id"], maintain_order=True):
        y_true = g.get_column("true_label").to_numpy()
        y_score = g.get_column("score").to_numpy()

        metrics = evaluate_group(y_true, y_score)
        if metrics is None:
            continue

        rows.append(
            {
                "method": method,
                "protein_id": protein_id,
                **metrics,
            }
        )

    if not rows:
        raise ValueError("No evaluable method/protein_id groups were found in predictions.")

    by_query = pl.DataFrame(rows)
    summary = summarize_by_method(by_query)

    by_query.write_parquet(args.by_query_out, compression="zstd", statistics=True)
    summary.write_parquet(args.summary_out, compression="zstd", statistics=True)


if __name__ == "__main__":
    main()