# Snakemake Plan

## Goal
Keep one small workflow that:
1. Builds the manifest.
2. Generates validation scores for baseline methods.
3. Evaluates ranking metrics per protein and summary metrics across proteins.

## Current DAG
1. prepare_data
2. score_methods
3. evaluate

## Inputs and Outputs
1. Input manifest: dataset/pdbscreen_manifest.parquet
2. Predictions: results/ranking_baselines/val_predictions.parquet
3. By-protein metrics: results/ranking_baselines/val_metrics_by_query.parquet
4. Summary metrics: results/ranking_baselines/val_metrics_summary.parquet

## Methods in Scope
1. lambdamart_ligand
2. tanimoto_random_ref
3. logistic_regression_ligand (planned)

## Metric Contract
Per-protein metrics:
1. ef@1%, ef@5%, ef@10%
2. nef@1%, nef@5%, nef@10%
3. ndcg@1%, ndcg@5%, ndcg@10%
4. roc_auc
5. bedroc_a20

Summary metrics:
1. mean and median for every metric above
2. n_queries

## Logistic Regression Addition Plan
1. Add a new scorer in workflow/scripts/score_methods.py named _logistic_regression_scores.
2. Reuse the existing ligand descriptor feature set from _feature_cols.
3. Cast boolean descriptor columns to numeric values exactly as in other methods.
4. Train on rows with non-null features using split=train.
5. Use sklearn LogisticRegression with:
- max_iter=1000
- random_state=seed
- class_weight=balanced
6. Score validation rows with predict_proba(... )[:, 1].
7. Merge scores back to validation rows by path, protein_id, and label to preserve row order.
8. Raise if merged score column has null values.
9. Add method handling in main:
- if logistic_regression_ligand in methods, append method predictions in the same schema.
10. Keep output schema unchanged:
- path, protein_id, true_label, method, score, split

## Config Update
In config.yaml, add logistic_regression_ligand to methods.

## Acceptance Checks
1. snakemake --cores 1 --forcerun score_methods evaluate completes.
2. val_predictions.parquet contains the new method label.
3. val_metrics_by_query.parquet contains rows for the new method.
4. val_metrics_summary.parquet contains mean and median columns for the new method.

## Non-goals
1. No new Snakemake rules.
2. No new output files.
3. No change to evaluator formulas.
# Snakemake Plan (Detailed + Code)

## 1) Objective
Create one reproducible, low-bloat workflow that:
1. Builds the enriched manifest.
2. Trains and scores ligand-only methods on train, predicts on val.
3. Evaluates ranking metrics in one shared evaluator.

Current scope:
1. lambdamart_ligand
2. tanimoto_random_ref

Deferred:
1. docking baselines
2. test-split reporting notebook

## 2) Design Constraints
1. Keep config minimal.
2. Keep Snakemake DAG small (3 rules).
3. Keep method logic extensible inside one scoring script.
4. Keep output schema fixed so downstream analysis does not change when methods change.

## 3) Workflow DAG
1. prepare_data
2. score_methods
3. evaluate

## 4) File Layout
1. Snakefile
2. config.yaml
3. workflow/scripts/score_methods.py
4. workflow/scripts/evaluate_predictions.py
5. results/ranking_baselines/... outputs

## 5) Minimal Config
Use only the fields below:

~~~yaml
seed: 42
manifest: dataset/pdbscreen_manifest.parquet
train_split: train
val_split: val
output_dir: results/ranking_baselines
methods:
  - lambdamart_ligand
  - tanimoto_random_ref
~~~

Notes:
1. query_col, label_col, and row_id_col remain code defaults to avoid config bloat.
2. Additional knobs can be added later only when needed.

## 6) Output Contract

### 6.1 Validation predictions
Path:
1. results/ranking_baselines/val_predictions.parquet

Columns:
1. path
2. protein_id
3. true_label
4. method
5. score
6. split

### 6.2 Summary metrics
Path:
1. results/ranking_baselines/val_metrics_summary.parquet

Columns:
1. method
2. ndcg@1%_mean
3. ndcg@1%_median
4. ndcg@5%_mean
5. ndcg@5%_median
6. ndcg@10%_mean
7. ndcg@10%_median
8. nef@1%_mean
9. nef@1%_median
10. nef@5%_mean
11. nef@5%_median
12. nef@10%_mean
13. nef@10%_median
14. roc_auc_mean
15. roc_auc_median
16. bedroc_a20_mean
17. bedroc_a20_median
18. n_queries

### 6.3 Per-target metrics
Path:
1. results/ranking_baselines/val_metrics_by_query.parquet

Columns:
1. method
2. protein_id
3. ndcg@1%
4. ndcg@5%
5. ndcg@10%
6. nef@1%
7. nef@5%
8. nef@10%
9. roc_auc
10. bedroc_a20

## 7) Method Definitions

### 7.1 lambdamart_ligand
1. Uses ligand-only descriptors from manifest.
2. Trains LightGBM LambdaRank on train split grouped by protein_id.
3. Produces ranking score for each validation row.

### 7.2 tanimoto_random_ref
1. Uses train active ligands as reference pool.
2. Picks one random reference per validation protein_id using fixed seed.
3. Scores each validation ligand by Tanimoto similarity to that reference.

## 8) Snakemake Code

### 8.1 Snakefile
~~~python
configfile: "config.yaml"


rule all:
  input:
    "results/ranking_baselines/val_predictions.parquet",
    "results/ranking_baselines/val_metrics_summary.parquet",
    "results/ranking_baselines/val_metrics_by_query.parquet"


rule prepare_data:
  output:
    config["manifest"]
  shell:
    """
    if [ -x .venv/bin/python ]; then
      PYTHON_BIN=.venv/bin/python bash bash_scripts/Manifest.sh
    else
      bash bash_scripts/Manifest.sh
    fi
    """


rule score_methods:
  input:
    config["manifest"]
  output:
    "results/ranking_baselines/val_predictions.parquet"
  params:
    seed=config["seed"],
    train_split=config["train_split"],
    val_split=config["val_split"],
    methods=",".join(config["methods"])
  shell:
    """
    if [ -x .venv/bin/python ]; then
      PYTHON_BIN=.venv/bin/python
    else
      PYTHON_BIN=python3
    fi

    "$PYTHON_BIN" workflow/scripts/score_methods.py \
      --manifest {input} \
      --out {output} \
      --seed {params.seed} \
      --train-split {params.train_split} \
      --val-split {params.val_split} \
      --methods {params.methods}
    """


rule evaluate:
  input:
    "results/ranking_baselines/val_predictions.parquet"
  output:
    summary="results/ranking_baselines/val_metrics_summary.parquet",
    by_query="results/ranking_baselines/val_metrics_by_query.parquet"
  shell:
    """
    if [ -x .venv/bin/python ]; then
      PYTHON_BIN=.venv/bin/python
    else
      PYTHON_BIN=python3
    fi

    "$PYTHON_BIN" workflow/scripts/evaluate_predictions.py \
      --predictions {input} \
      --summary-out {output.summary} \
      --by-query-out {output.by_query}
    """
~~~

### 8.2 score_methods.py
~~~python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from typing import Dict, List

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--manifest", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--train-split", default="train")
  p.add_argument("--val-split", default="val")
  p.add_argument("--methods", default="lambdamart_ligand,tanimoto_random_ref")
  return p.parse_args()


def feature_cols(df: pl.DataFrame) -> List[str]:
  cols = [
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
  return [c for c in cols if c in df.columns]


def group_sizes(df: pl.DataFrame, group_col: str) -> List[int]:
  return (
    df.group_by(group_col)
    .len()
    .sort(group_col)
    .get_column("len")
    .to_list()
  )


def mol_fp(smiles: str | None):
  if smiles is None:
    return None
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def tanimoto(a, b) -> float:
  if a is None or b is None:
    return 0.0
  return float(DataStructs.TanimotoSimilarity(a, b))


def lambdamart_scores(train_df: pl.DataFrame, val_df: pl.DataFrame, seed: int) -> np.ndarray:
  from lightgbm import LGBMRanker

  feats = feature_cols(train_df)
  if not feats:
    raise ValueError("No ligand descriptor columns found.")

  cast_expr = []
  for c in feats:
    if train_df.schema[c] == pl.Boolean:
      cast_expr.append(pl.col(c).cast(pl.Int8).alias(c))
  if cast_expr:
    train_df = train_df.with_columns(cast_expr)
    val_df = val_df.with_columns(cast_expr)

  tr = train_df.drop_nulls(feats + ["label", "protein_id"]).sort("protein_id")
  va = val_df.drop_nulls(feats + ["label", "protein_id"]).sort("protein_id")

  X_train = tr.select(feats).to_numpy()
  y_train = tr.get_column("label").to_numpy()
  g_train = group_sizes(tr, "protein_id")
  X_val = va.select(feats).to_numpy()

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
  scored = va.select(["path", "protein_id", "label"]).with_columns(pl.Series("score", val_scores))
  merged = val_df.join(scored, on=["path", "protein_id", "label"], how="left")
  return merged.get_column("score").fill_null(0.0).to_numpy()


def tanimoto_random_ref_scores(train_df: pl.DataFrame, val_df: pl.DataFrame, seed: int) -> np.ndarray:
  rng = random.Random(seed)

  train_active = train_df.filter((pl.col("label") == 1) & pl.col("ligand_smiles").is_not_null())
  global_refs = train_active.get_column("ligand_smiles").to_list()
  if not global_refs:
    return np.zeros(val_df.height, dtype=float)

  refs_by_query: Dict[str, List[str]] = {}
  for row in train_active.select(["protein_id", "ligand_smiles"]).iter_rows(named=True):
    refs_by_query.setdefault(row["protein_id"], []).append(row["ligand_smiles"])

  selected = {}
  for q in val_df.get_column("protein_id").unique().to_list():
    pool = refs_by_query.get(q, global_refs)
    selected[q] = pool[rng.randrange(len(pool))]

  ref_fp = {q: mol_fp(smi) for q, smi in selected.items()}
  ligand_fps = [mol_fp(smi) for smi in val_df.get_column("ligand_smiles").to_list()]
  query_vals = val_df.get_column("protein_id").to_list()
  return np.array([tanimoto(fp, ref_fp[q]) for fp, q in zip(ligand_fps, query_vals)], dtype=float)


def main() -> None:
  args = parse_args()
  methods = [m.strip() for m in args.methods.split(",") if m.strip()]

  df = pl.read_parquet(args.manifest)
  needed = {"path", "split", "protein_id", "label", "ligand_smiles"}
  miss = sorted(needed - set(df.columns))
  if miss:
    raise ValueError(f"Manifest missing required columns: {miss}")

  df = df.with_columns(
    [
      pl.col("label").cast(pl.Int8),
      pl.when(pl.col("ligand_smiles") == "")
      .then(None)
      .otherwise(pl.col("ligand_smiles"))
      .alias("ligand_smiles"),
    ]
  )

  train_df = df.filter(pl.col("split") == args.train_split)
  val_df = df.filter(pl.col("split") == args.val_split)

  if train_df.height == 0:
    raise ValueError("No training rows found.")
  if val_df.height == 0:
    raise ValueError("No validation rows found.")

  outputs = []
  if "lambdamart_ligand" in methods:
    s = lambdamart_scores(train_df, val_df, args.seed)
    outputs.append(
      val_df.select(["path", "protein_id", "label"]).with_columns(
        pl.lit("lambdamart_ligand").alias("method"),
        pl.Series("score", s),
      )
    )

  if "tanimoto_random_ref" in methods:
    s = tanimoto_random_ref_scores(train_df, val_df, args.seed)
    outputs.append(
      val_df.select(["path", "protein_id", "label"]).with_columns(
        pl.lit("tanimoto_random_ref").alias("method"),
        pl.Series("score", s),
      )
    )

  if not outputs:
    raise ValueError("No enabled methods produced predictions.")

  pred = pl.concat(outputs, how="vertical").with_columns(pl.lit(args.val_split).alias("split"))
  pred.write_parquet(args.out, compression="zstd", statistics=True)


if __name__ == "__main__":
  main()
~~~

### 8.3 evaluate_predictions.py
~~~python
#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from sklearn.metrics import ndcg_score


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--predictions", required=True)
  p.add_argument("--summary-out", required=True)
  p.add_argument("--by-query-out", required=True)
  return p.parse_args()


def ef_at_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
  n = len(y_true)
  if n == 0:
    return float("nan")
  n_top = max(1, int(np.ceil(frac * n)))
  order = np.argsort(-y_score)
  y_sorted = y_true[order]
  top_hits = y_sorted[:n_top].sum()
  total_hits = y_true.sum()
  if total_hits == 0:
    return float("nan")
  return float((top_hits / n_top) / (total_hits / n))


def eval_group(y_true: np.ndarray, y_score: np.ndarray) -> dict | None:
  if y_true.sum() == 0:
    return None
  y_true = y_true.astype(float)
  y_score = y_score.astype(float)
  return {
    "ndcg@10": float(ndcg_score([y_true], [y_score], k=10)),
    "ndcg@50": float(ndcg_score([y_true], [y_score], k=50)),
    "ef@1%": ef_at_fraction(y_true, y_score, 0.01),
    "ef@5%": ef_at_fraction(y_true, y_score, 0.05),
  }


def main() -> None:
  args = parse_args()

  pred = pl.read_parquet(args.predictions)
  needed = {"method", "protein_id", "label", "score"}
  miss = sorted(needed - set(pred.columns))
  if miss:
    raise ValueError(f"Predictions missing required columns: {miss}")

  rows = []
  for method in pred.get_column("method").unique().to_list():
    pm = pred.filter(pl.col("method") == method)
    for q in pm.get_column("protein_id").unique().to_list():
      pq = pm.filter(pl.col("protein_id") == q)
      m = eval_group(
        pq.get_column("label").to_numpy(),
        pq.get_column("score").to_numpy(),
      )
      if m is None:
        continue
      row = {"method": method, "protein_id": q}
      row.update(m)
      rows.append(row)

  by_query = pl.DataFrame(rows)
  summary = by_query.group_by("method").agg(
    [
      pl.col("ndcg@10").mean().alias("ndcg@10"),
      pl.col("ndcg@50").mean().alias("ndcg@50"),
      pl.col("ef@1%").mean().alias("ef@1%"),
      pl.col("ef@5%").mean().alias("ef@5%"),
      pl.len().alias("n_queries"),
    ]
  ).sort("method")

  by_query.write_parquet(args.by_query_out, compression="zstd", statistics=True)
  summary.write_parquet(args.summary_out, compression="zstd", statistics=True)


if __name__ == "__main__":
  main()
~~~

## 9) Run Commands
1. Prepare environment (once):

~~~bash
pip install snakemake polars numpy scikit-learn lightgbm rdkit
~~~

2. Run full workflow:

~~~bash
snakemake -j 1 --configfile config.yaml
~~~

3. Dry-run only:

~~~bash
snakemake -n -j 1 --configfile config.yaml
~~~

## 10) Future Extension (Without DAG Changes)
1. Add a new method adapter in score_methods.py.
2. Append method name to methods in config.
3. Keep evaluator untouched as long as prediction schema is unchanged.

## 11) Optional Pretrained Convention (No Config Growth)
1. Store artifacts under workdir/models/{method}.pkl.
2. In each adapter:
   1. if model file exists -> load and score.
   2. else -> train, save, score.
3. Add optional CLI switches later if needed:
   1. --force-train
   2. --force-load
