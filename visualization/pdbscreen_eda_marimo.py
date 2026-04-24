import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # PDBscreen Descriptor EDA (Manifest-Backed)

    This notebook reads precomputed chemistry features from:
    - `dataset/pdbscreen_manifest.parquet`

    Expected feature columns:
    - TPSA / molecular weight / rotatable bonds
    - HBD / HBA / Lipinski / Verber
    - Ring counts / aromatic ring counts
    - `ligand_smiles` (full molecule)
    - `scaffold` (Bemis-Murcko core, null for acyclic compounds)

    Leakage-safe defaults:
    1. Analyze only train split by default.
    2. Use val/test only for post-hoc reporting (not for feature decisions).
    3. Quantify scaffold overlap between train and holdout splits.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import polars as pl
    return Path, pl


@app.cell
def _(Path, pl):
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "dataset" / "pdbscreen_manifest.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run bash_scripts/Manifest.sh first."
        )

    manifest = pl.read_parquet(manifest_path)
    required = {
        "path",
        "split",
        "tpsa",
        "molecular_weight",
        "rotatable_bonds",
        "ring_count",
        "aromatic_ring_count",
        "hbd",
        "hba",
        "lipinski_compliant",
        "verber_compliant",
        "ligand_smiles",
        "scaffold",
    }
    missing = required.difference(set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    manifest = manifest.with_columns(
        [
            pl.when(pl.col("scaffold") == "").then(None).otherwise(pl.col("scaffold")).alias("scaffold"),
            pl.when(pl.col("ligand_smiles") == "")
            .then(None)
            .otherwise(pl.col("ligand_smiles"))
            .alias("ligand_smiles"),
        ]
    )
    return manifest, manifest_path


@app.cell
def _(manifest, manifest_path, mo, pl):
    total_rows = manifest.height
    scaffold_nulls = manifest.select(pl.col("scaffold").is_null().sum()).item()
    ligand_nulls = manifest.select(pl.col("ligand_smiles").is_null().sum()).item()
    mo.md(
        f"""
        **Manifest loaded:** `{manifest_path}`  
        **Rows:** {total_rows:,}  
        **Null scaffold:** {scaffold_nulls:,}  
        **Null ligand_smiles:** {ligand_nulls:,}
        """
    )
    return


@app.cell
def _(manifest, mo, pl):
    splits = sorted(manifest.select(pl.col("split").drop_nulls().unique()).to_series().to_list())
    default_splits = ["train"] if "train" in splits else (splits[:1] if splits else [])

    split_selector = mo.ui.multiselect(
        options=splits,
        value=default_splits,
        label="Splits to analyze (keep train-only for leakage-safe EDA)",
    )

    split_selector
    return (split_selector,)


@app.cell
def _(manifest, pl, split_selector):
    selected = split_selector.value
    if not selected:
        desc_df = manifest.head(0)
    else:
        desc_df = manifest.filter(pl.col("split").is_in(selected))
    return (desc_df,)


@app.cell
def _(desc_df):
    desc_df 
    return


@app.cell
def _(desc_df, mo, pl):
    if desc_df.height == 0:
        mo.md("Descriptor table is empty.")
        summary = pl.DataFrame()
    else:
        summary = (
            desc_df.group_by("split")
            .agg(
                [
                    pl.len().alias("n"),
                    pl.col("tpsa").mean().alias("tpsa_mean"),
                    pl.col("molecular_weight").mean().alias("mw_mean"),
                    pl.col("rotatable_bonds").mean().alias("rot_bonds_mean"),
                    pl.col("ring_count").mean().alias("ring_count_mean"),
                    pl.col("aromatic_ring_count").mean().alias("aromatic_ring_mean"),
                    pl.col("lipinski_compliant").cast(pl.Int8).mean().alias("lipinski_rate"),
                    pl.col("verber_compliant").cast(pl.Int8).mean().alias("verber_rate"),
                    pl.col("scaffold").drop_nulls().n_unique().alias("n_unique_scaffolds"),
                ]
            )
            .sort("split")
        )

    mo.md("## Descriptor Summary by Split")
    summary
    return


@app.cell
def _(desc_df, mo, pl):
    if desc_df.height == 0:
        examples = pl.DataFrame(schema={"path": pl.Utf8, "split": pl.Utf8, "ligand_smiles": pl.Utf8})
    else:
        examples = (
            desc_df.filter(pl.col("scaffold").is_null() & pl.col("ligand_smiles").is_not_null())
            .select(["path", "split", "ligand_smiles"])
            .head(20)
        )

    mo.md("## Examples: Null Scaffold with Valid Ligand SMILES")
    examples
    return


@app.cell
def _(desc_df, mo, pl):
    if desc_df.height == 0 or "train" not in desc_df.select("split").drop_nulls().to_series().to_list():
        overlap_df = pl.DataFrame(schema={"holdout_split": pl.Utf8, "overlap_rate_vs_train": pl.Float64})
    else:
        train_scaffolds = set(
            desc_df.filter(pl.col("split") == "train")
            .select(pl.col("scaffold").drop_nulls().unique())
            .to_series()
            .to_list()
        )

        overlap_rows = []
        for split in ["val", "test"]:
            holdout_scaffolds = set(
                desc_df.filter(pl.col("split") == split)
                .select(pl.col("scaffold").drop_nulls().unique())
                .to_series()
                .to_list()
            )
            if holdout_scaffolds:
                overlap = len(train_scaffolds & holdout_scaffolds) / len(holdout_scaffolds)
                overlap_rows.append({"holdout_split": split, "overlap_rate_vs_train": overlap})

        overlap_df = pl.DataFrame(overlap_rows) if overlap_rows else pl.DataFrame(schema={"holdout_split": pl.Utf8, "overlap_rate_vs_train": pl.Float64})

    mo.md("## Scaffold Overlap Check (Leakage Risk Indicator)")
    overlap_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Leakage-Safe Workflow

    1. Use this notebook on train split for all feature selection and threshold decisions.
    2. Keep val/test for locked, post-hoc reporting only.
    3. Do not tune descriptor filters after viewing test statistics.
    4. If scaffold overlap is high, prefer scaffold-aware splitting before model development.
    """)
    return


if __name__ == "__main__":
    app.run()
