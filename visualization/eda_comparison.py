"""Comparative EDA: BigBind vs PDBScreen (train+val)."""
import marimo

__all__ = ["app"]

app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    return pl, np, plt, Path


@app.cell
def _(pl, Path):
    """Load both datasets (train+val only)."""
    bb_path = Path("dataset/bigbind_manifest.parquet")
    pdb_path = Path("dataset/pdbscreen_manifest.parquet")
    
    bb_df = pl.read_parquet(bb_path).filter(pl.col("split").is_in(["train", "val"]))
    pdb_df = pl.read_parquet(pdb_path).filter(pl.col("split").is_in(["train", "val"]))
    
    return bb_df, pdb_df


@app.cell
def _(bb_df, pdb_df, marimo):
    """Size and composition comparison."""
    comparison = pl.DataFrame({
        "Metric": [
            "Total Rows",
            "Unique Proteins",
            "Unique Ligands",
            "Actives",
            "Inactives",
            "Active Ratio (%)",
        ],
        "BigBind": [
            bb_df.height,
            bb_df.select("protein_id").n_unique(),
            bb_df.select("ligand_smiles").n_unique(),
            bb_df.filter(pl.col("label") == 1).height,
            bb_df.filter(pl.col("label") == 0).height,
            f"{100 * bb_df.filter(pl.col('label') == 1).height / bb_df.height:.1f}%",
        ],
        "PDBScreen": [
            pdb_df.height,
            pdb_df.select("protein_id").n_unique(),
            pdb_df.select("ligand_smiles").n_unique(),
            pdb_df.filter(pl.col("label") == 1).height,
            pdb_df.filter(pl.col("label") == 0).height,
            f"{100 * pdb_df.filter(pl.col('label') == 1).height / pdb_df.height:.1f}%",
        ],
    })
    
    marimo.md(f"""
    ## Size & Composition
    {comparison.to_markdown()}
    """)


@app.cell
def _(bb_df, pdb_df, plt):
    """Ligand property distribution overlays."""
    desc_cols = ["tpsa", "molecular_weight", "rotatable_bonds", "hbd", "hba"]
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    
    for i, col in enumerate(desc_cols):
        bb_vals = bb_df.select(col).to_pandas()[col].dropna()
        pdb_vals = pdb_df.select(col).to_pandas()[col].dropna()
        
        axes[i].hist(bb_vals, bins=30, alpha=0.6, label="BigBind", edgecolor="k")
        axes[i].hist(pdb_vals, bins=30, alpha=0.6, label="PDBScreen", edgecolor="k")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


@app.cell
def _(bb_df, pdb_df, plt, pl):
    """Active ratio distribution per protein."""
    def get_protein_stats(df):
        return (
            df.group_by("protein_id")
            .agg([
                pl.len().alias("total"),
                (pl.col("label") == 1).sum().alias("actives"),
            ])
            .with_columns((pl.col("actives") / pl.col("total")).alias("active_ratio"))
        )
    
    bb_stats = get_protein_stats(bb_df)
    pdb_stats = get_protein_stats(pdb_df)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(bb_stats["active_ratio"].to_numpy(), bins=30, alpha=0.7, label="BigBind", edgecolor="k")
    axes[0].hist(pdb_stats["active_ratio"].to_numpy(), bins=30, alpha=0.7, label="PDBScreen", edgecolor="k")
    axes[0].set_xlabel("Active Ratio per Protein")
    axes[0].set_ylabel("Number of Proteins")
    axes[0].set_title("Class Balance Difficulty Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(bb_stats["total"].to_numpy(), bins=30, alpha=0.7, label="BigBind", edgecolor="k")
    axes[1].hist(pdb_stats["total"].to_numpy(), bins=30, alpha=0.7, label="PDBScreen", edgecolor="k")
    axes[1].set_xlabel("Ligands per Protein")
    axes[1].set_ylabel("Number of Proteins")
    axes[1].set_title("Protein Complexity Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


@app.cell
def _(bb_df, pdb_df, marimo):
    """Target (protein) overlap analysis."""
    bb_proteins = set(bb_df.select("protein_id").unique().to_series().to_list())
    pdb_proteins = set(pdb_df.select("protein_id").unique().to_series().to_list())
    
    overlap = len(bb_proteins & pdb_proteins)
    bb_only = len(bb_proteins - pdb_proteins)
    pdb_only = len(pdb_proteins - bb_proteins)
    
    marimo.md(f"""
    ## Target (Protein) Overlap
    - **Targets in both datasets**: {overlap}
    - **BigBind-only targets**: {bb_only}
    - **PDBScreen-only targets**: {pdb_only}
    - **Total unique targets**: {len(bb_proteins | pdb_proteins)}
    """)


@app.cell
def _(bb_df, pdb_df, marimo, pl):
    """Feature completeness comparison."""
    feat_cols = ["tpsa", "molecular_weight", "rotatable_bonds", "ring_count", 
                 "aromatic_ring_count", "hbd", "hba", "ligand_fingerprint"]
    
    bb_missing = {col: 100 * bb_df.select(col).null_count()[0, 0] / bb_df.height for col in feat_cols if col in bb_df.columns}
    pdb_missing = {col: 100 * pdb_df.select(col).null_count()[0, 0] / pdb_df.height for col in feat_cols if col in pdb_df.columns}
    
    comparison = pl.DataFrame({
        "feature": list(bb_missing.keys()),
        "BigBind_%_missing": list(bb_missing.values()),
        "PDBScreen_%_missing": [pdb_missing.get(f, None) for f in bb_missing.keys()],
    })
    
    marimo.md(f"""
    ## Feature Completeness
    {comparison.to_markdown()}
    """)
