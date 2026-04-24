"""PDBScreen EDA: train+val analysis."""
import marimo

__all__ = ["app"]

app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    return pl, np, plt, sns, Path


@app.cell
def _(pl, Path):
    """Load PDBScreen train+val data."""
    manifest_path = Path("dataset/pdbscreen_manifest.parquet")
    df = pl.read_parquet(manifest_path)
    
    # Filter to train+val only
    df = df.filter(pl.col("split").is_in(["train", "val"]))
    
    return df


@app.cell
def _(df, marimo):
    """Dataset overview: size, composition, splits."""
    total_rows = df.height
    unique_proteins = df.select("protein_id").n_unique()
    unique_ligands = df.select("ligand_smiles").n_unique()
    
    splits = df.group_by("split").agg(pl.len().alias("count")).sort("split")
    actives = df.filter(pl.col("label") == 1).height
    inactives = df.filter(pl.col("label") == 0).height
    
    marimo.md(f"""
    ## Dataset Overview
    - **Total rows**: {total_rows:,}
    - **Unique proteins**: {unique_proteins}
    - **Unique ligands**: {unique_ligands:,}
    - **Actives**: {actives:,} ({100*actives/total_rows:.1f}%)
    - **Inactives**: {inactives:,} ({100*inactives/total_rows:.1f}%)
    
    ### Splits
    {splits.to_markdown()}
    """)


@app.cell
def _(df, marimo):
    """Interactive protein filter."""
    proteins = sorted(df.select("protein_id").unique().to_series().to_list())
    selected_protein = marimo.ui.dropdown(proteins, label="Filter by protein (optional):")
    return selected_protein


@app.cell
def _(df, selected_protein):
    """Active/inactive ratio per protein."""
    protein_stats = (
        df.group_by("protein_id")
        .agg([
            pl.len().alias("total"),
            (pl.col("label") == 1).sum().alias("actives"),
            (pl.col("label") == 0).sum().alias("inactives"),
        ])
        .with_columns((pl.col("actives") / pl.col("total")).alias("active_ratio"))
        .sort("active_ratio", descending=True)
    )
    
    if selected_protein.value is not None:
        protein_stats = protein_stats.filter(pl.col("protein_id") == selected_protein.value)
    
    return protein_stats


@app.cell
def _(protein_stats, plt):
    """Visualize active ratio and ligand coverage per protein."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(protein_stats["active_ratio"].to_numpy(), bins=30, edgecolor="k", alpha=0.7)
    axes[0].set_xlabel("Active Ratio")
    axes[0].set_ylabel("Number of Proteins")
    axes[0].set_title("Active/Inactive Balance per Protein")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(protein_stats["total"].to_numpy(), bins=30, edgecolor="k", alpha=0.7)
    axes[1].set_xlabel("Ligands per Protein")
    axes[1].set_ylabel("Number of Proteins")
    axes[1].set_title("Ligand Coverage Distribution")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


@app.cell
def _(df, marimo):
    """Ligand descriptor statistics."""
    desc_cols = ["tpsa", "molecular_weight", "rotatable_bonds", "ring_count", 
                 "aromatic_ring_count", "hbd", "hba"]
    
    desc_stats = df.select(desc_cols).describe().to_pandas()
    
    marimo.md(f"""
    ## Ligand Descriptors
    {desc_stats.to_markdown()}
    """)


@app.cell
def _(df, plt):
    """Visualize descriptor distributions."""
    desc_cols = ["tpsa", "molecular_weight", "rotatable_bonds", "ring_count", 
                 "aromatic_ring_count", "hbd", "hba"]
    
    df_desc = df.select(desc_cols).to_pandas()
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    
    for i, col in enumerate(desc_cols):
        axes[i].hist(df_desc[col].dropna(), bins=30, edgecolor="k", alpha=0.7)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].axis("off")
    plt.tight_layout()
    return fig


@app.cell
def _(df, plt, sns):
    """Descriptor correlations heatmap."""
    desc_cols = ["tpsa", "molecular_weight", "rotatable_bonds", "ring_count", 
                 "aromatic_ring_count", "hbd", "hba"]
    
    corr_matrix = df.select(desc_cols).to_pandas().corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Descriptor Correlations")
    plt.tight_layout()
    return fig


@app.cell
def _(df, pl, marimo):
    """Feature missingness."""
    feat_cols = ["tpsa", "molecular_weight", "rotatable_bonds", "ring_count", 
                 "aromatic_ring_count", "hbd", "hba", "lipinski_compliant", 
                 "verber_compliant", "ligand_fingerprint", "scaffold"]
    
    missingness = {}
    for col in feat_cols:
        if col in df.columns:
            null_count = df.select(col).null_count()[0, 0]
            missingness[col] = 100 * null_count / df.height
    
    miss_df = pl.DataFrame({
        "feature": list(missingness.keys()),
        "missing_%": list(missingness.values())
    }).sort("missing_%", descending=True)
    
    marimo.md(f"""
    ## Feature Completeness
    {miss_df.to_markdown()}
    """)


@app.cell
def _(df, marimo, pl):
    """Compliance rates."""
    total = df.height
    lipinski_compliant = df.filter(pl.col("lipinski_compliant") == True).height
    verber_compliant = df.filter(pl.col("verber_compliant") == True).height
    
    marimo.md(f"""
    ## Compliance Rates
    - **Lipinski compliant**: {lipinski_compliant} ({100*lipinski_compliant/total:.1f}%)
    - **Verber compliant**: {verber_compliant} ({100*verber_compliant/total:.1f}%)
    """)
