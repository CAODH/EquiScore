"""BigBind EDA: train+val analysis including SNA."""

import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import marimo as mo

    import altair as alt
    import plotly.express as px

    return Path, mo, np, pl, plt, px, sns


@app.cell
def _(Path, pl):
    """Load BigBind train+val data."""
    manifest_path = Path("dataset/bigbind_manifest.parquet")
    df = pl.read_parquet(manifest_path)

    # Filter to train+val only
    df = df.filter(pl.col("split").is_in(["train", "val"]))
    return (df,)


@app.cell
def _(df, mo, pl):
    """Dataset overview: size, composition, splits."""
    total_rows = df.height
    unique_proteins = df.select("protein_id").n_unique()
    unique_ligands = df.select("ligand_smiles").n_unique()

    splits = df.group_by("split").agg(pl.len().alias("count")).sort("split")
    sna_dist = df.group_by("sna").agg(pl.len().alias("count"))
    actives = df.filter(pl.col("label") == 1).height
    inactives = df.filter(pl.col("label") == 0).height

    mo.md(f"""
    ## Dataset Overview
    - **Total rows**: {total_rows:,}
    - **Unique proteins**: {unique_proteins}
    - **Unique ligands**: {unique_ligands:,}
    - **Actives**: {actives:,} ({100*actives/total_rows:.1f}%)
    - **Inactives**: {inactives:,} ({100*inactives/total_rows:.1f}%)
    - **Ligands per protein**: {total_rows/unique_proteins}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ligand Descriptors
    """)
    return


@app.cell
def _(df, mo):
    mo.ui.dataframe(df)
    return


@app.cell
def _(df, mo, px):
    tpsa = mo.ui.plotly(
        px.violin(
            df.sample(100000),
            y="tpsa",
            box=True,     # keep summary
            points=False, # avoid huge payload
            width=650
        )
    )

    tpsa
    return


@app.cell
def _():
    return


@app.cell
def _(df, mo):
    """Interactive protein filter."""
    proteins = sorted(df.select("protein_id").unique().to_series().to_list())
    selected_protein = mo.ui.dropdown(proteins, label="Filter by protein (optional):")
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    def _():
        """Ligand descriptor statistics."""
        desc_cols = ["tpsa", "molecular_weight", "rotatable_bonds", "ring_count", 
        "aromatic_ring_count", "hbd", "hba"]

        desc_stats = df.select(desc_cols)
        return desc_stats


    _()
    return


@app.cell
def _(df, plt):
    def _():
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
        return plt.tight_layout()


    _()
    return


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
    return


@app.cell
def _(df, pl):
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

    miss_df
    return


@app.cell
def _(df, mo, pl):
    """Compliance rates."""
    total = df.height
    lipinski_compliant = df.filter(pl.col("lipinski_compliant") == True).height
    verber_compliant = df.filter(pl.col("verber_compliant") == True).height

    mo.md(f"""
    ## Compliance Rates
    - **Lipinski compliant**: {lipinski_compliant} ({100*lipinski_compliant/total:.1f}%)
    - **Verber compliant**: {verber_compliant} ({100*verber_compliant/total:.1f}%)
    """)
    return


@app.cell
def _(df, pl):
    """Assay type distribution."""
    assay_dist = df.group_by("standard_type").agg(pl.len().alias("count")).sort("count", descending=True)

    assay_dist
    return


@app.cell
def _(df, np, pl, plt):
    def _():
        """Assay type and standard_value distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Assay type counts
        assay_data = df.group_by("standard_type").agg(pl.len().alias("count")).sort("count", descending=True).to_pandas()
        assay_data.plot(x="standard_type", y="count", kind="barh", ax=axes[0], legend=False)
        axes[0].set_xlabel("Count")
        axes[0].set_ylabel("Assay Type")
        axes[0].set_title("Assay Type Distribution")
        axes[0].grid(True, alpha=0.3, axis="x")

        # Standard value distribution (non-null only)
        sv_data = df.filter(pl.col("standard_value").is_not_null()).get_column("standard_value").to_numpy()
        axes[1].hist(np.log10(sv_data + 1e-6), bins=40, edgecolor="k", alpha=0.7)
        axes[1].set_xlabel("log10(standard_value + 1e-6)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Standard Value Distribution (log scale)")
        axes[1].grid(True, alpha=0.3)
        return plt.tight_layout()


    _()
    return


@app.cell
def _(df, pl):
    """Assay metadata vs active label correlation."""
    df_check = df.filter(pl.col("standard_value").is_not_null())

    by_label = df_check.group_by("label").agg(
        pl.col("standard_value").mean().alias("mean_sv"),
        pl.col("standard_value").median().alias("median_sv"),
        pl.len().alias("count")
    )

    by_label
    return


@app.cell
def _(df, pl, plt):
    """SNA breakdown: compare SNA=false vs SNA=true."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Row counts by SNA and split
    sna_split = df.group_by(["sna", "split"]).agg(pl.len().alias("count")).to_pandas()
    sna_split_pivot = sna_split.pivot(index="split", columns="sna", values="count")
    sna_split_pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Rows by Split and SNA")
    axes[0].legend(title="SNA")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Active ratio by SNA
    sna_actives = df.group_by("sna").agg(
        pl.col("label").sum().alias("actives"),
        pl.len().alias("total")
    ).with_columns((pl.col("actives") / pl.col("total")).alias("active_ratio"))

    sna_actives_pd = sna_actives.to_pandas()
    sna_actives_pd.plot(x="sna", y="active_ratio", kind="bar", ax=axes[1], legend=False)
    axes[1].set_xlabel("SNA")
    axes[1].set_ylabel("Active Ratio")
    axes[1].set_title("Active Ratio by SNA")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    return


if __name__ == "__main__":
    app.run()
