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
    # PDBscreen Descriptor EDA (Leakage-Safe)

    This notebook computes ligand descriptors from pocket pickle files:
    - TPSA
    - Molecular weight
    - Lipinski compliance
    - Verber compliance
    - Rotatable bonds
    - Scaffold (Bemis-Murcko)
    - Ring count and aromaticity

    Leakage-safe defaults:
    1. Analyze only train split by default.
    2. Use val/test only for post-hoc reporting (not for feature decisions).
    3. Quantify scaffold overlap between train and holdout splits.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import pickle

    import polars as pl
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return (
        Crippen,
        Descriptors,
        Lipinski,
        MurckoScaffold,
        Path,
        pickle,
        pl,
        rdMolDescriptors,
    )


@app.cell
def _(Path, pl):
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "PDBscreen" / "pdbscreen_manifest.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run bash_scripts/Manifest.sh first."
        )

    manifest = pl.read_parquet(manifest_path)
    required = {"path", "split"}
    missing = required.difference(set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return manifest, repo_root


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
        selected_manifest = manifest.head(0)
    else:
        selected_manifest = manifest.filter(pl.col("split").is_in(selected))
    return (selected_manifest,)


@app.cell
def _(
    Crippen,
    Descriptors,
    Lipinski,
    MurckoScaffold,
    pickle,
    rdMolDescriptors,
    repo_root,
    selected_manifest,
):
    def _load_ligand(path_str: str):
        abs_path = repo_root / path_str
        try:
            with open(abs_path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            return None

        if isinstance(payload, tuple) and len(payload) >= 2:
            return payload[0]
        return None

    def _descriptor_row(path_str: str, split):
        mol = _load_ligand(path_str)
        if mol is None:
            return None

        try:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            mw = Descriptors.MolWt(mol)
            rot_bonds = Lipinski.NumRotatableBonds(mol)
            ring_count = rdMolDescriptors.CalcNumRings(mol)
            aromatic_ring_count = rdMolDescriptors.CalcNumAromaticRings(mol)
            aromatic_atom_count = sum(int(a.GetIsAromatic()) for a in mol.GetAtoms())
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            return None

        lipinski_ok = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
        verber_ok = (rot_bonds <= 10) and (tpsa <= 140)

        return {
            "path": path_str,
            "split": split,
            "tpsa": tpsa,
            "molecular_weight": mw,
            "rotatable_bonds": rot_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aromatic_atom_count": aromatic_atom_count,
            "logp": logp,
            "hbd": hbd,
            "hba": hba,
            "lipinski_compliant": lipinski_ok,
            "verber_compliant": verber_ok,
            "scaffold": scaffold,
        }

    descriptor_rows = []
    for row in selected_manifest.select(["path", "split"]).iter_rows(named=True):
        desc = _descriptor_row(row["path"], row["split"])
        if desc is not None:
            descriptor_rows.append(desc)
    return (descriptor_rows,)


@app.cell
def _(descriptor_rows, mo, pl):
    if not descriptor_rows:
        desc_df = pl.DataFrame(
            schema={
                "path": pl.Utf8,
                "split": pl.Utf8,
                "tpsa": pl.Float64,
                "molecular_weight": pl.Float64,
                "rotatable_bonds": pl.Int64,
                "ring_count": pl.Int64,
                "aromatic_ring_count": pl.Int64,
                "aromatic_atom_count": pl.Int64,
                "logp": pl.Float64,
                "hbd": pl.Int64,
                "hba": pl.Int64,
                "lipinski_compliant": pl.Boolean,
                "verber_compliant": pl.Boolean,
                "scaffold": pl.Utf8,
            }
        )
        mo.md("No descriptor rows were computed. Check selected paths and file availability.")
    else:
        desc_df = pl.DataFrame(descriptor_rows)
    return (desc_df,)


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
                    pl.col("scaffold").n_unique().alias("n_unique_scaffolds"),
                ]
            )
            .sort("split")
        )

    mo.md("## Descriptor Summary by Split")
    summary
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
