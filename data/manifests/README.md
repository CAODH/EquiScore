# PDBscreen Manifest Tools

This folder documents the Polars-based manifest script at utils/generate_manifest.py.

## Install dependencies

Use the lightweight dependency file in the repository root:

pip install -r requirements-manifest.txt

## Generate manifest

python utils/generate_manifest.py \
  --dataset-root ./PDBscreen \
  --out-parquet ./PDBscreen/pdbscreen_manifest.parquet

## Run in one command

bash bash_scripts/Manifest.sh

## Output columns

The generated manifest includes:

- path
- filename
- source_folder
- data_quality
- sample_family
- protein_id
- pose_id
- label
- split
