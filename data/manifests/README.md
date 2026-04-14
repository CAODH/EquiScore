# PDBscreen Manifest Tools

This folder documents the manifest and feature materialization scripts:

- utils/generate_manifest.py
- utils/materialize_features.py

## Install dependencies

Use the lightweight dependency file in the repository root:

pip install -r requirements-manifest.txt

## Generate manifest

python utils/generate_manifest.py \
  --dataset-root ./PDBscreen \
  --out-parquet ./dataset/pdbscreen_manifest.parquet

## Run in one command

bash bash_scripts/Manifest.sh

The shell script now runs both steps back to back:

1. manifest generation
2. chemistry feature materialization

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
- ecfp_morgan_r2_2048_bits
- hbd
- hba
- tpsa
- molecular_weight
- rotatable_bonds
- ring_count
- aromatic_ring_count
- lipinski_compliant
- verber_compliant
- scaffold
