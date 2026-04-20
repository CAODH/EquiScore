configfile: "config.yaml"

OUTPUT_DIR = config.get("output_dir", "results/ranking_baselines").rstrip("/")
SNA_VALUES = [str(v).lower() for v in config.get("sna_values", ["false", "true"])]
MANIFEST = config["manifest"]
MANIFEST_SUMMARY = config.get("manifest_summary", "dataset/bigbind_manifest_summary.parquet")
MANIFEST_FEATURE_STAMP = config.get("manifest_feature_stamp", "dataset/bigbind_manifest.features.done")


def predictions_path(sna):
    return f"{OUTPUT_DIR}/val_predictions_sna_{sna}.parquet"


def summary_path(sna):
    return f"{OUTPUT_DIR}/val_metrics_summary_sna_{sna}.parquet"


def by_query_path(sna):
    return f"{OUTPUT_DIR}/val_metrics_by_query_sna_{sna}.parquet"


COMPARISON = f"{OUTPUT_DIR}/val_metrics_sna_comparison.parquet"


rule all:
    input:
        [predictions_path(s) for s in SNA_VALUES],
        [summary_path(s) for s in SNA_VALUES],
        [by_query_path(s) for s in SNA_VALUES],
        COMPARISON


rule build_manifest:
    output:
        manifest=MANIFEST,
        summary=MANIFEST_SUMMARY
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" utils/generate_big_bind.py \
          --out-parquet {output.manifest} \
          --summary-parquet {output.summary}
        """


rule add_features:
    input:
        manifest=MANIFEST
    output:
        MANIFEST_FEATURE_STAMP
    params:
        feature_workers=config.get("feature_workers", 8),
        feature_chunksize=config.get("feature_chunksize", 200)
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" utils/features.py \
          --manifest {input.manifest} \
          --workers {params.feature_workers} \
          --chunksize {params.feature_chunksize}

        touch {output}
        """


rule score_methods:
    input:
        manifest=MANIFEST,
        features=MANIFEST_FEATURE_STAMP
    output:
        f"{OUTPUT_DIR}/val_predictions_sna_{{sna}}.parquet"
    params:
        seed=config["seed"],
        train_split=config["train_split"],
        val_split=config["val_split"],
        methods=",".join(config["methods"]),
        sna=lambda wildcards: wildcards.sna
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/score_methods.py \
          --manifest {input.manifest} \
          --out {output} \
          --seed {params.seed} \
          --train-split {params.train_split} \
          --val-split {params.val_split} \
          --sna {params.sna} \
          --methods {params.methods}
        """


rule evaluate:
    input:
        f"{OUTPUT_DIR}/val_predictions_sna_{{sna}}.parquet"
    output:
        summary=f"{OUTPUT_DIR}/val_metrics_summary_sna_{{sna}}.parquet",
        by_query=f"{OUTPUT_DIR}/val_metrics_by_query_sna_{{sna}}.parquet"
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/evaluate_predictions.py \
          --predictions {input} \
          --summary-out {output.summary} \
          --by-query-out {output.by_query}
        """


rule compare_sna:
    input:
        summaries=[summary_path(s) for s in SNA_VALUES]
    output:
        COMPARISON
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/compare_sna_summaries.py \
          --summaries {input.summaries} \
          --out {output}
        """
