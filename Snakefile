configfile: "config.yaml"

OUTPUT_DIR = config.get("output_dir", "results/ranking_baselines").rstrip("/")
PREDICTIONS = f"{OUTPUT_DIR}/val_predictions.parquet"
SUMMARY = f"{OUTPUT_DIR}/val_metrics_summary.parquet"
BY_QUERY = f"{OUTPUT_DIR}/val_metrics_by_query.parquet"


rule all:
    input:
        PREDICTIONS,
        SUMMARY,
        BY_QUERY


rule prepare_data:
    output:
        config["manifest"]
    shell:
        """
        PYTHON_BIN=venv/bin/python bash bash_scripts/Manifest.sh
        """


rule score_methods:
    input:
        config["manifest"]
    output:
        PREDICTIONS
    params:
        seed=config["seed"],
        train_split=config["train_split"],
        val_split=config["val_split"],
        methods=",".join(config["methods"])
    shell:
        """
        PYTHON_BIN=venv/bin/python

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
        PREDICTIONS
    output:
        summary=SUMMARY,
        by_query=BY_QUERY
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/evaluate_predictions.py \
          --predictions {input} \
          --summary-out {output.summary} \
          --by-query-out {output.by_query}
        """
