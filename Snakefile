configfile: "config.yaml"

from datetime import datetime, timezone


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")

RUN_ID = config.get("run_id") or _now_run_id()
OUTPUT_ROOT = config.get("output_root", "results/runs").rstrip("/")
RUN_DIR = f"{OUTPUT_ROOT}/{RUN_ID}"


def _dataset_configs() -> dict:
    datasets = config.get("datasets")
    if datasets:
        normalized = {}
        for name, cfg in datasets.items():
            normalized[name] = {
                "manifest": cfg["manifest"],
                "train_split": cfg.get("train_split", config.get("train_split", "train")),
                "val_split": cfg.get("val_split", config.get("val_split", "val")),
            }
        return normalized

    dataset_name = config.get("dataset_name", "bigbind")
    return {
        dataset_name: {
            "manifest": config["manifest"],
            "train_split": config.get("train_split", "train"),
            "val_split": config.get("val_split", "val"),
        }
    }


DATASETS = _dataset_configs()
DATASET_NAMES = list(DATASETS.keys())


def dcfg(dataset_name):
    return DATASETS[dataset_name]


def predictions_path(dataset_name):
    return f"{RUN_DIR}/_{dataset_name}_val_predictions.parquet"


def by_query_path(dataset_name):
    return f"{RUN_DIR}/_{dataset_name}_val_metrics_by_query.parquet"


def dataset_output_path(dataset_name):
    return f"{RUN_DIR}/{dataset_name}.parquet"


def summary_output_path():
    return f"{RUN_DIR}/summary.parquet"


def run_yaml_path():
    return f"{RUN_DIR}/run.yaml"


rule all:
    input:
        [dataset_output_path(d) for d in DATASET_NAMES],
        summary_output_path(),
        run_yaml_path()


rule score_methods:
    wildcard_constraints:
        dataset="|".join(DATASET_NAMES)
    input:
        manifest=lambda wildcards: dcfg(wildcards.dataset)["manifest"]
    output:
        temp(predictions_path("{dataset}"))
    params:
        seed=config["seed"],
        train_split=lambda wildcards: dcfg(wildcards.dataset)["train_split"],
        val_split=lambda wildcards: dcfg(wildcards.dataset)["val_split"],
        methods=",".join(config["methods"])
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/score_methods.py \
          --manifest {input.manifest} \
          --out {output} \
          --seed {params.seed} \
          --train-split {params.train_split} \
          --val-split {params.val_split} \
          --methods {params.methods}
        """


rule evaluate:
    wildcard_constraints:
        dataset="|".join(DATASET_NAMES)
    input:
        predictions_path("{dataset}")
    output:
        summary=dataset_output_path("{dataset}"),
        by_query=by_query_path("{dataset}")
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/evaluate.py per-query \
          --predictions {input} \
          --summary-out {output.summary} \
          --by-query-out {output.by_query}
        """


rule combine_dataset_summaries:
    input:
        [dataset_output_path(d) for d in DATASET_NAMES]
    output:
        summary=summary_output_path()
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/evaluate.py combine-datasets \
          --inputs {input} \
          --out {output.summary}
        """


rule write_run_yaml:
    output:
        run_yaml=run_yaml_path()
    params:
        run_id=RUN_ID,
        created_at=datetime.now(timezone.utc).isoformat(),
        output_root=OUTPUT_ROOT,
        datasets_yaml="\n".join([f"  - {name}" for name in DATASET_NAMES]),
        methods_yaml="\n".join([f"  - {method}" for method in config["methods"]]),
        per_dataset_yaml="\n".join(
            [
                f"  {name}:\n"
                f"    manifest: {dcfg(name)['manifest']}\n"
                f"    train_split: {dcfg(name)['train_split']}\n"
                f"    val_split: {dcfg(name)['val_split']}"
                for name in DATASET_NAMES
            ]
        )
    shell:
        """
        mkdir -p "$(dirname {output.run_yaml})"
        cat > {output.run_yaml} <<EOF
run_id: {params.run_id}
    created_at_utc: {params.created_at}
output_root: {params.output_root}
datasets:
{params.datasets_yaml}
methods:
{params.methods_yaml}
per_dataset:
{params.per_dataset_yaml}
EOF
        """
