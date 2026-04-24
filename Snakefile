configfile: "config.yaml"

from datetime import datetime, timezone


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")


def _as_sna_list(values):
    return [str(v).lower() for v in values]


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
                "sna_values": _as_sna_list(cfg.get("sna_values", config.get("sna_values", ["false", "true"]))),
            }
        return normalized

    dataset_name = config.get("dataset_name", "bigbind")
    return {
        dataset_name: {
            "manifest": config["manifest"],
            "train_split": config.get("train_split", "train"),
            "val_split": config.get("val_split", "val"),
            "sna_values": _as_sna_list(config.get("sna_values", ["false", "true"])),
        }
    }


DATASETS = _dataset_configs()
DATASET_NAMES = list(DATASETS.keys())
ALL_SNA_VALUES = sorted({s for d in DATASET_NAMES for s in DATASETS[d]["sna_values"]})
SNA_REGEX = "|".join(ALL_SNA_VALUES)


def dcfg(dataset_name):
    return DATASETS[dataset_name]


def dataset_sna_values(dataset_name):
    return dcfg(dataset_name)["sna_values"]


def predictions_path(dataset_name, sna):
    return f"{RUN_DIR}/_{dataset_name}_val_predictions_sna_{sna}.parquet"


def summary_path(dataset_name, sna):
    return f"{RUN_DIR}/_{dataset_name}_val_metrics_summary_sna_{sna}.parquet"


def by_query_path(dataset_name, sna):
    return f"{RUN_DIR}/_{dataset_name}_val_metrics_by_query_sna_{sna}.parquet"


def intermediate_output_path(dataset_name):
    return f"{RUN_DIR}/{dataset_name}_intermediate.parquet"


def dataset_output_path(dataset_name):
    return f"{RUN_DIR}/{dataset_name}.parquet"


def summary_output_path():
    return f"{RUN_DIR}/summary.parquet"


def run_yaml_path():
    return f"{RUN_DIR}/run.yaml"


rule all:
    input:
        [intermediate_output_path(d) for d in DATASET_NAMES],
        [dataset_output_path(d) for d in DATASET_NAMES],
        summary_output_path(),
        run_yaml_path()


rule score_methods:
    wildcard_constraints:
        dataset="|".join(DATASET_NAMES),
        sna=SNA_REGEX
    input:
        manifest=lambda wildcards: dcfg(wildcards.dataset)["manifest"]
    output:
        temp(f"{RUN_DIR}/_{{dataset}}_val_predictions_sna_{{sna}}.parquet")
    params:
        seed=config["seed"],
        train_split=lambda wildcards: dcfg(wildcards.dataset)["train_split"],
        val_split=lambda wildcards: dcfg(wildcards.dataset)["val_split"],
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


rule combine_intermediate:
    wildcard_constraints:
        dataset="|".join(DATASET_NAMES)
    input:
        lambda wildcards: [predictions_path(wildcards.dataset, s) for s in dataset_sna_values(wildcards.dataset)]
    output:
        f"{RUN_DIR}/{{dataset}}_intermediate.parquet"
    params:
        dataset=lambda wildcards: wildcards.dataset
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/combine_dataset_intermediates.py \
          --inputs {input} \
          --out {output} \
          --dataset {params.dataset}
        """


rule evaluate:
    wildcard_constraints:
        dataset="|".join(DATASET_NAMES),
        sna=SNA_REGEX
    input:
        f"{RUN_DIR}/_{{dataset}}_val_predictions_sna_{{sna}}.parquet"
    output:
        summary=temp(f"{RUN_DIR}/_{{dataset}}_val_metrics_summary_sna_{{sna}}.parquet"),
        by_query=temp(f"{RUN_DIR}/_{{dataset}}_val_metrics_by_query_sna_{{sna}}.parquet")
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/evaluate.py per-query \
          --predictions {input} \
          --summary-out {output.summary} \
          --by-query-out {output.by_query}
        """


rule compare_sna:
    wildcard_constraints:
        dataset="|".join(DATASET_NAMES)
    input:
        summaries=lambda wildcards: [summary_path(wildcards.dataset, s) for s in dataset_sna_values(wildcards.dataset)]
    output:
        f"{RUN_DIR}/{{dataset}}.parquet"
    params:
        labels=lambda wildcards: " ".join(dataset_sna_values(wildcards.dataset))
    shell:
        """
        PYTHON_BIN=venv/bin/python

        "$PYTHON_BIN" workflow/scripts/evaluate.py compare-sna \
          --summaries {input.summaries} \
          --labels {params.labels} \
          --out {output}
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
                f"    val_split: {dcfg(name)['val_split']}\n"
                + "    sna_values:\n"
                + "\n".join([f"      - {s}" for s in dcfg(name)["sna_values"]])
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
