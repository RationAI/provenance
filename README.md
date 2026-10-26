# rationai-provenance

Records **who** ran **what** on **which data** for every MLflow run — as tags,
params and artifacts. An external tool then assembles W3C PROV documents from
these records (this package deliberately does not generate PROV-JSON itself).

Works with any MLflow run — Lightning or not.

## Install

```bash
uv add "rationai-provenance @ git+ssh://git@github.com/RationAI/provenance.git"
```

## Setup

Two environment variables (in cluster jobs these are set for you):

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server   # where runs are logged
export MLFLOW_USER=jdoe                                # your username
```

## Quick start — 3 steps

### 1. Register yourself (once per person)

```python
from rationai.provenance import register_new_user

register_new_user(
    username="jdoe",
    real_name="Jane Doe",
    email="jdoe@example.org",
    organization="RationAI",
    lead_name="Prof. Lead",
    lead_email="lead@example.org",
)
```

### 2. Register your dataset (once per dataset version)

```python
from rationai.provenance import register_dataset

register_dataset(dataset_dir="/data/my-dataset", dataset_name="my-dataset", version="1.0.0")
```

Records every file's size and mtime (plus its manifest, if present) into the
`Dataset_Registry` MLflow experiment.

### 3. Verify + log in your training script

```python
import mlflow

from rationai.provenance import log_provenance, verify_dataset

verification = verify_dataset(manifest_path="/data/my-dataset/manifest.csv",
                              data_root="/data/my-dataset")

with mlflow.start_run(run_name="my-experiment"):
    log_provenance(config=cfg, name="my-dataset")  # the one call
    mlflow.set_tags({
        "dataset_verified": str(verification["verified"]),
        "dataset_run_id": str(verification["dataset_run_id"]),
    })
    # ... your training ...
```

`log_provenance` needs an **active MLflow run**. Its `config` argument takes any
OmegaConf/DictConfig or plain dict — your whole config is flattened into
`cfg_*` params (interpolations resolved). If you have a `train_df`, pass
`output=train_df` to also log split stats and frame manifests.

## Lightning + Hydra apps

If your repo uses [rationai-mlkit](https://github.com/RationAI/mlkit) (the standard
RationAI training stack), everything above is automatic — two additions:

**1. Decorate your hydra main** (creates the MLflow run, logs config, hyperparams
and console stream):

```python
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

@hydra.main(config_path="../configs", config_name="project_name")
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    ...
    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    trainer.fit(model, datamodule=data)
```

**2. Add the callback to your trainer config** (verifies the dataset, snapshots
the environment, writes `run_summary.json`; `fail_fast: true` aborts training if
verification fails):

```yaml
trainer:
  callbacks:
    - _target_: rationai.mlkit.lightning.callbacks.ProvenanceCallback
      fail_fast: true
```

Lightning without mlkit? Install the `lightning` extra
(`uv add 'rationai-provenance[lightning]'`) and add the callbacks from
`rationai.provenance.lightning` (`ProvenanceCallback`, `DatasetVerificationCallback`,
`EnvironmentCallback`) to your trainer directly.

## What lands in MLflow per run

| Where | What |
| --- | --- |
| tags | user + dataset run links, git commit, docker image, hardware, seeds, `job_*` (CPU/memory/GPU limits, cluster) |
| params | your whole config (`cfg_*`), dataset verification counts |
| artifacts | `requirements_frozen.txt`, split CSVs, frame manifests, `run_summary.json` |

Inside [kube-jobs](https://gitlab.ics.muni.cz/rationai/infrastructure/kube-jobs),
the `job_*` tags are picked up automatically from the injected `JOB_*` env vars.

A full end-to-end example (create → register → verify + train, cluster job
included): [`ulcerative-colitis-dysplasia`](https://github.com/jiribuchta/ulcerative-colitis-dysplasia),
branch `feature/provenance-demo`, module `provenance_demo.py`.

## Dependencies

`mlflow`, `pandas`, `omegaconf`. `torch` is optional — GPU details are omitted
from the environment snapshot when it is not installed.

## Development

```bash
uv venv && uv sync
uv run pytest
uv run ruff check --no-fix
```
