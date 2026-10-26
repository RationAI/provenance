# rationai-provenance

PROV-O-aware provenance capture for MLflow runs: **who** ran **what** on **which data**.

Framework-agnostic — works with any MLflow run, Lightning or not, pathology or not.
Extracted from [rationai-mlkit](../mlkit), which now consumes this package and adds the
Lightning/PyTorch integration on top (`@autolog`, `ProvenanceCallback`, console capture).

## What it does

- **User registry** — `register_new_user` / `lookup_user_run`: register people once, link every run to an agent.
- **Dataset registry** — `register_dataset` / `lookup_dataset_run`: file sizes, mtimes, manifests, verification.
- **Environment snapshot** — `capture_environment`: Python packages, hardware (GPU/CPU), docker image, seeds.
- **PROV documents** — `log_provenance` attaches a W3C PROV-O JSON document (`prov.json` artifact) to the active MLflow run, covering inputs, outputs, splits, metrics and agents.

## Install

```bash
uv add "rationai-provenance @ git+https://gitlab.ics.muni.cz/rationai/digital-pathology/libraries/provenance.git"
```

## Usage

```python
import mlflow

from rationai.provenance import log_provenance

with mlflow.start_run(run_name="my-experiment"):
    log_provenance(train_df=df, data_root="data/")  # prov.json lands in the run's artifacts
```

For Lightning training, install the `lightning` extra and add the callbacks
to your trainer (they use mlkit's `MLFlowLogger` under the hood, but the core
works with any MLflow run):

```python
from rationai.provenance.lightning import ProvenanceCallback

trainer = Trainer(callbacks=[ProvenanceCallback()], logger=mlflow_logger)
```

See `examples/example_provenance_train.py` for a full mlkit integration.

## Dependencies

`mlflow`, `pandas`, `omegaconf`. `torch` is optional — GPU details are simply
omitted from the environment snapshot when it is not installed.

## Development

```bash
uv venv && uv sync
uv run pytest
uv run ruff check --no-fix
```
