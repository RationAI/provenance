"""Universal provenance logging for any pipeline run.

One entry point — ``log_provenance`` — logs everything that describes a run
to MLflow regardless of its type (mask creation, tiling, filtering,
embeddings, training, inference, …). The PROV-O document itself is built
by an external tool from these records.

* environment + hardware + git + seeds (via ``capture_environment``)
* flattened scalar config leaves as MLflow params
* output dataframe statistics (+ optional file manifest and label breakdown)
* upstream inputs (explicit or auto-detected ``mlflow-artifacts:/`` URIs in
  the config) — links preprocessing steps into a chain
* frame manifests as CSV artifacts

Every argument is optional; the function derives what it can from the
Hydra config and the active MLflow run.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from rationai.provenance.environment import capture_environment


log = logging.getLogger(__name__)


def safe_id(name: str) -> str:
    """Sanitise a name so it works as an MLflow param/tag fragment."""
    import re as _re

    return _re.sub(r"[^a-zA-Z0-9_]", "_", name)


# Columns probed (in order) when looking for file paths / labels.
_PATH_COLUMN_CANDIDATES = (
    "slide_path",
    "wsi_path",
    "image_path",
    "filepath",
    "path",
    "tile_path",
    "mask_path",
    "annot_path",
)
_LABEL_COLUMN_CANDIDATES = ("label", "annotation", "class", "diagnosis")

# Config paths probed (in order) for dataset name / version / label semantics.
_NAME_CONFIG_PATHS = ("dataset.name", "dataset_name")
_VERSION_CONFIG_PATHS = ("dataset.version", "dataset_version")
_POSITIVE_LABEL_CONFIG_PATHS = ("dataset.positive_label", "positive_label")

# URI schemes recognised as references to upstream run artifacts.
_INPUT_URI_SCHEMES = ("mlflow-artifacts:/", "runs:/", "models:/")

_MAX_PARAMS = 200  # safety cap for flattened config leaves
_MAX_LABEL_CLASSES = 40  # per-class count params only for small cardinalities


def _select_config(config: Any, path: str) -> Any:
    """Read a dotted path from a Hydra config / dict, returning None if absent."""
    if config is None:
        return None
    try:
        if isinstance(config, DictConfig):
            return OmegaConf.select(config, path, default=None)
        node: Any = config
        for part in path.split("."):
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        return node
    except Exception:
        return None


def _select_any(config: Any, paths: tuple[str, ...]) -> Any:
    """First non-None value among candidate config paths."""
    for path in paths:
        value = _select_config(config, path)
        if value is not None:
            return value
    return None


def _flatten_scalars(node: Any, prefix: str, out: dict[str, str]) -> None:
    """Flatten config leaves into ``key.subkey: str(value)`` scalar params."""
    if len(out) >= _MAX_PARAMS:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten_scalars(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(node, (list, tuple)):
        if len(node) <= 5 and all(not isinstance(v, (dict, list, tuple)) for v in node):
            out[prefix] = json.dumps([str(v) for v in node])
    elif node is None:
        out[prefix] = "null"
    else:
        out[prefix] = str(node)


def _config_to_container(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(config, resolve=False)
        if isinstance(container, dict):
            return cast("dict[str, Any]", container)
        return {}
    if isinstance(config, dict):
        return config
    return {}


def _find_input_uris(node: Any, found: set[str]) -> None:
    """Collect upstream-artifact URI strings from config leaves."""
    if isinstance(node, dict):
        for value in node.values():
            _find_input_uris(value, found)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _find_input_uris(value, found)
    elif isinstance(node, str) and node.startswith(_INPUT_URI_SCHEMES):
        found.add(node)


def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _file_manifest(
    df: pd.DataFrame, path_column: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per-file manifest rows + size map for an existing path column."""
    rows: list[dict[str, Any]] = []
    sizes: dict[str, int] = {}
    for raw in df[path_column]:
        path_str = str(raw)
        basename = os.path.basename(path_str)
        fpath = Path(path_str)
        size = -1
        mtime = "unknown"
        if fpath.exists():
            try:
                stat = fpath.stat()
            except OSError:
                pass
            else:
                size = int(stat.st_size)
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        sizes[basename] = size
        rows.append(
            {
                "filename": basename,
                "path": path_str,
                "size_bytes": size,
                "modified_at": mtime,
            }
        )
    return rows, sizes


def _label_breakdown(
    df: pd.DataFrame,
    label_column: str,
    positive_label: str | None,
    frame_name: str,
) -> dict[str, int]:
    """Positive/negative counts, or per-class counts for small cardinalities."""
    prefix = f"{frame_name}_" if frame_name != "output" else ""
    if positive_label is not None:
        num_positive = int((df[label_column] != positive_label).sum())
        return {
            f"{prefix}num_positive": num_positive,
            f"{prefix}num_negative": len(df) - num_positive,
        }
    uniques = df[label_column].dropna().unique()
    if 1 < len(uniques) <= _MAX_LABEL_CLASSES:
        counts = df[label_column].value_counts()
        return {
            f"{prefix}label_{safe_id(str(k))}_count": int(v) for k, v in counts.items()
        }
    return {}


def _frame_stats(
    frames: dict[str, pd.DataFrame],
    *,
    positive_label: str | None,
    path_column: str | None,
    label_column: str | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    """Compute per-frame stats, manifests and file sizes."""
    stats: dict[str, Any] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    file_size_maps: dict[str, dict[str, int]] = {}
    for frame_name, df in frames.items():
        prefix = f"{frame_name}_" if frame_name != "output" else ""
        stats[f"{prefix}num_rows"] = len(df)
        stats[f"{prefix}num_columns"] = len(df.columns)
        stats[f"{prefix}columns"] = json.dumps([str(c) for c in df.columns])

        pcol = (
            path_column
            if path_column in df.columns
            else _detect_column(df, _PATH_COLUMN_CANDIDATES)
        )
        if pcol is not None:
            rows, sizes = _file_manifest(df, pcol)
            manifests[frame_name] = rows
            file_size_maps[frame_name] = sizes
            stats[f"{prefix}path_column"] = pcol
            stats[f"{prefix}files_existing"] = sum(1 for s in sizes.values() if s >= 0)

        lcol = (
            label_column
            if label_column in df.columns
            else _detect_column(df, _LABEL_COLUMN_CANDIDATES)
        )
        if lcol is not None:
            stats[f"{prefix}label_column"] = lcol
            stats.update(_label_breakdown(df, lcol, positive_label, frame_name))
    return stats, manifests, file_size_maps


# ──────────────────────────────────────────────
# Universal provenance entry point
# ──────────────────────────────────────────────


def log_provenance(
    output: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    logger: Any = None,
    config: DictConfig | dict[str, Any] | None = None,
    *,
    name: str | None = None,
    version: str | None = None,
    positive_label: str | None = None,
    path_column: str | None = None,
    label_column: str | None = None,
    inputs: list[str] | None = None,
    snapshot_env: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log end-to-end provenance for the current run to MLflow.

    Works for any pipeline step (dataset creation, masking, tiling,
    filtering, embeddings, training, inference, ...).  Everything is
    optional; values are derived from *config* and the active run whenever
    possible.

    Args:
        output: Produced dataframe, or mapping of split name → dataframe.
        logger: MLflow logger of the run (falls back to the active run).
        config: Hydra config / plain dict (used for name/version derivation,
            scalar param logging and upstream-URI detection).
        name: Output name (default: ``config.dataset.name`` →
            ``config.dataset_name`` → run name).
        version: Output version string (default: config value).
        positive_label: Value of the label column counting as the *negative*
            class; other values count as positive.  When None and the label
            column is categorical (≤ 40 classes), per-class counts are
            logged instead.
        path_column: Column with file paths (auto-detected from common
            names when omitted; file manifest skipped when no match).
        label_column: Column with labels (auto-detected likewise).
        inputs: Upstream references (``mlflow-artifacts:/...`` URIs or
            paths).  Merged with URIs auto-detected inside *config*.
        snapshot_env: Whether to snapshot the python environment.
        extra: Extra key/value pairs for the PROV metadata bundle.

    Returns:
        Dict with the computed stats and tags.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run — call inside @autolog")
    run_id = active_run.info.run_id

    # ── Environment (hardware, git, docker, seeds, freeze) ──
    capture_environment(snapshot_env=snapshot_env)

    # ── Derive identity ───────────────────────────────────
    run_name = active_run.data.tags.get("mlflow.runName", run_id)
    if name is None:
        raw_name = _select_any(config, _NAME_CONFIG_PATHS)
        name = str(raw_name) if raw_name is not None else str(run_name)
    if version is None:
        raw_version = _select_any(config, _VERSION_CONFIG_PATHS)
        version = str(raw_version) if raw_version is not None else None
    if positive_label is None:
        raw_label = _select_any(config, _POSITIVE_LABEL_CONFIG_PATHS)
        positive_label = str(raw_label) if raw_label is not None else None

    # ── Inputs: explicit + config-detected upstream URIs ──
    all_inputs: set[str] = set(inputs or [])
    _find_input_uris(_config_to_container(config), all_inputs)
    input_list = sorted(all_inputs)

    # ── Flatten scalar config leaves to params ────────────
    container = _config_to_container(config)
    if container:
        flat: dict[str, str] = {}
        _flatten_scalars(container, "", flat)
        if flat:
            mlflow.log_params(
                {f"cfg_{safe_id(k)}": v for k, v in list(flat.items())[:_MAX_PARAMS]}
            )

    # ── Output frame stats ────────────────────────────────
    frames: dict[str, pd.DataFrame] = {}
    if isinstance(output, pd.DataFrame):
        frames["output"] = output
    elif isinstance(output, dict):
        frames = {str(k): v for k, v in output.items()}

    stats, manifests, file_size_maps = _frame_stats(
        frames,
        positive_label=positive_label,
        path_column=path_column or "",
        label_column=label_column or "",
    )
    if stats:
        mlflow.log_params(
            {
                k: (v if isinstance(v, (int, float)) else str(v))
                for k, v in stats.items()
            }
        )

    # ── Tags ──────────────────────────────────────────────
    tags: dict[str, str] = {
        "output_name": name,
        "provenance_logged": "true",
    }
    if version:
        tags["output_version"] = version
    if input_list:
        tags["input_uris"] = json.dumps(input_list)
    mlflow.set_tags(tags)

    def _log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        if logger is not None:
            logger.log_artifact(local_path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        prov_dir = Path(tmpdir) / "provenance"
        prov_dir.mkdir(exist_ok=True)

        for frame_name, rows in manifests.items():
            manifest_csv = prov_dir / f"{frame_name}_manifest.csv"
            pd.DataFrame(rows).to_csv(manifest_csv, index=False)
            _log_artifact(str(manifest_csv), artifact_path="provenance")

    return {
        "name": name,
        "version": version,
        "inputs": input_list,
        "stats": stats,
        "file_sizes": file_size_maps,
    }
