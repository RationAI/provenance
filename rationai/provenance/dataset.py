"""Dataset registration and verification in MLflow.

``register_dataset`` captures per-file metadata so downstream provenance
tooling can inspect the exact state of the data; ``verify_dataset`` /
``verify_manifest`` re-check a dataset against its recorded state.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlflow
import pandas as pd


# ──────────────────────────────────────────────
# MLflow registration & verification helpers
# ──────────────────────────────────────────────


def lookup_experiment(name: str) -> str | None:
    """Return the MLflow experiment ID for *name*, or None."""
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id if exp else None


def lookup_dataset_run() -> str | None:
    """Return the latest Dataset_Registry run ID.

    Falls back to the most recent run so that workflows with only one
    registered dataset still work.
    """
    exp_id = lookup_experiment("Dataset_Registry")
    if exp_id is None:
        return None

    runs_df = mlflow.search_runs(
        experiment_ids=[exp_id],
        order_by=["start_time DESC"],
    )
    if pd.DataFrame(runs_df).empty:
        return None

    return pd.DataFrame(runs_df).iloc[0]["run_id"]


def detect_manifest() -> tuple[str | None, str | None]:
    """Walk data/ looking for manifest.csv.

    Returns (manifest_path, data_root) or (None, None).
    """
    for root_dir in ("data", "test_data", "."):
        for dirpath, _, filenames in os.walk(root_dir):
            if "manifest.csv" in filenames:
                return (
                    os.path.join(dirpath, "manifest.csv"),
                    os.path.dirname(
                        os.path.abspath(
                            os.path.join(dirpath, "manifest.csv"),
                        )
                    ),
                )
    return None, None


# ──────────────────────────────────────────────
# Manifest loading (shared with ProvenanceCallback)
# ──────────────────────────────────────────────


PATH_COLUMNS = ("wsi_path", "path", "slide", "slide_path", "file")
LABEL_COLUMNS = ("cancer", "label")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Column whose (stripped, lowercased) name is in *candidates*."""
    columns = {str(c).strip().lower(): c for c in df.columns}
    return next((columns[c] for c in candidates if c in columns), None)


def load_manifest(manifest_path: str, data_root: str) -> list[dict[str, Any]]:
    """Load a manifest.csv and resolve file paths.

    Only a path column is required — ``wsi_path`` / ``path`` / ``slide`` /
    ``file`` (matched case-insensitively). An optional ``cancer`` / ``label``
    column supplies integer labels (``0`` when absent — labels matter only
    for stratified splits / classification, not for registration or
    verification, which use file sizes + mtimes).

    Returns a list of dicts with keys ``path`` (absolute) and ``label``.

    Shared by ``register_dataset``, ``verify_dataset``, and
    ``ProvenanceCallback`` to avoid duplicating the CSV iteration pattern.
    """
    df = pd.read_csv(manifest_path)
    path_col = _pick_column(df, PATH_COLUMNS)
    if path_col is None:
        raise ValueError(
            f"{manifest_path}: no path column — name one of {PATH_COLUMNS}"
        )
    label_col = _pick_column(df, LABEL_COLUMNS)
    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rel = row[path_col]
        full = os.path.join(data_root, rel) if not os.path.isabs(rel) else rel
        label = int(row[label_col]) if label_col else 0
        samples.append({"path": full, "label": label})
    return samples


# ──────────────────────────────────────────────
# Dataset verification
# ──────────────────────────────────────────────


def verify_dataset(
    manifest_path: str | None = None,
    data_root: str | None = None,
) -> dict[str, Any]:
    """Public entry point — verify the current dataset against MLflow.

    Auto-detects the manifest if *manifest_path* is not given.

    Returns a dict with keys::

        {
            "verified": bool,
            "file_sizes_match": bool | None,
            "files_missing": int,
            "files_total": int,
            "details": list[str],
        }
    """
    if manifest_path is None:
        manifest_path, data_root = detect_manifest()

    if manifest_path is None:
        return {
            "verified": False,
            "dataset_run_id": None,
            "file_sizes_match": None,
            "files_missing": 0,
            "files_total": 0,
            "details": ["No manifest.csv found — skipping verification"],
        }

    if data_root is None:
        data_root = os.path.dirname(os.path.abspath(manifest_path))

    dataset_run_id = lookup_dataset_run()
    return verify_manifest(manifest_path, data_root, dataset_run_id)


def verify_manifest(
    manifest_path: str,
    data_root: str,
    dataset_run_id: str | None,
) -> dict[str, Any]:
    """Verify the current dataset against the registered version in MLflow.

    Checks:
      1. Per-file sizes match (file-level integrity)
      2. All WSI files exist on disk

    Returns a dict with verification results.
    """
    result: dict[str, Any] = {
        "verified": False,
        "dataset_run_id": dataset_run_id,
        "file_sizes_match": None,
        "files_missing": 0,
        "files_total": 0,
        "details": [],
    }

    if not dataset_run_id:
        result["details"].append(
            "No Dataset_Registry run found — skipping verification"
        )
        return result

    # Fetch registered metadata
    try:
        reg_run = mlflow.get_run(dataset_run_id)
        reg_tags = reg_run.data.tags
        reg_file_sizes_str = reg_tags.get("file_sizes", "")
        reg_file_sizes = json.loads(reg_file_sizes_str) if reg_file_sizes_str else {}
    except Exception as e:
        result["details"].append(f"Failed to fetch Dataset_Registry run: {e}")
        return result

    samples = load_manifest(manifest_path, data_root)
    curr_file_sizes: dict[str, int] = {}
    for s in samples:
        basename = os.path.basename(s["path"])
        if os.path.isfile(s["path"]):
            curr_file_sizes[basename] = os.stat(s["path"]).st_size
        else:
            curr_file_sizes[basename] = -1  # missing

    manifest_match = set(curr_file_sizes) == set(reg_file_sizes)
    if not manifest_match:
        result["details"].append(
            f"File manifest mismatch: expected {len(reg_file_sizes)} file(s), found {len(curr_file_sizes)} file(s)"
        )
    sizes_match = all(
        curr_file_sizes.get(k) == reg_file_sizes[k] for k in reg_file_sizes
    )
    result["file_sizes_match"] = manifest_match and sizes_match

    if not result["file_sizes_match"]:
        mismatched = [
            name
            for name in reg_file_sizes
            if curr_file_sizes.get(name) != reg_file_sizes[name]
        ]
        if mismatched:
            result["details"].append(
                f"File size mismatch on {len(mismatched)} file(s): "
                + ", ".join(sorted(mismatched)[:5])
                + ("…" if len(mismatched) > 5 else ""),
            )

    missing = sum(1 for s in samples if not os.path.isfile(s["path"]))
    result["files_total"] = len(samples)
    result["files_missing"] = missing

    if missing > 0:
        result["details"].append(f"{missing}/{len(samples)} WSI files missing on disk")

    result["verified"] = result["file_sizes_match"] and missing == 0

    if result["verified"]:
        result["details"].append("✅ Dataset verified — matches registered version")
    else:
        result["details"].append("❌ Dataset verification FAILED")

    return result


# ──────────────────────────────────────────────
# Hash-based registration (preferred)
# ──────────────────────────────────────────────


def register_dataset(
    csv: str,
    path_column: str | None = None,
    experiment_name: str = "Dataset_Registry",
    dataset_name: str | None = None,
    version: str = "1.0.0",
) -> str:
    """Register a dataset from its cohort csv.

    Takes the csv, the column with the file paths, and the experiment —
    that's it. Paths are absolute or relative to the csv's folder.
    Labels are deliberately not part of registration — classes live in
    the cohort table itself, in whatever format it has; whoever needs
    them joins on the cohort. Writes the canonical ``manifest.csv``
    (paths only) next to the csv — the data itself stays in place — and
    stores the per-file fingerprint (size + mtime) in the registration
    run.

    Args:
        csv: the cohort csv (or xlsx) listing the sample files.
        path_column: column holding the file paths — auto-detected
            from *csv*'s columns (case-insensitive) when omitted.
        experiment_name: MLflow experiment for the registration run.
        dataset_name: defaults to the csv folder's name.
        version: dataset version string.

    Returns:
        The run_id of the registration run.

    Example:
        from rationai.provenance import register_dataset

        run_id = register_dataset(
            "data/mammaprint/slides.csv",
            path_column="slide_path",
            experiment_name="Mammaprint",
        )
    """
    csv = os.path.abspath(csv)
    dataset_dir = os.path.dirname(csv)
    df = (
        pd.read_excel(csv)
        if str(csv).lower().endswith((".xlsx", ".xls"))
        else pd.read_csv(csv)
    )

    if path_column is None:
        path_column = _pick_column(df, PATH_COLUMNS)
        if path_column is None:
            raise ValueError(
                f"{csv}: no path column — pass path_column= or "
                f"name one of {PATH_COLUMNS}"
            )
    elif path_column not in df.columns:
        raise ValueError(
            f"{csv}: no column {path_column!r} — found {list(df.columns)}"
        )

    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rel = str(row[path_column]).strip()
        full = rel if os.path.isabs(rel) else os.path.join(dataset_dir, rel)
        samples.append({"path": full})
    missing = [s["path"] for s in samples if not os.path.isfile(s["path"])]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(samples)} files not found, "
            f"e.g. {missing[:3]} — check the paths in {csv}"
        )

    # canonical manifest (paths only) next to the csv — data stays in place
    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    pd.DataFrame(
        {
            "wsi_path": [
                os.path.relpath(s["path"], dataset_dir)
                if s["path"].startswith(dataset_dir + os.sep)
                else s["path"]
                for s in samples
            ],
        }
    ).to_csv(manifest_path, index=False)

    file_sizes = {}
    file_mtimes = {}
    for s in samples:
        st = os.stat(s["path"])
        file_sizes[os.path.basename(s["path"])] = st.st_size
        file_mtimes[os.path.basename(s["path"])] = st.st_mtime

    if dataset_name is None:
        dataset_name = os.path.basename(dataset_dir)

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(
        run_name=f"Dataset_{dataset_name}_{version}", nested=True
    )  # nested: usable inside an active @autolog run
    run_id = run.info.run_id
    run_active = True

    try:
        # ── Environment + user tags ────────────────────────
        from rationai.provenance.environment import capture_environment

        capture_environment(snapshot_env=False)

        mlflow.log_params(
            {
                "dataset_root": dataset_dir,
                "source_csv": os.path.basename(csv),
                "num_samples": len(samples),
            }
        )

        mlflow.set_tags(
            {
                "dataset_name": dataset_name,
                "version": version,
                "file_sizes": json.dumps(file_sizes),
                "file_mtimes": json.dumps(file_mtimes),
            }
        )

    finally:
        if run_active:
            mlflow.end_run()

    print(f"  [register_dataset] {dataset_name} v{version} → run_id={run_id}")
    return run_id
