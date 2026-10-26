"""Example: full preprocessing pipeline with provenance at every step.

Demonstrates a complete data-preprocessing workflow where each stage
creates its own MLflow run and PROV document, linked together through
tags and the provenance graph:

    1. Register user           →  User_Registry experiment
    2. Register raw dataset    →  Dataset_Registry experiment
    3. Create train/test split →  Data_Splits experiment (links to dataset run)
    4. Generate masks          →  Preprocessing_Pipeline experiment (links to split run)

Each step logs:
  - Environment (git, hardware, docker, user)
  - Parameters and metrics
  - PROV document (prov.json) as artifact
  - Links to parent runs via tags

Run::

    python example_preprocessing_pipeline.py
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from rationai.provenance import (
    build_dataset_prov,
    build_training_run_prov,
    capture_environment,
    register_new_user,
)


# ── Config ──────────────────────────────────────────────────────────────

TRACKING_URI = "http://localhost:5000"
DATASET_DIR = "data/example_pipeline_dataset"
NUM_SAMPLES = 10
USERNAME = "jiribuchta"
ORGANIZATION = "RationAI"


def _make_artifact_dir(name: str) -> str:
    """Create temp dir for artifact logging."""
    d = f"_mlflow_{name}_{uuid.uuid4().hex[:8]}"
    os.makedirs(d, exist_ok=True)
    return d


def _log_prov(prov_doc: dict, run_id: str, artifact_path: str = "provenance") -> None:
    """Log a PROV document as prov.json artifact."""
    d = _make_artifact_dir("prov")
    try:
        path = os.path.join(d, "prov.json")
        with open(path, "w") as f:
            json.dump(prov_doc, f, indent=2)
        mlflow.log_artifact(path, artifact_path=artifact_path)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── Step 0: Create dummy data ──────────────────────────────────────────

def _create_dummy_data() -> Path:
    """Create synthetic dataset files."""
    dataset_dir = Path(DATASET_DIR)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(NUM_SAMPLES):
        fname = f"slide_{i:03d}.png"
        fpath = dataset_dir / fname
        # Random tensor as dummy "image"
        x = torch.randn(3, 64, 64)
        torch.save(x, str(fpath))
        label = i % 2
        rows.append({"wsi_path": str(fpath), "cancer": label})

    manifest = dataset_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Created {NUM_SAMPLES} dummy slides at {dataset_dir}")
    return dataset_dir


# ── Step 1: Register user ──────────────────────────────────────────────

def step_register_user() -> str:
    """Register user in User_Registry. Returns run_id."""
    print("\n" + "=" * 60)
    print("STEP 1: Register user")
    print("=" * 60)

    run_id = register_new_user(
        username=USERNAME,
        real_name="Jiri Buchta",
        email="jiri@example.com",
        organization=ORGANIZATION,
        lead_name="Lead PI",
        lead_email="lead@example.com",
    )
    print(f"  User registered → run_id={run_id}")
    return run_id


# ── Step 2: Register raw dataset ───────────────────────────────────────

def _get_dataset_experiment(dataset_name: str) -> int:
    """Get or create a per-dataset experiment."""
    exp_name = f"dataset_{dataset_name}"
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        return mlflow.create_experiment(exp_name)
    return exp.experiment_id


def step_register_dataset(dataset_dir: Path) -> str:
    """Register dataset run under per-dataset experiment. Returns run_id."""
    print("\n" + "=" * 60)
    print("STEP 2: Register raw dataset")
    print("=" * 60)

    exp_id = _get_dataset_experiment(dataset_dir.name)
    manifest_path = str(dataset_dir / "manifest.csv")

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"Dataset_raw_{dataset_dir.name}",
    ) as run:
        run_id = run.info.run_id

        # Load manifest
        df = pd.read_csv(manifest_path)
        num_samples = len(df)
        num_positive = int(df["cancer"].sum())
        num_negative = num_samples - num_positive

        # File sizes
        file_sizes = {}
        for _, row in df.iterrows():
            basename = os.path.basename(row["wsi_path"])
            fpath = Path(row["wsi_path"])
            file_sizes[basename] = int(fpath.stat().st_size) if fpath.exists() else -1

        # Environment capture
        capture_environment(snapshot_env=False)

        # Log params
        mlflow.log_params({
            "dataset_root": str(dataset_dir),
            "num_samples": num_samples,
            "num_positive": num_positive,
            "num_negative": num_negative,
        })

        # Log tags
        mlflow.set_tags({
            "dataset_name": dataset_dir.name,
            "version": "1.0.0",
            "file_sizes": json.dumps(file_sizes),
            "manifest_path": manifest_path,
        })

        # Build PROV document
        prov_doc = build_dataset_prov(
            run_id=run_id,
            dataset_name=dataset_dir.name,
            version="1.0.0",
            dataset_root=str(dataset_dir),
            num_samples=num_samples,
            num_positive=num_positive,
            num_negative=num_negative,
            file_sizes=file_sizes,
            manifest_path=manifest_path,
        )
        _log_prov(prov_doc, run_id)

        print(f"  Dataset registered → run_id={run_id}")
        print(f"  Samples: {num_samples} ({num_positive}+, {num_negative}-)")

    return run_id


# ── Step 3: Create train/test splits ───────────────────────────────────

def step_create_splits(dataset_dir: Path, dataset_run_id: str) -> str:
    """Create train/test split. Links to dataset_run_id. Returns run_id."""
    print("\n" + "=" * 60)
    print("STEP 3: Create train/test splits")
    print("=" * 60)

    exp_id = _get_dataset_experiment(dataset_dir.name)

    manifest_path = str(dataset_dir / "manifest.csv")
    df = pd.read_csv(manifest_path)

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"Split_{dataset_dir.name}_80_20",
    ) as run:
        run_id = run.info.run_id

        # Environment capture
        capture_environment(snapshot_env=False)

        # Link to parent dataset run
        mlflow.set_tag("dataset_run_id", dataset_run_id)

        # Create split
        test_size = 0.2
        random_state = 42
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["cancer"])

        num_train = len(train_df)
        num_test = len(test_df)

        # Log params
        mlflow.log_params({
            "test_size": test_size,
            "random_state": random_state,
            "num_train": num_train,
            "num_test": num_test,
            "train_positive": int(train_df["cancer"].sum()),
            "train_negative": num_train - int(train_df["cancer"].sum()),
            "test_positive": int(test_df["cancer"].sum()),
            "test_negative": num_test - int(test_df["cancer"].sum()),
        })

        # Save split files as artifacts
        d = _make_artifact_dir("split")
        train_csv = os.path.join(d, "train_split.csv")
        test_csv = os.path.join(d, "test_split.csv")
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        mlflow.log_artifacts(d, artifact_path="split")
        shutil.rmtree(d, ignore_errors=True)

        # Build PROV document (reusing build_training_run_prov for split activity)
        prov_doc = build_training_run_prov(
            run_id=run_id,
            run_name=f"Split_{dataset_dir.name}_80_20",
            params={
                "dataset_name": dataset_dir.name,
                "test_size": str(test_size),
                "random_state": str(random_state),
                "num_train": str(num_train),
                "num_test": str(num_test),
            },
            metrics={},
            tags={"dataset_run_id": dataset_run_id},
            split_data={
                "test_size": test_size,
                "random_state": random_state,
                "train": train_df.to_dict("records"),
                "test": test_df.to_dict("records"),
            },
        )
        _log_prov(prov_doc, run_id)

        print(f"  Split created → run_id={run_id}")
        print(f"  Train: {num_train}, Test: {num_test}")
        print(f"  Linked to dataset run: {dataset_run_id}")

    return run_id


# ── Step 4: Generate masks ─────────────────────────────────────────────

def step_generate_masks(dataset_dir: Path, split_run_id: str) -> str:
    """Generate dummy masks. Links to split_run_id. Returns run_id."""
    print("\n" + "=" * 60)
    print("STEP 4: Generate masks")
    print("=" * 60)

    exp_id = _get_dataset_experiment(dataset_dir.name)

    manifest_path = str(dataset_dir / "manifest.csv")
    df = pd.read_csv(manifest_path)

    # Generate dummy masks
    masks_dir = dataset_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    mask_paths = []
    for _, row in df.iterrows():
        slide_path = Path(row["wsi_path"])
        mask_name = slide_path.stem + "_mask.png"
        mask_path = masks_dir / mask_name
        # Dummy binary mask (random noise thresholded)
        mask = (torch.rand(64, 64) > 0.5).float()
        torch.save(mask, str(mask_path))
        mask_paths.append(str(mask_path))

    num_masks = len(mask_paths)

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"Masks_{dataset_dir.name}",
    ) as run:
        run_id = run.info.run_id

        # Environment capture
        capture_environment(snapshot_env=False)

        # Link to parent runs
        mlflow.set_tag("split_run_id", split_run_id)

        # Log params
        mlflow.log_params({
            "num_masks": num_masks,
            "mask_size": "64x64",
            "mask_type": "binary_segmentation",
            "method": "dummy_random",
        })

        # Save mask manifest as artifact
        d = _make_artifact_dir("masks")
        mask_manifest = os.path.join(d, "mask_manifest.csv")
        pd.DataFrame({"slide_path": df["wsi_path"].tolist(), "mask_path": mask_paths}).to_csv(mask_manifest, index=False)
        mlflow.log_artifacts(d, artifact_path="masks")
        shutil.rmtree(d, ignore_errors=True)

        # Build PROV document
        prov_doc = build_training_run_prov(
            run_id=run_id,
            run_name=f"Masks_{dataset_dir.name}",
            params={
                "num_masks": str(num_masks),
                "mask_size": "64x64",
                "mask_type": "binary_segmentation",
                "method": "dummy_random",
            },
            metrics={},
            tags={"split_run_id": split_run_id},
        )
        _log_prov(prov_doc, run_id)

        print(f"  Masks generated → run_id={run_id}")
        print(f"  Masks: {num_masks}")
        print(f"  Linked to split run: {split_run_id}")

    return run_id


# ── Step 5: Extract tiles ──────────────────────────────────────────────

def step_extract_tiles(dataset_dir: Path, masks_run_id: str) -> str:
    """Extract dummy tiles from masked regions. Links to masks_run_id. Returns run_id."""
    print("\n" + "=" * 60)
    print("STEP 5: Extract tiles")
    print("=" * 60)

    exp_id = _get_dataset_experiment(dataset_dir.name)

    manifest_path = str(dataset_dir / "manifest.csv")
    df = pd.read_csv(manifest_path)

    # Generate dummy tile metadata (parquet-style as CSV for simplicity)
    tiles_dir = dataset_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    TILE_SIZE = 224
    tiles_all = []

    for _, row in df.iterrows():
        slide_path = Path(row["wsi_path"])
        slide_name = slide_path.stem
        # Simulate extracting N tiles per slide from masked regions
        n_tiles = np.random.randint(50, 200)
        for _i in range(n_tiles):
            x = np.random.randint(0, 64 - TILE_SIZE // 4)  # dummy coords (slide is 64x64)
            y = np.random.randint(0, 64 - TILE_SIZE // 4)
            tiles_all.append({
                "slide_name": slide_name,
                "slide_path": str(slide_path),
                "x": x,
                "y": y,
                "tile_size_x": TILE_SIZE,
                "tile_size_y": TILE_SIZE,
                "level": 0,
            })

    tiles_df = pd.DataFrame(tiles_all)
    num_tiles = len(tiles_df)
    unique_slides = tiles_df["slide_name"].nunique()

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"Tiles_{dataset_dir.name}",
    ) as run:
        run_id = run.info.run_id

        # Environment capture
        capture_environment(snapshot_env=False)

        # Link to parent run
        mlflow.set_tag("masks_run_id", masks_run_id)

        # Log params
        mlflow.log_params({
            "num_tiles": num_tiles,
            "unique_slides": unique_slides,
            "tile_size": str(TILE_SIZE),
            "method": "dummy_random",
        })

        # Save tiles manifest as artifact
        d = _make_artifact_dir("tiles")
        tiles_csv = os.path.join(d, "tiles.csv")
        tiles_df.to_csv(tiles_csv, index=False)
        mlflow.log_artifacts(d, artifact_path="tiles")
        shutil.rmtree(d, ignore_errors=True)

        # Build PROV document
        prov_doc = build_training_run_prov(
            run_id=run_id,
            run_name=f"Tiles_{dataset_dir.name}",
            params={
                "num_tiles": str(num_tiles),
                "unique_slides": str(unique_slides),
                "tile_size": str(TILE_SIZE),
                "method": "dummy_random",
            },
            metrics={},
            tags={"masks_run_id": masks_run_id},
        )
        _log_prov(prov_doc, run_id)

        print(f"  Tiles extracted → run_id={run_id}")
        print(f"  Tiles: {num_tiles} from {unique_slides} slides")
        print(f"  Linked to masks run: {masks_run_id}")

    return run_id


# ── Step 6: Compute embeddings ─────────────────────────────────────────

def step_compute_embeddings(dataset_dir: Path, tiles_run_id: str) -> str:
    """Compute dummy embeddings for each tile. Links to tiles_run_id. Returns run_id."""
    print("\n" + "=" * 60)
    print("STEP 6: Compute embeddings")
    print("=" * 60)

    exp_id = _get_dataset_experiment(dataset_dir.name)

    EMBED_DIM = 768  # e.g. ViT-B/16 feature dim

    # Load tiles manifest
    tiles_csv = dataset_dir / "tiles" / "tiles.csv"
    if not tiles_csv.exists():
        # Fallback: regenerate from manifest (tiles dir might not exist on disk)
        tiles_df = pd.read_csv(str(dataset_dir / "manifest.csv"))  # placeholder
        num_tiles = len(tiles_df) * 100  # approximate
    else:
        tiles_df = pd.read_csv(tiles_csv)
        num_tiles = len(tiles_df)

    # Generate dummy embeddings
    embeddings = np.random.randn(num_tiles, EMBED_DIM).astype(np.float32)

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"Embeddings_{dataset_dir.name}",
    ) as run:
        run_id = run.info.run_id

        # Environment capture
        capture_environment(snapshot_env=False)

        # Link to parent run
        mlflow.set_tag("tiles_run_id", tiles_run_id)

        # Log params
        mlflow.log_params({
            "num_tiles": num_tiles,
            "embedding_dim": EMBED_DIM,
            "model": "dummy_vit_b16",
            "method": "dummy_random",
        })

        # Save embeddings as artifact
        d = _make_artifact_dir("embeddings")
        embed_path = os.path.join(d, "embeddings.npy")
        np.save(embed_path, embeddings)

        # Also save tile→embedding index mapping
        if tiles_csv.exists():
            index_path = os.path.join(d, "tile_index.csv")
            tiles_df.to_csv(index_path, index=False)
        else:
            # Create synthetic index from tiles_run_id artifacts
            index_path = os.path.join(d, "tile_index.csv")
            pd.DataFrame({
                "tile_idx": range(num_tiles),
                "slide_name": [f"slide_{i % 10:03d}" for i in range(num_tiles)],
            }).to_csv(index_path, index=False)

        mlflow.log_artifacts(d, artifact_path="embeddings")
        shutil.rmtree(d, ignore_errors=True)

        # Build PROV document
        prov_doc = build_training_run_prov(
            run_id=run_id,
            run_name=f"Embeddings_{dataset_dir.name}",
            params={
                "num_tiles": str(num_tiles),
                "embedding_dim": str(EMBED_DIM),
                "model": "dummy_vit_b16",
                "method": "dummy_random",
            },
            metrics={},
            tags={"tiles_run_id": tiles_run_id},
        )
        _log_prov(prov_doc, run_id)

        print(f"  Embeddings computed → run_id={run_id}")
        print(f"  Shape: ({num_tiles}, {EMBED_DIM})")
        print(f"  Linked to tiles run: {tiles_run_id}")

    return run_id


# ── Step 7: Train dummy classifier on embeddings ─────────────────────

def step_train_classifier(dataset_dir: Path, embeddings_run_id: str, split_run_id: str) -> str:
    """Train a dummy classifier on tile embeddings. Links to embeddings + split. Returns run_id."""
    print("\n" + "=" * 60)
    print("STEP 7: Train classifier")
    print("=" * 60)

    exp_id = _get_dataset_experiment(dataset_dir.name)

    EMBED_DIM = 768

    # Load embeddings
    embed_path = dataset_dir / "embeddings" / "embeddings.npy"
    if not embed_path.exists():
        # Fallback: regenerate dummy
        num_tiles = 1000
        X = np.random.randn(num_tiles, EMBED_DIM).astype(np.float32)
    else:
        X = np.load(str(embed_path))

    # Load labels from manifest (slide-level label → tile-level)
    manifest_df = pd.read_csv(dataset_dir / "manifest.csv")
    slide_to_label = dict(zip(manifest_df["wsi_path"].str.split("/").str[-1], manifest_df["cancer"], strict=False))

    # Load tile index to map tiles → slides
    tile_index_path = dataset_dir / "embeddings" / "tile_index.csv"
    if tile_index_path.exists():
        tile_index_df = pd.read_csv(tile_index_path)
        labels = tile_index_df["slide_name"].map(slide_to_label).values.astype(np.float32)
    else:
        # Fallback: random labels matching manifest distribution
        n_pos = int(manifest_df["cancer"].sum())
        n_neg = len(manifest_df) - n_pos
        tiles_per_slide = len(X) // len(manifest_df)
        labels = np.array([1] * (n_pos * tiles_per_slide) + [0] * (n_neg * tiles_per_slide), dtype=np.float32)
        np.random.shuffle(labels)
        labels = labels[:len(X)]

    num_train = int(len(X) * 0.8)
    X_train, X_test = X[:num_train], X[num_train:]
    y_train, y_test = labels[:num_train], labels[num_train:]

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"Classifier_{dataset_dir.name}",
    ) as run:
        run_id = run.info.run_id

        # Environment capture
        capture_environment(snapshot_env=False)

        # Link to parent runs
        mlflow.set_tag("embeddings_run_id", embeddings_run_id)
        mlflow.set_tag("split_run_id", split_run_id)

        # Log params
        lr = 1e-3
        epochs = 10
        mlflow.log_params({
            "num_tiles": len(X),
            "embedding_dim": EMBED_DIM,
            "model": "dummy_linear_classifier",
            "lr": lr,
            "epochs": epochs,
            "num_train": num_train,
            "num_test": len(X) - num_train,
        })

        # Train dummy linear classifier (single-layer)
        W = np.random.randn(EMBED_DIM, 1).astype(np.float32) * 0.01
        b = np.zeros(1, dtype=np.float32)

        train_losses = []
        for _epoch in range(epochs):
            # Forward
            logits = X_train @ W + b
            probs = 1 / (1 + np.exp(-logits.flatten()))

            # BCE loss
            loss = -np.mean(y_train * np.log(probs + 1e-7) + (1 - y_train) * np.log(1 - probs + 1e-7))
            train_losses.append(loss)

            # Gradient
            diff = (probs - y_train).reshape(-1, 1)
            dW = X_train.T @ diff / len(X_train)
            db = np.mean(diff)

            # Update
            W -= lr * dW
            b -= lr * db

        # Evaluate on test set
        logits_test = X_test @ W + b
        probs_test = 1 / (1 + np.exp(-logits_test.flatten()))
        preds_test = (probs_test >= 0.5).astype(np.float32)

        accuracy = np.mean(preds_test == y_test)
        precision_pos = np.sum(preds_test == 1) / max(np.sum(preds_test == 1), 1) if np.sum(y_test == 1) > 0 else 0
        recall_pos = np.sum((preds_test == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)

        # Log metrics
        mlflow.log_metrics({
            "final_train_loss": train_losses[-1],
            "test_accuracy": accuracy,
            "test_precision_pos": precision_pos,
            "test_recall_pos": recall_pos,
        })
        mlflow.log_metric("best_train_loss", min(train_losses))

        # Log loss curve as artifact
        d = _make_artifact_dir("classifier")
        np.save(os.path.join(d, "train_loss.npy"), np.array(train_losses))
        np.save(os.path.join(d, "weights.npy"), W)

        # Confusion matrix
        tp = int(np.sum((preds_test == 1) & (y_test == 1)))
        fp = int(np.sum((preds_test == 1) & (y_test == 0)))
        tn = int(np.sum((preds_test == 0) & (y_test == 0)))
        fn = int(np.sum((preds_test == 0) & (y_test == 1)))
        pd.DataFrame([[tp, fp], [fn, tn]], columns=["pred_pos", "pred_neg"], index=["true_pos", "true_neg"]).to_csv(
            os.path.join(d, "confusion_matrix.csv")
        )

        mlflow.log_artifacts(d, artifact_path="classifier")
        shutil.rmtree(d, ignore_errors=True)

        # Build PROV document
        prov_doc = build_training_run_prov(
            run_id=run_id,
            run_name=f"Classifier_{dataset_dir.name}",
            params={
                "num_tiles": str(len(X)),
                "embedding_dim": str(EMBED_DIM),
                "model": "dummy_linear_classifier",
                "lr": str(lr),
                "epochs": str(epochs),
            },
            metrics={
                "test_accuracy": accuracy,
                "final_train_loss": train_losses[-1],
            },
            tags={"embeddings_run_id": embeddings_run_id, "split_run_id": split_run_id},
        )
        _log_prov(prov_doc, run_id)

        print(f"  Classifier trained → run_id={run_id}")
        print(f"  Test accuracy: {accuracy:.3f} (dummy model)")
        print(f"  Linked to embeddings run: {embeddings_run_id}")

    return run_id


# ── Summary ────────────────────────────────────────────────────────────

def print_summary(user_run_id: str, dataset_run_id: str, split_run_id: str, masks_run_id: str, tiles_run_id: str, embeddings_run_id: str, classifier_run_id: str) -> None:
    """Print pipeline summary with run links."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    client = mlflow.tracking.MlflowClient()

    runs_info = [
        ("User registration", user_run_id),
        ("Dataset registration", dataset_run_id),
        ("Train/test split", split_run_id),
        ("Mask generation", masks_run_id),
        ("Tile extraction", tiles_run_id),
        ("Embeddings", embeddings_run_id),
        ("Classifier", classifier_run_id),
    ]

    for name, run_id in runs_info:
        run = client.get_run(run_id)
        print(f"\n  {name}:")
        print(f"    Run ID:   {run_id}")
        print(f"    Name:     {run.info.run_name}")
        exp = client.get_experiment(run.info.experiment_id)
        print(f"    Experiment: {exp.name}")

        # Check artifacts for prov.json
        arts = [a.path for a in client.list_artifacts(run_id)]
        has_prov = any("prov.json" in a for a in arts) or any(
            "prov.json" in str(x) for x in _list_all_artifacts(client, run_id)
        )
        print(f"    PROV:     {'✓' if has_prov else '✗'}")

    print("\n  Provenance chain:")
    print(f"    User({user_run_id[:8]}) → Dataset({dataset_run_id[:8]}) → Split({split_run_id[:8]}) → Masks({masks_run_id[:8]}) → Tiles({tiles_run_id[:8]}) → Embeddings({embeddings_run_id[:8]}) → Classifier({classifier_run_id[:8]})")
    print(f"\n  View: {TRACKING_URI}")


def _list_all_artifacts(client: mlflow.tracking.MlflowClient, run_id: str) -> list[str]:
    """Recursively list all artifact paths."""
    def _walk(path: str):
        items = client.list_artifacts(run_id, path)
        for item in items:
            if item.is_dir:
                yield from _walk(item.path)
            else:
                yield item.path
    return list(_walk(""))


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    # Tell provenance system which user to link runs to.
    # _lookup_user_run() checks MLFLOW_USER > git config user.name > $USER
    os.environ["MLFLOW_USER"] = USERNAME

    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow tracking: {TRACKING_URI}")

    # Create dummy data
    dataset_dir = _create_dummy_data()

    # Step 1: Register user
    user_run_id = step_register_user()

    # Step 2: Register dataset
    dataset_run_id = step_register_dataset(dataset_dir)

    # Step 3: Create splits (linked to dataset)
    split_run_id = step_create_splits(dataset_dir, dataset_run_id)

    # Step 4: Generate masks (linked to split)
    masks_run_id = step_generate_masks(dataset_dir, split_run_id)

    # Step 5: Extract tiles (linked to masks)
    tiles_run_id = step_extract_tiles(dataset_dir, masks_run_id)

    # Step 6: Compute embeddings (linked to tiles)
    embeddings_run_id = step_compute_embeddings(dataset_dir, tiles_run_id)

    # Step 7: Train classifier (linked to embeddings + split)
    classifier_run_id = step_train_classifier(dataset_dir, embeddings_run_id, split_run_id)

    # Summary
    print_summary(user_run_id, dataset_run_id, split_run_id, masks_run_id, tiles_run_id, embeddings_run_id, classifier_run_id)


if __name__ == "__main__":
    main()
