"""Example: register users and a dataset in MLflow (prerequisite for training).

Run this *before* the training script so that provenance can reference
existing user and dataset runs.  In real workflows this corresponds to
one-time setup — you don't re-register on every training run.

Usage:
    python example_provenance_setup.py
"""

from __future__ import annotations

import csv
import os

import torch

from rationai.provenance import register_dataset, register_new_user


# ── 1. Register users ───────────────────────────────────────────────────

def register_users():
    """Register researchers into MLflow's User_Registry experiment."""
    register_new_user(
        username="jiribuchta",
        real_name="Jiří Buchta",
        email="524981@mail.muni.cz",
        organization="RationAI",
        lead_name="Tomáš Brázdil",
        lead_email="brazdil@muni.cz",
    )
    register_new_user(
        username="matejpekar",
        real_name="Matěj Pekár",
        email="matejpekar@mail.muni.cz",
        organization="RationAI",
        lead_name="Tomáš Brázdil",
        lead_email="brazdil@muni.cz",
    )


# ── 2. Create dummy dataset & register it ───────────────────────────────

def create_dummy_data(dataset_dir: str = "data/example_dataset"):
    """Create a minimal dataset with manifest.csv for demonstration."""
    os.makedirs(dataset_dir, exist_ok=True)

    samples = [
        {"wsi_path": f"slide_{i:03d}.png", "cancer": i % 2}
        for i in range(10)
    ]

    for s in samples:
        path = os.path.join(dataset_dir, s["wsi_path"])
        if not os.path.exists(path):
            with open(path, "wb") as f:
                torch.save(torch.randn(3, 64, 64), f)

    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wsi_path", "cancer"])
            writer.writeheader()
            writer.writerows(samples)

    print(f"  Created dummy dataset at {dataset_dir} ({len(samples)} samples)")
    return dataset_dir


def register_example_dataset(dataset_dir: str = "data/example_dataset"):
    """Register the dummy dataset into MLflow's Dataset_Registry experiment."""
    run_id = register_dataset(
        csv=os.path.join(dataset_dir, "manifest.csv"),
        dataset_name="example_cohort",
        experiment_name="Dataset_Registry",
    )
    return run_id


# ── Main ────────────────────────────────────────────────────────────────

def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print("=" * 60)
    print("Provenance Setup — register users & dataset")
    print(f"MLflow tracking URI: {tracking_uri}")
    print("=" * 60)

    print("\n[1/2] Registering users…")
    register_users()

    print("\n[2/2] Creating dummy dataset and registering it…")
    dataset_dir = create_dummy_data()
    register_example_dataset(dataset_dir)

    print("\n✅ Setup done! Check MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    main()
