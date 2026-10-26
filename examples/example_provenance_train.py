"""Example: train a simple model with full provenance tracking.

Requires :data:`example_provenance_setup.py` to have been run first
(so that users and dataset are registered in MLflow).

This script demonstrates the ``@autolog`` decorator together with
``ProvenanceCallback`` — every training run is captured as a PROV document
and logged to MLflow.

Usage::

    python example_provenance_train.py
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import hydra
import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule, seed_everything
from omegaconf import DictConfig
from rationai.mlkit import Trainer, autolog
from torch.utils.data import DataLoader


if TYPE_CHECKING:
    from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# ── Minimal model ───────────────────────────────────────────────────────

class SimpleNet(LightningModule):
    """Tiny 2-layer net for binary classification of image tiles."""

    def __init__(self, num_classes: int = 2, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ── Minimal data module ────────────────────────────────────────────────

class ExampleDataModule(LightningDataModule):
    """Load the dummy dataset created by :data:`example_provenance_setup.py`."""

    def __init__(self, dataset_dir: str = "data/example_dataset", batch_size: int = 4) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        import glob

        paths = sorted(glob.glob(os.path.join(self.dataset_dir, "slide_*.png")))
        # Labels: same pattern as setup script — idx % 2
        self._data = [
            (p, int(os.path.basename(p)[6:9]) % 2)
            for p in paths
        ]

    def _make_dataloader(self, split_ratio: float = 1.0):
        samples = self._data[: int(len(self._data) * split_ratio)]

        class TensorDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                x = torch.load(self.samples[idx][0], weights_only=False)
                y = torch.tensor(self.samples[idx][1], dtype=torch.long)
                return x, y

        return DataLoader(TensorDataset(samples), batch_size=self.batch_size, shuffle=True)

    def train_dataloader(self):
        return self._make_dataloader(split_ratio=0.8)

    def val_dataloader(self):
        samples = self._data[int(len(self._data) * 0.8):]

        class TensorDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                x = torch.load(self.samples[idx][0], weights_only=False)
                y = torch.tensor(self.samples[idx][1], dtype=torch.long)
                return x, y

        return DataLoader(TensorDataset(samples), batch_size=self.batch_size)


# ── Main ────────────────────────────────────────────────────────────────

@hydra.main(config_path=".", config_name="example_provenance_train_cfg", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # Apply user override from config (must happen before any callback runs)
    if hasattr(config, "user") and config.user is not None:
        username = config.user.get("username", config.user.username if hasattr(config.user, "username") else None)
        if username:
            os.environ["MLFLOW_USER"] = username

    seed_everything(config.seed, workers=True)

    torch.set_float32_matmul_precision(precision="medium")

    # Instantiate model and data module from config
    model = SimpleNet(num_classes=config.num_classes, lr=config.lr)
    data = ExampleDataModule(batch_size=config.batch_size)

    # Trainer *and its callbacks* come from the hydra config — nothing
    # provenance-related is hardcoded here (see example_provenance_train_cfg.yaml).
    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
