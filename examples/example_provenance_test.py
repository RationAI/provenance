"""Test script: verify environment capture works via all three entry points.

1. `capture_environment()` — standalone function call
2. `@log_environment()` — decorator
3. `EnvironmentCallback` — Lightning callback

Requires `example_provenance_setup.py` to have been run first.

Usage:
    python example_provenance_test.py
"""

from __future__ import annotations

import os

import lightning as pl
import mlflow


TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ── 1. Standalone capture_environment() ────────────────────────────────

def test_standalone_capture():
    """Test capture_environment() directly."""
    from rationai.provenance import capture_environment

    print("\n" + "=" * 60)
    print("TEST 1: capture_environment() — standalone function")
    print("=" * 60)

    mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run(run_name="test_standalone_capture"):
        result = capture_environment(
            skip_hardware=False,
            snapshot_env=True,
            strict=False,
        )

        print(f"  git_commit: {result.get('git_commit', 'MISSING')}")
        print(f"  git_url: {result.get('git_url', 'MISSING')}")
        print(f"  git_branch: {result.get('git_branch', 'MISSING')}")
        print(f"  user_run_id: {result.get('user_run_id', 'MISSING')}")
        hardware = result.get("hardware") or {}
        print(f"  hardware keys: {list(hardware.keys())}")
        print(f"  docker: {result.get('docker', {})}")
        print(f"  frozen_requirements: {'(present)' if result.get('frozen_requirements') else '(missing)'}")

        assert result.get("git_commit"), "git_commit missing"
        assert result.get("hardware"), "hardware missing"
        assert result.get("docker") is not None, "docker missing"
        print("  ✅ PASS")


# ── 2. @log_environment() decorator ────────────────────────────────────

def test_decorator():
    """Test @log_environment() decorator."""
    from rationai.provenance import log_environment

    print("\n" + "=" * 60)
    print("TEST 2: @log_environment() — decorator")
    print("=" * 60)

    @log_environment(skip_hardware=False, snapshot_env=True)
    def my_training_fn():
        """Dummy training function wrapped with environment logging."""
        print("  Inside training fn...")
        return 42

    mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run(run_name="test_decorator"):
        result = my_training_fn()

        # Verify tags were set
        active = mlflow.active_run()
        assert active is not None
        run = mlflow.get_run(active.info.run_id)
        tags = run.data.tags

        assert tags.get("git_commit"), "git_commit tag missing"
        assert tags.get("prov_start_time"), "prov_start_time tag missing"
        print(f"  git_commit tag: {tags['git_commit']}")
        print(f"  prov_start_time tag: {tags['prov_start_time']}")
        if tags.get("user_run_id"):
            print(f"  user_run_id tag: {tags['user_run_id']}")
        print(f"  fn returned: {result}")
        print("  ✅ PASS")


# ── 3. EnvironmentCallback (Lightning) ─────────────────────────────────

def test_lightning_callback():
    """Test EnvironmentCallback inside a minimal Lightning Trainer."""
    import torch
    import torch.nn as nn
    from lightning.pytorch import Trainer
    from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger
    from torch.utils.data import DataLoader, TensorDataset

    from rationai.provenance.lightning.environment import EnvironmentCallback

    print("\n" + "=" * 60)
    print("TEST 3: EnvironmentCallback — Lightning callback")
    print("=" * 60)

    # Minimal model
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 2)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())

        def train_dataloader(self):
            xs = torch.randn(20, 10)
            ys = torch.randint(0, 2, (20,))
            return DataLoader(TensorDataset(xs, ys), batch_size=4)

    # Create data
    model = DummyModel()
    callback = EnvironmentCallback(skip_hardware=False, snapshot_env=True)

    logger = MLFlowLogger(
        experiment_name="test_environment_callback",
        tracking_uri=TRACKING_URI,
        run_name="test_lightning_callback",
    )

    trainer = Trainer(
        callbacks=[callback],
        logger=logger,
        max_epochs=1,
        accelerator="cpu",
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )

    trainer.fit(model)

    # Verify callback populated fields
    print(f"  _git_commit: {callback._git_commit}")
    print(f"  _git_url: {callback._git_url}")
    print(f"  _hardware keys: {list(callback._hardware.keys())}")
    print(f"  _docker: {callback._docker}")
    print(f"  _user_run_id: {callback._user_run_id}")

    assert callback._git_commit != "unknown", "git_commit not captured"
    assert len(callback._hardware) > 0, "hardware not captured"
    print("  ✅ PASS")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Environment Provenance Tests")
    print(f"MLflow tracking URI: {TRACKING_URI}")
    print("=" * 60)

    test_standalone_capture()
    test_decorator()
    test_lightning_callback()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print(f"Check MLflow UI at {TRACKING_URI}")
    print("=" * 60)


if __name__ == "__main__":
    main()
