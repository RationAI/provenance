"""Smoke tests: the package is a pure MLflow ingredient logger.

It must import without torch/lightning/mlkit, expose the logging API, and
contain no PROV document builders (those live in an external tool).
"""

import sys


def test_import_is_standalone():
    import rationai.provenance  # noqa: F401

    assert "torch" not in sys.modules
    assert "lightning" not in sys.modules
    assert "rationai.mlkit" not in sys.modules


def test_public_api_is_logging_only():
    import rationai.provenance as rp

    for name in (
        "register_new_user",
        "register_dataset",
        "verify_dataset",
        "capture_environment",
        "log_environment",
        "log_provenance",
    ):
        assert hasattr(rp, name), name

    # PROV document builders were removed; PROV-JSON is built externally.
    assert not hasattr(rp, "build_user_prov")
    assert not hasattr(rp, "build_dataset_prov")
    assert not hasattr(rp, "build_training_run_prov")


def test_safe_id():
    from rationai.provenance.log_provenance import safe_id

    assert safe_id("hello world") == "hello_world"
    assert safe_id("a/b:c") == "a_b_c"


def test_detect_job_limits(monkeypatch):
    from rationai.provenance.environment import detect_job_limits

    for key in (
        "JOB_CPU_LIMIT",
        "JOB_MEMORY_LIMIT",
        "JOB_USER",
        "JOB_NAME",
    ):
        monkeypatch.delenv(key, raising=False)

    assert detect_job_limits() == {}

    monkeypatch.setenv("JOB_CPU_LIMIT", "4")
    monkeypatch.setenv("JOB_MEMORY_LIMIT", "8Gi")
    monkeypatch.setenv("JOB_USER", "jdoe")

    assert detect_job_limits() == {
        "cpu_limit": "4",
        "memory_limit": "8Gi",
        "user": "jdoe",
    }


def test_flatten_scalars_resolves_per_leaf():
    from omegaconf import OmegaConf

    from rationai.provenance.log_provenance import (
        _config_to_container,
        _flatten_scalars,
    )

    cfg = OmegaConf.structured(
        {"a": {"name": "x"}, "good": "${a.name}", "bad": "${a.missing}"}
    )
    flat: dict[str, str] = {}
    _flatten_scalars(_config_to_container(cfg), "", flat, cfg)
    assert flat["a.name"] == "x"  # plain value
    assert flat["good"] == "x"  # resolved interpolation
    assert flat["bad"] == "${a.missing}"  # dangling: raw, does not crash


def test_flatten_scalars_without_dictconfig_keeps_raw():
    from rationai.provenance.log_provenance import _flatten_scalars

    flat: dict[str, str] = {}
    _flatten_scalars({"a": {"b": "${c}"}}, "", flat)
    assert flat["a.b"] == "${c}"
