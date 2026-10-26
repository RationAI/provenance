"""Smoke tests: PROV builders are pure functions; imports stay torch/lightning/mlkit-free."""

import sys


def test_import_is_standalone():
    import rationai.provenance  # noqa: F401

    assert "torch" not in sys.modules
    assert "lightning" not in sys.modules
    assert "rationai.mlkit" not in sys.modules


def test_build_training_run_prov_document():
    from rationai.provenance.run import build_training_run_prov

    doc = build_training_run_prov(
        run_id="0123456789abcdef0123456789abcdef",
        run_name="smoke-run",
        params={"lr": "0.1"},
        metrics={"acc": 0.9},
        tags={"user_id": "u1"},
    )

    inner = doc["bundle"]["storage:0123456789abcdef0123456789abcdef"]
    assert "prefix" in inner
    assert "entity" in inner
    assert "activity" in inner
    assert "agent" in inner
    assert "wasAssociatedWith" in inner


def test_common_helpers():
    from rationai.provenance.common import get_prov_prefixes, safe_id

    assert safe_id("hello world") == "hello_world"
    assert "prov" in get_prov_prefixes()
