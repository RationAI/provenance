"""Provenance ingredient logging to MLflow.

Submodules:
    user            - register_new_user (``User_Registry``)
    dataset         - register_dataset + verify_dataset + lookup helpers
    environment     - capture_environment (git, hardware, docker, seeds)
    log_provenance  - universal ingredient logging (params, stats, manifests)

This package logs the *ingredients* of provenance (who, which data, which
environment, which config); the W3C PROV-O document itself is assembled by
an external tool from these MLflow records.
"""

from rationai.provenance.dataset import register_dataset, verify_dataset
from rationai.provenance.environment import capture_environment, log_environment
from rationai.provenance.log_provenance import log_provenance
from rationai.provenance.user import register_new_user


__all__ = [
    "capture_environment",
    "log_environment",
    "log_provenance",
    "register_dataset",
    "register_new_user",
    "verify_dataset",
]
