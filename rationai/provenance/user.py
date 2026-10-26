"""User registration in the MLflow ``User_Registry``.

``register_new_user`` logs identity params/tags so downstream provenance
tooling can link runs to people. The PROV-O document itself is built by
an external tool from these records.
"""

from __future__ import annotations

import mlflow


def register_new_user(
    username: str,
    real_name: str,
    email: str,
    organization: str,
    lead_name: str,
    lead_email: str,
) -> str:
    """Register a user in the ``User_Registry`` experiment.

    Creates an MLflow run with identity params (username, real name, email,
    organization, lead) and tags (username, organization).

    Returns:
        The MLflow run_id of the registration run.
    """
    experiment_name = "User_Registry"
    experiment_id: str | None
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        exp = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = exp.experiment_id if exp else None

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"User_{username}",
    ) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "username": username,
                "real_name": real_name,
                "email": email,
                "organization": organization,
                "lead_name": lead_name,
                "lead_email": lead_email,
            }
        )
        mlflow.set_tags(
            {
                "username": username,
                "organization": organization,
            }
        )

    print(f"  [register_new_user] {username} → run_id={run_id}")
    return run_id


if __name__ == "__main__":
    register_new_user(
        username="researcher_01",
        real_name="Jane Doe",
        email="jane.doe@example.com",
        organization="Example Org",
        lead_name="John Smith",
        lead_email="john.smith@example.com",
    )
