"""
tracking/mlflow_tracker.py

Responsible for initialising MLflow tracking and providing
reusable logging helpers for all pipeline phases.

Used by:
    - Phase 3: feature engineering (parameters, metrics, artifacts)
    - Phase 4: model training (hyperparameters, model metrics, artifacts)

Six helpers:
    setup_mlflow()          — configure tracking URI and experiment
    start_run()             — open a new run inside the experiment
    log_feature_params()    — log input parameters for a run
    log_feature_metrics()   — log output metrics for a run
    log_feature_artifact()  — log a file artifact for a run
    end_run()               — close the active run cleanly
"""

import mlflow
from loguru import logger


def setup_mlflow(experiment_name: str) -> None:
    """
    Configure MLflow tracking URI and set the active experiment.

    Creates the experiment if it does not exist. Connects to it
    if it already does. All subsequent runs will be logged here.

    Args:
        experiment_name: Name of the MLflow experiment to log into

    Returns:
        None
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    logger.success(f"MLflow setup complete | Experiment: {experiment_name}")


def start_run(run_name: str):
    """
    Start a new MLflow run inside the active experiment.

    Args:
        run_name: Human-readable name for this run

    Returns:
        Active MLflow run object
    """
    run = mlflow.start_run(run_name=run_name)
    logger.info(f"MLflow run started | Run name: {run_name} | Run ID: {run.info.run_id}")
    return run


def log_feature_params(params: dict) -> None:
    """
    Log input parameters for the current run.

    Args:
        params: inputs used during the run

    Returns:
        None
    """
    mlflow.log_params(params)
    logger.success(f"Parameters({list(params.keys())}) recorded successfully")


def log_feature_metrics(metrics: dict) -> None:
    """
    Log output metrics for the current run.

    Args:
        metrics: results gotten during the run

    Returns:
        None
    """
    mlflow.log_metrics(metrics)
    logger.success(f"Metrics({list(metrics.keys())}) recorded successfully")


def log_feature_artifact(artifact_path: str) -> None:
    """
    Log a file artifact for the current run.

    Args:
        artifact_path: File path to the output file being logged
                       e.g. a feature distribution CSV or correlation matrix

    Returns:
        None
    """
    mlflow.log_artifact(artifact_path)
    logger.success(f"Artifact({artifact_path}) recorded successfully")


def end_run() -> None:
    """
    End the current active MLflow run cleanly.

    Returns:
        None
    """
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else "unknown"
    mlflow.end_run()
    logger.success(f"MLflow run ended | Run ID: {run_id}")