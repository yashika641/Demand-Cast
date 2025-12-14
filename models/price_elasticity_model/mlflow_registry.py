# mlflow_registry.py

from typing import Dict, Any
import mlflow
import mlflow.sklearn


def log_price_elasticity_model_to_mlflow(
    model_artifact: Dict[str, Any],
    run_params: Dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "price_elasticity",
    registered_model_name: str = "price_elasticity_model",
):
    """
    Logs model + metrics to MLflow and registers it.

    model_artifact: output dict from train_price_elasticity_model()
    """

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    model = model_artifact["model"]
    metrics = model_artifact.get("metrics", {})
    feature_names = model_artifact.get("feature_names", [])
    elasticity_coef = model_artifact.get("elasticity_coef_global")

    with mlflow.start_run(run_name="price_elasticity_training") as run:
        run_id = run.info.run_id

        # Params
        if run_params:
            mlflow.log_params(run_params)

        mlflow.log_param("model_type", model_artifact.get("model_type", "elasticnet_loglog"))
        mlflow.log_param("n_features", len(feature_names))
        if elasticity_coef is not None:
            mlflow.log_param("global_price_elasticity", elasticity_coef)

        # Metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        model_uri = f"runs:/{run_id}/model"

    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "registered_model_name": registered_model_name,
    }
