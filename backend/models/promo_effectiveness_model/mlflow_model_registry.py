import mlflow
import mlflow.sklearn

def log_promo_model_to_mlflow(model, model_type, X_train, y_train, X_test, y_test, metrics, lookback, discount_sensitivity, run_name):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("lookback", lookback)
        mlflow.log_param("discount_sensitivity", discount_sensitivity)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        return run.info.run_id
