import mlflow
import mlflow.sklearn
from datetime import datetime

def log_leadtime_model_to_mlflow(model, model_name, metrics, params=None, experiment_name="LeadTimeModel"):
    """
    Logs the trained lead time model to MLflow with metrics and parameters.
    
    Parameters:
        model: trained model
        model_name: name of the model (e.g., "xgboost")
        metrics: dict of rmse, mae, r2
        params: model parameters (optional)
        experiment_name: which MLflow experiment to log to
    """

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now()}"):

        # Log model metadata
        mlflow.log_param("model_type", model_name)

        # Log parameters
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        # Log metrics
        for mname, mvalue in metrics.items():
            mlflow.log_metric(mname, float(mvalue))

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"📦 Model logged to MLflow with run_id: {run_id}")

        return run_id
def load_leadtime_model_from_mlflow(run_id):
    """
    Loads a lead time model from MLflow using run_id.
    """
    model_uri = f"runs:/{run_id}/model"
    print(f"🔄 Loading model from {model_uri} ...")

    model = mlflow.sklearn.load_model(model_uri)
    return model
import pandas as pd

def predict_leadtime_mlflow(model, X_input):
    """
    Predict lead time using MLflow-loaded model.
    Ensures X_input is a DataFrame and columns are aligned.
    """
    if not isinstance(X_input, pd.DataFrame):
        X_input = pd.DataFrame([X_input])

    preds = model.predict(X_input)

    return preds
