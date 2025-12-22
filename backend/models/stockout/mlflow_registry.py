import mlflow
import mlflow.sklearn
import pandas as pd
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import os

def log_stockout_model_to_mlflow(model, model_type, X_train, y_train, X_test, y_test, metrics, run_name="stockout_model_run"):
    """
    Logs model, dataset, and metrics to MLflow and returns run_id.
    """

    with mlflow.start_run(run_name=run_name) as run:

        # Get run_id
        run_id = run.info.run_id

        # 2. Log Parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("num_features", X_train.shape[1])

        # 3. Log Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # 4. Log Dataset Artifacts
        os.makedirs("mlflow_datasets", exist_ok=True)

        train_path = "mlflow_datasets/train_data.csv"
        test_path = "mlflow_datasets/test_data.csv"

        train_df = X_train.copy()
        train_df["target"] = y_train
        train_df.to_csv(train_path, index=False)

        test_df = X_test.copy()
        test_df["target"] = y_test
        test_df.to_csv(test_path, index=False)

        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)

        # 5. Log the Model
        if model_type == "lstm":
            mlflow.tensorflow.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        print(f"✔ Model logged successfully. Run ID: {run_id}")

        return run_id

import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

def predict_from_mlflow_run(run_id: str, X_test: pd.DataFrame, return_proba: bool = True):
    """
    Loads a model from MLflow using run_id and performs predictions.

    Parameters:
        run_id: MLflow run ID (returned during training)
        X_test: Pandas DataFrame of test features
        return_proba: bool → whether to return probabilities

    Returns:
        dict → {"predictions": np.array, "probabilities": np.array or None}
    """

    # -----------------------------
    # 1. Load model from MLflow
    # -----------------------------
    model_uri = f"runs:/{run_id}/model"

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise Exception(f"❌ Failed to load model from MLflow: {e}")

    # -----------------------------
    # 2. Make Predictions
    # -----------------------------
    try:
        preds = model.predict(X_test)
    except Exception as e:
        raise Exception(f"❌ Prediction failed: {e}")

    # -----------------------------
    # 3. Try extracting probabilities (if supported)
    # -----------------------------
    proba = None
    if return_proba:
        try:
            # sklearn models
            if hasattr(model._model_impl, "predict_proba"):
                proba = model._model_impl.predict_proba(X_test)

            # Keras models (LSTM)
            elif hasattr(model._model_impl, "predict"):
                raw = model._model_impl.predict(X_test)
                proba = raw if raw.ndim > 1 else np.column_stack([1 - raw, raw])

        except Exception:
            print("⚠ Probability extraction not supported for this model.")

    return preds, proba

