
def log_demand_model_to_mlflow(
    model,
    model_type,
    X_train, y_train,
    X_test, y_test,
    metrics,
    run_name="demand_forecast_run",
):
    import mlflow
    import mlflow.sklearn
    import mlflow.keras
    """
    Logs demand forecast model to MLflow model registry.
    Supports SKLearn, XGBoost, Keras, Prophet, NeuralForecast.

    Returns:
        run_id (str)
    """

    mlflow.set_experiment("DemandForecasting")
    
    with mlflow.start_run(run_name=run_name) as run:

        run_id = run.info.run_id

        # -------------------------------------------------------
        # 1. Log parameters + metrics
        # -------------------------------------------------------
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics(metrics)

        # -------------------------------------------------------
        # 2. Log feature columns
        # -------------------------------------------------------
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # -------------------------------------------------------
        # 3. Log model according to type
        # -------------------------------------------------------
        if model_type in ["XGBOOST", "RANDOMFOREST"]:
            mlflow.sklearn.log_model(model, artifact_path="model")

        elif model_type == "LSTM":
            mlflow.keras.log_model(model, artifact_path="model")

        elif model_type == "PROPHET":
            import mlflow.pyfunc
            mlflow.prophet.log_model(model, artifact_path="model")

        elif model_type in ["TFT", "N-BEATS", "DEEPAR"]:
            # Save entire NeuralForecast object
            import pickle
            bk = f"neuralforecast_model.pkl"
            with open(bk, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(bk, artifact_path="model")

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # -------------------------------------------------------
        # 4. Register model in MLflow Registry
        # -------------------------------------------------------
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="DemandForecastModel"
        )

        return run_id

def predict_from_demand_mlflow_run(run_id, X_input):
    """
    Loads a demand forecast model from MLflow by run ID and performs predictions.

    Supports:
      - SKLearn / XGBoost
      - Keras LSTM
      - Prophet
      - NeuralForecast models
    """

    import mlflow
    import numpy as np
    import pandas as pd

    # 1. Load model using MLflow generic loader
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    # 2. PyFunc models expect Pandas DataFrame
    if not isinstance(X_input, pd.DataFrame):
        X_input = pd.DataFrame(X_input)

    # 3. Prediction
    preds = model.predict(X_input)

    # Optional probability output
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)
    except Exception:
        proba = None

    return preds, proba
