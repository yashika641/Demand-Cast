import os 
import sys
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.stockout.col_mapper import map_column
from models.stockout.data_preprocess import preprocess_data
from models.stockout.utils import *
from models.demand_forecast.feature_extraction import extract_demand_forecast_features
from models.demand_forecast.train_test_split import train_test_split_time_from_xy
from models.stockout.class_imbalance_check import handle_class_imbalance
from models.demand_forecast.model_training import train_demand_forecast_model
from models.demand_forecast.mlflow_model_regostry import log_demand_model_to_mlflow, predict_from_demand_mlflow_run

import json
from tqdm import tqdm
# run_stockout_pipeline.py

def run_demand_pipeline(
    sales_df,
    mlflow_run_name="leadtime_training_run",
    callback=None,
    horizon=7,seasonality='auto',model_type="xgboost"# kept for compatibility but NOT used
):
    """
    Clean synchronous full pipeline.
    NO SSE.
    NO callback usage.
    PRINT LOGGING ADDED for debugging.
    """

    print("📌 Pipeline Started")

    # -------------------------------------------
    # STEP 1: COLUMN MAPPING
    # -------------------------------------------
    print("🔄 Step 1: Column Mapping Started")

    date_col_sales = map_column(sales_df, date_col_syn)
    prod_col_sales = map_column(sales_df, product_id_col_syn)
    sales_col = map_column(sales_df, sales_col_syn)
    price_col = map_column(sales_df, price_col_syn)
    # prod_col_products = map_column(products_df, product_id_col_syn)

    sales_df = sales_df.rename(columns={
        date_col_sales: "date",
        prod_col_sales: "product_id",
        sales_col: "sales"
    })
    print("date_col_sales:", date_col_sales)
    print("prod_col_sales:", prod_col_sales)
    print("sales_col:", sales_col)
    print("✅ Step 1 Complete: Columns mapped")
    
    # -------------------------------------------
    # STEP 2: DATA PREPROCESSING
    # -------------------------------------------
    print("🔄 Step 2: Data Preprocessing Started")
    preprocess_df = preprocess_data(sales_df)
    print("✅ Step 2 Complete: Data Preprocessed")

    # -------------------------------------------
    # STEP 3: FEATURE EXTRACTION
    # -------------------------------------------
    print("🔄 Step 3: Feature Extraction Started")
    feature_df = extract_demand_forecast_features(preprocess_df)
    print("✅ Step 3 Complete: Features Extracted")

    # -------------------------------------------
    # STEP 4: TIME-SERIES SPLIT
    # -------------------------------------------
    print("🔄 Step 4: Time-Series Split Started")
    X_train, X_val, X_test, y_train, y_val, y_test= train_test_split_time_from_xy(
        feature_df,
        feature_df["sales"],
        date_col="date"
    )
    print("✅ Step 4 Complete: Time-Series Split Complete")
    
    # -------------------------------------------
    # STEP 5: CLASS IMBALANCE HANDLING
    # -------------------------------------------
    print("🔄 Step 5: Class Imbalance Handling Started")
    X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)
    print("✅ Step 5 Complete: Class Imbalance Handled")
    
    X_train.drop(columns=["date"], inplace=True)
    X_val.drop(columns=["date"], inplace=True)
    X_test.drop(columns=["date"], inplace=True)
    
    # -------------------------------------------
    # STEP 6: MODEL TRAINING
    # -------------------------------------------
    print("🔄 Step 6: Model Training Started")
    results= train_demand_forecast_model(
        X_train_bal, y_train_bal,
        X_val, y_val,
        X_test, y_test,
        full_df=feature_df,
        target_col="sales",
        horizon=horizon,
        seasonality=seasonality,
        model_type=model_type
    )
    print("✅ Step 6 Complete: Model Trained")

    print(results["metrics"],"results metrics")
    # -------------------------------------------
    # STEP 7: MLFLOW LOGGING
    # -------------------------------------------
    print("🔄 Step 7: MLflow Logging Started")
    run_id = log_demand_model_to_mlflow(
        model=results["model"],
        model_type=results["model_type"],
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        metrics=results["metrics"],
        run_name=mlflow_run_name
    )
    print("run_id:", run_id)
    print("✅ Step 7 Complete: Model Logged to MLflow")
    
    # -------------------------------------------
    # STEP 8: PREDICTION
    # # -------------------------------------------
    # print("🔄 Step 8: Prediction Started")
    # print("x_test:",X_test.dtypes)
    # preds, proba = predict_from_demand_mlflow_run(run_id, X_test)
    # print("✅ Step 8 Complete: Predictions Generated")
    # print(preds)
    # print(proba)
    # print("🎉 Pipeline Completed Successfully")
    return run_id


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
    import mlflow
    import pandas as pd
    import numpy as np

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

def main():
    # -----------------------------
    # 1. Load Data
    # -----------------
    sales_df = pd.read_csv(r"C:\Users\palya\Desktop\demancast\data\testB\SALES.csv")
    
    # -----------------------------
    # 2. Run Pipeline
    # -----------------
    run_demand_pipeline(
        sales_df,
        mlflow_run_name="demand_training_run"
    )

if __name__ == "__main__":
    main()