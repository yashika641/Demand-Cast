import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.promo_effectiveness_model.feature_extraction import extract_promo_features , merge_transactions_with_promos
from models.promo_effectiveness_model.train_test_split import promo_train_test_split
from models.promo_effectiveness_model.model_training import train_promo_model_auto
from models.promo_effectiveness_model.mlflow_model_registry import log_promo_model_to_mlflow    
from models.stockout.col_mapper import map_column
from models.stockout.data_preprocess import preprocess_data
from models.stockout.utils import *

import pandas as pd

def run_promo_pipeline(
    trans_df: pd.DataFrame,
    promo_df: pd.DataFrame,
    lookback: int = 180,
    discount_sensitivity: bool = True,
    mlflow_run_name: str = "promo_effectiveness_run",
    callback=None
):
    """
    FULL Promotional Effectiveness Pipeline.

    Steps:
    1) Merge transaction-level data with promo master data
    2) Feature engineering (lookback + discount_sensitivity)
    3) Train/Val/Test split (time-based)
    4) Auto-select model & train
    5) Log to MLflow
    6) Return run_id, metrics, predictions

    Returns:
        dict {
            run_id,
            model_type,
            metrics,
            preds,
            full_features_df
        }
    """

    # ------------------------------
    # Helper for progress logging
    # ------------------------------
    def report(stage, pct, msg):
        print(f"[{stage}][{pct}%] {msg}")
        if callback:
            try:
                callback(stage, pct, msg)
            except:
                pass

    # ------------------------------
    # STEP 1 — MERGE DATASETS
    # ------------------------------
    report("STEP 1", 10, "Merging transactions with promo table")

    merged_df = merge_transactions_with_promos(trans_df, promo_df)

    if merged_df.empty:
        raise ValueError("Merged dataset is empty. Promo or transaction file mismatch.")

    report("STEP 1", 20, f"Merged shape = {merged_df.shape}")

    # ------------------------------
    # STEP 2 — FEATURE EXTRACTION
    # ------------------------------
    report("STEP 2", 30, "Extracting promo features")

    features_df = extract_promo_features(
        merged_df,
        lookback=lookback,
        discount_sensitivity=discount_sensitivity
    )

    if features_df.empty:
        raise ValueError("Feature extraction returned empty dataset (insufficient lookback).")

    report("STEP 2", 45, f"Final engineered features shape = {features_df.shape}")

    # ------------------------------
    # STEP 3 — TRAIN/VAL/TEST SPLIT
    # ------------------------------
    report("STEP 3", 55, "Performing train-val-test split")

    X_train, y_train, X_val, y_val, X_test, y_test = promo_train_test_split(features_df)
    print(X_train.dtypes)
    print(X_train.head())
    cols=X_train.select_dtypes(include=['object']).columns
    X_train.drop(columns=cols, inplace=True)
    X_val.drop(columns=cols, inplace=True)
    X_test.drop(columns=cols, inplace=True)


    print(X_train.dtypes)
    print(X_train.head())

    report("STEP 3", 65, "Split completed")

    # ------------------------------
    # STEP 4 — AUTO MODEL SELECTION + TRAINING
    # ------------------------------
    report("STEP 4", 70, "Auto-selecting & training best promo model")

    result = train_promo_model_auto(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    model = result["model"]
    model_type = result["model_type"]
    metrics = result["metrics"]
    preds = result["preds"]

    report("STEP 4", 90, f"Training done — model={model_type}, metrics={metrics}")

    # ------------------------------
    # STEP 5 — LOG TO MLFLOW
    # ------------------------------
    report("STEP 5", 95, "Logging promo model to MLflow")

    run_id = log_promo_model_to_mlflow(
        model=model,
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metrics=metrics,
        lookback=lookback,
        discount_sensitivity=discount_sensitivity,
        run_name=mlflow_run_name
    )

    report("STEP 5", 100, f"Model logged → run_id={run_id}")

    # ------------------------------
    # FINAL OUTPUT
    # ------------------------------
    return {
        "run_id": run_id,
        "model_type": model_type,
        "metrics": metrics,
        "predictions": preds,
        "full_features_df": features_df
    }

def main():
    trans_df = pd.read_csv(r'C:\Users\palya\Desktop\demancast\data\testB\SALES.csv')
    promo_df = pd.read_csv(r'C:\Users\palya\Desktop\demancast\data\testB\PRICING_PROMO.csv')
    results=run_promo_pipeline(trans_df, promo_df)
    print(results)

if __name__ == "__main__":
    main()