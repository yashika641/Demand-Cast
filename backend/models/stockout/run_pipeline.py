import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from models.stockout.col_mapper import map_column
from models.stockout.function_extraction import extract_stockout_features
from models.stockout.class_imbalance_check import handle_class_imbalance
from models.stockout.train_test_split import train_test_split_time_from_xy
from models.stockout.model_training import train_stockout_model
from models.stockout.mlflow_registry import log_stockout_model_to_mlflow, predict_from_mlflow_run
from models.stockout.data_preprocess import preprocess_data
from models.stockout.utils import date_col_syn as date_col_syn, delivery_date_col_syn as delivery_date_col_syn, order_date_col_syn as order_date_col_syn, price_col_syn as price_col_syn, product_id_col_syn as product_id_col_syn, quantity_col_syn as quantity_col_syn, sales_col_syn as sales_col_syn, stock_col_syn as stock_col_syn, stockout_target_syn as stockout_target_syn, supplier_col_syn as supplier_col_syn



import json
from tqdm import tqdm
# run_stockout_pipeline.py

from typing import Any, Dict, Optional

def run_stockout_pipeline(
    sales_df,
    inventory_df,
    products_df,
    mlflow_run_name: str = "stockout_training_run",
    callback: Optional[callable] = None,
    class_balancing: str = 'auto',
    model_type: str = 'xgboost'
) -> Dict[str, Any]:
    """
    Full stockout training pipeline with support for:
    ✔ custom class balancing mode
    ✔ manual model_type override from UI
    ✔ MLflow logging
    """

    def report(stage: str, percent: int, message: str):
        print(f"[{stage:>10}][{percent:3d}%] {message}")
        if callback:
            try:
                callback(stage, percent, message)
            except Exception as e:
                print(f"[CALLBACK ERROR] {e}")

    report("PIPELINE", 0, "📌 Pipeline started")

    # ------------------------------------------------------------------
    # STEP 1: COLUMN MAPPING
    # ------------------------------------------------------------------
    report("STEP 1", 5, "Column mapping started")

    date_col_sales = map_column(sales_df, date_col_syn)
    prod_col_sales = map_column(sales_df, product_id_col_syn)
    sales_col = map_column(sales_df, sales_col_syn)

    prod_col_inventory = map_column(inventory_df, product_id_col_syn)
    stock_col = map_column(inventory_df, stock_col_syn)
    prod_col_products = map_column(products_df, product_id_col_syn)

    sales_df = sales_df.rename(columns={
        date_col_sales: "date",
        prod_col_sales: "product_id",
        sales_col: "sales",
    })
    inventory_df = inventory_df.rename(columns={
        prod_col_inventory: "product_id",
        stock_col: "stock",
    })
    products_df = products_df.rename(columns={
        prod_col_products: "product_id",
    })

    report("STEP 1", 15, "Columns mapped")

    # ------------------------------------------------------------------
    # STEP 2: MERGE
    # ------------------------------------------------------------------
    report("STEP 2", 20, "Merging datasets")

    date_col_inv = map_column(inventory_df, date_col_syn)
    inventory_df = inventory_df.rename(columns={date_col_inv: "date"})

    merged = (
        sales_df.merge(inventory_df, on=["product_id", "date"], how="left")
                .merge(products_df, on="product_id", how="left")
                .dropna()
                .reset_index(drop=True)
    )

    report("STEP 2", 30, f"Merged shape = {merged.shape}")

    # ------------------------------------------------------------------
    # STEP 3: PREPROCESSING
    # ------------------------------------------------------------------
    report("STEP 3", 35, "Preprocessing started")

    df_pre = preprocess_data(merged)
    report("STEP 3", 45, f"Preprocessed shape = {df_pre.shape}")

    # ------------------------------------------------------------------
    # STEP 4: FEATURE ENGINEERING
    # ------------------------------------------------------------------
    report("STEP 4", 50, "Feature engineering")

    df_feat = extract_stockout_features(df_pre)

    target_col = "stockout_label"
    if target_col not in df_feat.columns:
        raise ValueError("Missing 'stockout_label' target")

    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]

    # Encode objects
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str).factorize()[0]

    report("STEP 4", 60, "Feature engineering complete")

    # ------------------------------------------------------------------
    # STEP 5: TIME SERIES SPLIT
    # ------------------------------------------------------------------
    if "date" not in X.columns:
        raise ValueError("'date' column missing for split")

    report("STEP 5", 65, "Splitting time-series data")

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_time_from_xy(
        X, y, date_col="date"
    )

    for df in [X_train, X_val, X_test]:
        df.drop(columns=["date"], inplace=True)

    report("STEP 5", 72, f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # ------------------------------------------------------------------
    # STEP 6: CLASS IMBALANCE HANDLING
    # ------------------------------------------------------------------
    report("STEP 6", 75, f"Applying class balancing: {class_balancing}")

    if class_balancing == "auto":
        X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)
    elif class_balancing == "none":
        X_train_bal, y_train_bal = X_train, y_train
    else:
        raise ValueError("Unknown class_balancing mode")

    report("STEP 6", 80, f"Balancing complete: {y_train_bal.value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # STEP 7: MODEL TRAINING (manual override)
    # ------------------------------------------------------------------
    report("STEP 7", 85, f"Training model_type={model_type}")

    result = train_stockout_model(
        X_train=X_train_bal,
        y_train=y_train_bal,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        forced_model_type=model_type  # <<< ADD THIS TO YOUR TRAINER
    )
    
    model = result["model"]
    metrics = {"accuracy": result["accuracy"], "f1": result["f1"]}

    # ------------------------------------------------------------------
    # STEP 8: MLflow LOGGING
    # ------------------------------------------------------------------
    run_id = log_stockout_model_to_mlflow(
        model=model,
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metrics=metrics,
        run_name=mlflow_run_name,
    )

    report("STEP 8", 95, f"MLflow logged → {run_id}")

    preds, proba = predict_from_mlflow_run(run_id, X_test)

    report("PIPELINE", 100, "🎉 STOCKOUT PIPELINE COMPLETED")

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

import numpy as np
import pandas as pd

def monte_carlo_stock_curve(
    initial_stock: int,
    demand_history: list,
    future_days: int = 30,
    n_simulations: int = 500,
    
):
    """
    Generate Monte Carlo stock projection curves.

    Parameters:
        initial_stock (int): current stock
        demand_history (list): past daily demand values
        future_days (int): how many future days to simulate
        n_simulations (int): how many simulation runs

    Returns:
        DataFrame: shape (future_days, n_simulations)
                   each column = one simulation trajectory
    """

    demand_history = np.array(demand_history)

    # Output matrix
    simulations = np.zeros((future_days, n_simulations))

    for sim in range(n_simulations):
        stock = initial_stock
        trajectory = []

        for day in range(future_days):
            # Randomly sample demand from history
            demand = np.random.choice(demand_history)

            stock = max(stock - demand, 0)  # stock cannot go below zero
            trajectory.append(stock)

        simulations[:, sim] = trajectory

    # Convert to DataFrame for plotting
    df_simulations = pd.DataFrame(simulations)
    df_simulations.index = np.arange(1, future_days+1)

    return df_simulations


def main():
    # -----------------------------
    # 1. Load Data
    # -----------------
    sales_df = pd.read_csv(r"C:\Users\palya\Desktop\demancast\data\testB\SALES.csv")
    inventory_df= pd.read_csv(r"C:\Users\palya\Desktop\demancast\data\testB\INVENTORY.csv")
    products_df = pd.read_csv(r"C:\Users\palya\Desktop\demancast\data\testB\PRODUCTS.csv")
    
    # -----------------------------
    # 2. Run Pipeline
    # -----------------
    result = run_stockout_pipeline(
        sales_df,
        inventory_df,
        products_df,
        mlflow_run_name="stockout_training_run"
    )
    print(result)

if __name__ == "__main__":
    main()