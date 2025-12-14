# run_pipeline.py

import argparse
import pandas as pd

from feature_extraction import extract_price_elasticity_features
from train_test_split import time_based_train_val_test_split
from model_training import train_price_elasticity_model
from mlflow_registry import log_price_elasticity_model_to_mlflow

# column mapper
from models.stockout.col_mapper import map_column   # <-- YOUR EXISTING FUNCTION


# ---------------------------------------------------------
# UNIVERSAL COLUMN DETECTOR
# ---------------------------------------------------------
def auto_detect_columns(df):
    """
    Detect correct column names using the column mapper logic.
    Returns a dict with canonical column names.
    """

    required_cols = {
        "date": ["date", "transaction_date", "day", "datetime"],
        "product_id": ["product_id", "sku", "item_id", "product", "sku_id"],
        "price": ["price", "unit_price", "selling_price"],
        "quantity": ["quantity", "qty", "units_sold", "sales_units"],
        "promo_flag": ["promo_flag", "promotion", "is_promo", "on_promo"],
        "competitor_price": ["competitor_price", "comp_price", "market_price"]
    }

    detected = {}
    for canonical, synonyms in required_cols.items():
        detected[canonical] = map_column(df, synonyms)

    return detected


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def run_price_elasticity_pipeline(
        df: pd.DataFrame | None = None,
        input_csv_path: str | None = None,
        tracking_uri: str | None = None,
        experiment_name: str = "price_elasticity",
        registered_model_name: str = "price_elasticity_model",
):

    # 1. Load / validate df
    if df is None:
        if input_csv_path is None:
            raise ValueError("Either df or input_csv_path must be provided.")
        print(f"📥 Loading data from {input_csv_path}")
        df = pd.read_csv(input_csv_path)
    else:
        print("📥 Using dataframe passed into pipeline...")

    print(f"🔎 Incoming DF Shape: {df.shape}")

    # 2. Automatic column detection
    print("🧠 Auto-detecting column names using col_mapper...")
    cols = auto_detect_columns(df)

    date_col = cols["date"]
    product_col = cols["product_id"]
    price_col = cols["price"]
    qty_col = cols["quantity"]
    promo_flag_col = cols["promo_flag"]
    competitor_price_col = cols["competitor_price"]

    print("📌 Column Mapping:")
    print(cols)

    # 3. Feature Extraction
    print("🔧 Extracting price elasticity features...")
    X, y, meta = extract_price_elasticity_features(
        df,
        date_col=date_col,
        product_col=product_col,
        price_col=price_col,
        qty_col=qty_col,
        promo_flag_col=promo_flag_col,
        competitor_price_col=competitor_price_col,
    )

    # ensure date is present in meta for splitting
    if date_col not in meta.columns:
        meta[date_col] = pd.to_datetime(df.loc[meta.index, date_col])

    # 4. Train/Val/Test split
    print("✂️ Performing time-based split...")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        meta_train,
        meta_val,
        meta_test,
    ) = time_based_train_val_test_split(
        X, y, meta, date_col=date_col, val_size=0.1, test_size=0.2
    )

    # 5. Train model
    print("🤖 Training model...")
    model_artifact = train_price_elasticity_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # 6. Evaluate
    print("📊 Evaluating on test set...")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred_test = model_artifact["model"].predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    test_r2 = r2_score(y_test, y_pred_test)

    model_artifact["metrics"].update({
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
    })

    print("✅ Final Test Metrics:")
    print(model_artifact["metrics"])

    # 7. MLflow Logging
    print("📝 Logging model to MLflow...")
    mlflow_info = log_price_elasticity_model_to_mlflow(
        model_artifact=model_artifact,
        run_params={"used_dataframe": df.shape},
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
    )

    print("🎯 MLflow Run Completed:")
    print(mlflow_info)

    return {
        "model_artifact": model_artifact,
        "mlflow": mlflow_info,
        "columns_used": cols
    }


# ---------------------------------------------------------
# CLI SUPPORT (still available)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 👉 Just edit this path whenever you want to test on a new file
    csv_path = r"C:\Users\palya\Desktop\demancast\data\price_elasticity_sample.csv"

    print(f"📥 Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 👉 Run full pipeline using the dataframe
    results = run_price_elasticity_pipeline(df=df)

    print("\n🎉 Pipeline Completed Successfully!")
    print("Results Summary:")
    print(results)

