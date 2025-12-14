import os 
import sys
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.stockout.col_mapper import map_column
from models.stockout.data_preprocess import preprocess_data
from models.stockout.utils import *
from models.leadtime.feature_extraction import extract_leadtime_features
from models.stockout.class_imbalance_check import handle_class_imbalance
from models.leadtime.traintestsplit import train_test_split_time_from_xy
from models.leadtime.model_training import train_leadtime_model_auto
from models.leadtime.model_registry_predictions import log_leadtime_model_to_mlflow,load_leadtime_model_from_mlflow,predict_leadtime_mlflow


import json
from tqdm import tqdm
# run_stockout_pipeline.py

def run_leadtime_pipeline(
    sales_df,
    inventory_df,
    suppliers_df,
    mlflow_run_name="leadtime_training_run",
    callback=None ,
    smoothing=7,
    variability=False# kept for compatibility but NOT used
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
    order_date_col = map_column(suppliers_df, order_date_col_syn)
    delivery_date_col = map_column(suppliers_df, delivery_date_col_syn)
    supplier_col = map_column(suppliers_df, supplier_col_syn)
    prod_col_inventory = map_column(inventory_df, product_id_col_syn)
    stock_col = map_column(inventory_df, stock_col_syn)
    sku_id_col = map_column(suppliers_df, product_id_col_syn)
    quantity_col = map_column(suppliers_df, quantity_col_syn)
    price_col = map_column(sales_df, price_col_syn)
    # prod_col_products = map_column(products_df, product_id_col_syn)

    sales_df = sales_df.rename(columns={
        date_col_sales: "date",
        prod_col_sales: "product_id",
        sales_col: "sales"
    })

    inventory_df = inventory_df.rename(columns={
        prod_col_inventory: "product_id",
        stock_col: "stock"
    })

    suppliers_df = suppliers_df.rename(columns={
        order_date_col: "order_date",
        delivery_date_col: "delivery_date",
        supplier_col: "supplier_id",
        sku_id_col: "product_id",
        quantity_col: "quantity",
        price_col: "price"
    })

    # FIX HERE — overwrite old column names with the new ones
    order_date_col = "order_date"
    delivery_date_col = "delivery_date"
    supplier_col = "supplier_id"
    sku_id_col = "product_id"
    quantity_col = "quantity"
    price_col = "price"

    print("order_date_col",order_date_col)
    print("delivery_date_col",delivery_date_col)
    print("supplier_col",supplier_col)
    print("✅ Step 1 Complete: Columns mapped")

    # -------------------------------------------
    # STEP 2: MERGING
    # -------------------------------------------
    print("🔄 Step 2: Merging Datasets")

    date_col_inv = map_column(inventory_df, date_col_syn)
    inventory_df.rename(columns={date_col_inv: "date"}, inplace=True)

    merged = sales_df.merge(inventory_df, on=["product_id", "date"], how="left")
    merged = merged.merge(suppliers_df, on="product_id", how="left")
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)

    print(f"✅ Step 2 Complete: Merged shape → {merged.shape}")

    # -------------------------------------------
    # STEP 3: PREPROCESSING
    # -------------------------------------------
    print("🔄 Step 3: Preprocessing Started")

    df_preprocess = preprocess_data(merged)

    print(f"✅ Step 3 Complete: Preprocessed shape → {df_preprocess.shape}")
    print("🔄 Step 4: Feature Engineering Started")

    print(df_preprocess.columns,df_preprocess.head())
    df_features=extract_leadtime_features(df_preprocess,
                              order_date_col,
                              delivery_date_col,
                              supplier_col,
                              sku_id_col,
                              quantity_col,
                              price_col)

    print(f"✅ Step 4 Complete: Features shape → {df_features.shape}")
    
    target_col = "lead_time_days"

    X = df_features.drop(columns=[target_col])
    y = df_features[target_col]

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype(str).factorize()[0]
            
    X_train, X_test, y_train, y_test = train_test_split_time_from_xy(
    X, y, date_col=order_date_col, split_ratio=0.8
)

    print("train test split done!")

    datetime_cols = [col for col in X_train.columns if "date" in col.lower() or 
                 str(X_train[col].dtype).startswith("datetime")]

    if datetime_cols:
        print("⚠️ FINAL CLEANUP: Dropping datetime columns:", datetime_cols)
        X_train = X_train.drop(columns=datetime_cols)
        X_test = X_test.drop(columns=datetime_cols)
    
    model,model_name,metrics=train_leadtime_model_auto(X_train,y_train,X_test,y_test,smoothing=smoothing,variability=variability)
    print("model trained!")
    
    run_id = log_leadtime_model_to_mlflow(
        model=model,
        model_name=model_name,
        metrics=metrics)
    model=load_leadtime_model_from_mlflow(run_id)
    preds=predict_leadtime_mlflow(model,X_test)
    print("predictions made")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    
    return {
        "model": model,
        "metrics": metrics, 
        "merged_df": merged,
        "final_features_df": df_features
    }
    
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
    inventory_df= pd.read_csv(r"C:\Users\palya\Desktop\demancast\data\testB\INVENTORY.csv")
    suppliers_df = pd.read_csv(r"C:\Users\palya\Desktop\demancast\data\testB\SUPPLIERS.csv")
    
    # -----------------------------
    # 2. Run Pipeline
    # -----------------
    result = run_leadtime_pipeline(
        sales_df,
        inventory_df,
        suppliers_df,
        mlflow_run_name="leadtime_training_run"
    )
    print(result)

if __name__ == "__main__":
    main()