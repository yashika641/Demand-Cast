import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

try:
    from lightgbm import LGBMRegressor
    lgb_available = True
except:
    lgb_available = False

try:
    from catboost import CatBoostRegressor
    cb_available = True
except:
    cb_available = False



# ------------------------------------------------------------
# Helper: evaluate regression model
# ------------------------------------------------------------
def evaluate_reg_model(model, X_val, y_val):
    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, preds)

    return {"mae": mae, "rmse": rmse, "r2": r2}



# ------------------------------------------------------------
# FINAL: Auto-select + train ONE model only
# NOW SUPPORTS:
#   - smoothing (MA window)
#   - variability (supplier variability feature)
# ------------------------------------------------------------
def train_leadtime_model_auto(
    X_train, y_train, 
    X_val, y_val,
    smoothing=7,
    variability=False
):
    """
    Selects ONE best model based on dataset size & structure.
    Adds:
        ✔ smoothing (rolling mean window)
        ✔ variability feature engineering
    """

    print("🔍 Detecting dataset characteristics...")

    num_features = X_train.select_dtypes(include=[np.number]).shape[1]
    obj_features = X_train.select_dtypes(include=['object']).shape[1]
    n_samples = len(X_train)

    print(f"➡ Numeric features: {num_features}")
    print(f"➡ Categorical features: {obj_features}")
    print(f"➡ Rows: {n_samples}")


    # ------------------------------------------------------------
    # 1️⃣ APPLY SMOOTHING (MOVING AVERAGE)
    # ------------------------------------------------------------
    if smoothing and smoothing > 1:
        print(f"📉 Applying smoothing: Rolling Mean (window={smoothing})")

        # Only smooth numeric features
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns

        X_train[numeric_cols] = (
            X_train[numeric_cols].rolling(window=smoothing, min_periods=1).mean()
        )
        X_val[numeric_cols] = (
            X_val[numeric_cols].rolling(window=smoothing, min_periods=1).mean()
        )


    # ------------------------------------------------------------
    # 2️⃣ ADD VARIABILITY FEATURE
    # ------------------------------------------------------------
    if variability:
        print("📈 Adding supplier variability feature...")

        # If supplier_id exists → create variability by supplier
        if "supplier_id" in X_train.columns:
            # Compute variance for each supplier
            var_map = (
                pd.concat([X_train, y_train], axis=1)
                .groupby("supplier_id")[y_train.name]
                .std()
                .fillna(0)
            )

            # Map to train and val
            X_train["supplier_variability"] = X_train["supplier_id"].map(var_map)
            X_val["supplier_variability"] = X_val["supplier_id"].map(var_map)
        else:
            print("⚠ variability=True but supplier_id column NOT found. Skipping.")



    # --------------------------------------------------------------------
    # MODEL SELECTION LOGIC
    # --------------------------------------------------------------------
    if obj_features > 10 and cb_available:
        print("🤖 Selecting CatBoost (many categorical features detected)...")
        model = CatBoostRegressor(
            depth=10,
            learning_rate=0.05,
            iterations=700,
            loss_function="RMSE",
            verbose=False
        )
        selected = "catboost"

    elif n_samples < 20_000:
        print("⚡ Selecting Linear Regression (small dataset)...")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearRegression())
        ])
        selected = "linear_regression"

    elif n_samples < 200_000:
        print("🌲 Selecting Random Forest (medium dataset)...")
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=50,
            n_jobs=-1
        )
        selected = "random_forest"

    elif n_samples >= 200_000 and n_samples <= 2_000_000 and lgb_available:
        print("💡 Selecting LightGBM (large dataset)...")
        model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.7,
            verbose=1
        )
        selected = "lightgbm"

    elif xgb_available:
        print("🚀 Selecting XGBoost (very large dataset / fallback)...")
        model = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=1
        )
        selected = "xgboost"

    else:
        print("🌲 Defaulting to RandomForest (fallback)...")
        model = RandomForestRegressor(
            n_estimators=400,
            n_jobs=-1
        )
        selected = "random_forest"


    # --------------------------------------------------------------------
    # TRAIN ONLY THE SELECTED MODEL
    # --------------------------------------------------------------------
    print(f"\n🚀 Training selected model: {selected} ...\n")
    model.fit(X_train, y_train)

    # --------------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------------
    metrics = evaluate_reg_model(model, X_val, y_val)

    print("📊 Evaluation:", metrics)
    print(f"🏆 FINAL MODEL USED: {selected}")

    return model, selected, metrics
