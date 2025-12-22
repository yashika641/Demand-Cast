from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

def evaluate(y, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred))
    }

def train_promo_model_auto(X_train, y_train, X_val, y_val, X_test, y_test):
    n = len(X_train)

    if n < 5000:
        model_type = "linear"
        model = LinearRegression()

    elif n < 100000:
        model_type = "random_forest"
        model = RandomForestRegressor(n_estimators=400, max_depth=35, n_jobs=-1)

    else:
        model_type = "xgboost"
        model = XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror"
        )

    print(f"📌 Selected model: {model_type}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model,
        "model_type": model_type,
        "metrics": evaluate(y_test, preds),
        "preds": preds
    }
