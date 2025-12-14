# model_training.py

from typing import Dict, Any
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_price_elasticity_model(
    X_train,
    y_train,
    X_val,
    y_val,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a log-log demand model:
        log(Q) ~ f(log(P), promo, lags, etc.)

    Currently uses ElasticNetCV.
    """

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=None,
        cv=5,
        n_jobs=-1,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # Eval on val set
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)

    metrics = {
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "val_r2": float(r2),
    }

    # Approximate price elasticity:
    # If "log_price" is one of the features, its coefficient is elasticity
    feature_names = list(X_train.columns)
    elasticity_coef = None
    if "log_price" in feature_names:
        idx = feature_names.index("log_price")
        elasticity_coef = float(model.coef_[idx])

    return {
        "model": model,
        "metrics": metrics,
        "feature_names": feature_names,
        "elasticity_coef_global": elasticity_coef,
        "model_type": "elasticnet_loglog",
    }
