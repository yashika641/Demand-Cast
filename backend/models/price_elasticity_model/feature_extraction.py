# feature_extraction.py

import pandas as pd
import numpy as np


def extract_price_elasticity_features(
    df: pd.DataFrame,
    date_col: str = "date",
    product_col: str = "product_id",
    price_col: str = "price",
    qty_col: str = "quantity",
    promo_flag_col: str | None = "promo_flag",
    competitor_price_col: str | None = "competitor_price",
    min_history: int = 30,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build features for price elasticity estimation using a log-log demand model.

    Assumptions:
      - You have transactional data per product per date.
      - quantity is the demand (units sold).
      - price is the observed selling price.

    Returns:
      X: feature matrix
      y: target (log_quantity)
      meta: dataframe with [date, product_id, raw_price, raw_quantity] for later use
    """

    df = df.copy()

    # Basic cleaning
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([product_col, date_col])

    # Drop rows with non-positive price or quantity (log cannot handle <= 0)
    df = df[(df[price_col] > 0) & (df[qty_col] > 0)].copy()

    # Log transforms (classic price elasticity in log-log space)
    df["log_price"] = np.log(df[price_col])
    df["log_quantity"] = np.log(df[qty_col])

    # Time-based features
    df["dow"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)

    # Within-product rolling stats
    def add_group_lags(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col)

        # Lags on quantity
        g["lag_qty_1"] = g[qty_col].shift(1)
        g["lag_qty_7"] = g[qty_col].shift(7)
        g["lag_qty_28"] = g[qty_col].shift(28)

        # Rolling quantity
        g["rolling_qty_7"] = g[qty_col].rolling(window=7, min_periods=3).mean()
        g["rolling_qty_28"] = g[qty_col].rolling(window=28, min_periods=7).mean()

        # Price lags & rolling
        g["lag_price_1"] = g[price_col].shift(1)
        g["lag_price_7"] = g[price_col].shift(7)
        g["rolling_price_7"] = g[price_col].rolling(window=7, min_periods=3).mean()

        return g

    df = df.groupby(product_col, group_keys=False).apply(add_group_lags)

    # Optional: competitor price gap
    if competitor_price_col is not None and competitor_price_col in df.columns:
        df["competitor_gap"] = df[competitor_price_col] - df[price_col]

    # Optional: promo flag as int
    if promo_flag_col is not None and promo_flag_col in df.columns:
        df["promo_flag_int"] = df[promo_flag_col].astype(int)
    else:
        df["promo_flag_int"] = 0

    # Drop early-history rows with too many NaNs
    df = df.groupby(product_col).filter(lambda g: len(g) >= min_history)
    df = df.dropna(subset=["lag_qty_1", "lag_price_1"])

    # Meta (for later reporting / curves)
    meta_cols = [date_col, product_col, price_col, qty_col]
    meta = df[meta_cols].rename(
        columns={
            price_col: "raw_price",
            qty_col: "raw_quantity",
        }
    )

    # Feature set
    feature_cols = [
        "log_price",
        "promo_flag_int",
        "dow",
        "month",
        "weekofyear",
        "lag_qty_1",
        "lag_qty_7",
        "lag_qty_28",
        "rolling_qty_7",
        "rolling_qty_28",
        "lag_price_1",
        "lag_price_7",
        "rolling_price_7",
    ]

    if "competitor_gap" in df.columns:
        feature_cols.append("competitor_gap")

    X = df[feature_cols].copy()
    y = df["log_quantity"].copy()

    return X, y, meta
