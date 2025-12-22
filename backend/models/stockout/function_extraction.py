import pandas as pd
import numpy as np
from .utils import *
from .col_mapper import map_column

def extract_stockout_features(df):
    df = df.copy()

    # -----------------------------------
    # 1. Detect columns using synonyms
    # -----------------------------------
    date_col = map_column(df, date_col_syn)
    product_col = map_column(df, product_id_col_syn)
    sales_col = map_column(df, sales_col_syn)
    stock_col = map_column(df, stock_col_syn)

    if any(x is None for x in [date_col, product_col, sales_col, stock_col]):
        raise ValueError("Required columns missing. Check synonyms or input dataframe.")

    # -----------------------------------
    # 2. Sort properly for time-series logic
    # -----------------------------------
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([product_col, date_col])

    # -----------------------------------
    # 3. Group by product for per-item features
    # -----------------------------------
    g = df.groupby(product_col)

    # -----------------------------------
    # 4. Lag Features (Demand History)
    # -----------------------------------
    df["lag_1"] = g[sales_col].shift(1)
    df["lag_7"] = g[sales_col].shift(7)
    df["lag_14"] = g[sales_col].shift(14)

    # -----------------------------------
    # 5. Rolling Mean / Std (Demand Trend)
    # -----------------------------------
    df["roll_mean_7"] = g[sales_col].rolling(7).mean().reset_index(level=0, drop=True)
    df["roll_std_7"] = g[sales_col].rolling(7).std().reset_index(level=0, drop=True)

    df["roll_mean_14"] = g[sales_col].rolling(14).mean().reset_index(level=0, drop=True)
    df["roll_std_14"] = g[sales_col].rolling(14).std().reset_index(level=0, drop=True)

    # -----------------------------------
    # 6. Stock Coverage Features
    # -----------------------------------
    df["stock_coverage_days"] = df[stock_col] / (df["roll_mean_7"] + 1e-6)

    # More coverage calculations
    df["stock_to_lag1_ratio"] = df[stock_col] / (df["lag_1"] + 1e-6)
    df["stock_to_lag7_ratio"] = df[stock_col] / (df["lag_7"] + 1e-6)

    # -----------------------------------
    # 7. Demand Surge Flags
    # -----------------------------------
    df["demand_surge"] = (df[sales_col] > df["roll_mean_7"] * 1.5).astype(int)

    # -----------------------------------
    # 8. Days Since Last Stockout
    # -----------------------------------
    df["was_stockout"] = (df[stock_col] <= 0).astype(int)
    df["days_since_stockout"] = (
        g["was_stockout"].cumsum() - g["was_stockout"].cumsum().where(df["was_stockout"]==1).ffill()
    )

    # -----------------------------------
    # 9. Create Target Column: Stockout Next Day
    # -----------------------------------
    df["stock_tomorrow"] = g[stock_col].shift(-1)
    df["stockout_label"] = (df["stock_tomorrow"] <= 0).astype(int)

    # -----------------------------------
    # 10. Drop rows with NaNs from lags/rolling
    # -----------------------------------
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    non_cat_cols = numerical_cols + ([date_col] if date_col else [])
    categorical_cols = [c for c in df.columns if c not in non_cat_cols]
    df.drop(columns=categorical_cols, inplace=True)

    df = df.dropna().reset_index(drop=True)
    
    return df
