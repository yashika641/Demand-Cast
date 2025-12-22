def extract_demand_forecast_features(df, target_col="sales"):
    """
    Feature engineering for demand forecasting.
    Assumes df has: 
        - 'date' (datetime)
        - 'product_id'
        - target column (e.g., 'sales')

    Output:
        Returns df with rich time-series derived features:
        - time features
        - lag features
        - rolling windows
        - expanding features
        - product-level encodings
        - demand volatility indicators
    """

    import numpy as np
    import pandas as pd

    df = df.copy()

    # -------------------------------------------------------------------
    # 1. Ensure correct sorting
    # -------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(["product_id", "date"], inplace=True)

    # -------------------------------------------------------------------
    # 2. Time-based features
    # -------------------------------------------------------------------
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # -------------------------------------------------------------------
    # 3. Lag features
    # -------------------------------------------------------------------
    lag_list = [1, 2, 3, 7, 14, 30]

    for l in lag_list:
        df[f"{target_col}_lag_{l}"] = df.groupby("product_id")[target_col].shift(l)

    # -------------------------------------------------------------------
    # 4. Rolling window features
    # -------------------------------------------------------------------
    windows = [3, 7, 14, 30]

    for w in windows:
        df[f"{target_col}_rolling_mean_{w}"] = (
            df.groupby("product_id")[target_col].shift(1).rolling(w).mean()
        )
        df[f"{target_col}_rolling_std_{w}"] = (
            df.groupby("product_id")[target_col].shift(1).rolling(w).std()
        )
        df[f"{target_col}_rolling_min_{w}"] = (
            df.groupby("product_id")[target_col].shift(1).rolling(w).min()
        )
        df[f"{target_col}_rolling_max_{w}"] = (
            df.groupby("product_id")[target_col].shift(1).rolling(w).max()
        )

    # -------------------------------------------------------------------
    # 5. Expanding / Cumulative features
    # -------------------------------------------------------------------
    df["expanding_mean"] = (
        df.groupby("product_id")[target_col].expanding().mean().reset_index(level=0, drop=True)
    )
    df["expanding_std"] = (
        df.groupby("product_id")[target_col].expanding().std().reset_index(level=0, drop=True)
    )
    df["cumulative_sales"] = (
        df.groupby("product_id")[target_col].cumsum()
    )

    # -------------------------------------------------------------------
    # 6. Demand volatility
    # -------------------------------------------------------------------
    df["sales_diff_1"] = df.groupby("product_id")[target_col].diff(1)
    df["sales_diff_7"] = df.groupby("product_id")[target_col].diff(7)

    df["volatility_7"] = (
        df.groupby("product_id")[target_col]
        .rolling(7)
        .std()
        .reset_index(level=0, drop=True)
    )

    # -------------------------------------------------------------------
    # 7. Stock interaction features (only if stock column exists)
    # -------------------------------------------------------------------
    if "stock" in df.columns:
        df["stock_to_sales_ratio"] = df["stock"] / (df[target_col] + 1)
        df["is_low_stock"] = (df["stock"] < df[target_col] * 1.2).astype(int)

    # -------------------------------------------------------------------
    # 8. Product-level encodings
    # -------------------------------------------------------------------
    df["product_mean_sales"] = (
        df.groupby("product_id")[target_col].transform("mean")
    )
    df["product_std_sales"] = (
        df.groupby("product_id")[target_col].transform("std")
    )
    df["product_sales_rank"] = (
        df.groupby("date")[target_col].rank("dense")
    )

    # -------------------------------------------------------------------
    # 9. Fill missing values from lags/rolling windows
    # -------------------------------------------------------------------
    df.fillna(0, inplace=True)

    return df
