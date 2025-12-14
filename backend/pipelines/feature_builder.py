import pandas as pd, numpy as np

def build_features_churn(customers, transactions, activity=None, support=None):
    # Base customers
    df = customers[["customer_id"]].drop_duplicates().copy()

    # Transactions (RFM)
    if transactions is not None:
        transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")
        ref_date = transactions["date"].max()
        rfm = transactions.groupby("customer_id").agg({
            "date": "max", "amount": ["sum", "count"]
        })
        rfm.columns = ["last_purchase", "total_spent", "purchase_count"]
        rfm["recency_days"] = (ref_date - rfm["last_purchase"]).dt.days
        rfm["frequency_mo"] = rfm["purchase_count"] / ((ref_date - transactions["date"].min()).days/30)
        df = df.merge(rfm.drop(columns=["last_purchase"]), on="customer_id", how="left")

    # Activity (Engagement)
    if activity is not None and "sessions" in activity.columns:
        eng = activity.groupby("customer_id").agg({"sessions":"sum","session_duration":"mean"}).reset_index()
        df = df.merge(eng, on="customer_id", how="left")

    # Support (Complaints)
    if support is not None:
        s = support.groupby("customer_id").agg({"ticket_id":"count","sentiment_score":"mean"}).reset_index()
        s.columns = ["customer_id","complaints","avg_sentiment"]
        df = df.merge(s, on="customer_id", how="left")

    # Fill missing
    df = df.fillna(0)
    df["engagement_index"] = (0.5*np.tanh(df["sessions"]/10) + 0.5*np.tanh(df["session_duration"]/30))
    df["support_burden"] = df["complaints"]*(1-df["avg_sentiment"])

    return df

import pandas as pd
import numpy as np

def sales_forecasting_feature_builder(df, 
                             date_col='date', 
                             sales_col='sales',
                             group_cols=None,  # e.g. ['product_id', 'region']
                             lags=[1, 7, 30], 
                             rolling_windows=[7, 30], 
                             add_date_parts=True):
    """
    Universal feature builder for sales forecasting.

    Params:
    - df: input dataframe, must contain date_col and sales_col
    - date_col: name of the date column
    - sales_col: name of the sales column
    - group_cols: columns to group by in rolling/lags (optional)
    - lags: list of lags (in days)
    - rolling_windows: list of days for rolling features
    - add_date_parts: whether to add date decomposition features

    Returns:
    - df_fe: dataframe with new features added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(date_col)

    if add_date_parts:
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['week'] = df[date_col].dt.isocalendar().week
        df['day'] = df[date_col].dt.day
        df['weekday'] = df[date_col].dt.weekday
        df['quarter'] = df[date_col].dt.quarter
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # Lags and rolling
    base_group = group_cols if group_cols else []
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(base_group)[sales_col].shift(lag) if base_group else df[sales_col].shift(lag)

    for window in rolling_windows:
        df[f'sales_roll_mean_{window}'] = df.groupby(base_group)[sales_col].shift(1).rolling(window).mean().reset_index(level=base_group, drop=True) if base_group else df[sales_col].shift(1).rolling(window).mean()
        df[f'sales_roll_std_{window}'] = df.groupby(base_group)[sales_col].shift(1).rolling(window).std().reset_index(level=base_group, drop=True) if base_group else df[sales_col].shift(1).rolling(window).std()

    # Optionally: aggregate statistics by group or date
    if group_cols:
        for col in group_cols:
            df[f'{col}_sales_mean'] = df.groupby(col)[sales_col].transform('mean')
            df[f'{col}_sales_median'] = df.groupby(col)[sales_col].transform('median')
            df[f'{col}_sales_max'] = df.groupby(col)[sales_col].transform('max')

    # Fill NAs (from lags/rolling)
    df = df.fillna(0)

    return df
