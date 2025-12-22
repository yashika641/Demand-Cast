import pandas as pd
import numpy as np

def extract_leadtime_features(df,
                              order_date_col,
                              delivery_date_col,
                              supplier_col,
                              sku_col=None,
                              quantity_col=None,
                              price_col=None):
    """
    Extracts all ML-ready features for Lead Time Prediction.
    Works even if some optional columns are missing.
    """

    df = df.copy()

    # -------------------------------------------------------
    # 1. BASIC VALIDATION
    # -------------------------------------------------------
    # if order_date_col not in df.columns or delivery_date_col not in df.columns:
    #     raise ValueError("Order date and delivery date columns are required.")

    # Ensure datetime formats
    df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
    df[delivery_date_col] = pd.to_datetime(df[delivery_date_col], errors='coerce')

    # -------------------------------------------------------
    # 2. LEAD TIME TARGET
    # -------------------------------------------------------
    df["lead_time_days"] = (df[delivery_date_col] - df[order_date_col]).dt.days

    # -------------------------------------------------------
    # 3. DATE FEATURES (Order & Delivery)
    # -------------------------------------------------------
    # Order date features
    df["order_day"] = df[order_date_col].dt.day
    df["order_week"] = df[order_date_col].dt.isocalendar().week.astype(int)
    df["order_month"] = df[order_date_col].dt.month
    df["order_quarter"] = df[order_date_col].dt.quarter
    df["order_year"] = df[order_date_col].dt.year
    df["order_day_of_week"] = df[order_date_col].dt.dayofweek
    df["order_is_weekend"] = df["order_day_of_week"].isin([5, 6]).astype(int)

    # Delivery date features
    df["delivery_day"] = df[delivery_date_col].dt.day
    df["delivery_week"] = df[delivery_date_col].dt.isocalendar().week.astype(int)
    df["delivery_month"] = df[delivery_date_col].dt.month
    df["delivery_quarter"] = df[delivery_date_col].dt.quarter
    df["delivery_year"] = df[delivery_date_col].dt.year
    df["delivery_day_of_week"] = df[delivery_date_col].dt.dayofweek
    df["delivery_is_weekend"] = df["delivery_day_of_week"].isin([5, 6]).astype(int)

    # -------------------------------------------------------
    # 4. SUPPLIER HISTORICAL AGGREGATES
    # -------------------------------------------------------
    if supplier_col in df.columns:
        supplier_agg = df.groupby(supplier_col)["lead_time_days"].agg(
            supplier_leadtime_mean="mean",
            supplier_leadtime_std="std",
            supplier_order_count="count"
        ).reset_index()

        df = df.merge(supplier_agg, on=supplier_col, how="left")

        # Delay rate calculation
        df["is_delayed"] = (df["lead_time_days"] > df["lead_time_days"].median()).astype(int)
        supplier_delay = df.groupby(supplier_col)["is_delayed"].mean().reset_index()
        supplier_delay.columns = [supplier_col, "supplier_delay_rate"]
        df = df.merge(supplier_delay, on=supplier_col, how="left")

    else:
        df["supplier_leadtime_mean"] = np.nan
        df["supplier_leadtime_std"] = np.nan
        df["supplier_order_count"] = np.nan
        df["supplier_delay_rate"] = np.nan

    # -------------------------------------------------------
    # 5. PRODUCT / SKU FEATURE AGGREGATION
    # -------------------------------------------------------
    if sku_col and sku_col in df.columns:
        sku_agg = df.groupby(sku_col)["lead_time_days"].agg(
            sku_leadtime_mean="mean",
            sku_leadtime_std="std",
            sku_order_count="count"
        ).reset_index()

        df = df.merge(sku_agg, on=sku_col, how="left")
    else:
        df["sku_leadtime_mean"] = np.nan
        df["sku_leadtime_std"] = np.nan
        df["sku_order_count"] = np.nan

    # -------------------------------------------------------
    # 6. ORDER / PURCHASE ORDER FEATURES
    # -------------------------------------------------------
    if quantity_col and quantity_col in df.columns:
        df["is_bulk_order"] = (df[quantity_col] > df[quantity_col].median()).astype(int)
        df["order_quantity"] = df[quantity_col]
    else:
        df["is_bulk_order"] = np.nan
        df["order_quantity"] = np.nan

    if price_col and price_col in df.columns:
        df["order_value"] = df["order_quantity"] * df[price_col]
    else:
        df["order_value"] = np.nan

    # -------------------------------------------------------
    # 7. ROLLING WINDOW FEATURES (SUPPLIER & SKU)
    # -------------------------------------------------------
    # Rolling windows require sorting
    df = df.sort_values(order_date_col)

    if supplier_col in df.columns:
        df["supplier_lt_7d_avg"] = (
            df.groupby(supplier_col)["lead_time_days"]
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df["supplier_lt_30d_avg"] = (
            df.groupby(supplier_col)["lead_time_days"]
            .rolling(30, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    else:
        df["supplier_lt_7d_avg"] = np.nan
        df["supplier_lt_30d_avg"] = np.nan

    if sku_col and sku_col in df.columns:
        df["sku_lt_7d_avg"] = (
            df.groupby(sku_col)["lead_time_days"]
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df["sku_lt_30d_avg"] = (
            df.groupby(sku_col)["lead_time_days"]
            .rolling(30, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    else:
        df["sku_lt_7d_avg"] = np.nan
        df["sku_lt_30d_avg"] = np.nan

    return df
