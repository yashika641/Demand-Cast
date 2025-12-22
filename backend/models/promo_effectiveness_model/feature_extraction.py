import pandas as pd

def merge_transactions_with_promos(trans_df, promo_df):
    """
    Merges transaction-level data with daily promo master.

    Steps:
    1) Aggregate transactions → daily product sales
    2) Merge with promo info
    """

    # --- Aggregate transactions ---
    daily_sales = (
        trans_df.groupby(["date", "sku_id", "location_id"])
        .agg(
            units_sold=("units_sold", "sum"),
            total_revenue=("price", "sum"),
            avg_price=("price", "mean"),
            on_promo=("on_promotion", "max")
        )
        .reset_index()
    )

    # --- Merge with promo master ---
    merged = daily_sales.merge(
        promo_df,
        on=["date", "sku_id", "location_id"],
        how="left"
    )

    # Fill missing promo fields (days without promo)
    merged["discount_percent"] = merged["discount_percent"].fillna(0)
    merged["promo_type"] = merged["promo_type"].fillna("None")

    # Final dataset
    return merged


def extract_promo_features(df, lookback=180, discount_sensitivity=True):
    df = df.sort_values(["sku_id", "location_id", "date"]).copy()

    # ---- Lag features ----
    df["lag_1"] = df.groupby(["sku_id", "location_id"])["units_sold"].shift(1)
    df[f"lag_{lookback}"] = df.groupby(["sku_id", "location_id"])["units_sold"].shift(lookback)

    # ---- Rolling baseline ----
    df[f"roll_mean_{lookback}"] = (
        df.groupby(["sku_id", "location_id"])["units_sold"]
        .shift(1)
        .rolling(lookback)
        .mean()
    )

    # ---- Uplift ----
    df["baseline"] = df[f"roll_mean_{lookback}"]
    df["promo_uplift"] = (df["units_sold"] - df["baseline"]) / df["baseline"]

    # ---- Discount Sensitivity ----
    if discount_sensitivity:
        df["elasticity"] = (df["avg_price"].pct_change() * -1).fillna(0)
        df["discount_effect"] = df["discount_percent"] / 100
    else:
        df["elasticity"] = 0
        df["discount_effect"] = 0

    # ---- Promo type encoding ----
    df["promo_type"] = df["promo_type"].astype("category").cat.codes

    df = df.dropna()
    return df
