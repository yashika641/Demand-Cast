import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .utils import *
from .col_mapper import map_column


def preprocess_data(df):
    df = df.copy()

    # -----------------------------
    # 1. Detect date column
    # -----------------------------
    date_col = map_column(df, date_col_syn)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        print("Warning: No date column detected")

    # -----------------------------
    # 2. Detect numeric columns safely
    # -----------------------------
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove product_id or ID-like columns
    product_col = map_column(df, product_id_col_syn)
    if product_col and product_col in numerical_cols:
        numerical_cols.remove(product_col)

    # Remove stockout target if exists
    target_col = map_column(df, stockout_target_syn)
    if target_col and target_col in numerical_cols:
        numerical_cols.remove(target_col)

    # Remove date column if mistakenly included
    if date_col in numerical_cols:
        numerical_cols.remove(date_col)

    # -----------------------------
    # 3. Scale numeric columns
    # -----------------------------
    if numerical_cols:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # -----------------------------
    # 4. Encode categorical columns
    # -----------------------------
    non_cat_cols = numerical_cols + ([date_col] if date_col else [])
    categorical_cols = [c for c in df.columns if c not in non_cat_cols]

    encoder = LabelEncoder()

    for col in categorical_cols:
        if df[col].dtype == "object" or df[col].dtype == "category":
            df[col] = encoder.fit_transform(df[col].astype(str))
    # df.drop(columns=categorical_cols, inplace=True)

    return df
