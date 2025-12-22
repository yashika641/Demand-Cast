# train_test_split.py

import pandas as pd
from typing import Tuple


def time_based_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    date_col: str = "date",
    val_size: float = 0.1,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series,
           pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split based on meta[date_col].

    Returns:
      X_train, X_val, X_test,
      y_train, y_val, y_test,
      meta_train, meta_val, meta_test
    """

    # We assume meta index aligns 1:1 with X and y
    df_all = meta.copy()
    df_all = df_all.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    df_all["__idx__"] = df_all.index
    df_all = df_all.sort_values(date_col)

    n = len(df_all)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("Not enough data to split into train/val/test.")

    train_idx = df_all["__idx__"].iloc[:n_train].tolist()
    val_idx = df_all["__idx__"].iloc[n_train:n_train + n_val].tolist()
    test_idx = df_all["__idx__"].iloc[n_train + n_val:].tolist()

    X_train = X.loc[train_idx].reset_index(drop=True)
    X_val = X.loc[val_idx].reset_index(drop=True)
    X_test = X.loc[test_idx].reset_index(drop=True)

    y_train = y.loc[train_idx].reset_index(drop=True)
    y_val = y.loc[val_idx].reset_index(drop=True)
    y_test = y.loc[test_idx].reset_index(drop=True)

    meta_train = meta.loc[train_idx].reset_index(drop=True)
    meta_val = meta.loc[val_idx].reset_index(drop=True)
    meta_test = meta.loc[test_idx].reset_index(drop=True)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        meta_train,
        meta_val,
        meta_test,
    )
