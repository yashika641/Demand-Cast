def train_test_split_time_from_xy(X, y, date_col="date", 
                                  train_ratio=0.70,
                                  val_ratio=0.15,
                                  test_ratio=0.15):
    """
    Time-series aware split for demand forecasting.

    Parameters:
        X (pd.DataFrame) : features including 'date'
        y (pd.Series)    : target values
        date_col (str)   : name of date column
        train_ratio      : train %
        val_ratio        : validation %
        test_ratio       : test %

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    import pandas as pd
    import numpy as np

    # -------------------------------------------------------------------
    # 1. Ensure date column exists
    # -------------------------------------------------------------------
    if date_col not in X.columns:
        raise ValueError(f"Date column '{date_col}' not found in X.")

    # -------------------------------------------------------------------
    # 2. Sort by date (very important)
    # -------------------------------------------------------------------
    df = X.copy()
    df["__target__"] = y.values
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -------------------------------------------------------------------
    # 3. Compute split indices
    # -------------------------------------------------------------------
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # -------------------------------------------------------------------
    # 4. Create splits
    # -------------------------------------------------------------------
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    X_train = train_df.drop(columns=["__target__"])
    y_train = train_df["__target__"]

    X_val   = val_df.drop(columns=["__target__"])
    y_val   = val_df["__target__"]

    X_test  = test_df.drop(columns=["__target__"])
    y_test  = test_df["__target__"]

    return X_train, X_val, X_test, y_train, y_val, y_test
