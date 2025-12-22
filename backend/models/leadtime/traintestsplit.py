import pandas as pd

def train_test_split_time_from_xy(X, y, date_col, split_ratio=0.8):
    """
    Time-series aware train-test split for X and y.
    Sorts by the date column in X and then splits both X and y
    while keeping the index alignment intact.
    
    Parameters:
        X (pd.DataFrame): Feature matrix containing the date_col
        y (pd.Series or pd.DataFrame): Target values
        date_col (str): Date column inside X to sort by
        split_ratio (float): Train split ratio
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Combine X & y temporarily for aligned sorting
    df_temp = X.copy()
    df_temp["__target__"] = y.values

    # Ensure datetime
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')

    # Sort by date
    df_temp = df_temp.sort_values(date_col)

    # Compute split index
    split_idx = int(len(df_temp) * split_ratio)

    # Split
    train = df_temp.iloc[:split_idx]
    test = df_temp.iloc[split_idx:]

    # Separate X and y again
    X_train = train.drop(columns=["__target__"])
    y_train = train["__target__"]
    X_test = test.drop(columns=["__target__"])
    y_test = test["__target__"]

    return X_train, X_test, y_train, y_test
