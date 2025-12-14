def train_test_split_time_from_xy(
    X, 
    y, 
    date_col, 
    train_ratio=0.7, 
    val_ratio=0.15
):
    """
    Time-series split into train, validation, and test.
    
    Parameters:
        X: feature dataframe
        y: target series
        date_col: name of the date column in X
        train_ratio: proportion for training set (default 0.7)
        val_ratio: proportion for validation set (default 0.15)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    # 1. Combine X & y (so sorting aligns)
    temp = X.copy()
    temp["__target__"] = y.values

    # 2. Sort strictly by date
    temp = temp.sort_values(date_col).reset_index(drop=True)

    # 3. Compute split indices
    n = len(temp)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # 4. Create splits
    train = temp.iloc[:train_end]
    val   = temp.iloc[train_end:val_end]
    test  = temp.iloc[val_end:]

    # 5. Split back
    X_train = train.drop(columns=["__target__"])
    y_train = train["__target__"]

    X_val = val.drop(columns=["__target__"])
    y_val = val["__target__"]

    X_test = test.drop(columns=["__target__"])
    y_test = test["__target__"]

    return X_train, X_val, X_test, y_train, y_val, y_test
