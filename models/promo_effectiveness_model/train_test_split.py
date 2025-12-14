def promo_train_test_split(df, target="promo_uplift", date_col="date"):
    df = df.sort_values(date_col)

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_val = val.drop(columns=[target])
    y_val = val[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    # remove 'date'
    for X in [X_train, X_val, X_test]:
        if date_col in X.columns:
            X.drop(columns=[date_col], inplace=True)

    return X_train, y_train, X_val, y_val, X_test, y_test
