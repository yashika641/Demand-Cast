def train_demand_forecast_model(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    full_df=None,
    target_col="sales",
    callback=None,
    horizon=7,
    seasonality="auto",
    model_type="xgboost"
):

    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # ---------------------------------------------------------
    def report(stage, pct, msg):
        text = f"[{stage}] {pct}% - {msg}"
        print(text)
        if callback:
            try:
                callback(stage, pct, msg)
            except:
                pass

    report("MODEL_TRAINING", 1, "Initializing model selection...")

    # ---------------------------------------------------------
    def eval_preds(y_true, y_pred):
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))
        }

    # ---------------------------------------------------------
    n_samples = len(X_train) + len(X_val) + len(X_test)
    n_features = X_train.shape[1]
    n_products = int(full_df["product_id"].nunique()) if full_df is not None else 1

    report("MODEL_TRAINING", 10, f"Dataset scanned. Samples={n_samples}, Features={n_features}, Products={n_products}")

    # ------------------ FIXED MODEL SELECTION ----------------
    chosen = None

        # ------------------ MODEL SELECTION ----------------
    # If user selects a model explicitly → override auto selection
    if model_type:
        chosen = model_type.upper()
        report("MODEL_TRAINING", 15, f"User selected model: {chosen}")
    else:
        # fallback automatic selector
        if n_samples < 10000:
            chosen = "PROPHET" if "date" in X_train.columns else "RANDOMFOREST"
        elif 10000 <= n_samples <= 150000:
            chosen = "XGBOOST"
        else:
            chosen = "LSTM"

        report("MODEL_TRAINING", 15, f"Auto-selected model: {chosen}")


    report("MODEL_TRAINING", 20, f"Selected model: {chosen}")

    # ---------------------------------------------------------
    # PROPHET
    if chosen == "PROPHET":
        report("PROPHET", 30, "Initializing Prophet...")
        from prophet import Prophet

        model = Prophet(
            yearly_seasonality=True if seasonality == "auto" else False,
            weekly_seasonality=True,
            daily_seasonality=False
        )

        dfp = full_df[["date", target_col]].rename(columns={"date": "ds", target_col: "y"})

        report("PROPHET", 40, "Training Prophet...")
        model.fit(dfp)

        future = model.make_future_dataframe(periods=horizon)
        pred = model.predict(future)["yhat"].values[-len(y_test):]
        report("PROPHET", 100, "Training complete.")
        return {"model": model, "model_type": "PROPHET",
                "metrics": eval_preds(y_test, pred), "preds": pred}

    # ---------------------------------------------------------
    # RANDOM FOREST
    if chosen == "RANDOMFOREST":
        report("RF", 30, "Initializing RandomForest...")
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=300, n_jobs=-1)
        report("RF", 50, "Training RandomForest...")
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        report("RF", 100, "Training complete.")
        return {"model": model, "model_type": "RANDOMFOREST",
                "metrics": eval_preds(y_test, pred), "preds": pred}

    # ---------------------------------------------------------
    # XGBOOST
    if chosen == "XGBOOST":
        report("XGBOOST", 30, "Initializing XGBoost...")
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=200 + horizon,
            max_depth=6 if horizon < 30 else 8,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
        )


        report("XGBOOST", 50, "Training XGBoost...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        report("XGBOOST", 100, "Training complete.")
        return {"model": model, "model_type": "XGBOOST",
                "metrics": eval_preds(y_test, pred), "preds": pred}

    # ---------------------------------------------------------
    # LSTM
    if chosen == "LSTM":
        report("LSTM", 30, "Preparing LSTM sequences...")

        from tensorflow.keras.models import Sequential #type:ignore
        from tensorflow.keras.layers import LSTM, Dense, Dropout #type:ignore


        SEQ = horizon
        series = full_df[target_col].values

        Xs, Ys = [], []
        for i in range(len(series) - SEQ):
            Xs.append(series[i:i+SEQ])
            Ys.append(series[i+SEQ])

        Xs = np.array(Xs).reshape(-1, SEQ, 1)
        Ys = np.array(Ys)

        split_1 = int(len(Xs) * 0.7)
        split_2 = int(len(Xs) * 0.85)

        Xtr, Xva, Xte = Xs[:split_1], Xs[split_1:split_2], Xs[split_2:]
        ytr, yva, yte = Ys[:split_1], Ys[split_1:split_2], Ys[split_2:]

        report("LSTM", 50, "Building LSTM model...")

        model = Sequential([
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(loss="mse", optimizer="adam")

        report("LSTM", 70, "Training LSTM...")
        model.fit(Xtr, ytr, epochs=5, batch_size=32, verbose=1)

        pred = model.predict(Xte).flatten()

        report("LSTM", 100, "Training complete.")
        return {"model": model, "model_type": "LSTM",
                "metrics": eval_preds(yte, pred), "preds": pred}
