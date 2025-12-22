import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    lgbm_available = True
except:
    lgbm_available = False

from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import LSTM, Dense #type:ignore
from tensorflow.keras.callbacks import EarlyStopping #type:ignore
from tensorflow.keras.optimizers import Adam #type:ignore


def train_stockout_model(
    X_train, y_train,
    X_test, y_test,
    X_val, y_val,
    forced_model_type: str = None
):
    """
    Smart + manual override stockout model trainer.

    Supports:
    - Auto model selection
    - forced_model_type = 'logistic' | 'random_forest' | 'xgboost' | 'lightgbm' | 'lstm'
    """

    print("training start with :", X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape)

    n_rows = len(X_train)
    n_features = X_train.shape[1]
    columns = X_train.columns.tolist()

    # -----------------------------------------------------------
    # 1. MODEL SELECTION LOGIC (AUTO + OVERRIDE)
    # -----------------------------------------------------------

    sequential_features = any(col.startswith("lag_") for col in columns)

    # Override takes highest priority
    if forced_model_type:
        model_type = forced_model_type.lower()
        print(f"📌 Forced model type → {model_type.upper()}")
    else:
        # AUTO MODE
        if sequential_features:
            model_type = "lstm"
        elif n_rows < 5000:
            model_type = "logistic"
        elif n_rows < 50000:
            model_type = "random_forest"
        else:
            model_type = "xgboost"

        print(f"📌 Auto-selected model type → {model_type.upper()}")

    # -----------------------------------------------------------
    # 2. TRAIN THE MODEL
    # -----------------------------------------------------------

    # -----------------
    # LOGISTIC
    # -----------------
    if model_type == "logistic":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train_scaled, y_train)

        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated_model.fit(X_val, y_val)

        preds = calibrated_model.predict(X_test_scaled)

    # -----------------
    # RANDOM FOREST
    # -----------------
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight="balanced_subsample",
            random_state=42
        )
        model.fit(X_train, y_train)

        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated_model.fit(X_val, y_val)

        preds = calibrated_model.predict(X_test)

    # -----------------
    # LIGHTGBM
    # -----------------
    elif model_type == "lightgbm":
        if not lgbm_available:
            raise RuntimeError("LightGBM not installed")

        model = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated_model.fit(X_val, y_val)

        preds = calibrated_model.predict(X_test)

    # -----------------
    # XGBOOST
    # -----------------
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=float(len(y_train) / sum(y_train)),
        )
        model.fit(X_train, y_train)

        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated_model.fit(X_val, y_val)

        preds = calibrated_model.predict(X_test)

    # -----------------
    # LSTM
    # -----------------
    elif model_type == "lstm":
        X_train_lstm = np.expand_dims(X_train.values, axis=1)
        X_test_lstm = np.expand_dims(X_test.values, axis=1)

        model = Sequential([
            LSTM(64, activation="tanh", input_shape=(1, n_features)),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=Adam(0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            X_train_lstm, y_train,
            validation_split=0.1,
            epochs=20,
            batch_size=32,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=1
        )

        preds = (model.predict(X_test_lstm) > 0.5).astype(int)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # -----------------------------------------------------------
    # 3. EVALUATE
    # -----------------------------------------------------------
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n📊 MODEL METRICS:")
    print("Accuracy:", round(accuracy, 3))
    print("F1 Score:", round(f1, 3))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    return {
        "model_type": model_type,
        "model": model,
        "preds": preds,
        "accuracy": accuracy,
        "f1": f1,
    }
