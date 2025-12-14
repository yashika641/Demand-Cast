import pandas as pd, joblib, os
from app.core.model_registry import load_model

def predict_churn(uid: str, payload: list[dict]):
    df = pd.DataFrame(payload)
    model = load_model(uid)
    probs = model.predict_proba(df)[:,1]
    preds = (probs > 0.5).astype(int)
    return {"predictions": preds.tolist(), "probabilities": probs.round(4).tolist()}
