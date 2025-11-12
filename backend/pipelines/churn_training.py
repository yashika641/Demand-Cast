import pandas as pd, shap, json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from joblib import dump
from app.pipelines.feature_builder import build_features
from app.pipelines.label_builder import create_labels
from app.core.supabase_client import fetch_user_files
from app.utils.column_mapper import map_columns
from app.core.model_registry import save_model

def train_churn(uid: str):
    files = fetch_user_files(uid)
    customers = map_columns(pd.read_csv(files["customers"]))
    transactions = map_columns(pd.read_csv(files["transactions"]))
    activity = map_columns(pd.read_csv(files.get("activity_logs", None))) if "activity_logs" in files else None
    support = map_columns(pd.read_csv(files.get("support_tickets", None))) if "support_tickets" in files else None

    X_df = build_features(customers, transactions, activity, support)
    y_df = create_labels(transactions, activity)
    df = X_df.merge(y_df, on="customer_id", how="left").fillna({"churn_flag":0})

    X = df.drop(columns=["customer_id","churn_flag"])
    y = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=7, subsample=0.9, colsample_bytree=0.8)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:,1]
    metrics = {"f1": f1_score(y_test,(y_pred>0.5).astype(int)), "auc": roc_auc_score(y_test,y_pred)}

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    importance = pd.Series(abs(shap_vals).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
    shap_summary = importance.head(10).to_dict()

    save_model(uid, model, metrics, list(X.columns), shap_summary)
    return metrics, shap_summary
