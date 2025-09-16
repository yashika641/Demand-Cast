import numpy as np 
import pandas as pd 
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

def churn_prediction_model(transactions_df, customer_df, transaction_id_col, order_date_col, order_value_col):
    st.header("Churn Prediction Model")

    # ---------------------------
    # Step 1: Preprocessing
    # ---------------------------
    transactions_df[order_date_col] = pd.to_datetime(transactions_df[order_date_col], errors="coerce")

    # Aggregate at customer level
    customer_sales = transactions_df.groupby("customer_id").agg({
        order_date_col: "max",          # last purchase date
        order_value_col: "sum",         # total spent
        transaction_id_col: "count"     # purchase frequency
    }).reset_index()

    customer_sales.rename(columns={
        order_date_col: "last_purchase",
        order_value_col: "monetary",
        transaction_id_col: "frequency"
    }, inplace=True)

    # Ensure datetime
    customer_sales["last_purchase"] = pd.to_datetime(customer_sales["last_purchase"], errors="coerce")

    # Reference date = latest transaction
    today = pd.to_datetime(transactions_df[order_date_col].max(), errors="coerce")

    # Recency calculation
    customer_sales["recency"] = (today - customer_sales["last_purchase"]).dt.days

    # Churn label (threshold = 60 days since last purchase)
    customer_sales["churn"] = (customer_sales["recency"] > 60).astype(int)

    # ---------------------------
    # Step 2: Model Training
    # ---------------------------
    features = ["monetary", "frequency", "recency"]
    X = customer_sales[features]
    y = customer_sales["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predict churn probability for ALL customers (not just test set)
    y_pred_prob = model.predict_proba(X)[:, 1]

    customer_sales["churn_probability"] = y_pred_prob

    # Risk segmentation
    customer_sales["risk_level"] = pd.cut(
        customer_sales["churn_probability"],
        bins=[0, 0.33, 0.66, 1],
        labels=["Low", "Medium", "High"]
    )

    # ---------------------------
    # Step 3: Visualizations
    # ---------------------------
    st.subheader("Churn Risk Distribution")
    fig_risk = px.histogram(customer_sales, x="risk_level", color="risk_level", title="Churn Risk Levels")
    st.plotly_chart(fig_risk, use_container_width=True, key="churn_dist")

    st.subheader("Top Customers at Risk")
    st.dataframe(
        customer_sales[["customer_id", "recency", "frequency", "monetary", "churn_probability", "risk_level"]]
        .sort_values("churn_probability", ascending=False)
        .head(10)
    )

    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(importance, x="Feature", y="Importance", title="Drivers of Churn")
    st.plotly_chart(fig_imp, use_container_width=True, key="churn_feat_imp")

    # ---------------------------
    # Step 4: Summary
    # ---------------------------
    st.success("✅ Churn prediction model trained successfully. Use this to design retention campaigns.")

    # return customer_sales
