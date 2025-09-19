import numpy as np 
import pandas as pd 
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

def churn_prediction_model(transactions_df, customer_df, transaction_id_col, order_date_col, order_value_col):
    # ---- Custom CSS Styling ----
    st.markdown("""
        <style>
        /* Background */
        .stApp {
            background-image: url("https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/Gemini_Generated_Image_6uxpod6uxpod6uxp.png");
            background-size: cover;
            background-attachment: fixed;
            font-family: 'Montserrat', sans-serif;
            color: #ffffff;
            background-position : center;
        }
        /* Section headers */
        .css-1v0mbdj h1, h2, h3, h4, h5 {
            color: #f2f2f2;
            text-shadow: 1px 1px 4px #00000050;
        }
        /* Plotly charts background */
        .plotly-graph-div {
            background-color: rgba(255,255,255,0.05) !important;
            border-radius: 10px;
            padding: 10px;
            transition: transform 0.3s ease-in-out;
        }
        .plotly-graph-div:hover {
            transform: scale(1.03);
        }
        /* Dataframe styling */
        .stDataFrame td, .stDataFrame th {
            background-color: rgba(255,255,255,0.05);
            color: #ffffff;
            border-radius: 5px;
            padding: 5px;
        }
        /* Success messages */
        .stAlert {
            background-color: #4b5fff80 !important;
            color: white !important;
            border-radius: 10px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("🛡️ Churn Prediction Model")

    # ---------------------------
    # Step 1: Preprocessing
    # ---------------------------
    transactions_df[order_date_col] = pd.to_datetime(transactions_df[order_date_col], errors="coerce")

    # Aggregate at customer level
    customer_sales = transactions_df.groupby("customer_id").agg({
        order_date_col: "max",
        order_value_col: "sum",
        transaction_id_col: "count"
    }).reset_index()

    customer_sales.rename(columns={
        order_date_col: "last_purchase",
        order_value_col: "monetary",
        transaction_id_col: "frequency"
    }, inplace=True)

    customer_sales["last_purchase"] = pd.to_datetime(customer_sales["last_purchase"], errors="coerce")
    today = pd.to_datetime(transactions_df[order_date_col].max(), errors="coerce")
    customer_sales["recency"] = (today - customer_sales["last_purchase"]).dt.days
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

    customer_sales["churn_probability"] = model.predict_proba(X)[:, 1]
    customer_sales["risk_level"] = pd.cut(
        customer_sales["churn_probability"],
        bins=[0, 0.33, 0.66, 1],
        labels=["Low", "Medium", "High"]
    )

    # ---------------------------
    # Step 3: Visualizations
    # ---------------------------
    st.subheader("🔹 Churn Risk Distribution")
    fig_risk = px.histogram(
        customer_sales,
        x="risk_level",
        color="risk_level",
        title="Churn Risk Levels",
        color_discrete_sequence=["#4b5fff", "#b24cff", "#7f5aff"]
    )
    st.plotly_chart(fig_risk, use_container_width=True, key="churn_dist")

    st.subheader("🟣 Top Customers at Risk")
    st.dataframe(
        customer_sales[["customer_id", "recency", "frequency", "monetary", "churn_probability", "risk_level"]]
        .sort_values("churn_probability", ascending=False)
        .head(10)
    )

    st.subheader("⚡ Feature Importance")
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        importance,
        x="Feature",
        y="Importance",
        title="Drivers of Churn",
        color="Feature",
        color_discrete_sequence=["#4b5fff", "#b24cff", "#7f5aff"]
    )
    st.plotly_chart(fig_imp, use_container_width=True, key="churn_feat_imp")

    # ---------------------------
    # Step 4: Summary
    # ---------------------------
    st.success("✅ Churn prediction model trained successfully. Use this to design retention campaigns.")
