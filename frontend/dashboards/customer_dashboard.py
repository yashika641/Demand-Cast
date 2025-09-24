import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.column_finder import column_finder
from models.churn_prediction_model import churn_prediction_model

# ---------------- Constants ----------------
TRANSACTION_ID_SYNONYMS = [
    "transaction_id", "trans_id", "txn_id", "trx_id",
    "order_id", "orderid", "purchase_id", "invoice_id",
    "invoice_no", "bill_id", "checkout_id", "sale_id",
    "receipt_id", "id", "record_id", "entry_id",
    "doc_id", "document_id", "voucher_id", "payment_id"
]

PRIMARY_COLOR = "#4b5fff"
SECONDARY_COLOR = "#b24cff"
TERTIARY_COLOR = "#7f5aff"
CARD_BG = "#ffffff"
CARD_SHADOW = "0 8px 24px rgba(0,0,0,0.15)"
FONT_FAMILY = "'Montserrat', sans-serif"
ANIMATION = "all 0.3s ease-in-out"

# ---------------- Apply CSS ----------------
st.markdown(f"""
<style>
.stApp {{
        background: transparent !important;
    }}
video.bg-video {{
        position: fixed;     /* allow scrolling */
        top: 0;
        left: 0;
        display:flex;
        align-items: center;
        justify-content: center;    
        width: 110%;           /* fill full width */
        height: 100%;           /* scale height proportionally */
        min-height: 100%;       /* ensures it covers vertically */
        object-fit: cover;    /* show entire video (no cropping) */
        z-index: -1;            /* push behind content */
        background: black;  
        background-position:center;
        background-size: cover;
    }}

/* Section headers */
h1, h2, h3, h4 {{
    color: {PRIMARY_COLOR};
    font-weight: bold;
}}

/* Metrics styling */
.stMetricValue {{
    color: {SECONDARY_COLOR};
    font-weight: bold;
    transition: {ANIMATION};
}}
.stMetricValue:hover {{
    color: {TERTIARY_COLOR};
    transform: scale(1.05);
}}

/* DataFrames styling */
.stDataFrame td, .stDataFrame th {{
    padding: 0.5rem;
    border: none;
}}
.stDataFrame {{
    background-color: {CARD_BG};
    border-radius: 12px;
    box-shadow: {CARD_SHADOW};
    padding: 1rem;
    transition: {ANIMATION};
}}
.stDataFrame:hover {{
    box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}}

/* Tabs customization */
.stTabs [role="tab"] {{
    background-color: {PRIMARY_COLOR};
    color: white;
    border-radius: 10px 10px 0 0;
    font-weight: bold;
    transition: {ANIMATION};
}}
.stTabs [role="tab"]:hover {{
    background-color: {SECONDARY_COLOR};
}}

/* Plotly chart container */
.css-1d391kg e1fqkh3o2 {{
    transition: {ANIMATION};
}}
.css-1d391kg e1fqkh3o2:hover {{
    transform: scale(1.02);
}}

/* Add smooth fade-in for subheaders */
h3 {{
    animation: fadeIn 1s ease-in-out;
}}
@keyframes fadeIn {{
    from {{opacity: 0; transform: translateY(10px);}}
    to {{opacity: 1; transform: translateY(0);}}
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Dashboard Function ----------------
def customer_dashboard(customers_df, transactions_df=None):
    st.markdown(f"<h1 style='color:{PRIMARY_COLOR}; font-family:{FONT_FAMILY}'>👥 Customer Analytics Dashboard</h1>", unsafe_allow_html=True)

    st.markdown("""<video autoplay muted loop class="bg-video">
                    <source src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/bg-video1.mp4" type="video/mp4" >
                </video>""",unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Customer Demographics",
        "Customer Lifetime Value",
        "Churn Prediction",
        "Segmentation & Personas",
        "Loyalty & Retention"
    ])

    df = customers_df.copy()

    # --- Column detection ---
    id_col = column_finder(df, ["customer_id", "id", "cust_id"])
    gender_col = column_finder(df, ["gender", "sex"])
    age_col = column_finder(df, ["age"])
    region_col = column_finder(df, ["region", "state", "country", "location"])
    signup_col = column_finder(df, ["signup_date", "sign_up", "date_joined"])
    member_col = column_finder(df, ["membership_tier", "membership", "tier", "loyalty_level"])
    points_col = column_finder(df, ["loyalty_points", "points", "reward_points"])
    items_col = column_finder(df, ["highest_freq_purchased_items", "fav_item", "popular_item"])

    order_value_col = column_finder(transactions_df, ["order_value", "amount", "sales", "revenue","total","line_total"]) if transactions_df is not None else None
    order_date_col = column_finder(transactions_df, ["date", "order_date", "purchase_date", "datetime"]) if transactions_df is not None else None
    transaction_id_col = column_finder(transactions_df, TRANSACTION_ID_SYNONYMS)

    # ---------------- Tab 1: Demographics ----------------
    with tab1:
        if gender_col:
            st.subheader("Gender Distribution")
            fig_gender = px.pie(df, names=gender_col, title="Customer Gender Split",
                                color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR])
            st.plotly_chart(fig_gender, use_container_width=True)

        if age_col:
            st.subheader("Age Distribution")
            fig_age = px.histogram(df, x=age_col, nbins=10, color=gender_col,
                                   color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR],
                                   title="Customer Age Histogram")
            st.plotly_chart(fig_age, use_container_width=True)

            df["age_group"] = pd.cut(df[age_col], bins=[0, 18, 25, 35, 45, 60, 100],
                                     labels=["<18", "18-25", "26-35", "36-45", "46-60", "60+"])
            age_group_counts = df["age_group"].value_counts().sort_index().reset_index()
            age_group_counts.columns = ["age_group", "count"]
            fig_agegroup = px.bar(age_group_counts, x="age_group", y="count",
                                  color_discrete_sequence=[PRIMARY_COLOR],
                                  title="Customers by Age Group")
            st.plotly_chart(fig_agegroup, use_container_width=True)

        if region_col:
            st.subheader("Customers by Region")
            region_counts = df[region_col].value_counts().reset_index()
            region_counts.columns = ["region", "count"]
            fig_region = px.bar(region_counts, x="region", y="count",
                                color_discrete_sequence=[SECONDARY_COLOR],
                                title="Customers by Region")
            st.plotly_chart(fig_region, use_container_width=True)

        if member_col:
            st.subheader("Membership Tier Breakdown")
            fig_tier = px.pie(df, names=member_col,
                              color_discrete_sequence=[PRIMARY_COLOR, TERTIARY_COLOR],
                              title="Membership Tiers")
            st.plotly_chart(fig_tier, use_container_width=True)

        if points_col:
            st.subheader("Loyalty Points Distribution")
            fig_points = px.box(df, y=points_col, points="all",
                                color_discrete_sequence=[PRIMARY_COLOR],
                                title="Loyalty Points Spread")
            st.plotly_chart(fig_points, use_container_width=True)

        if signup_col:
            st.subheader("Customer Signups Over Time")
            df[signup_col] = pd.to_datetime(df[signup_col], errors="coerce")
            signup_trend = df.dropna(subset=[signup_col]).groupby(df[signup_col].dt.to_period("M")).size().reset_index(name="signups")
            signup_trend[signup_col] = signup_trend[signup_col].astype(str)
            fig_signup = px.line(signup_trend, x=signup_col, y="signups",
                                 line_shape='spline', color_discrete_sequence=[PRIMARY_COLOR],
                                 title="Monthly Signups")
            st.plotly_chart(fig_signup, use_container_width=True)

        if items_col:
            st.subheader("Most Frequently Purchased Items")
            top_items = df[items_col].value_counts().reset_index()
            top_items.columns = ["item", "count"]
            fig_items = px.bar(top_items.head(10), x="item", y="count",
                               color_discrete_sequence=[SECONDARY_COLOR],
                               title="Top 10 Purchased Items")
            st.plotly_chart(fig_items, use_container_width=True)

    # ---------------- Tab 2: Customer Lifetime Value ----------------
    with tab2:
        if transactions_df is not None and id_col and order_value_col:
            st.header("💰 Customer Lifetime Value (CLV)")
            clv = transactions_df.groupby(id_col)[order_value_col].sum().reset_index()
            clv.columns = [id_col, "CLV"]
            clv_demo = clv.merge(df, on=id_col, how="left")

            st.subheader("CLV Distribution")
            fig_clv_dist = px.histogram(clv_demo, x="CLV", nbins=20,
                                        color_discrete_sequence=[PRIMARY_COLOR],
                                        title="Distribution of CLV")
            st.plotly_chart(fig_clv_dist, use_container_width=True)

            st.subheader("Top 10 Customers by CLV")
            top_customers = clv_demo.sort_values("CLV", ascending=False).head(10)
            fig_top = px.bar(top_customers, x=id_col, y="CLV",
                             color_discrete_sequence=[SECONDARY_COLOR],
                             title="Top 10 Customers by CLV")
            st.plotly_chart(fig_top, use_container_width=True)

            if gender_col:
                st.subheader("CLV by Gender")
                fig_gender_clv = px.box(clv_demo, x=gender_col, y="CLV",
                                        color_discrete_sequence=[PRIMARY_COLOR],
                                        title="CLV by Gender")
                st.plotly_chart(fig_gender_clv, use_container_width=True)

            if "age_group" in clv_demo.columns:
                st.subheader("CLV by Age Group")
                fig_age_clv = px.bar(clv_demo.groupby("age_group")["CLV"].mean().reset_index(),
                                     x="age_group", y="CLV",
                                     color_discrete_sequence=[SECONDARY_COLOR],
                                     title="Average CLV by Age Group")
                st.plotly_chart(fig_age_clv, use_container_width=True)

            if region_col:
                st.subheader("CLV by Region")
                fig_region_clv = px.bar(clv_demo.groupby(region_col)["CLV"].mean().reset_index(),
                                        x=region_col, y="CLV",
                                        color_discrete_sequence=[TERTIARY_COLOR],
                                        title="Average CLV by Region")
                st.plotly_chart(fig_region_clv, use_container_width=True)

            if member_col:
                st.subheader("CLV by Membership Tier")
                fig_member_clv = px.bar(clv_demo.groupby(member_col)["CLV"].mean().reset_index(),
                                        x=member_col, y="CLV",
                                        color_discrete_sequence=[PRIMARY_COLOR],
                                        title="Average CLV by Membership Tier")
                st.plotly_chart(fig_member_clv, use_container_width=True)

    # ---------------- Tab 3: Churn Prediction ----------------
    with tab3:
        churn_prediction_model(transactions_df, df, transaction_id_col, order_date_col, order_value_col)

    # ---------------- Tab 4: Segmentation & Personas ----------------
    with tab4:
        st.header("🧩 Customer Segmentation & Personas")
        if transactions_df is not None and id_col and order_value_col:
            clv = transactions_df.groupby(id_col)[order_value_col].sum().reset_index()
            clv.columns = [id_col, "CLV"]
            clv_demo = clv.merge(df, on=id_col, how="left")
            clv_demo["CLV_segment"] = pd.qcut(clv_demo["CLV"], 3, labels=["Low Value", "Medium Value", "High Value"])

            st.subheader("Customer Segments by CLV")
            fig_seg = px.pie(clv_demo, names="CLV_segment",
                             color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR],
                             title="Customer Segmentation (CLV-based)")
            st.plotly_chart(fig_seg, use_container_width=True)

            st.subheader("Customer Personas")
            persona_summary = clv_demo.groupby("CLV_segment").agg({
                age_col: "mean" if age_col else "count",
                "CLV": "mean",
                points_col: "mean" if points_col else "count"
            }).reset_index()
            st.dataframe(persona_summary)

    # ---------------- Tab 5: Loyalty & Retention ----------------
    with tab5:
        st.header("🏆📈 Loyalty & Retention Dashboard")
        if points_col:
            st.subheader("Loyalty Points Distribution")
            fig_points = px.box(df, y=points_col, points="all",
                                color_discrete_sequence=[PRIMARY_COLOR],
                                title="Spread of Loyalty Points")
            st.plotly_chart(fig_points, use_container_width=True)

            if gender_col:
                st.subheader("Average Loyalty Points by Gender")
                gender_loyalty = df.groupby(gender_col)[points_col].mean().reset_index()
                fig_gender_loyalty = px.bar(gender_loyalty, x=gender_col, y=points_col,
                                            color_discrete_sequence=[SECONDARY_COLOR],
                                            title="Avg Loyalty Points by Gender")
                st.plotly_chart(fig_gender_loyalty, use_container_width=True)

            st.subheader("Top 10 Loyal Customers")
            top_loyal = df.nlargest(10, points_col)[[id_col, points_col]]
            st.dataframe(top_loyal)

        st.markdown("---")

        if transactions_df is not None and id_col and order_date_col:
            sales = transactions_df.copy()
            sales[order_date_col] = pd.to_datetime(sales[order_date_col], errors="coerce")
            sales = sales.dropna(subset=[order_date_col])

            if signup_col and id_col:
                df[signup_col] = pd.to_datetime(df[signup_col], errors="coerce")
                df = df.dropna(subset=[signup_col])
                df["signup_month"] = df[signup_col].dt.to_period("M").astype(str)
                sales["purchase_month"] = sales[order_date_col].dt.to_period("M").astype(str)
                cohort = sales.merge(df[[id_col, "signup_month"]], on=id_col)
                cohort_group = cohort.groupby(["signup_month", "purchase_month"])[id_col].nunique().reset_index()
                retention_matrix = cohort_group.pivot(index="signup_month", columns="purchase_month", values=id_col).fillna(0)
                retention_matrix = retention_matrix.div(retention_matrix.iloc[:, 0], axis=0).replace([np.inf, -np.inf], 0).fillna(0)

                st.subheader("Cohort Retention Heatmap")
                fig_retention = px.imshow(retention_matrix, text_auto=True, aspect="auto",
                                          color_continuous_scale=[PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR],
                                          title="Retention Heatmap (Cohorts)")
                st.plotly_chart(fig_retention, use_container_width=True)

            st.subheader("Churn Analysis")
            last_purchase = sales.groupby(id_col)[order_date_col].max().reset_index()
            last_purchase["days_since_purchase"] = (pd.Timestamp.today() - last_purchase[order_date_col]).dt.days
            churned = last_purchase[last_purchase["days_since_purchase"] > 180]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(df))
            col2.metric("Churned Customers (6+ months inactive)", len(churned))
            col3.metric("Retention Rate", f"{100 - (len(churned)/len(df))*100:.2f}%")

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "page2"
        st.rerun()
