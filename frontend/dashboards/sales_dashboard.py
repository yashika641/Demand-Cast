import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from utils.column_finder import column_finder
from utils.sales_data_aggregation import sales_aggregation_over_time
from utils.google_trends_api import fetch_google_trends_best_keyword
from models.hybrid_sales_model import hybrid_sales_forecast_plot
# ---------------------------
# Column name mappings
# ---------------------------
POSSIBLE_SALES_COLS = [
    "sales",
    "Sales",
    "SALES",
    "revenue",
    "Revenue",
    "REVENUE",
    "amount",
    "Amount",
    "AMOUNT",
    "price",
    "Price",
    "PRICE",
]

POSSIBLE_DATE_COLS = [
    "date", "Date", "DATE",
    "order_date", "Order_Date", "ORDER_DATE",
    "orderDate", "created_at", "createdAt",
    "timestamp", "Timestamp", "TIMESTAMP",
]

POSSIBLE_PRODUCT_COLS = [
    "product",
    "product_name",
    "productname",
    "product description",
    "product_title",
    "item",
    "item_name",
    "item description",
    "model",
    "model_name",
]

POSSIBLE_CATEGORY_COLS = [
    "category",
    "category_name",
    "category_title",
    "category_type",
    "subcategory",
    "sub_category",
    "sub_category_name",
    "main_category",
    "department",
    "division",
    "section",
    "product_category",
    "product_type",
    "product_group",
    "product_family",
    "line_of_business",
    "market_segment",
    "class",
    "class_name",
    "class_id",
]


# ---------------------------
# Sales Dashboard
# ---------------------------
def sales_dashboard(sales_df):
    # Use a separate key for internal navigation
    if sales_df is None:
        st.warning("no sales data uploaded yet. Please upload sales data first.")
        st.stop()
        return

    df = sales_df.copy()
    st.title("📊 Sales and Revenue Dashboard")

    st.write(
        "Welcome to the Sales and Revenue section of DemandCast. "
        "Upload your sales data (CSV) to analyze trends, revenue, and insights."
    )
    st.warning(
        "⚠️ Please ensure your CSV has columns like 'Date', 'Sales', and 'Revenue'."
    )
    # ---------------------------
    # Dashboard Page
    # --------------------------

    # Detect Columns
    date_col = column_finder(df, POSSIBLE_DATE_COLS)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # df = df.set_index(date_col)
        
        
    st.write("Available columns:", df.columns.tolist())
    st.write("Looking for:", date_col)


    product_col = column_finder(df, POSSIBLE_PRODUCT_COLS)
    category_col = column_finder(df, POSSIBLE_CATEGORY_COLS)
    sales_col = column_finder(df, POSSIBLE_SALES_COLS)
    if sales_col:
        df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

    if not date_col or not sales_col:
        st.error("❌ Required columns ('Date', 'Sales/Revenue') not found.")
        st.stop()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Overview",
            "Sales Trends",
            "Filtering & Segmentation",
            "Forecasting",
            "Google Trends",
            "Revenue Analysis",
        ]
    )

    # ---------------------------
    # Tab 1: Overview
    # ---------------------------
    with tab1:
        st.subheader("📋 Data Summary")
        if st.button("Show Summary"):
            st.write(df.head())
            st.write(df.describe())
            st.write("Data Types", df.dtypes)
            st.write("Missing Values", df.isnull().sum())
            st.write("Duplicates", df.duplicated().sum())
            st.write("Detected sales column:", sales_col)
            st.write(df[sales_col].head(10))

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"{df[sales_col].sum():,.0f}")
        col2.metric("Total Revenue", f"{df[sales_col].sum():,.0f}")
        col3.metric("Avg Order Value", f"{df[sales_col].mean():,.2f}")

    # ---------------------------
    # Tab 2: Sales Trends
    # ---------------------------
    with tab2:
        st.subheader("📈 Sales Trend Visualization")

        if "agg_freq" not in st.session_state:
            st.session_state.agg_freq = "Daily"

        col1, col2, col3 = st.columns(3)
        if col1.button("Daily"):
            st.session_state.agg_freq = "Daily"
        if col2.button("Monthly"):
            st.session_state.agg_freq = "Monthly"
        if col3.button("Yearly"):
            st.session_state.agg_freq = "Yearly"

        fig = sales_aggregation_over_time(
            df, date_col, sales_col, st.session_state.agg_freq
        )
        st.pyplot(fig)

    # ---------------------------
    # Tab 3: Filtering & Segmentation
    # ---------------------------
    with tab3:
        st.subheader("🔍 Filter Data")

        # Sidebar date filter
        date_range = st.sidebar.date_input(
            "Date Range", value=(df[date_col].min(), df[date_col].max())
        )

        filtered_df = df.copy()
        if date_range:
            filtered_df = filtered_df[
                (filtered_df[date_col] >= pd.to_datetime(date_range[0]))
                & (filtered_df[date_col] <= pd.to_datetime(date_range[1]))
            ]

        # Sales trend line chart
        st.line_chart(filtered_df[[date_col, sales_col]].set_index(date_col))

        # Side-by-side charts
        col1, col2 = st.columns(2)

        with col1:
            st.write("Product Sales Share")
            if product_col:
                df_product = filtered_df.groupby(product_col, as_index=False)[
                    sales_col
                ].sum()
                fig_product = px.pie(
                    df_product,
                    values=sales_col,
                    names=product_col,
                    title="Product Sales Share",
                )
                st.plotly_chart(fig_product)

        with col2:
            st.write("Category Sales Share")
            if category_col:
                df_category = filtered_df.groupby(category_col, as_index=False)[
                    sales_col
                ].sum()
                fig_category = px.pie(
                    df_category,
                    values=sales_col,
                    names=category_col,
                    title="Category Sales Share",
                )
                st.plotly_chart(fig_category)

    # ---------------------------
    # Tab 4: Forecasting (Placeholder)
    # ---------------------------
    with tab4:
        hybrid_sales_forecast_plot(sales_df, sales_col, date_col, forecast_periods=30)


    # ---------------------------
    # Tab 5: Google Trends
    # ---------------------------
    with tab5:
        st.subheader("📊 Google Trends Analysis")
        agg_col = st.selectbox("Select Column for Trends", df.columns.tolist())

        if agg_col and agg_col.lower() in [
            c.lower() for c in POSSIBLE_PRODUCT_COLS + POSSIBLE_CATEGORY_COLS
        ]:
            product_list = df[agg_col].dropna().unique().tolist()
            selected_product = st.selectbox("Select Product", product_list)

            timeframe = st.selectbox(
                "Timeframe",
                ["now 7-d", "today 1-m", "today 3-m", "today 12-m", "today 5-y"],
            )
            country = st.selectbox("Country", ["Worldwide", "India", "United States"])

            country_geo = {"Worldwide": "", "India": "IN", "United States": "US"}
            df_trends = fetch_google_trends_best_keyword(
                selected_product, timeframe=timeframe, geo=country_geo[country]
            )

            if not df_trends.empty:
                st.line_chart(df_trends)

    # ---------------------------
    # Tab 6: Revenue Analysis
    # ---------------------------
    with tab6:
        st.subheader("💰 Revenue Analysis")
        df[date_col] = pd.to_datetime(df[date_col])
        total_revenue = df[sales_col].sum()
        avg_revenue = df[sales_col].mean()

        col1, col2 = st.columns(2)
        col1.metric("Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("Avg Revenue", f"${avg_revenue:,.2f}")

        st.line_chart(df.set_index(date_col)[sales_col])

    # ---------------------------
    # Back Buttons
    # ---------------------------
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "page2"
        st.rerun()
