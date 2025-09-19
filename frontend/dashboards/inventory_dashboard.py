import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from  utils.column_finder import column_finder
from models.stockout_model import stockout_dashboard
# Column name possibilities
possible_date_cols = [
    "date",
    "Date",
    "DATE",
    "day",
    "Day",
    "DAY",
    "timestamp",
    "Timestamp",
    "TIMESTAMP",
]
possible_inventory_cols = [
    "inventory",
    "Inventory",
    "INVENTORY",
    "stock",
    "Stock",
    "STOCK",
    "inventory_level",
    "Inventory_Level",
    "INVENTORY_LEVEL",
    "stock_level",
    "Stock_Level",
    "STOCK_LEVEL",
    "stock_quantity",
    "STOCK_QUAN",
    "STOCK_NUMBER",
    "STOCK_VALUE",
]
possible_product_cols = [
    "product",
    "Product",
    "PRODUCT",
    "item",
    "Item",
    "ITEM",
    "product_name",
    "Product_Name",
    "PRODUCT_NAME",
    "item_name",
    "Item_Name",
    "ITEM_NAME",
]
price_columns = [
    "Price",
    "Unit_Price",
    "Cost",
    "Cost_Price",
    "Purchase_Price",
    "Selling_Price",
    "MRP",
    "Retail_Price",
    "Wholesale_Price",
    "List_Price",
    "Standard_Price",
    "Discounted_Price",
    "Final_Price",
    "Net_Price",
    "Gross_Price",
    "Sale_Price",
    "Current_Price",
    "Base_Price",
    "Original_Price",
    "Offer_Price",
    "total amount",
    'total_amount'
]
possible_order_date_cols = [
    "order_date",
    "Order_Date",
    "ORDER_DATE",
    "orderday",
    "OrderDay",
    "ORDERDAY",
    "ship_date",
    "Ship_Date",
    "SHIP_DATE",
    "shipday",
    "ShipDay",
    "SHIPDAY",
]
possible_delivery_date_cols = [
    "delivery_date",
    "Delivery_Date",
    "DELIVERY_DATE",
    "deliveryday",
    "DeliveryDay",
    "DELIVERYDAY",
    "received_date",
    "Received_Date",
    "RECEIVED_DATE",
    "receivedday",
    "ReceivedDay",
    "RECEIVEDDAY",
]
possible_supplier_cols = [
    "supplier",
    "Supplier",
    "SUPPLIER",
    "vendor",
    "Vendor",
    "VENDOR",
    "supplier_name",
    "Supplier_Name",
    "SUPPLIER_NAME",
    "vendor_name",
    "Vendor_Name",
    "VENDOR_NAME",
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
# ================== INVENTORY DASHBOARD ==================
def inventory_dashboard(inventory_df, transactions_df):
    # ================== GLOBAL CSS ==================
    st.markdown("""
        <style>
        /* Background */
        .stApp {
            background-image: url("https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/Gemini_Generated_Image_6uxpod6uxpod6uxp.png");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            font-family: 'Montserrat', sans-serif;
        }

        /* Titles */
        h1, h2, h3 {
            font-weight: 700;
        }
        h1 {
            text-align: center;
            color: #ffffff !important;
            font-size: 2.4rem !important;
            padding-bottom: 0.3rem;
        }
        h2 {
            color: #1f3b73 !important;
        }
        h3 {
            color: #334e68 !important;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            border-radius: 15px;
            padding: 15px 10px;
            margin: 5px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.12);
            transition: all 0.3s ease-in-out;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px) scale(1.03);
            box-shadow: 0px 6px 16px rgba(0,0,0,0.2);
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 12px;
            font-weight: 600;
            color: #334e68;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.35);
            color: #1f3b73;
        }
        .stTabs [aria-selected="true"] {
            background: #1f3b73 !important;
            color: white !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #1f3b73, #334e68);
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #334e68, #1f3b73);
            transform: translateY(-3px);
            box-shadow: 0px 6px 14px rgba(0,0,0,0.2);
        }

        /* Dataframes */
        .stDataFrame {
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 8px;
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(8px);
        }
        </style>
    """, unsafe_allow_html=True)

    # ================== TITLE ==================
    st.markdown("""
        <div>
            <h1>📦 Inventory & Supply Chain Insights</h1>
            <p style="color:#f5f5f5; font-size:18px; text-align:center;">
                Analyze inventory levels, lead times, stockouts, and supply chain performance with interactive dashboards.
            </p>
            <p style="color:#dbeafe; font-size:16px; text-align:center;">
                Ensure CSVs contain columns like 'Date', 'Inventory Level', 'Order Date', 'Delivery Date', and 'Price'.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ================== COLUMN DETECTION ==================
    date_col = column_finder(inventory_df, possible_date_cols)
    inventory_col = column_finder(inventory_df, possible_inventory_cols)
    product_col = column_finder(inventory_df, possible_product_cols)

    price_col_transaction = column_finder(transactions_df, price_columns)
    price_col_inventory = column_finder(inventory_df, price_columns)
    order_date_col = column_finder(inventory_df, possible_order_date_cols)
    delivery_date_col = column_finder(inventory_df, possible_delivery_date_cols)
    supplier_col = column_finder(inventory_df, possible_supplier_cols)
    category_col = column_finder(transactions_df, POSSIBLE_CATEGORY_COLS)
    region_col = column_finder(transactions_df, ['region','Region','state','city','country','location'])
    lead_time_col = column_finder(inventory_df, ['leadtime','lead_time','leadtime_days','lead_time_col'])
    transaction_date_col = column_finder(transactions_df, possible_date_cols)

    df = inventory_df.copy()
    df1 = transactions_df.copy()

    # ================== TABS ==================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Current Inventory Levels",
        "⏱️ Lead Time Analysis",
        "🚨 Stockout Analysis",
        "📦 Supply Chain Performance",
        "🔮 Forecasts & Predictions"
    ])

    # ================== TAB 1: INVENTORY LEVELS ==================
    with tab1:
        st.subheader("Current Inventory Levels")
        c1, c2, c3, _ = st.columns(4)
        c1.metric("Total Inventory", f"{df[inventory_col].sum():,.0f}")

        if price_col_transaction:
            df = df.merge(df1[[product_col, price_col_transaction]], on=product_col, how="left")
            df["total_money_acquired"] = df[inventory_col] * df[price_col_transaction]
            c2.metric("Inventory Value", f"${df['total_money_acquired'].sum():,.0f}")
            c3.metric("Avg Value", f"${df['total_money_acquired'].mean():,.2f}")

        product = st.selectbox("📌 Choose Product", options=df[product_col].unique())

        if not date_col or not inventory_col:
            st.error("⚠️ Missing required columns: 'Date' and 'Inventory Level'")
        else:
            st.subheader(f"📈 Inventory Trend for {product}")
            plt.figure(figsize=(10, 5))
            df_temp = df[df[product_col] == product]
            plt.bar(df_temp[date_col], df_temp[inventory_col], color="#1f77b4")
            plt.xlabel("Date")
            plt.ylabel("Inventory Level")
            plt.title(f"Inventory Levels - {product}")
            st.pyplot(plt)

    # ================== TAB 2: LEAD TIME ==================
    with tab2:
        st.subheader("Lead Time Insights")

        if not order_date_col or not delivery_date_col:
            st.error("⚠️ Missing 'Order Date' or 'Delivery Date'")
        else:
            if lead_time_col is None:
                df[order_date_col] = pd.to_datetime(df[order_date_col])
                df[delivery_date_col] = pd.to_datetime(df[delivery_date_col])
                df["lead_time_days"] = (df1[delivery_date_col] - df1[order_date_col]).dt.days
                lead_time_col = "lead_time_days"

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Avg", f"{df[lead_time_col].mean():.1f}d")
            c2.metric("Median", f"{df[lead_time_col].median():.1f}d")
            c3.metric("Min", f"{df[lead_time_col].min()}d")
            c4.metric("Max", f"{df[lead_time_col].max()}d")
            c5.metric("Std Dev", f"{df[lead_time_col].std():.1f}")

            st.subheader("📊 Lead Time Distribution")
            plt.figure(figsize=(10,5))
            plt.hist(df[lead_time_col].dropna(), bins=20, color="#2ca02c", edgecolor="black")
            plt.xlabel("Lead Time (days)")
            plt.ylabel("Frequency")
            plt.title("Lead Time Distribution")
            st.pyplot(plt)

            if supplier_col:
                st.subheader("Supplier-wise Lead Times")
                avg_supplier_lead_time = df.groupby(supplier_col)[lead_time_col].mean()
                st.bar_chart(avg_supplier_lead_time)

    # ================== TAB 3: STOCKOUT ==================
    with tab3:
        st.subheader("🚨 Stockout Insights")
        sales_df, top_products, region_df, cat_df, products_to_order, forecast = stockout_dashboard(
            inventory_df, transactions_df, inventory_col, product_col,
            category_col, region_col, price_col_transaction, lead_time_col
        )
        st.info("Detailed stockout models integrated from `stockout_dashboard`.")

    # ================== TAB 4: SUPPLY CHAIN ==================
    with tab4:
        st.subheader("📦 Supply Chain KPIs")

        total_demand = sales_df['forecasted_demand'].sum()
        total_sales = sales_df['actual_sales'].sum()
        fill_rate = (total_sales / total_demand * 100) if total_demand > 0 else 0
        stockout_rate = (df['stockout_flag'].mean() * 100)
        avg_inventory = df[inventory_col].mean()
        inventory_turnover = (total_sales / avg_inventory) if avg_inventory > 0 else 0
        avg_lead_time = df[lead_time_col].mean() if lead_time_col else np.nan

        c1, c2, c3, _, c5 = st.columns(5)
        c1.metric("Fill Rate (%)", f"{fill_rate:.2f}%")
        c2.metric("Stockout Rate (%)", f"{stockout_rate:.2f}%")
        c3.metric("Inventory Turnover", f"{inventory_turnover:.2f}")
        c5.metric("Avg Lead Time (Days)", f"{avg_lead_time:.1f}" if not np.isnan(avg_lead_time) else "N/A")

        st.subheader("📈 KPI Trends")
        sales_df['fill_rate'] = (sales_df['actual_sales'] / sales_df['forecasted_demand']).clip(0,1)

        fig_fill = px.line(sales_df, x=transaction_date_col, y='fill_rate', title="Fill Rate Over Time")
        st.plotly_chart(fig_fill, use_container_width=True, key="fill_rate_chart")

        fig_loss = px.line(sales_df, x=transaction_date_col, y='lost_sales', title="Lost Sales Over Time")
        st.plotly_chart(fig_loss, use_container_width=True, key="lost_sales_chart")

        df['inventory_turnover'] = total_sales / (df[inventory_col].replace(0, np.nan))
        fig_turn = px.histogram(df, x='inventory_turnover', title="Inventory Turnover Distribution")
        st.plotly_chart(fig_turn, use_container_width=True, key="inv_turnover_chart")

        st.subheader("📊 Performance Summary")
        perf_summary = {
            "Fill Rate (%)": [round(fill_rate,2)],
            "Stockout Rate (%)": [round(stockout_rate,2)],
            "Inventory Turnover": [round(inventory_turnover,2)],
            "Avg Lead Time (Days)": [round(avg_lead_time,1) if not np.isnan(avg_lead_time) else "N/A"]
        }
        st.dataframe(pd.DataFrame(perf_summary))

    # ================== TAB 5: FORECAST ==================
    with tab5:
        st.subheader("🔮 Forecasting & Predictions (to be implemented)")

    # ================== BACK BUTTON ==================
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "page2"
        st.rerun()
