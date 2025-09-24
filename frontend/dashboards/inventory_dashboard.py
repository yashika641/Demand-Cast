import plotly.express as px
import plotly.graph_objects as go
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
        .stApp {
            background: transparent !important;
        }
        video.bg-video {
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
            font-size: 25px;
            color: #ccd4db;
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
        <video autoplay muted loop class="bg-video">
            <source src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/bg-video1.mp4" type="video/mp4" >
        </video>
    """, unsafe_allow_html=True)

    # ================== TITLE ==================
    st.markdown("""
        <div>
            <h1>📦 Inventory & Supply Chain Insights</h1>
            <p style="color:#f5f5f5; font-size:25px; text-align:center;">
                Analyze inventory levels, lead times, stockouts, and supply chain performance with interactive dashboards.
            </p>
            <p style="color:#dbeafe; font-size:20px; text-align:center;">
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
        st.header("📊Current Inventory Levels")
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

            df_temp = df[df[product_col] == product]

            # Sort by date to avoid jagged lines
            df_temp = df_temp.sort_values(by=date_col)

            # Line chart with smooth curve
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_temp[date_col],
                    y=df_temp[inventory_col],
                    mode="lines+markers",
                    line=dict(shape="spline", color="#1f77b4", width=3),
                    marker=dict(size=6, color="#ff7f0e", line=dict(width=1, color="white")),
                    name="Inventory Level",
                    hovertemplate="Date: %{x}<br>Inventory: %{y}<extra></extra>"
                )
            )

            # Style layout
            fig.update_layout(
                title=dict(text=f"📦 Inventory Levels - {product}", x=0.5, font=dict(size=20, color="#333")),
                xaxis_title="Date",
                yaxis_title="Inventory Level",
                template="plotly_white",
                hovermode="x unified",
                plot_bgcolor="#f9f9f9",
                margin=dict(l=40, r=40, t=60, b=40),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

    # ================== TAB 2: LEAD TIME ==================
    with tab2:
        st.subheader("📊 Lead Time Distribution (Interactive)")

        # Histogram with stronger blue-purple-pink theme
        fig = px.histogram(
            df,
            x=lead_time_col,
            nbins=20,
            title="Lead Time Distribution",
            color_discrete_sequence=["#1f77b4", "#8a2be2", "#ff1493"],  # vivid blue, purple, pink
            opacity=0.85
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1.2)
        fig.update_layout(
            xaxis_title="Lead Time (days)",
            yaxis_title="Frequency",
            template="plotly_white",
            title_font=dict(size=20, color="#8a2be2"),
            plot_bgcolor="rgba(230,230,250,0.9)",  # lavender
            paper_bgcolor="rgba(255,255,255,1)"
        )
        st.plotly_chart(fig, use_container_width=True)


        if supplier_col and supplier_col in df.columns:
            st.subheader("📦 Supplier-wise Lead Times (Interactive)")

            # Dropdown for supplier selection
            supplier_list = df[supplier_col].dropna().unique()
            selected_supplier = st.selectbox("Select Supplier", supplier_list)

            # ✅ Filter supplier data
            supplier_data = df[df[supplier_col] == selected_supplier].copy()

            # Ensure order_date is datetime
            if not pd.api.types.is_datetime64_any_dtype(supplier_data[order_date_col]):
                supplier_data[order_date_col] = pd.to_datetime(
                    supplier_data[order_date_col], errors="coerce"
                )

            # Drop invalid dates
            supplier_data = supplier_data.dropna(subset=[order_date_col])

            if not supplier_data.empty:
                # 👇 Let user choose granularity
                freq_choice = st.radio(
                    "Select Time Granularity:",
                    ["Daily", "Weekly", "Monthly"],
                    horizontal=True
                )

                if freq_choice == "Daily":
                    supplier_data["Date"] = supplier_data[order_date_col].dt.date
                    trend_df = (
                        supplier_data.groupby("Date")[lead_time_col].mean().reset_index()
                    )
                    x_col = "Date"

                elif freq_choice == "Weekly":
                    supplier_data["YearWeek"] = supplier_data[order_date_col].dt.to_period("W").astype(str)
                    trend_df = (
                        supplier_data.groupby("YearWeek")[lead_time_col].mean().reset_index()
                    )
                    x_col = "YearWeek"

                else:  # Monthly
                    supplier_data["YearMonth"] = supplier_data[order_date_col].dt.to_period("M").astype(str)
                    trend_df = (
                        supplier_data.groupby("YearMonth")[lead_time_col].mean().reset_index()
                    )
                    x_col = "YearMonth"

                # Line chart for supplier trend
                fig2 = px.line(
                    trend_df,
                    x=x_col,
                    y=lead_time_col,
                    markers=True,
                    title=f"📈 {freq_choice} Lead Time Trend for {selected_supplier}",
                    line_shape="spline",
                    color_discrete_sequence=["#0077b6"],
                )
                fig2.update_traces(line=dict(width=3), marker=dict(size=8, color="#ff4da6"))
                fig2.update_layout(
                    xaxis_title=freq_choice,
                    yaxis_title="Average Lead Time (days)",
                    template="plotly_white",
                    title_font=dict(size=20, color="#6f42c1"),
                    plot_bgcolor="rgba(240,240,255,0.95)",
                )
                st.plotly_chart(fig2, use_container_width=True)

            else:
                st.warning(f"No valid date/lead time data available for supplier: {selected_supplier}")

            # Supplier comparison bar chart (all suppliers)
            st.subheader("📊 Supplier Comparison (All Suppliers)")
            avg_supplier_lead_time = (
                df.groupby(supplier_col)[lead_time_col].mean().reset_index()
            )
            fig3 = px.bar(
                avg_supplier_lead_time,
                x=supplier_col,
                y=lead_time_col,
                color=lead_time_col,
                color_discrete_sequence=px.colors.qualitative.Dark24,  # 24 distinct colors
                barmode="group",    
                title="Average Supplier Lead Times (All Suppliers)",
            )
            fig3.update_traces(marker_line_color="white", marker_line_width=1.2)
            fig3.update_layout(
                xaxis_title="Supplier",
                yaxis_title="Average Lead Time (days)",
                template="plotly_white",
                title_font=dict(size=20, color="#8a2be2"),
                plot_bgcolor="rgba(250,245,255,0.95)",
            )
            st.plotly_chart(fig3, use_container_width=True)

        else:
            st.warning("Supplier column not found in the dataset.")

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
