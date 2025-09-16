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


def inventory_dashboard(inventory_df, transactions_df):
    st.title("Inventory and Supply Chain Insights")

    st.write(
        "Welcome to the Inventory and Supply Chain section of DemandCast. Here, you can analyze your inventory levels, "
        "track supply chain performance, and gain insights into your operations."
    )
    st.warning(
        "Please ensure your CSV files contain columns such as 'Date', 'Inventory Level', 'Order Date', 'Delivery Date', and 'Price'."
    )

    # ================== Column Detection ==================
    # Inventory dataframe provides inventory-related cols
    date_col = column_finder(inventory_df, possible_date_cols)
    inventory_col = column_finder(inventory_df, possible_inventory_cols)
    product_col = column_finder(inventory_df, possible_product_cols)
    
    # Sales dataframe provides price & order/delivery/supplier cols
    price_col_transaction = column_finder(transactions_df, price_columns)
    price_col_inventory = column_finder(inventory_df,price_columns)
    order_date_col = column_finder(inventory_df, possible_order_date_cols)
    delivery_date_col = column_finder(inventory_df, possible_delivery_date_cols)
    supplier_col = column_finder(inventory_df, possible_supplier_cols)
    category_col=column_finder(transactions_df,POSSIBLE_CATEGORY_COLS)
    region_col=column_finder(transactions_df,['region','Region','state','city','country','location'])
    possible_leadtime_col =['leadtime','lead_time','leadtime_days','lead_time_col']
    lead_time_col=column_finder(inventory_df,possible_leadtime_col)
    transaction_date_col = column_finder(transactions_df,possible_date_cols)

    # Copy dfs to avoid mutation
    df = inventory_df.copy()
    df1 = transactions_df.copy()

    # ================== Tabs ==================
    tab1, tab2, tab3, tab4, tab5, = st.tabs(
        [
            "Current Inventory Levels",
            "Lead Time Analysis",
            "Stockout Analysis",
            "Supply Chain Performance",
            "Forecasts and Predictions",
        ]
    )

    # ============== TAB 1: Current Inventory Levels ==============
    with tab1:
            st.subheader("Your inventory looks something like this")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Inventory in Warehouse", f"{df[inventory_col].sum():,.0f}"
            )

            if price_col_transaction:  # ensure price exists in sales data
                df = df.merge(df1[[product_col, price_col_transaction]], on=product_col, how="left")
                df["total_money_acquired"] = df[inventory_col] * df[price_col_transaction]

                col2.metric(
                    "Total Amount in Inventory",
                    f"{df['total_money_acquired'].sum():,.0f}",
                )
                col3.metric(
                    "Average Amount of Inventory",
                    f"{df['total_money_acquired'].mean():,.2f}",
                )

            
            st.subheader("Choose a product to display inventory levels")
            product = st.selectbox(
                    "Select product name", options=df[product_col].unique()
                )

            if not date_col or not inventory_col:
                    st.error(
                        "Required columns not found. Please ensure your CSV file contains 'Date' and 'Inventory Level'."
                    )
            else:
                    st.subheader("Your current level of inventory")
                    plt.figure(figsize=(10, 5))
                    df_temp = df[df[product_col] == product]
                    plt.bar(df_temp[date_col], df_temp[inventory_col], color="blue")
                    plt.xlabel("Date")
                    plt.ylabel("Inventory Level")
                    plt.title("Inventory Levels")
                    st.pyplot(plt)

    # ============== TAB 2: Lead Time Analysis ==============
    with tab2:
            st.title("Lead Time Analysis Insights")
            st.write(df.columns)
            st.write('order_column_found',order_date_col)
            st.write('delivery_column_found',delivery_date_col)

            if not order_date_col or not delivery_date_col:
                st.error(
                    "Required columns not found in sales data. Please ensure your sales CSV contains 'Order Date' and 'Delivery Date'."
                )
            else:
                if lead_time_col is None:
                    df[order_date_col] = pd.to_datetime(df[order_date_col])
                    df[delivery_date_col] = pd.to_datetime(df[delivery_date_col])
                    df["lead_time_days"] = (
                        df1[delivery_date_col] - df1[order_date_col]
                    ).dt.days

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Avg Lead Time", f"{df[lead_time_col].mean():.1f} days")
                col2.metric("Median", f"{df[lead_time_col].median():.1f}")
                col3.metric("Min", f"{df[lead_time_col].min()} days")
                col4.metric("Max", f"{df[lead_time_col].max()} days")
                col5.metric("Std Dev", f"{df[lead_time_col].std():.1f}")

                st.subheader("Lead Time Distribution")
                plt.figure(figsize=(10, 5))
                plt.hist(
                        df[lead_time_col].dropna(),
                        bins=20,
                        color="green",
                        edgecolor="black",
                    )
                plt.xlabel("Lead Time (days)")
                plt.ylabel("Frequency")
                plt.title("Lead Time Distribution")
                st.pyplot(plt)

                    # Supplier-level analysis
                if supplier_col:
                        st.subheader("Supplier-wise Lead Time Analysis")
                        avg_supplier_lead_time = df.groupby(supplier_col)[
                            lead_time_col
                        ].mean()
                        st.bar_chart(avg_supplier_lead_time)

    # ============== Other Tabs Skeleton ==============
    with tab3:
        st.subheader("Stockout analysis insights (to be implemented)")
        sales_df, top_products, region_df, cat_df, products_to_order,forecast=stockout_dashboard(inventory_df,transactions_df,inventory_col,product_col,category_col,region_col,price_col_transaction,lead_time_col)
            
    with tab4:
        st.subheader("📦 Supply Chain Performance Dashboard")
        st.write(forecast)
        st.write(sales_df)
        
        # --------------------------
        # 1. Fill Rate
        # --------------------------
        total_demand = sales_df['forecasted_demand'].sum()
        total_sales = sales_df['actual_sales'].sum()
        fill_rate = (total_sales / total_demand * 100) if total_demand > 0 else 0

        # --------------------------
        # 2. Stockout Rate
        # --------------------------
        stockout_rate = (df['stockout_flag'].mean() * 100)

        # --------------------------
        # 3. Inventory Turnover
        # --------------------------
        avg_inventory = df[inventory_col].mean()
        inventory_turnover = (total_sales / avg_inventory) if avg_inventory > 0 else 0

        # --------------------------
        # 4. Service Level (from earlier)
        # --------------------------
        # service_level = service_level_pct

        # --------------------------
        # 5. Lead Time Performance (if column exists)
        # --------------------------
        if lead_time_col:
            avg_lead_time = df[lead_time_col].mean()
        else:
            avg_lead_time = np.nan

        # KPI Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fill Rate (%)", f"{fill_rate:.2f}%")
        c2.metric("Stockout Rate (%)", f"{stockout_rate:.2f}%")
        c3.metric("Inventory Turnover", f"{inventory_turnover:.2f}")
        # c4.metric("Service Level (%)", f"{service_level:.2f}%")
        c5.metric("Avg Lead Time (Days)", f"{avg_lead_time:.1f}" if not np.isnan(avg_lead_time) else "N/A")

        # --------------------------
        # Trend Charts
        # --------------------------
        st.subheader("KPI Trends Over Time")

        # Fill rate trend (rolling window)
        sales_df['fill_rate'] = (sales_df['actual_sales'] / sales_df['forecasted_demand']).clip(0,1)
        fig_fill = px.line(sales_df, x=transaction_date_col, y='fill_rate',
                        title="Fill Rate Over Time")
        st.plotly_chart(fig_fill, use_container_width=True, key="fill_rate_chart")

        # Lost sales trend
        fig_loss = px.line(sales_df, x=transaction_date_col, y='lost_sales',
                        title="Lost Sales Over Time")
        st.plotly_chart(fig_loss, use_container_width=True, key="lost_sales_chart")

        # Inventory turnover over time (approx)
        df['inventory_turnover'] = total_sales / (df[inventory_col].replace(0, np.nan))
        fig_turn = px.histogram(df, x='inventory_turnover',
                                title="Distribution of Inventory Turnover Across Products")
        st.plotly_chart(fig_turn, use_container_width=True, key="inv_turnover_chart")

        # --------------------------
        # Performance Summary Table
        # --------------------------
        st.subheader("Performance Summary")
        perf_summary = {
            "Fill Rate (%)": [round(fill_rate,2)],
            "Stockout Rate (%)": [round(stockout_rate,2)],
            "Inventory Turnover": [round(inventory_turnover,2)],
            # "Service Level (%)": [round(service_level,2)],
            "Avg Lead Time (Days)": [round(avg_lead_time,1) if not np.isnan(avg_lead_time) else "N/A"]
        }
        st.dataframe(pd.DataFrame(perf_summary))


    with tab5:
            st.subheader("Forecasting and prediction insights (to be implemented)")

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "page2"
        st.rerun()
