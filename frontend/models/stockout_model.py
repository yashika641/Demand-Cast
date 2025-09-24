import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from utils.column_finder import column_finder
import plotly.express as px

# --------------------------
# Column synonyms
# --------------------------
FULFILLED_SYNONYMS = [
    "fulfilled_qty", "sales_qty", "sold_qty", "units_sold", "quantity_sold",
    "product_quantity", "qty", "order_fulfilled_qty", "shipped_qty",
    "delivered_qty", "completed_qty", "quantity_purchased", "purchase_qty",
    "items_sold", "num_items", "checkout_qty", "dispatched_qty",
    "issued_qty", "outgoing_qty", "allocation_qty", "picked_qty",
    "actual_qty", "completed_units", "sales_units", "billed_qty"
]

REORDER_LEVEL_SYNONYMS = [
    "reorder_level", "reorder_point", "reorder_threshold", "reorder_qty",
    "replenishment_level", "minimum_stock", "min_stock_level", "stock_reorder_level",
    "replenish_point", "safety_stock", "par_level", "restock_level",
    "refill_point", "min_level", "trigger_level", "buffer_stock",
    "threshold_qty", "critical_stock_level", "order_point", "alert_level",
    "rl", "reorder_limit", "low_stock_trigger"
]

transaction_date_col_names = [
    "date", "transaction_date", "order_date", "sales_date", "purchase_date",
    "invoice_date", "billing_date", "delivery_date", "order_placed_date",
    "order_completed_date", "date_of_transaction", "date_of_order",
    "posting_date", "checkout_date", "sale_date", "shipment_date",
    "dispatch_date", "payment_date", "date_x", "date_y",
    "trx_date", "txn_date", "doc_date"
]

# --------------------------
# Stockout Dashboard Function with CSS
# --------------------------
def stockout_dashboard(inventory_df, transactions_df, inventory_col, product_col,
                       category_col=None, region_col=None, unit_price_col=None,
                       lead_time_col=None):

    # ---- Custom CSS Styling ----
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
        /* Section headers */
        .css-1v0mbdj h1, h2, h3, h4, h5 {
            color: #f2f2f2;
            text-shadow: 1px 1px 4px #00000050;
        }
        /* Plotly charts background with hover effect */
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
        /* Metric cards */
        .stMetric {
            background-color: rgba(255,255,255,0.1) !important;
            border-radius: 15px;
            padding: 10px;
            text-align: center;
            transition: transform 0.3s ease-in-out;
        }
        .stMetric:hover {
            transform: scale(1.05);
        }
        </style>
        <video autoplay muted loop class="bg-video">
            <source src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/bg-video1.mp4" type="video/mp4" >
        </video>
    """, unsafe_allow_html=True)

    df_inv = inventory_df.copy()
    df_txn = transactions_df.copy()

    # --------------------------
    # Detect dynamic columns
    # --------------------------
    reorder_col = column_finder(df_inv, REORDER_LEVEL_SYNONYMS)
    fulfilled_col = column_finder(df_txn, FULFILLED_SYNONYMS)
    transaction_date_col = column_finder(df_txn, transaction_date_col_names)

    # --------------------------
    # Flags
    # --------------------------
    df_inv['stockout_flag'] = (df_inv[inventory_col] <= 0).astype(int)
    df_inv['at_risk_flag'] = (df_inv[inventory_col] <= df_inv[reorder_col]).astype(int)

    # --------------------------
    # Forecast & Lost Sales
    # --------------------------
    sales_df = df_txn.groupby(transaction_date_col)[fulfilled_col].sum().reset_index()
    sales_df.rename(columns={fulfilled_col:'actual_sales'}, inplace=True)

    prophet_df = sales_df.rename(columns={transaction_date_col:'ds', "actual_sales":'y'})
    m = Prophet()
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    sales_df[transaction_date_col] = pd.to_datetime(sales_df[transaction_date_col], errors="coerce")
    forecast['ds'] = pd.to_datetime(forecast['ds'], errors="coerce")
    sales_df = sales_df.merge(forecast[['ds','yhat']], left_on=transaction_date_col, right_on='ds', how='left')
    sales_df.rename(columns={'yhat': "forecasted_demand"}, inplace=True)
    sales_df['lost_sales'] = (sales_df['forecasted_demand'] - sales_df['actual_sales']).clip(lower=0)

    total_actual_sales = sales_df['actual_sales'].sum()
    total_forecasted = sales_df['forecasted_demand'].sum()
    total_lost_sales = sales_df['lost_sales'].sum()
    lost_sales_pct = (total_lost_sales / total_forecasted * 100) if total_forecasted>0 else 0
    service_level_pct = ((total_forecasted - total_lost_sales)/total_forecasted*100) if total_forecasted>0 else 0
    lost_sales_value = (sales_df['lost_sales_value'].sum() if 'lost_sales_value' in sales_df.columns else np.nan)

    # Top 5 products summary
    prod_df = df_txn.groupby(product_col)[fulfilled_col].sum().reset_index()
    prod_df.rename(columns={fulfilled_col:'actual_sales'}, inplace=True)
    prod_df['forecasted_demand'] = prod_df['actual_sales'].rolling(3, min_periods=1).mean()
    prod_df['lost_sales'] = (prod_df['forecasted_demand'] - prod_df['actual_sales']).clip(lower=0)
    top_products = prod_df.sort_values('lost_sales', ascending=False).head(5)
    top5_products_str = ", ".join(top_products[product_col].head(5).tolist())

    # --------------------------
    # Streamlit Dashboard
    # --------------------------
    st.title("📦 Stockout & Reordering Dashboard")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Lost Sales Qty', round(total_lost_sales,2))
    col2.metric('Lost Sales %', round(lost_sales_pct,2))
    col3.metric('Service Level %', round(service_level_pct,2))
    col4.metric('Total Forecasted Demand', round(total_forecasted,2))

    # Stockout Trend
    st.subheader("📈 Stockout Trend Over Time")
    fig_trend = px.line(sales_df, x=transaction_date_col, y='lost_sales',
                        title='Lost Sales Over Time', color_discrete_sequence=['#b24cff'])
    st.plotly_chart(fig_trend)

    # Top Products
    st.subheader("🏆 Top 5 Products by Lost Sales")
    st.dataframe(top_products[[product_col,'lost_sales']])
    fig_prod = px.bar(top_products, x=product_col, y='lost_sales',
                      title='Top 5 Products Lost Sales', color_discrete_sequence=['#7f5aff'])
    st.plotly_chart(fig_prod)

    # Region Impact
    if region_col:
        st.subheader("🌐 Lost Sales by Region")
        region_df = df_txn.groupby(region_col)[fulfilled_col].sum().reset_index()
        region_df.rename(columns={fulfilled_col:'actual_sales'}, inplace=True)
        region_df['forecasted_demand'] = region_df['actual_sales'].rolling(2, min_periods=1).mean()
        region_df['lost_sales'] = (region_df['forecasted_demand'] - region_df['actual_sales']).clip(lower=0)
        st.dataframe(region_df[[region_col,'lost_sales']])
        fig_region = px.bar(region_df, x=region_col, y='lost_sales',
                            title='Region-wise Lost Sales', color_discrete_sequence=['#4b5fff'])
        st.plotly_chart(fig_region)

    # Category Analysis
    if category_col:
        st.subheader("📊 Lost Sales by Category")
        cat_df = df_txn.groupby(category_col)[fulfilled_col].sum().reset_index()
        cat_df.rename(columns={fulfilled_col:'actual_sales'}, inplace=True)
        cat_df['forecasted_demand'] = cat_df['actual_sales'].rolling(2, min_periods=1).mean()
        cat_df['lost_sales'] = (cat_df['forecasted_demand'] - cat_df['actual_sales']).clip(lower=0)
        st.dataframe(cat_df[[category_col,'lost_sales']])
        fig_cat = px.pie(cat_df, names=category_col, values='lost_sales',
                         title='Category-wise Lost Sales', color_discrete_sequence=px.colors.sequential.Purple)
        st.plotly_chart(fig_cat)

    # At-Risk SKUs & Safety Stock
    st.subheader("⚠️ Products at Risk of Stockout")
    at_risk = df_inv[df_inv['at_risk_flag']==1][product_col].unique()
    st.write(at_risk)

    if lead_time_col:
        df_inv['daily_forecast'] = total_forecasted / len(sales_df)
        df_inv['safety_stock_adj'] = np.maximum(0, (df_inv['daily_forecast'] * df_inv[lead_time_col]) - df_inv[inventory_col])
        df_inv['days_until_reorder'] = np.ceil((df_inv[inventory_col] - df_inv[reorder_col]) / df_inv['daily_forecast'])
        df_inv['next_reorder_date'] = pd.to_datetime('today') + pd.to_timedelta(df_inv['days_until_reorder'], unit='D')
        products_to_order = df_inv[df_inv['at_risk_flag']==1][
            [product_col, inventory_col, reorder_col, 'safety_stock_adj', 'next_reorder_date']]
        st.subheader("🔄 Recommended Products to Reorder")
        st.dataframe(products_to_order.sort_values('next_reorder_date'))
    else:
        products_to_order = None

    return sales_df, top_products, region_df if region_col else None, cat_df if category_col else None, products_to_order, forecast
