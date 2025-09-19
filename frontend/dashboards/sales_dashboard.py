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
    # Keep original guard
    if sales_df is None:
        st.warning("No sales data uploaded yet. Please upload sales data first.")
        st.stop()
        return

    # --- Page Styling (beige-brown theme + glass cards) ---
    st.markdown(
        """
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

        /* Page title */
        .page-title {
            color: #56ade8;
            font-family: 'Montserrat', sans-serif;
            font-weight: 800;
            font-size: 36px;
            margin-bottom: 4px;
        }
        .page-sub {
            color: #ffffff;
            font-size: 43px;
            margin-top: 0px;
            margin-bottom: 18px;
        }

        /* Glass card for summaries / sections */
        .glass-card {
            background: rgba(255,255,255,0.7);
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.12);
            border: 1px solid rgba(111,78,55,0.12);
        }

        /* Metrics style */
        .metric-title {
            font-size: 25px;
            color: #59b9f5;
            font-weight: 700;
        }
        .metric-value {
            font-size: 40px;
            color: #dfdde6;
            font-weight: 800;
            margin-top: 6px;
        }

        /* Tabs customization (makes tabs look cleaner) */
        .stTabs [role="tablist"] button[role="tab"] {
            background: rgba(255,255,255,0.85);
            color: #27062c;
            border-radius: 10px;
            padding: 8px 14px;
            border: 1px solid rgba(111,78,55,0.06);
        }
        .stTabs [role="tablist"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg,#27062c,#001539);
            color: white;
            box-shadow: 0 6px 18px rgba(111,78,55,0.18);
        }

        /* Buttons used for aggregation */
        .agg-btn > button {
            background: rgba(255,255,255,0.9);
            color: #1c3f78;
            border-radius: 10px;
            padding: 8px 12px;
            border: 1px solid rgba(111,78,55,0.08);
            font-weight: 700;
        }
        .agg-btn > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 18px rgba(111,78,55,0.12);
        }

        /* Small helper text under charts */
        .chart-note {
            font-size: 18px;
            color: #c3c0d6;
            font-weight: 700;
            margin-top: 6px;
        }

        /* Back button style */
        .back-btn > button {
            background: linear-gradient(90deg, #061c49, #28062d);
            color: white;
            padding: 10px 18px;
            border-radius: 10px;
            border: none;
            font-weight: 700;
        }
        .back-btn > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 28px rgba(111,78,55,0.18);
        }
        
        .h1{
            font-size:50px;
            text-align:center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Data & Title ---
    df = sales_df.copy()
    st.markdown('<div class="page-title">📊 Sales & Revenue Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
    '<div class="page-sub" style="text-align:left; width:70%; color:#ffffff; font-size:36px; margin-top:10px;">'
    'Explore trends,<br> segment performance,and<br> forecasting tools to make data-driven decisions.'
    '</div>',
    unsafe_allow_html=True,
    )


    # st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""<h4 style="text-align:left;color:#56ade8;">Welcome to the Sales and Revenue section of DemandCast.<br>
        Upload your sales data (CSV/XLSX) to analyze trends, revenue, and actionable insights.<br>
        The interface auto-detects common column names — if detection fails, please check your headers.</h4>"""

    ,unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Small warning but styled by Streamlit (functionality unchanged)
    st.warning("⚠️ Please ensure your CSV/XLSX has columns like a Date-type column and a Sales/Revenue numeric column.")

    # ---------------------------
    # Column detection & preprocessing
    # ---------------------------
    date_col = column_finder(df, POSSIBLE_DATE_COLS)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    product_col = column_finder(df, POSSIBLE_PRODUCT_COLS)
    category_col = column_finder(df, POSSIBLE_CATEGORY_COLS)
    sales_col = column_finder(df, POSSIBLE_SALES_COLS)
    if sales_col:
        df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

    if not date_col or not sales_col:
        st.error("❌ Required columns ('Date' and 'Sales/Revenue') not found. Please check your file and column headers.")
        st.stop()

    # Show available columns info in a compact card
    # st.markdown('<div class="glass-card" style="margin-top:14px;">', unsafe_allow_html=True)
    # st.write("**Detected Columns:**", df.columns.tolist())
    # st.write("**Detected date column:**", date_col)
    # st.write("**Detected sales column:**", sales_col)
    # st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # Tabs
    # ---------------------------
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
        st.markdown("<div class='h1'>📋 Data Summary & Key Metrics</div>", unsafe_allow_html=True)
        # Compact summary card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        if st.button("Show Summary"):
            st.write(df.head())
            st.write(df.describe(include="all"))
            st.write("Data Types", df.dtypes)
            st.write("Missing Values", df.isnull().sum())
            st.write("Duplicates", df.duplicated().sum())
            st.write("Detected sales column:", sales_col)
            st.write(df[sales_col].head(10))

        col1, col2, col3 = st.columns(3)
        total_sales = df[sales_col].sum(skipna=True)
        avg_order = df[sales_col].mean(skipna=True)
        col1.markdown(f"<div class='metric-title'>Total Sales</div><div class='metric-value'> {total_sales:,.0f} </div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-title'>Total Revenue</div><div class='metric-value'> {total_sales:,.0f} </div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-title'>Avg Order Value</div><div class='metric-value'> {avg_order:,.2f} </div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Small descriptive elevator pitch about this service (2-3 lines)
        st.markdown(
            """
            <div style="margin-top:12px; color:#59b9f5; font-size:25px;">
                <h4><b>What this service does:</b></h4>
                DemandCast's Sales & Revenue module helps you uncover revenue drivers, identify seasonal trends,
                and highlight underperforming SKUs. Use these insights to improve stocking, pricing, and promotion decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------
    # Tab 2: Sales Trends
    # ---------------------------
    with tab2:
        st.markdown("<div class='h1'>📈 Sales Trend Visualization</div>", unsafe_allow_html=True)
        # default aggregation frequency in session state
        if "agg_freq" not in st.session_state:
            st.session_state.agg_freq = "Daily"

        # aggregation buttons (styled)
        # st.markdown('<div class="agg-btn">', unsafe_allow_html=True)

        # ---- CSS Styling ----
        st.markdown("""
            <style>
            /* General button wrapper */
            div.agg-btn > button[kind="secondary"] {
                background: rgba(255, 255, 255, 0.9);
                color: #1f77b4;
                border-radius: 10px;
                padding: 8px 18px;
                border: 1px solid rgba(111, 78, 55, 0.08);
                font-weight: 700;
                transition: all 0.2s ease-in-out;
                cursor: pointer;
            }

            /* Hover effect */
            div.agg-btn > button[kind="secondary"]:hover {
                background: #1f77b4;
                color: white;
                transform: translateY(-3px);
                box-shadow: 0 6px 18px rgba(111, 78, 55, 0.15);
            }

            /* Active/Selected style */
            div.agg-btn.active > button[kind="secondary"] {
                background: #1f77b4 !important;
                color: white !important;
                box-shadow: 0 6px 18px rgba(31, 119, 180, 0.3);
            }
            </style>
        """, unsafe_allow_html=True)

        # ---- Session State ----
        if "agg_freq" not in st.session_state:
            st.session_state.agg_freq = "Daily"  # default selected

        # ---- Layout ----
        cols = st.columns([4, 1, 3, 3, 3, 1, 2], gap="small")

        with cols[2]:
            active_class = "active" if st.session_state.agg_freq == "Daily" else ""
            st.markdown(f'<div class="agg-btn {active_class}">', unsafe_allow_html=True)
            if st.button("Daily", key="daily_btn"):
                st.session_state.agg_freq = "Daily"
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[3]:
            active_class = "active" if st.session_state.agg_freq == "Monthly" else ""
            st.markdown(f'<div class="agg-btn {active_class}">', unsafe_allow_html=True)
            if st.button("Monthly", key="monthly_btn"):
                st.session_state.agg_freq = "Monthly"
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[4]:
            active_class = "active" if st.session_state.agg_freq == "Yearly" else ""
            st.markdown(f'<div class="agg-btn {active_class}">', unsafe_allow_html=True)
            if st.button("Yearly", key="yearly_btn"):
                st.session_state.agg_freq = "Yearly"
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Debugging/Check ----
        st.write("Selected Aggregation:", st.session_state.agg_freq)


        # create aggregation plot using existing utility
        fig = sales_aggregation_over_time(df, date_col, sales_col, st.session_state.agg_freq)

        # if fig is a matplotlib figure, show it; otherwise try plotly
        try:
            st.pyplot(fig)
        except Exception:
            st.plotly_chart(fig)

        st.markdown('<div class="chart-note">Tip: switch aggregation (Daily / Monthly / Yearly) to inspect short/long-term trends.</div>', unsafe_allow_html=True)

    # ---------------------------
    # Tab 3: Filtering & Segmentation
    # ---------------------------
    with tab3:
        st.markdown("<div class='h1'>🔍 Filter & Segment Data</div>", unsafe_allow_html=True)

        # Sidebar date filter (functionality unchanged)
        # date_range = st.sidebar.date_input(
        #     "Date Range", value=(df[date_col].min(), df[date_col].max())
        # )

        filtered_df = df.copy()
        # if date_range:
        #     filtered_df = filtered_df[
        #         (filtered_df[date_col] >= pd.to_datetime(date_range[0]))
        #         & (filtered_df[date_col] <= pd.to_datetime(date_range[1]))
        #     ]

        # Trend line
        import plotly.express as px
        import plotly.graph_objects as go

    # ===== Chart Style Function =====
        def apply_chart_style(fig, title=None):
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=20, family="Arial, sans-serif", color="#333"),
                    x=0.5,  # center title
                    xanchor="center"
                ),
                paper_bgcolor="rgba(0,0,0,0)",   # transparent background
                plot_bgcolor="rgba(0,0,0,0)",    # transparent inside chart
                font=dict(family="Arial, sans-serif", size=12, color="#444"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color="#333")
                ),
                margin=dict(l=20, r=20, t=50, b=40)
            )
            return fig
        def top_contributors(df, group_col, value_col, threshold=0.5):
            df_sorted = df.groupby(group_col, as_index=False)[value_col].sum()
            df_sorted = df_sorted.sort_values(value_col, ascending=False)
            df_sorted["cum_percent"] = df_sorted[value_col].cumsum() / df_sorted[value_col].sum()

            # keep until cumulative <= threshold
            top_df = df_sorted[df_sorted["cum_percent"] <= threshold]

            # put the rest as "Others"
            others = df_sorted[df_sorted["cum_percent"] > threshold][value_col].sum()
            if others > 0:
                top_df = pd.concat([top_df, pd.DataFrame({group_col: ["Others"], value_col: [others]})])

            return top_df
        # ===== Time Series Chart (Plotly instead of st.line_chart) =====
        try:
            df_line = filtered_df[[date_col, sales_col]].set_index(date_col)
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_line.index,
                y=df_line[sales_col],
                mode="lines+markers",
                line=dict(color="#4794EC", width=3),
                marker=dict(size=6, color="#b832c7", line=dict(width=1, color="white")),
                name="Sales"
            ))
            fig_line = apply_chart_style(fig_line, "📈 Sales Over Time")
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception:
            st.info("Unable to draw time series chart with provided columns.")

        # ===== Side-by-Side Charts =====
        col1, col2 = st.columns(2)
        with col1:
            if product_col:
                df_product = top_contributors(filtered_df, product_col, sales_col, threshold=0.5)
                st.markdown("<div class='h1'>📊 Top 50% Product Sales Share</div>", unsafe_allow_html=True)
                fig_product = px.pie(
                    df_product,
                    values=sales_col,
                    names=product_col,
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_product.update_traces(textinfo="percent+label", pull=[0.05]*len(df_product))
                fig_product = apply_chart_style(fig_product)
                st.plotly_chart(fig_product, use_container_width=True)
            else:
                st.info("Product column not detected.")

        with col2:
            if category_col:
                df_category = top_contributors(filtered_df, category_col, sales_col, threshold=0.5)
                st.markdown("<h3>📊 Top 50% Category Sales Share</h3>", unsafe_allow_html=True)
                fig_category = px.pie(
                    df_category,
                    values=sales_col,
                    names=category_col,
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                fig_category.update_traces(textinfo="percent+label", pull=[0.05]*len(df_category))
                fig_category = apply_chart_style(fig_category)
                st.plotly_chart(fig_category, use_container_width=True)
            else:
                st.info("Category column not detected.")
    # ---------------------------
    # Tab 4: Forecasting
    # ---------------------------
    with tab4:
        st.subheader("🔮 Forecasting (Hybrid Model)")
        # Delegate to your hybrid plotting function (keeps behaviour)
        try:
            hybrid_sales_forecast_plot(sales_df, sales_col, date_col, forecast_periods=30)
        except Exception as e:
            st.error(f"Forecasting failed: {e}")

    # ---------------------------
    # Tab 5: Google Trends
    # ---------------------------
    with tab5:
        st.subheader("📊 Google Trends Analysis")
        agg_col = st.selectbox("Select Column for Trends", df.columns.tolist())

        if agg_col and agg_col.lower() in [c.lower() for c in POSSIBLE_PRODUCT_COLS + POSSIBLE_CATEGORY_COLS]:
            product_list = df[agg_col].dropna().unique().tolist()
            selected_product = st.selectbox("Select Product", product_list)

            timeframe = st.selectbox("Timeframe", ["now 7-d", "today 1-m", "today 3-m", "today 12-m", "today 5-y"])
            country = st.selectbox("Country", ["Worldwide", "India", "United States"])

            country_geo = {"Worldwide": "", "India": "IN", "United States": "US"}
            df_trends = fetch_google_trends_best_keyword(selected_product, timeframe=timeframe, geo=country_geo[country])

            if df_trends is None or df_trends.empty:
                st.info("No Google Trends data returned for this selection.")
            else:
                st.line_chart(df_trends)
        else:
            st.info("Select a product/category-like column to run Google Trends analysis.")

    # ---------------------------
    # Tab 6: Revenue Analysis
    # ---------------------------
    with tab6:
        # Subheader with animation
        st.markdown("""
            <style>
            .revenue-subheader {
                font-size: 1.6rem;
                font-weight: 700;
                color: #5B2C98;
                background: linear-gradient(90deg, #4A00E0, #8E2DE2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: fadeIn 1.5s ease-in-out;
            }
            @keyframes fadeIn {
                from {opacity: 0; transform: translateY(-10px);}
                to {opacity: 1; transform: translateY(0);}
            }
            </style>
            <h3 class="revenue-subheader">💰 Revenue Analysis</h3>
        """, unsafe_allow_html=True)

        # Revenue metrics
        df[date_col] = pd.to_datetime(df[date_col])
        total_revenue = df[sales_col].sum()
        avg_revenue = df[sales_col].mean()

        col1, col2 = st.columns(2)
        col1.metric("Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("Avg Revenue", f"${avg_revenue:,.2f}")

        # Chart with styled container
        st.markdown("""
            <style>
            .chart-container {
                background: rgba(74, 0, 224, 0.07);
                border-radius: 16px;
                padding: 12px;
                box-shadow: 0px 4px 12px rgba(142, 45, 226, 0.25);
                animation: fadeUp 1.2s ease-in-out;
            }
            @keyframes fadeUp {
                from {opacity: 0; transform: translateY(15px);}
                to {opacity: 1; transform: translateY(0);}
            }
            </style>
            <div class="chart-container">
        """, unsafe_allow_html=True)

        try:
            st.line_chart(df.set_index(date_col)[sales_col])
        except Exception:
            st.write("Unable to draw revenue chart with provided data.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # Back Button with styling
    # ---------------------------
    st.markdown("""
        <style>
        .back-btn button {
            background: linear-gradient(90deg, #4A00E0, #8E2DE2);
            color: white !important;
            font-weight: 600;
            padding: 0.6rem 1.5rem;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0px 4px 12px rgba(142, 45, 226, 0.3);
        }
        .back-btn button:hover {
            transform: scale(1.08);
            box-shadow: 0px 6px 16px rgba(74, 0, 224, 0.4);
        }
        .back-btn button:active {
            transform: scale(0.96);
        }
        </style>
        <div class="back-btn">
    """, unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "page2"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
