import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
import logging
import warnings
from mlxtend.frequent_patterns import apriori, association_rules
from utils.column_finder import column_finder

# Fallback images
IMAGES_FALLBACK = [
    "https://cdn-icons-png.flaticon.com/512/4436/4436481.png",
    "https://cdn-icons-png.flaticon.com/512/2933/2933245.png",
    "https://cdn-icons-png.flaticon.com/512/3209/3209265.png",
    "https://cdn-icons-png.flaticon.com/512/2921/2921222.png",
    "https://cdn-icons-png.flaticon.com/512/2920/2920664.png",
    "https://cdn-icons-png.flaticon.com/512/891/891462.png",
    "https://cdn-icons-png.flaticon.com/512/3081/3081559.png",
    "https://cdn-icons-png.flaticon.com/512/2920/2920054.png",
    "https://cdn-icons-png.flaticon.com/512/1040/1040230.png",
    "https://cdn-icons-png.flaticon.com/512/1055/1055646.png",
    "https://cdn-icons-png.flaticon.com/512/4727/4727422.png",
    "https://cdn-icons-png.flaticon.com/512/2921/2921822.png",
    "https://cdn-icons-png.flaticon.com/512/2921/2921800.png",
    "https://cdn-icons-png.flaticon.com/512/2921/2921827.png",
    "https://cdn-icons-png.flaticon.com/512/1048/1048946.png",
    "https://cdn-icons-png.flaticon.com/512/1048/1048948.png",
    "https://cdn-icons-png.flaticon.com/512/1048/1048956.png",
    "https://cdn-icons-png.flaticon.com/512/3122/3122929.png",
    "https://cdn-icons-png.flaticon.com/512/892/892458.png",
    "https://cdn-icons-png.flaticon.com/512/891/891462.png",
]

# Candidate column name lists
POSSIBLE_PRODUCT_COLS = [
    "product",
    "product_name",
    "productname",
    "product_title",
    "item",
    "item_name",
    "model",
    "model_name",
]
POSSIBLE_PRICE_COLS = [
    "price",
    "unit_price",
    "cost",
    "cost_price",
    "purchase_price",
    "selling_price",
    "mrp",
    "retail_price",
    "wholesale_price",
    "list_price",
    "final_price",
    "sale_price",
]
POSSIBLE_DESC_COLS = [
    "description",
    "desc",
    "product_description",
    "details",
    "info",
    "product_info",
    "specs",
    "specifications",
    "overview",
    "summary",
    "text",
]
POSSIBLE_SALES_COLS = ["sales", "revenue", "amount", "price", "order_value"]
POSSIBLE_UNITS_SOLD_COLS = [
    "quantity",
    "quantity_sold",
    "units",
    "items_sold",
    "order_quantity",
    "sales_count",
]
POSSIBLE_CATEGORY_COLS = [
    "category",
    "category_name",
    "product_category",
    "product_type",
    "department",
    "subcategory",
]
POSSIBLE_DATE_COLS = [
    "date",
    "order_date",
    "purchase_date",
    "invoice_date",
    "sale_date",
    "transaction_date",
    "timestamp",
]
POSSIBLE_INVENTORY_COLS = [
    "inventory",
    "stock",
    "inventory_level",
    "stock_level",
    "stock_quantity",
    "on_hand",
]

# ---------------- Single Large product_dashboard ----------------
def product_dashboard(products_df=None, sales_df=None, inventory_df=None, transactions_df=None):
    """
    Single-file product dashboard (keeps everything in one big function as requested).
    Applies blue/purple glass styling, animations, and suppresses noisy logs.
    """
    # -------------------- Silence noisy libraries --------------------
    # Quiet CmdStan/Prophet/Matplotlib/transformers logs that spam the terminal
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # -------------------- Guards --------------------
    if products_df is None:
        st.warning("No products_df provided.")
        return

    # --- shallow copies to avoid mutating user data
    prod = products_df.copy()
    sales = sales_df.copy() if sales_df is not None else None
    inv = inventory_df.copy() if inventory_df is not None else None
    tx = transactions_df.copy() if transactions_df is not None else None

    # -------------------- Page Styling (blue / purple theme) --------------------
    st.markdown(
        """
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

        /* Main title */
        .prod-title {
            font-size: 38px;
            font-weight: 800;
            color: #dbeafe;
            letter-spacing: 0.2px;
            margin-bottom: 4px;
        }
        .prod-sub {
            color: #bbcfff;
            font-size: 50px;
            margin-top: 0;
            margin-bottom: 14px;
        }

        /* Glass card */
        .glass {
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 8px 28px rgba(31,41,55,0.45);
            border: 1px solid rgba(123, 85, 255, 0.08);
            backdrop-filter: blur(6px);
            margin-bottom: 12px;
        }

        /* Animated subheaders */
        .section-h {
            font-size:25px;
            font-weight:700;
            color: #c145b8;
            margin-bottom:6px;
            background: linear-gradient(90deg,#7c3aed,#2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: fadeInUp 0.9s ease both;
        }

        @keyframes fadeInUp {
            from {opacity:0; transform: translateY(8px);}
            to {opacity:1; transform: translateY(0);}
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg,#4f46e5,#2563eb);
            color: white;
            border-radius: 10px;
            padding: 8px 14px;
            font-weight:700;
            border: none;
            transition: transform .18s ease, box-shadow .18s ease;
        }
        .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(37,99,235,0.2); }

        /* Metrics */
        .metric-box {
            background: rgba(255,255,255,0.03);
            padding:12px;
            border-radius:12px;
            box-shadow: 0 6px 18px rgba(10,12,30,0.6);
            border: 1px solid rgba(99,102,241,0.06);
        }
        .metric-title { color:#c7d8ff; font-weight:700; font-size:14px; }
        .metric-value { color:#ffffff; font-weight:800; font-size:20px; margin-top:6px; }

        /* Dataframe card */
        .data-card { padding:10px; border-radius:10px; background: rgba(255,255,255,0.02); }

        /* Carousel card tweaks */
        .card { width:220px; height:340px; background: linear-gradient(180deg, #ffffff, #f3f6ff); border-radius:12px; padding:12px; text-align:center; box-shadow:0 8px 22px rgba(2,6,23,0.45); }
        .card h3 { font-size:14px; color:#0b1220; margin:8px 0 4px; }
        .card p { color:#334155; font-size:12px; }

        /* small helpers */
        .muted { color:#9fb3ff; font-size:13px; }

        /* reduce Streamlit default margins for denser layout */
        .css-1d391kg { padding-top: 8px; }
        </style>
        
        <video autoplay muted loop class="bg-video">
            <source src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/bg-video1.mp4" type="video/mp4" >
        </video>
        """,
        unsafe_allow_html=True,
    )

    # -------------------- Header --------------------
    st.markdown(f"<div class='prod-title'>🛍️ Product Dashboard</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='prod-sub'> Product Dashboard – <br>Track inventory, performance, and<br> trends in one place.</div>", unsafe_allow_html=True)

    # -------------------- Column detection --------------------
    product_col = column_finder(prod, POSSIBLE_PRODUCT_COLS)
    price_col = column_finder(prod, POSSIBLE_PRICE_COLS) or (column_finder(sales, POSSIBLE_PRICE_COLS) if sales is not None else None)
    desc_col = column_finder(prod, POSSIBLE_DESC_COLS)
    sales_col = column_finder(sales, POSSIBLE_SALES_COLS) if sales is not None else None
    units_sold_col = column_finder(sales, POSSIBLE_UNITS_SOLD_COLS) if sales is not None else None
    category_col_sales = column_finder(sales, POSSIBLE_CATEGORY_COLS) if sales is not None else None
    category_col_inv = column_finder(inv, POSSIBLE_CATEGORY_COLS) if inv is not None else None
    date_col_sales = column_finder(sales, POSSIBLE_DATE_COLS) if sales is not None else None
    date_col_inv = column_finder(inv, POSSIBLE_DATE_COLS) if inv is not None else None
    inventory_col = column_finder(inv, POSSIBLE_INVENTORY_COLS) if inv is not None else None

    # -------------------- Images handling --------------------
    image_col = None
    for c in prod.columns:
        if c.lower() in ["image", "images", "img_url", "picture", "photo", "image_url"]:
            image_col = c
            break
    if image_col:
        prod["images"] = prod[image_col].fillna("").replace("", np.nan)
        prod["images"] = prod["images"].fillna(random.choice(IMAGES_FALLBACK))
    else:
        prod["images"] = [random.choice(IMAGES_FALLBACK) for _ in range(len(prod))]

    # -------------------- Product carousel (html + swiper) --------------------
    slides_html = ""
    for _, p in prod.iterrows():
        pname = (
            p[product_col]
            if product_col and product_col in p and pd.notna(p[product_col])
            else "Unknown Product"
        )
        pdesc = (
            p[desc_col]
            if desc_col and desc_col in p and pd.notna(p[desc_col])
            else "No description available"
        )
        pprice = (
            p[price_col]
            if price_col and price_col in p and pd.notna(p[price_col])
            else "N/A"
        )
        pimg = p["images"]
        slides_html += f"""
        <div class="swiper-slide">
        <div class="card">
            <img src="{pimg}" alt="{pname}">
            <h3>{pname}</h3>
            <p>{pdesc}</p>
            <h4>₹ {pprice}</h4>
        </div>
        </div>
        """

    carousel_html = f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>

    <style>
    .glass {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 10px;
        margin-bottom: 16px;
        border-radius: 12px;
        color: #fff;
    }}
    .swiper-slide {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .card {{
        background: rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        width: 200px; /* increased size */
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .card:hover {{
        transform: translateY(-8px) scale(1.08); /* pop-up effect */
        box-shadow: 0 8px 25px rgba(0,0,0,0.45);
    }}
    .card img {{
        width: 100px;
        height: 100px;
        object-fit: contain;
        margin: 0 auto 8px;
        display: block;
    }}
    .card h3 {{
        font-size: 1.05rem;
        margin: 4px 0;
    }}
    .card p {{
        font-size: 0.85rem;
        height: 48px;
        overflow: hidden;
        margin: 4px 0;
    }}
    .card h4 {{
        color: #4fff4f;
        margin-top: 6px;
    }}
    </style>

    <div class="glass">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-weight:800;color:#e6eef8;">Featured Products</div>
        <div class="muted">Swipe →</div>
    </div>
    <div style="height:14px;"></div>
    <div class="swiper">
        <div class="swiper-wrapper">{slides_html}</div>
        <div class="swiper-button-next" style="color:#fff;"></div>
        <div class="swiper-button-prev" style="color:#fff;"></div>
    </div>
    </div>

    <script>
    new Swiper('.swiper', {{
        slidesPerView: 3,
        spaceBetween: 10,  // reduced spacing
        loop: true,
        centeredSlides: true,
        autoplay: {{
            delay: 2500,
            disableOnInteraction: false,
        }},
        pagination: false,  // removed dots
        navigation: {{ nextEl: '.swiper-button-next', prevEl: '.swiper-button-prev' }},
        breakpoints: {{
        320: {{ slidesPerView: 1 }},
        768: {{ slidesPerView: 2 }},
        1024: {{ slidesPerView: 3 }}
        }}
    }});
    </script>
    """

    try:
        components.html(carousel_html, height=500, scrolling=False)
    except Exception:
        st.info("Product carousel couldn't be displayed — showing product table instead.")
        if product_col and price_col and product_col in prod.columns and price_col in prod.columns:
            st.dataframe(prod[[product_col, price_col]].head(20))
        else:
            st.dataframe(prod.head(20))



    # -------------------- Top products by sales --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    if (
        sales is not None
        and sales_col
        and product_col
        and product_col in sales.columns
        and sales_col in sales.columns
    ):
        try:
            top_sales = (
                sales.groupby(product_col)[sales_col]
                .sum()
                .reset_index()
                .sort_values(sales_col, ascending=False)
                .head(10)
            )
            st.markdown("<div class='section-h'>📈 Top Products by Sales</div>", unsafe_allow_html=True)
            fig = px.bar(
                top_sales,
                x=product_col,
                y=sales_col,
                title=f"Top 10 Products by {sales_col}",
                color_discrete_sequence=px.colors.sequential.Blues,  # bar colors
            )

            # Update Plotly layout (chart styling)
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",   # transparent background
                paper_bgcolor="rgba(0,0,0,0)",  # transparent frame
                font=dict(size=14, color="#1E3A8A"),  # navy labels
                title=dict(font=dict(size=20, color="#0EA5E9"), x=0.5),  # center title
                xaxis=dict(showgrid=False, tickfont=dict(color="#2563EB")),  # blue axis
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", tickfont=dict(color="#2563EB")),
            )

            # Wrap chart in a styled div
            st.markdown(
                """
                <style>
                .chart-container {
                    background: linear-gradient(to right, #E0F2FE, #F0F9FF);
                    padding: 20px;
                    border-radius: 16px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    margin-bottom: 30px;
                }
                </style>
                <div class="chart-container">
                """,
                unsafe_allow_html=True,
            )

            st.plotly_chart(fig, use_container_width=True, key="top_sales_chart")

            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            raise 
        
    # -------------------- Inventory turnover --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    turnover_df = None
    if (
        sales is not None
        and inv is not None
        and product_col
        and product_col in sales.columns
        and product_col in inv.columns
    ):
        if units_sold_col and units_sold_col in sales.columns:
            price_for_cogs = column_finder(sales, POSSIBLE_PRICE_COLS) or price_col
            if price_for_cogs and price_for_cogs in sales.columns:
                try:
                    sales["cogs"] = sales[units_sold_col].astype(float) * sales[price_for_cogs].astype(float)
                    cogs_per_product = sales.groupby(product_col)["cogs"].sum().reset_index()
                    if inventory_col and inventory_col in inv.columns:
                        avg_inventory = (
                            inv.groupby(product_col)[inventory_col]
                            .mean()
                            .reset_index()
                            .rename(columns={inventory_col: "avg_inventory"})
                        )
                        turnover_df = pd.merge(cogs_per_product, avg_inventory, on=product_col, how="inner")
                        turnover_df["avg_inventory"] = turnover_df["avg_inventory"].replace(0, np.nan)
                        turnover_df["inventory_turnover"] = turnover_df["cogs"] / turnover_df["avg_inventory"]
                        turnover_df["inventory_turnover"] = turnover_df["inventory_turnover"].replace([np.inf, -np.inf], np.nan).fillna(0)
                        turnover_df["dsi"] = np.where(turnover_df["inventory_turnover"] > 0, 365 / turnover_df["inventory_turnover"], np.nan)
                        st.markdown("<div class='section-h'>🔁 Inventory Turnover (per product)</div>", unsafe_allow_html=True)
                        st.dataframe(
                            turnover_df[
                                [product_col, "cogs", "avg_inventory", "inventory_turnover", "dsi"]
                            ]
                            .sort_values("inventory_turnover", ascending=False)
                            .head(20)
                        )
                    else:
                        st.info("Inventory turnover: inventory column not found in inventory_df.")
                except Exception as e:
                    st.warning(f"Could not compute inventory turnover: {e}")
            else:
                st.info("Inventory turnover: price column for COGS not found in sales or products.")
        else:
            st.info("Inventory turnover: units_sold column not found in sales_df.")
    else:
        st.info("Inventory turnover: required data (sales_df, inventory_df, product_col) not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Category-wise sales/inventory --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    if (
        sales is not None
        and category_col_sales
        and category_col_sales in sales.columns
        and sales_col
        and sales_col in sales.columns
    ):
        try:
            cat_sales = sales.groupby(category_col_sales)[sales_col].sum().reset_index()
            st.markdown("<div class='section-h'>📊 Sales Distribution by Category</div>", unsafe_allow_html=True)
            fig_cat = px.pie(cat_sales, names=category_col_sales, values=sales_col, hole=0.4, title="Sales by Category",
                             color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig_cat, use_container_width=True, key="category_sales_pie")
        except Exception as e:
            st.warning(f"Category sales chart error: {e}")

    if (
        inv is not None
        and category_col_inv
        and category_col_inv in inv.columns
        and inventory_col
        and inventory_col in inv.columns
    ):
        try:
            cat_inv = inv.groupby(category_col_inv)[inventory_col].sum().reset_index()
            st.markdown("<div class='section-h'>📦 Inventory Distribution by Category</div>", unsafe_allow_html=True)
            fig_inv_cat = px.pie(cat_inv, names=category_col_inv, values=inventory_col, hole=0.4, title="Inventory by Category",
                                 color_discrete_sequence=px.colors.sequential.Blues)
            st.plotly_chart(fig_inv_cat, use_container_width=True, key="category_inv_pie")
        except Exception as e:
            st.warning(f"Category inventory chart error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Product Growth Trends (sales vs inventory) --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    if (
        sales is not None
        and inv is not None
        and product_col
        and product_col in sales.columns
        and product_col in inv.columns
        and date_col_sales
        and date_col_inv
    ):
        try:
            sales[date_col_sales] = pd.to_datetime(sales[date_col_sales], errors="coerce")
            inv[date_col_inv] = pd.to_datetime(inv[date_col_inv], errors="coerce")
            sales_monthly = sales.dropna(subset=[date_col_sales]).copy()
            sales_monthly["month"] = sales_monthly[date_col_sales].dt.to_period("M").dt.to_timestamp()
            if sales_col and sales_col in sales_monthly.columns:
                trend_sales = (
                    sales_monthly.groupby(["month", product_col])[sales_col]
                    .sum()
                    .reset_index()
                    .rename(columns={"month": "date", product_col: product_col, sales_col: "sales_value"})
                )
            else:
                trend_sales = (
                    sales_monthly.groupby(["month", product_col]).size().reset_index(name="sales_value").rename(columns={"month": "date"})
                )

            inv_monthly = inv.dropna(subset=[date_col_inv]).copy()
            inv_monthly["month"] = inv_monthly[date_col_inv].dt.to_period("M").dt.to_timestamp()
            trend_inv = (
                inv_monthly.groupby(["month", product_col])[inventory_col]
                .sum()
                .reset_index()
                .rename(columns={"month": "date", inventory_col: "inventory_value"})
            )

            product_trend = pd.merge(trend_sales, trend_inv, on=["date", product_col], how="outer").fillna(0)

            prod_list = product_trend[product_col].unique().tolist()
            if not prod_list:
                st.info("No product trend data available.")
            else:
                st.markdown("<div class='section-h'>📈 Product Growth Trends (Sales vs Inventory)</div>", unsafe_allow_html=True)
                selected_product = st.selectbox("Select a Product (trend)", prod_list, key="product_trend_select")
                product_data = product_trend[product_trend[product_col] == selected_product].sort_values("date")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=product_data["date"], y=product_data["sales_value"], mode="lines+markers", name="Sales", line=dict(color="#60a5fa")))
                fig.add_trace(go.Scatter(x=product_data["date"], y=product_data["inventory_value"], mode="lines+markers", name="Inventory", line=dict(color="#7c3aed")))
                fig.update_layout(title=f"Sales vs Inventory Trend: {selected_product}", xaxis_title="Date", yaxis_title="Value", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True, key="product_trend_chart")
        except Exception as e:
            st.warning(f"Product trend error: {e}")
    else:
        st.info("Product growth trends: missing data (sales/inventory/date/product columns).")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Profitability & ABC & Lifecycle --------------------
    # Profitability (if sales value and cost exist)
    try:
        cost_col = column_finder(sales, ["cost", "cost_price", "cogs", "unit_cost"]) or column_finder(prod, ["cost", "cost_price", "unit_cost"])
    except Exception:
        cost_col = None

    if sales is not None and sales_col and cost_col and cost_col in sales.columns:
        try:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            sales["profit"] = sales[sales_col].astype(float) - sales[cost_col].astype(float)
            profit_df = sales.groupby(product_col)["profit"].sum().reset_index().sort_values("profit", ascending=False)
            st.markdown("<div class='section-h'>💰 Product Profitability</div>", unsafe_allow_html=True)
            fig_profit = px.bar(profit_df.head(10), x=product_col, y="profit", title="Top 10 Profitable Products", color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig_profit, use_container_width=True, key="profit_chart")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Profitability calculation error: {e}")

    # ABC Analysis
    if sales is not None and sales_col:
        try:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='section-h'>🔠 ABC Analysis (Pareto)</div>", unsafe_allow_html=True)
            abc = sales.groupby(product_col)[sales_col].sum().sort_values(ascending=False).reset_index()
            abc["cum_percent"] = abc[sales_col].cumsum() / abc[sales_col].sum()
            abc["abc_category"] = pd.cut(abc["cum_percent"], bins=[0, 0.8, 0.95, 1.0], labels=["A", "B", "C"])
            fig_abc = go.Figure()
            fig_abc.add_trace(go.Bar(x=abc[product_col], y=abc[sales_col], name="Sales", marker_color='#60a5fa'))
            fig_abc.add_trace(go.Scatter(x=abc[product_col], y=abc["cum_percent"], mode="lines+markers", name="Cumulative %", marker_color='#7c3aed'))
            st.plotly_chart(fig_abc, use_container_width=True, key="abc_chart")
            st.dataframe(abc[[product_col, sales_col, "cum_percent", "abc_category"]].head(30))
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"ABC analysis error: {e}")

    # Product Lifecycle
    if sales is not None and date_col_sales and sales_col:
        try:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='section-h'>📈 Product Lifecycle Stages</div>", unsafe_allow_html=True)
            monthly = sales.dropna(subset=[date_col_sales]).copy()
            monthly[date_col_sales] = pd.to_datetime(monthly[date_col_sales], errors="coerce")
            monthly["month"] = monthly[date_col_sales].dt.to_period("M").dt.to_timestamp()
            monthly_agg = monthly.groupby([product_col, "month"])[sales_col].sum().reset_index()

            def compute_slope(df):
                df_sorted = df.sort_values("month")
                if len(df_sorted) < 3:
                    return 0.0
                y = df_sorted[sales_col].values
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                return float(slope)

            slopes = monthly_agg.groupby(product_col).apply(compute_slope).reset_index().rename(columns={0: "slope"})
            slopes["stage"] = pd.cut(slopes["slope"], bins=[-np.inf, -0.01, 0.01, np.inf], labels=["Declining", "Mature", "Growing"])
            fig_lifecycle = px.bar(slopes.sort_values("slope", ascending=False), x=product_col, y="slope", color="stage", title="Product Lifecycle (trend slope)", color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig_lifecycle, use_container_width=True, key="lifecycle_chart")
            st.dataframe(slopes.sort_values("slope", ascending=False).head(30))
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Lifecycle calculation error: {e}")

    # -------------------- Stockout & Overstock Risk --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    if inv is not None and not inv.empty and sales is not None and product_col in sales.columns and product_col in inv.columns:
        try:
            st.markdown("<div class='section-h'>⚠️ Inventory Risk (Stockout / Overstock)</div>", unsafe_allow_html=True)
            if date_col_sales:
                sales[date_col_sales] = pd.to_datetime(sales[date_col_sales], errors="coerce")
                period_days = max((sales[date_col_sales].max() - sales[date_col_sales].min()).days, 1)
                avg_daily = (sales.groupby(product_col)[sales_col].sum() / period_days).reindex(inv[product_col]).fillna(0).values
            else:
                avg_daily = (sales.groupby(product_col)[sales_col].sum() / 30).reindex(inv[product_col]).fillna(0).values

            inv_local = inv.copy()
            inv_local["avg_daily_demand"] = avg_daily
            inv_local["days_of_supply"] = np.where(inv_local["avg_daily_demand"] > 0, inv_local[inventory_col] / inv_local["avg_daily_demand"], np.nan)
            inv_local["stock_risk"] = np.where(inv_local["days_of_supply"] < 15, "Stockout Risk", np.where(inv_local["days_of_supply"] > 90, "Overstock Risk", "Healthy"))
            fig_inv_risk = px.scatter(inv_local, x=inventory_col, y="days_of_supply", color="stock_risk", hover_data=[product_col], title="Inventory vs Days of Supply", color_discrete_map={"Stockout Risk":"#ef4444","Overstock Risk":"#f59e0b","Healthy":"#34d399"})
            st.plotly_chart(fig_inv_risk, use_container_width=True, key="inv_risk_chart")
            st.dataframe(inv_local[[product_col, inventory_col, "avg_daily_demand", "days_of_supply", "stock_risk"]].sort_values("days_of_supply").head(50))
        except Exception as e:
            st.warning(f"Inventory risk computation error: {e}")
    else:
        st.info("Inventory risk: insufficient data (inventory_df and sales_df with product column needed).")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Price Sensitivity Analysis --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    if price_col and sales is not None and product_col in sales.columns and price_col in sales.columns:
        try:
            st.markdown("<div class='section-h'>💹 Price Sensitivity (Price vs Units Sold)</div>", unsafe_allow_html=True)
            price_sens = sales.groupby(product_col).agg({price_col: "mean", sales_col: "sum"}).reset_index().rename(columns={price_col: "avg_price", sales_col: "total_sales"})
            fig_price = px.scatter(price_sens, x="avg_price", y="total_sales", text=product_col, title="Average Price vs Total Sales (per product)", color_discrete_sequence=px.colors.sequential.Blues)
            st.plotly_chart(fig_price, use_container_width=True, key="price_sens_chart")
        except Exception as e:
            st.warning(f"Price sensitivity error: {e}")
    else:
        st.info("Price sensitivity: price or sales data missing.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Customer Engagement --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    customer_col = column_finder(sales, ["customer_id", "cust_id", "buyer_id", "user_id"]) if sales is not None else None
    if sales is not None and product_col in sales.columns and customer_col and customer_col in sales.columns:
        try:
            st.markdown("<div class='section-h'>🙋 Customer Engagement</div>", unsafe_allow_html=True)
            engagement = sales.groupby(product_col)[customer_col].nunique().reset_index().rename(columns={customer_col: "unique_buyers"})
            fig_eng = px.bar(engagement.sort_values("unique_buyers", ascending=False).head(10), x=product_col, y="unique_buyers", title="Top Products by Unique Buyers", color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig_eng, use_container_width=True, key="engagement_chart")
        except Exception as e:
            st.warning(f"Customer engagement error: {e}")
    else:
        st.info("Customer engagement: missing sales/customer/product columns.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Geo Insights --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    region_col = column_finder(sales, ["region", "state", "city", "location"]) if sales is not None else None
    if sales is not None and region_col and region_col in sales.columns and product_col in sales.columns and sales_col in sales.columns:
        try:
            st.markdown("<div class='section-h'>🗺️ Regional Bestsellers</div>", unsafe_allow_html=True)
            geo_sales = sales.groupby([region_col, product_col])[sales_col].sum().reset_index()
            top_geo = geo_sales.groupby(region_col).apply(lambda x: x.sort_values(sales_col, ascending=False).head(1)).reset_index(drop=True)
            st.dataframe(top_geo)
        except Exception as e:
            st.warning(f"Geo insights error: {e}")
    else:
        st.info("Geo insights: missing region/product/sales columns.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Product Correlation / Cannibalization --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    if sales is not None and product_col in sales.columns and date_col_sales:
        try:
            st.markdown("<div class='section-h'>🔎 Product Correlation (Cannibalization Hint)</div>", unsafe_allow_html=True)
            sales[date_col_sales] = pd.to_datetime(sales[date_col_sales], errors="coerce")
            monthly_pivot = sales.groupby([product_col, sales[date_col_sales].dt.to_period("M")])[sales_col].sum().unstack(fill_value=0)
            corr_df = monthly_pivot.transpose().corr()
            product_list = corr_df.columns.tolist()[:200]
            if product_list:
                sel = st.selectbox("Select product to inspect correlations", product_list, key="cannibal_select")
                top_corr = corr_df[sel].drop(sel).dropna().sort_values(ascending=False)
                st.write("Top positive correlations (may indicate bundles / co-demand):")
                st.write(top_corr.head(10))
                st.write("Top negative correlations (possible cannibalization):")
                st.write(top_corr.tail(10))
        except Exception as e:
            st.warning(f"Cannibalization/correlation error: {e}")
    else:
        st.info("Cannibalization: missing sales/product/date columns.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Cross-selling & bundling (transactions) --------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    possible_transaction_cols = [
        "transaction_id",
        "txn_id",
        "trans_id",
        "order_id",
        "invoice_id",
        "receipt_id",
        "purchase_id",
    ]
    transaction_id = column_finder(tx, possible_transaction_cols) if tx is not None else None

    if (
        tx is not None
        and sales is not None
        and transaction_id
        and product_col
        and product_col in tx.columns
        and product_col in sales.columns
        and transaction_id in tx.columns
        and transaction_id in sales.columns
    ):
        try:
            merged = pd.merge(
                tx[[transaction_id, product_col]],
                (sales[[transaction_id, product_col, units_sold_col]] if units_sold_col and units_sold_col in sales.columns else sales[[transaction_id, product_col]]),
                on=[transaction_id, product_col],
                how="inner",
            )
            basket = merged.assign(value=1).groupby([transaction_id, product_col])["value"].sum().unstack(fill_value=0)
            basket = basket.applymap(lambda x: 1 if x > 0 else 0)

            if basket.shape[0] < 10 or basket.shape[1] < 2:
                st.info("Not enough transaction data for robust association rules (need >=10 transactions and >=2 unique products).")
            else:
                frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
                if frequent_itemsets.empty:
                    st.info("No frequent itemsets found with min_support=0.05.")
                else:
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
                    if rules.empty:
                        st.info("No association rules found (increase support/threshold).")
                    else:
                        cross_sell_rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)]
                        cross_sell_rules = cross_sell_rules.sort_values("lift", ascending=False).head(10)

                        st.markdown("<div class='section-h'>🤝 Top Cross-Selling Suggestions</div>", unsafe_allow_html=True)

                        def single(x):
                            return list(x)[0] if len(x) else ""

                        cross_df = cross_sell_rules.copy()
                        cross_df["antecedent"] = cross_df["antecedents"].apply(single)
                        cross_df["consequent"] = cross_df["consequents"].apply(single)
                        fig1 = px.bar(cross_df, x="antecedent", y="lift", text="consequent", color="confidence", title="Top Cross-Selling Opportunities", color_continuous_scale=px.colors.sequential.Blues)
                        fig1.update_traces(textposition="outside")
                        st.plotly_chart(fig1, use_container_width=True, key="cross_sell_chart")

                        st.markdown("<div class='section-h'>📦 Top Product Bundles</div>", unsafe_allow_html=True)
                        bundle_rules = rules.copy()
                        bundle_rules["bundle"] = bundle_rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s)))) + " → " + bundle_rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
                        bundle_rules = bundle_rules.sort_values("support", ascending=False).head(10)
                        fig2 = px.bar(bundle_rules, x="bundle", y="support", color="confidence", title="Top Product Bundles", color_continuous_scale=px.colors.sequential.Purples)
                        fig2.update_traces(textposition="outside")
                        st.plotly_chart(fig2, use_container_width=True, key="bundle_chart")
        except Exception as e:
            st.warning(f"Cross-sell/bundle computation error: {e}")
    else:
        st.info("Cross-selling: transactions_df or required columns not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Back button --------------------
    if st.button("⬅️ Back to Main Home"):
        st.session_state.page = "page2"
        st.rerun()
