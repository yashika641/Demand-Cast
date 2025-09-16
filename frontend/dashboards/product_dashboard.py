import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
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

# ---------- Robust product_dashboard ----------
def product_dashboard(
    products_df=None, sales_df=None, inventory_df=None, transactions_df=None
):
    """
    products_df, sales_df, inventory_df, transactions_df are optional.
    The function tries to auto-detect useful columns and produce advanced insights.
    """
    if products_df is None:
        st.warning("No products_df provided.")
        return

    # shallow copies to avoid mutating user data
    prod = products_df.copy()
    sales = sales_df.copy() if sales_df is not None else None
    inv = inventory_df.copy() if inventory_df is not None else None
    tx = transactions_df.copy() if transactions_df is not None else None

    st.title("🛍️ Product Dashboard")
    st.write("Overview of products, sales, inventory, trends and cross-sell suggestions.")

    # --- find columns ---
    product_col = column_finder(prod, POSSIBLE_PRODUCT_COLS)
    price_col = column_finder(prod, POSSIBLE_PRICE_COLS) or column_finder(sales, POSSIBLE_PRICE_COLS)
    desc_col = column_finder(prod, POSSIBLE_DESC_COLS)
    sales_col = column_finder(sales, POSSIBLE_SALES_COLS) if sales is not None else None
    units_sold_col = column_finder(sales, POSSIBLE_UNITS_SOLD_COLS) if sales is not None else None
    category_col_sales = column_finder(sales, POSSIBLE_CATEGORY_COLS) if sales is not None else None
    category_col_inv = column_finder(inv, POSSIBLE_CATEGORY_COLS) if inv is not None else None
    date_col_sales = column_finder(sales, POSSIBLE_DATE_COLS) if sales is not None else None
    date_col_inv = column_finder(inv, POSSIBLE_DATE_COLS) if inv is not None else None
    inventory_col = column_finder(inv, POSSIBLE_INVENTORY_COLS) if inv is not None else None

    # --- images handling ---
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

    # --- product carousel (simple safe HTML) ---
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
            <img src="{pimg}" alt="{pname}" class="card-img">
            <h3>{pname}</h3>
            <p style="height:48px; overflow:hidden;">{pdesc}</p>
            <h4 style="color:green;">₹ {pprice}</h4>
          </div>
        </div>
        """

    html_code = f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>
    <style>
      .swiper {{ width: 100%; padding: 20px 0; box-sizing:border-box; }}
      .swiper-slide {{ display:flex; justify-content:center; align-items:center; }}
      .card {{ width:220px; height:340px; background:white; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.12); text-align:center; padding:12px; }}
      .card-img {{ width:80px; height:80px; object-fit:contain; margin:6px auto; display:block; }}
      .card h3 {{ margin:8px 0 4px 0; font-size:14px; color:#222; }}
      .card p {{ color:#555; font-size:12px; margin-bottom:8px; }}
    </style>
    <div class="swiper">
      <div class="swiper-wrapper">{slides_html}</div>
      <div class="swiper-pagination"></div>
      <div class="swiper-button-next"></div>
      <div class="swiper-button-prev"></div>
    </div>
    <script>
      var swiper = new Swiper('.swiper', {{
        slidesPerView: 3,
        spaceBetween: 20,
        loop: true,
        centeredSlides: true,
        pagination: {{ el: '.swiper-pagination', clickable: true }},
        navigation: {{ nextEl: '.swiper-button-next', prevEl: '.swiper-button-prev' }}
      }});
    </script>
    """
    try:
        st.components.v1.html(html_code, height=480, scrolling=False)
    except Exception:
        # fallback: show a simple table if HTML fails
        st.info("Product carousel couldn't be displayed — showing product table instead.")
        st.dataframe(prod[[product_col, price_col]].head(20) if product_col and price_col else prod.head(20))

    # ---------------- Top products by sales ----------------
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
            st.subheader(f"Top 10 Products by {sales_col}")
            fig = px.bar(top_sales, x=product_col, y=sales_col, title=f"Top 10 Products by {sales_col}")
            st.plotly_chart(fig, use_container_width=True, key="top_sales_chart")
        except Exception as e:
            st.warning(f"Could not compute top products by sales: {e}")
    else:
        st.info("Top products by sales: missing sales_df, sales column, or product column.")

    # ---------------- Inventory turnover ----------------
    turnover_df = None
    if (
        sales is not None
        and inv is not None
        and product_col
        and product_col in sales.columns
        and product_col in inv.columns
    ):
        # Need units_sold and a price column to compute COGS if price provided in sales or product
        if units_sold_col and units_sold_col in sales.columns:
            price_for_cogs = column_finder(sales, POSSIBLE_PRICE_COLS) or price_col
            if price_for_cogs and price_for_cogs in sales.columns:
                try:
                    sales["cogs"] = sales[units_sold_col].astype(float) * sales[price_for_cogs].astype(float)
                    cogs_per_product = sales.groupby(product_col)["cogs"].sum().reset_index()
                    # average inventory per product
                    if inventory_col and inventory_col in inv.columns:
                        avg_inventory = (
                            inv.groupby(product_col)[inventory_col]
                            .mean()
                            .reset_index()
                            .rename(columns={inventory_col: "avg_inventory"})
                        )
                        turnover_df = pd.merge(cogs_per_product, avg_inventory, on=product_col, how="inner")
                        # Avoid division by zero / NaN
                        turnover_df["avg_inventory"] = turnover_df["avg_inventory"].replace(0, np.nan)
                        turnover_df["inventory_turnover"] = turnover_df["cogs"] / turnover_df["avg_inventory"]
                        turnover_df["inventory_turnover"] = turnover_df["inventory_turnover"].replace([np.inf, -np.inf], np.nan).fillna(0)
                        turnover_df["dsi"] = np.where(turnover_df["inventory_turnover"] > 0, 365 / turnover_df["inventory_turnover"], np.nan)
                        st.subheader("Inventory Turnover (per product)")
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

    # ------------------ ADDITIONAL ADVANCED INSIGHTS ------------------

    # ---------------- Category-wise sales/inventory ----------------
    if (
        sales is not None
        and category_col_sales
        and category_col_sales in sales.columns
        and sales_col
        and sales_col in sales.columns
    ):
        try:
            cat_sales = sales.groupby(category_col_sales)[sales_col].sum().reset_index()
            st.subheader("Sales Distribution by Category")
            fig_cat = px.pie(cat_sales, names=category_col_sales, values=sales_col, hole=0.4, title="Sales by Category")
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
            st.subheader("Inventory Distribution by Category")
            fig_inv_cat = px.pie(cat_inv, names=category_col_inv, values=inventory_col, hole=0.4, title="Inventory by Category")
            st.plotly_chart(fig_inv_cat, use_container_width=True, key="category_inv_pie")
        except Exception as e:
            st.warning(f"Category inventory chart error: {e}")

    # ---------------- Product Growth Trends (sales vs inventory) ----------------
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

            # roll up by month for stability
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
                selected_product = st.selectbox("Select a Product (trend)", prod_list, key="product_trend_select")
                product_data = product_trend[product_trend[product_col] == selected_product].sort_values("date")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=product_data["date"], y=product_data["sales_value"], mode="lines+markers", name="Sales"))
                fig.add_trace(go.Scatter(x=product_data["date"], y=product_data["inventory_value"], mode="lines+markers", name="Inventory"))
                fig.update_layout(title=f"Sales vs Inventory Trend: {selected_product}", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True, key="product_trend_chart")
        except Exception as e:
            st.warning(f"Product trend error: {e}")
    else:
        st.info("Product growth trends: missing data (sales/inventory/date/product columns).")

    # ---------------- Product Profitability & ABC & Lifecycle ----------------
    # Profitability (if sales value and cost exist)
    try:
        # Try to derive cost column if present in sales or products
        cost_col = column_finder(sales, ["cost", "cost_price", "cogs", "unit_cost"]) or column_finder(prod, ["cost", "cost_price", "unit_cost"])
    except Exception:
        cost_col = None

    # Profitability
    if sales is not None and sales_col and cost_col and cost_col in sales.columns:
        try:
            sales["profit"] = sales[sales_col].astype(float) - sales[cost_col].astype(float)
            profit_df = sales.groupby(product_col)["profit"].sum().reset_index().sort_values("profit", ascending=False)
            st.subheader("💰 Product Profitability")
            fig_profit = px.bar(profit_df.head(10), x=product_col, y="profit", title="Top 10 Profitable Products")
            st.plotly_chart(fig_profit, use_container_width=True, key="profit_chart")
        except Exception as e:
            st.warning(f"Profitability calculation error: {e}")

    # ABC Analysis (Pareto)
    if sales is not None and sales_col:
        try:
            st.subheader("🔠 ABC Analysis (Pareto)")
            abc = sales.groupby(product_col)[sales_col].sum().sort_values(ascending=False).reset_index()
            abc["cum_percent"] = abc[sales_col].cumsum() / abc[sales_col].sum()
            abc["abc_category"] = pd.cut(abc["cum_percent"], bins=[0, 0.8, 0.95, 1.0], labels=["A", "B", "C"])
            fig_abc = go.Figure()
            fig_abc.add_trace(go.Bar(x=abc[product_col], y=abc[sales_col], name="Sales"))
            fig_abc.add_trace(go.Scatter(x=abc[product_col], y=abc["cum_percent"], mode="lines+markers", name="Cumulative %"))
            st.plotly_chart(fig_abc, use_container_width=True, key="abc_chart")
            st.dataframe(abc[[product_col, sales_col, "cum_percent", "abc_category"]].head(30))
        except Exception as e:
            st.warning(f"ABC analysis error: {e}")

    # Product Lifecycle (basic slope-based classification)
    if sales is not None and date_col_sales and sales_col:
        try:
            st.subheader("📈 Product Lifecycle Stages")
            monthly = sales.dropna(subset=[date_col_sales]).copy()
            monthly[date_col_sales] = pd.to_datetime(monthly[date_col_sales], errors="coerce")
            monthly["month"] = monthly[date_col_sales].dt.to_period("M").dt.to_timestamp()
            monthly_agg = monthly.groupby([product_col, "month"])[sales_col].sum().reset_index()

            # compute slope per product using linear fit if enough points
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
            fig_lifecycle = px.bar(slopes.sort_values("slope", ascending=False), x=product_col, y="slope", color="stage", title="Product Lifecycle (slope of monthly sales)")
            st.plotly_chart(fig_lifecycle, use_container_width=True, key="lifecycle_chart")
            st.dataframe(slopes.sort_values("slope", ascending=False).head(30))
        except Exception as e:
            st.warning(f"Lifecycle calculation error: {e}")

    # ---------------- Stockout & Overstock Risk ----------------
    if inv is not None and not inv.empty and sales is not None and product_col in sales.columns and product_col in inv.columns:
        try:
            st.subheader("⚠️ Inventory Risk (Stockout / Overstock)")
            # average daily demand per product (approx)
            if date_col_sales:
                sales[date_col_sales] = pd.to_datetime(sales[date_col_sales], errors="coerce")
                period_days = max((sales[date_col_sales].max() - sales[date_col_sales].min()).days, 1)
                avg_daily = (sales.groupby(product_col)[sales_col].sum() / period_days).reindex(inv[product_col]).fillna(0).values
            else:
                # fallback: use monthly average scaled to daily
                avg_daily = (sales.groupby(product_col)[sales_col].sum() / 30).reindex(inv[product_col]).fillna(0).values

            inv = inv.copy()
            inv["avg_daily_demand"] = avg_daily
            # days of supply (guard divide by zero)
            inv["days_of_supply"] = np.where(inv["avg_daily_demand"] > 0, inv[inventory_col] / inv["avg_daily_demand"], np.nan)
            inv["stock_risk"] = np.where(inv["days_of_supply"] < 15, "Stockout Risk", np.where(inv["days_of_supply"] > 90, "Overstock Risk", "Healthy"))
            fig_inv_risk = px.scatter(inv, x=inventory_col, y="days_of_supply", color="stock_risk", hover_data=[product_col], title="Inventory vs Days of Supply")
            st.plotly_chart(fig_inv_risk, use_container_width=True, key="inv_risk_chart")
            st.dataframe(inv[[product_col, inventory_col, "avg_daily_demand", "days_of_supply", "stock_risk"]].sort_values("days_of_supply").head(50))
        except Exception as e:
            st.warning(f"Inventory risk computation error: {e}")
    else:
        st.info("Inventory risk: insufficient data (inventory_df and sales_df with product column needed).")

    # ---------------- Price Sensitivity Analysis ----------------
    if price_col and sales is not None and product_col in sales.columns and price_col in sales.columns:
        try:
            st.subheader("💹 Price Sensitivity (Price vs Units Sold)")
            price_sens = sales.groupby(product_col).agg({price_col: "mean", sales_col: "sum"}).reset_index().rename(columns={price_col: "avg_price", sales_col: "total_sales"})
            fig_price = px.scatter(price_sens, x="avg_price", y="total_sales", text=product_col, title="Average Price vs Total Sales (per product)")
            st.plotly_chart(fig_price, use_container_width=True, key="price_sens_chart")
        except Exception as e:
            st.warning(f"Price sensitivity error: {e}")
    else:
        st.info("Price sensitivity: price or sales data missing.")

    # ---------------- Customer Engagement with Products ----------------
    customer_col = column_finder(sales, ["customer_id", "cust_id", "buyer_id", "user_id"]) if sales is not None else None
    if sales is not None and product_col in sales.columns and customer_col and customer_col in sales.columns:
        try:
            st.subheader("🙋 Customer Engagement")
            engagement = sales.groupby(product_col)[customer_col].nunique().reset_index().rename(columns={customer_col: "unique_buyers"})
            fig_eng = px.bar(engagement.sort_values("unique_buyers", ascending=False).head(10), x=product_col, y="unique_buyers", title="Top Products by Unique Buyers")
            st.plotly_chart(fig_eng, use_container_width=True, key="engagement_chart")
        except Exception as e:
            st.warning(f"Customer engagement error: {e}")
    else:
        st.info("Customer engagement: missing sales/customer/product columns.")

    # ---------------- Geo Insights (regional bestsellers) ----------------
    region_col = column_finder(sales, ["region", "state", "city", "location"]) if sales is not None else None
    if sales is not None and region_col and region_col in sales.columns and product_col in sales.columns and sales_col in sales.columns:
        try:
            st.subheader("🗺️ Regional Bestsellers")
            geo_sales = sales.groupby([region_col, product_col])[sales_col].sum().reset_index()
            top_geo = geo_sales.groupby(region_col).apply(lambda x: x.sort_values(sales_col, ascending=False).head(1)).reset_index(drop=True)
            st.dataframe(top_geo)
        except Exception as e:
            st.warning(f"Geo insights error: {e}")

    # ---------------- Product Correlation / Cannibalization Hint ----------------
    if sales is not None and product_col in sales.columns and date_col_sales:
        try:
            st.subheader("🔎 Product Correlation (Cannibalization Hint)")
            # pivot by month to get product x month sales matrix
            sales[date_col_sales] = pd.to_datetime(sales[date_col_sales], errors="coerce")
            monthly_pivot = sales.groupby([product_col, sales[date_col_sales].dt.to_period("M")])[sales_col].sum().unstack(fill_value=0)
            corr_df = monthly_pivot.transpose().corr()
            # show top positive/negative correlations for selected product
            product_list = corr_df.columns.tolist()[:200]  # guard size
            if product_list:
                sel = st.selectbox("Select product to inspect correlations", product_list, key="cannibal_select")
                top_corr = corr_df[sel].drop(sel).dropna().sort_values(ascending=False)
                st.write("Top positive correlations (may indicate bundles / co-demand):")
                st.write(top_corr.head(10))
                st.write("Top negative correlations (possible cannibalization):")
                st.write(top_corr.tail(10))
        except Exception as e:
            st.warning(f"Cannibalization/correlation error: {e}")

    # ---------------- Cross-selling & bundling (transactions) ----------------
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
            # Merge tx & sales to ensure we have product per transaction and units
            merged = pd.merge(
                tx[[transaction_id, product_col]],
                (sales[[transaction_id, product_col, units_sold_col]] if units_sold_col and units_sold_col in sales.columns else sales[[transaction_id, product_col]]),
                on=[transaction_id, product_col],
                how="inner",
            )
            # create basket: one-hot table (transaction x product)
            basket = merged.assign(value=1).groupby([transaction_id, product_col])["value"].sum().unstack(fill_value=0)
            basket = basket.applymap(lambda x: 1 if x > 0 else 0)

            # require reasonable size for apriori
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

                        st.subheader("Top Cross-Selling Suggestions")

                        # prepare display dataframe
                        def single(x):
                            return list(x)[0] if len(x) else ""

                        cross_df = cross_sell_rules.copy()
                        cross_df["antecedent"] = cross_df["antecedents"].apply(single)
                        cross_df["consequent"] = cross_df["consequents"].apply(single)
                        fig1 = px.bar(cross_df, x="antecedent", y="lift", text="consequent", color="confidence", title="Top Cross-Selling Opportunities")
                        fig1.update_traces(textposition="outside")
                        st.plotly_chart(fig1, use_container_width=True, key="cross_sell_chart")

                        st.subheader("Top Product Bundles")
                        bundle_rules = rules.copy()
                        bundle_rules["bundle"] = bundle_rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s)))) + " → " + bundle_rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
                        bundle_rules = bundle_rules.sort_values("support", ascending=False).head(10)
                        fig2 = px.bar(bundle_rules, x="bundle", y="support", color="confidence", title="Top Product Bundles")
                        fig2.update_traces(textposition="outside")
                        st.plotly_chart(fig2, use_container_width=True, key="bundle_chart")
        except Exception as e:
            st.warning(f"Cross-sell/bundle computation error: {e}")
    else:
        st.info("Cross-selling: transactions_df or required columns not available.")

    # ---------------- Back button ----------------
    if st.button("⬅️ Back to Main Home"):
        st.session_state.page = "page2"
        st.rerun()
