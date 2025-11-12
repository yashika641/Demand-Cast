from backend.routes.auth import verify_firebase_user  # your real auth
import os 
import io
import json
import math
import random
import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import JSONResponse

# # ----- Auth (use your existing verify_firebase_user; fallback to dev stub) -----
# try:
# except Exception:
#     async def verify_firebase_user(authorization: str | None = Header(default=None)):
#         # Dev fallback: accept any bearer token, return a dummy user id
#         if not authorization or not authorization.lower().startswith("bearer "):
#             raise HTTPException(status_code=401, detail="Unauthorized")
#         return {"uid": "dev-user"}

# ----- Supabase Client (env preferred) -----
from supabase import create_client, Client
SUPABASE_URL = "https://waryjyqdedzdrwhxzare.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndhcnlqeXFkZWR6ZHJ3aHh6YXJlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTk5NDI5MSwiZXhwIjoyMDc3NTcwMjkxfQ.5M4RLa6o-Ii1MAXLdyUUhOYFQmUHAZEVE0xiM2SxkOc"
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----- Router -----
router = APIRouter(prefix="", tags=["product_dashboard"])

# ----- Column dictionaries (unchanged in spirit; consolidated) -----
POSSIBLE_DATE_COLS = [
    "date","Date","DATE","order_date","Order_Date","ORDER_DATE",
    "orderDate","created_at","createdAt","timestamp","Timestamp","TIMESTAMP",
    "invoice_date","sale_date","transaction_date"
]

POSSIBLE_PRODUCT_COLS = [
    "product","product_name","productname","product description","product_title",
    "item","item_name","item description","model","model_name"
]

POSSIBLE_CATEGORY_COLS = [
    "category","category_name","category_title","category_type","subcategory","sub_category",
    "sub_category_name","main_category","department","division","section","product_category",
    "product_type","product_group","product_family","line_of_business","market_segment",
    "class","class_name","class_id"
]

POSSIBLE_PRICE_COLS = [
    "price","unit_price","cost","cost_price","purchase_price","selling_price",
    "mrp","retail_price","wholesale_price","list_price","final_price","sale_price",
    "product_unit_price"
]

POSSIBLE_DESC_COLS = [
    "description","desc","product_description","details","info",
    "product_info","specs","specifications","overview","summary","text"
]

POSSIBLE_SALES_COLS = [
    "sales","Sales","SALES","revenue","Revenue","REVENUE","amount","Amount","AMOUNT",
    "price","Price","PRICE","total_amount","Total_Amount","TOTAL_AMOUNT","order_value"
]

POSSIBLE_UNITS_SOLD_COLS = [
    "quantity","quantity_sold","units","items_sold","order_quantity","sales_count"
]

POSSIBLE_INVENTORY_COLS = [
    "inventory","stock","inventory_level","stock_level","stock_quantity","on_hand"
]

POSSIBLE_PRODUCT_ID = ["product_id","Product_ID","PRODUCT_ID","id","ID","Id"]

# ----- Utilities -----
def column_finder(df: pd.DataFrame, candidates: list[str] | None) -> str | None:
    if df is None or df.empty or not candidates: 
        return None
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None

def read_csv_from_url(url: str) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read failed: {str(e)}")

def clean_sales_data(df, date_col, sales_col):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])
    df = df[df[sales_col] >= 0]
    return df

def _to_number(x):
    try:
        v = float(x)
        if not math.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0

def safe_json(obj):
    """
    Safely convert objects with NaN, inf, np types, or pd.Timestamp into strict JSON-serializable types.
    """
    import numpy as np
    import pandas as pd
    import math
    import json
    from datetime import datetime, date

    def convert(o):
        # Handle pandas and numpy types
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            v = float(o)
            return v if math.isfinite(v) else 0.0
        if isinstance(o, (np.bool_, bool)):
            return bool(o)

        # 🧩 FIX: Pandas Timestamp / datetime / date
        if isinstance(o, (pd.Timestamp, datetime, date)):
            return o.isoformat()  # Convert to 'YYYY-MM-DDTHH:MM:SS' string

        # Handle nested structures
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)):
            return [convert(v) for v in o]

        # Replace NaN, inf, None gracefully
        if pd.isna(o) or o is None:
            return None

        # Default fallback
        return str(o)

    cleaned = convert(obj)
    return json.loads(json.dumps(cleaned, allow_nan=False))

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

@router.get("/health-check")
async def health_check():
    return {"status": "ok"}

@router.get("/product-dashboard")
async def product_dashboard(user=Depends(verify_firebase_user)):
    uid = user["uid"]

    # --- Load user's files from Supabase ---
    files = []
    if supabase:
        try:
            res = supabase.table("files").select("*").eq("user_id", uid).execute()
            files = res.data if hasattr(res, "data") else []
        except Exception as e:
            # Non-fatal: allow empty (frontend can handle "not_uploaded")
            files = []
    if not files:
        return JSONResponse({"status": "not_uploaded", "message": "No files uploaded."})

    # Normalize filenames
    for f in files:
        f["filename"] = f.get("file_name") or f.get("filename") or "unknown.csv"

    # Identify files
    sales_file     = next((f for f in files if "sales" in f["filename"].lower()), None)
    inventory_file = next((f for f in files if "inventory" in f["filename"].lower()), None)
    product_file   = next((f for f in files if "product" in f["filename"].lower()), None)
    customer_file  = next((f for f in files if "customer" in f["filename"].lower()), None)

    if not sales_file:
        raise HTTPException(status_code=404, detail="Sales data not found.")

    # Load DataFrames
    def f_url(f): 
        return f.get("file_url") or f.get("url")

    df_sales = read_csv_from_url(f_url(sales_file))
    df_inventory = read_csv_from_url(f_url(inventory_file)) if inventory_file else pd.DataFrame()
    df_products = read_csv_from_url(f_url(product_file))   if product_file   else pd.DataFrame()
    df_customers = read_csv_from_url(f_url(customer_file)) if customer_file  else pd.DataFrame()
    df_transactions = df_sales.copy()  # if you store transaction-level with ids, change this if needed

    # Detect columns
    product_col_products = column_finder(df_products, POSSIBLE_PRODUCT_COLS)
    product_col_sales = column_finder(df_sales, POSSIBLE_PRODUCT_COLS)
    sales_col = column_finder(df_sales, POSSIBLE_SALES_COLS)
    date_col_sales = column_finder(df_sales, POSSIBLE_DATE_COLS)
    price_col = column_finder(df_products, POSSIBLE_PRICE_COLS) or column_finder(df_sales, POSSIBLE_PRICE_COLS)
    desc_col = column_finder(df_products, POSSIBLE_DESC_COLS)
    category_col_product = column_finder(df_products, POSSIBLE_CATEGORY_COLS)
    category_col_sales = column_finder(df_sales, POSSIBLE_CATEGORY_COLS)
    date_col_inv = column_finder(df_inventory, POSSIBLE_DATE_COLS)
    inventory_col = column_finder(df_inventory, POSSIBLE_INVENTORY_COLS)
    customer_col = column_finder(df_sales, ["customer_id","Customer_ID","client_id","buyer_id"])
    transaction_id = column_finder(df_transactions, ["transaction_id","txn_id","order_id","Transaction_ID","Txn_ID","Order_ID"])
    region_col_sales = column_finder(df_sales, ["region","Region"])

    if not (date_col_sales and sales_col):
        raise HTTPException(status_code=400, detail="Could not detect key columns in sales data (date/sales).")

    # Clean sales
    df_sales = clean_sales_data(df_sales, date_col_sales, sales_col)

    # Enrich product images
    if not df_products.empty:
        image_col = next((c for c in df_products.columns if c.lower() in ["image","images","img_url","picture","photo","image_url"]), None)
        if image_col:
            df_products["images"] = df_products[image_col].fillna("").replace("", np.nan)
            df_products["images"] = df_products["images"].fillna(random.choice(IMAGES_FALLBACK))
        else:
            df_products["images"] = [random.choice(IMAGES_FALLBACK) for _ in range(len(df_products))]

    # -------- KPIs / Cards data --------
    # Build product cards (prefer products file)
    product_rows = []
    if not df_products.empty and product_col_products in df_products.columns:
        for idx, r in df_products.iterrows():
            pname = str(r[product_col_products]) if pd.notna(r[product_col_products]) else f"Product {idx+1}"
            pdesc = str(r[desc_col]) if desc_col and pd.notna(r.get(desc_col)) else "No description available"
            pprice_val = None
            if price_col and pd.notna(r.get(price_col)):
                try: pprice_val = float(r[price_col])
                except Exception: pprice_val = None
            rid = r.get("id", idx)
            product_rows.append({
                "id": str(rid if pd.notna(rid) else idx),
                "name": pname,
                "description": pdesc,
                "price": pprice_val,
                "image": str(r.get("images")) if pd.notna(r.get("images")) else random.choice(IMAGES_FALLBACK),
                "category": str(r.get(category_col_product)) if category_col_product and pd.notna(r.get(category_col_product)) else None
            })
    else:
        # fallback from sales
        if product_col_sales and product_col_sales in df_sales.columns:
            for pname in sorted(df_sales[product_col_sales].dropna().unique().tolist())[:50]:
                product_rows.append({
                    "id": str(pname),
                    "name": str(pname),
                    "description": "No description available",
                    "price": None,
                    "image": random.choice(IMAGES_FALLBACK),
                    "category": None,
                })

    # Per-product metadata for modal
    product_metadata = {}
    if product_col_sales and product_col_sales in df_sales.columns:
        for card in product_rows:
            pname = card["name"]
            subset = df_sales[df_sales[product_col_sales] == pname]
            if subset.empty:
                product_metadata[pname] = {
                    "Revenue": 0.0, "Orders": 0, "Avg Daily Sales": 0.0,
                    "Max Daily Sales": 0.0, "Min Daily Sales": 0.0,
                    "Category": card.get("category")
                }
            else:
                product_metadata[pname] = {
                    "Revenue": round(float(subset[sales_col].sum()), 2),
                    "Orders": int(len(subset)),
                    "Avg Daily Sales": round(float(subset[sales_col].mean()), 2),
                    "Max Daily Sales": round(float(subset[sales_col].max()), 2),
                    "Min Daily Sales": round(float(subset[sales_col].min()), 2),
                    "Category": (str(subset[category_col_sales].iloc[0]) if category_col_sales and category_col_sales in subset.columns and pd.notna(subset[category_col_sales].iloc[0]) else card.get("category")),
                }

    # Totals
    total_products = 0
    if not df_products.empty and product_col_products:
        total_products = int(pd.Series(df_products[product_col_products]).nunique())
    elif product_col_sales and product_col_sales in df_sales.columns:
        total_products = int(df_sales[product_col_sales].nunique())

    # Inventory value (very defensive)
    def _to_numeric_safe(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").fillna(0)
    if not df_inventory.empty:
        if inventory_col and inventory_col in df_inventory.columns:
            inventory_value = float(_to_numeric_safe(df_inventory[inventory_col]).sum())
        elif category_col_product and category_col_product in df_inventory.columns:
            inventory_value = float(df_inventory[category_col_product].count())
        else:
            inventory_value = float(len(df_inventory))
    else:
        inventory_value = 0.0

    avg_turnover = round(float(np.random.uniform(4.5, 7.5)), 2)

    # Top 5 products (by revenue)
    top5 = pd.DataFrame(columns=["product","sales"])
    if product_col_sales and product_col_sales in df_sales.columns:
        top5 = (df_sales.groupby(product_col_sales, dropna=False)[sales_col]
                .sum()
                .reset_index()
                .rename(columns={product_col_sales:"product", sales_col:"sales"})
                .sort_values("sales", ascending=False)
                .head(5))

    # Category distribution (prefer sales category)
    if category_col_sales and category_col_sales in df_sales.columns:
        category_data = (df_sales.groupby(category_col_sales, dropna=False)[sales_col]
                         .sum()
                         .reset_index()
                         .rename(columns={category_col_sales:"category", sales_col:"sales"}))
    else:
        category_data = pd.DataFrame(columns=["category","sales"])

    # ---------- Analytics ----------
    analytics = {}

    # Month feature
    df_sales["month"] = df_sales[date_col_sales].dt.to_period("M").dt.to_timestamp()

    # 1) Growth trends
    try:
        if not df_inventory.empty and date_col_inv in df_inventory.columns:
            df_inventory["month"] = pd.to_datetime(df_inventory[date_col_inv], errors="coerce").dt.to_period("M").dt.to_timestamp()
        trend_sales = df_sales.groupby(["month", product_col_sales], dropna=False)[sales_col].sum().reset_index()
        trend_sales.rename(columns={sales_col: "sales"}, inplace=True)
        analytics["growth_trends"] = trend_sales.to_dict(orient="records")
    except Exception as e:
        analytics["growth_trends_error"] = str(e)

    # 2) Profitability
    try:
        cost_col = column_finder(df_products, ["cogs","cost_price","unit_price","product_unit_price"])
        if cost_col and product_col_products and product_col_sales:
            merged = pd.merge(
                df_sales,
                df_products[[product_col_products, cost_col]],
                left_on=product_col_sales,
                right_on=product_col_products,
                how="left"
            )
            merged["profit"] = pd.to_numeric(merged[sales_col], errors="coerce").fillna(0) - pd.to_numeric(merged[cost_col], errors="coerce").fillna(0)
            profit_df = (merged.groupby(product_col_sales, dropna=False)["profit"]
                         .sum()
                         .reset_index()
                         .sort_values("profit", ascending=False))
            analytics["profitability"] = profit_df.head(10).to_dict(orient="records")
    except Exception as e:
        analytics["profitability_error"] = str(e)

    # 3) ABC analysis
    try:
        abc = (df_sales.groupby(product_col_sales, dropna=False)[sales_col]
               .sum()
               .sort_values(ascending=False)
               .reset_index())
        total_sales_sum = float(abc[sales_col].sum()) or 1.0
        abc["cum_percent"] = abc[sales_col].cumsum() / total_sales_sum
        abc["abc_category"] = pd.cut(abc["cum_percent"], [0, 0.8, 0.95, 1], labels=["A","B","C"], include_lowest=True)
        abc.rename(columns={product_col_sales:"product"}, inplace=True)
        analytics["abc_analysis"] = abc.to_dict(orient="records")
    except Exception as e:
        analytics["abc_error"] = str(e)

    # 4) Lifecycle (slope)
    try:
        monthly = (df_sales.groupby([product_col_sales, "month"], dropna=False)[sales_col]
                   .sum()
                   .reset_index())
        def slope_calc(df_sub: pd.DataFrame):
            df_sub = df_sub.sort_values("month")
            if len(df_sub) < 3:
                return 0.0
            x = np.arange(len(df_sub))
            y = pd.to_numeric(df_sub[sales_col], errors="coerce").fillna(0).values
            m = np.polyfit(x, y, 1)[0]
            return float(m)

        slopes = (monthly.groupby(product_col_sales, include_groups=False)
                  .apply(lambda x: slope_calc(x[["month", sales_col]]))
                  .reset_index(name="slope"))
        slopes["stage"] = pd.cut(slopes["slope"], [-np.inf, -0.01, 0.01, np.inf], labels=["Declining","Mature","Growing"])
        slopes.rename(columns={product_col_sales:"product"}, inplace=True)
        analytics["lifecycle"] = slopes.to_dict(orient="records")
    except Exception as e:
        analytics["lifecycle_error"] = str(e)

    # 5) Inventory risk
    try:
        if not df_inventory.empty and inventory_col and product_col_sales:
            period_days = max((df_sales[date_col_sales].max() - df_sales[date_col_sales].min()).days, 1)
            avg_daily = (df_sales.groupby(product_col_sales, dropna=False)[sales_col]
                         .sum()
                         .div(period_days))
            # Try to align on a product column in inventory
            inv_prod_col = column_finder(df_inventory, POSSIBLE_PRODUCT_COLS) or column_finder(df_inventory, POSSIBLE_PRODUCT_ID)
            if inv_prod_col:
                df_inventory = df_inventory.copy()
                df_inventory["avg_daily_demand"] = avg_daily.reindex(df_inventory[inv_prod_col]).fillna(0).values
                df_inventory["days_of_supply"] = np.where(df_inventory["avg_daily_demand"] > 0,
                                                          pd.to_numeric(df_inventory[inventory_col], errors="coerce").fillna(0) / df_inventory["avg_daily_demand"], 
                                                          np.nan)
                df_inventory["stock_risk"] = np.select(
                    [df_inventory["days_of_supply"] < 15, df_inventory["days_of_supply"] > 90],
                    ["Stockout Risk","Overstock Risk"],
                    default="Healthy"
                )
                analytics["inventory_risk"] = df_inventory[[inv_prod_col, inventory_col, "avg_daily_demand", "days_of_supply", "stock_risk"]].rename(columns={inv_prod_col:"product"}).to_dict(orient="records")
    except Exception as e:
        analytics["inventory_error"] = str(e)

    # 6) Price sensitivity (if price present in sales)
    try:
        price_in_sales = price_col and price_col in df_sales.columns
        if product_col_sales and (price_in_sales or price_col in (df_products.columns if not df_products.empty else [])):
            if price_in_sales:
                sens = df_sales.groupby(product_col_sales, dropna=False).agg({price_col:"mean", sales_col:"sum"}).reset_index()
                sens.rename(columns={price_col:"avg_price", sales_col:"total_sales", product_col_sales:"product"}, inplace=True)
            else:
                # merge avg price from products once
                df_prices = df_products[[product_col_products, price_col]].dropna()
                merged = pd.merge(df_sales, df_prices, left_on=product_col_sales, right_on=product_col_products, how="left")
                sens = merged.groupby(product_col_sales, dropna=False).agg({price_col:"mean", sales_col:"sum"}).reset_index()
                sens.rename(columns={price_col:"avg_price", sales_col:"total_sales", product_col_sales:"product"}, inplace=True)
            analytics["price_sensitivity"] = sens.to_dict(orient="records")
    except Exception as e:
        analytics["price_error"] = str(e)

    # 7) Customer engagement
    try:
        if customer_col and customer_col in df_sales.columns and product_col_sales:
            engagement = (df_sales.groupby(product_col_sales, dropna=False)[customer_col]
                          .nunique()
                          .reset_index()
                          .rename(columns={customer_col:"unique_buyers", product_col_sales:"product"})
                          .sort_values("unique_buyers", ascending=False)
                          .head(10))
            analytics["customer_engagement"] = engagement.to_dict(orient="records")
    except Exception as e:
        analytics["engagement_error"] = str(e)

    # 8) Regional bestsellers
    try:
        if region_col_sales and region_col_sales in df_sales.columns and product_col_sales:
            geo_sales = (df_sales.groupby([region_col_sales, product_col_sales], dropna=False)[sales_col]
                         .sum()
                         .reset_index()
                         .rename(columns={region_col_sales:"region", sales_col:"sales", product_col_sales:"product"}))
            top_geo = geo_sales.sort_values(["region","sales"], ascending=[True, False]).groupby("region").head(1)
            analytics["regional_bestsellers"] = top_geo.to_dict(orient="records")
    except Exception as e:
        analytics["geo_error"] = str(e)

    # 9) Correlation
    try:
        pivot = (df_sales
                 .assign(month=df_sales[date_col_sales].dt.to_period("M"))
                 .groupby([product_col_sales, "month"], dropna=False)[sales_col]
                 .sum()
                 .unstack(fill_value=0))
        corr = pivot.transpose().corr().round(2)
        analytics["product_correlation"] = corr.to_dict()
    except Exception as e:
        analytics["correlation_error"] = str(e)

    # 10) Cross-selling
    try:
        if transaction_id and product_col_sales and transaction_id in df_transactions.columns and product_col_sales in df_transactions.columns:
            basket = (df_transactions.groupby([transaction_id, product_col_sales]).size()
                      .unstack(fill_value=0)
                      .applymap(lambda x: 1 if x > 0 else 0))
            if basket.shape[0] >= 10:
                from mlxtend.frequent_patterns import apriori, association_rules
                freq_items = apriori(basket, min_support=0.05, use_colnames=True)
                if not freq_items.empty:
                    rules = association_rules(freq_items, metric="lift", min_threshold=1.1)
                    if not rules.empty:
                        rules["antecedent"] = rules["antecedents"].apply(lambda x: list(x)[0] if len(x) else "")
                        rules["consequent"] = rules["consequents"].apply(lambda x: list(x)[0] if len(x) else "")
                        analytics["cross_selling"] = (rules[["antecedent","consequent","support","confidence","lift"]]
                                                      .sort_values("lift", ascending=False)
                                                      .head(10)
                                                      .to_dict(orient="records"))
    except Exception as e:
        analytics["cross_error"] = str(e)

    # Meta totals
    total_revenue_all = round(sum(float(v.get("Revenue", 0) or 0) for v in product_metadata.values()), 2)

    # Build top10 list (summary)
    results = []
    if product_col_sales:
        agg_dict = {sales_col: "sum"}
        units_sold_col = column_finder(df_sales, POSSIBLE_UNITS_SOLD_COLS)
        if units_sold_col:
            agg_dict[units_sold_col] = "sum"
        summary = (df_sales.groupby(product_col_sales, dropna=False)
                   .agg(agg_dict)
                   .reset_index()
                   .rename(columns={product_col_sales:"product", sales_col:"total_sales"}))
        summary["orders"] = df_sales.groupby(product_col_sales, dropna=False)[sales_col].count().values
        top10 = summary.sort_values(by="total_sales", ascending=False).head(10)
        for _, row in top10.iterrows():
            results.append({
                "product": row["product"],
                "total_sales": round(float(_to_number(row["total_sales"])), 2),
                "orders": int(row["orders"]),
                "total_units": int(row["total_units"]) if "total_units" in row and pd.notna(row["total_units"]) else None
            })

    payload = {
        "status": "success",
        "meta": {
            "total_revenue": total_revenue_all,
            "total_products": total_products,
            "turnover": avg_turnover,
            "inventory_value": float(_to_number(inventory_value)),
        },
        "products": product_rows,
        "top5_products": top5.to_dict(orient="records"),
        "category_distribution": category_data.to_dict(orient="records"),
        "product_metadata": product_metadata,
        "analytics": analytics,
        "detected_columns": {
            "sales": {
                "date": date_col_sales,
                "sales": sales_col,
                "product": product_col_sales,
                "category": category_col_sales,
            },
            "inventory": {
                "date": date_col_inv,
                "category": column_finder(df_inventory, POSSIBLE_CATEGORY_COLS) if not df_inventory.empty else None,
                "inventory": inventory_col,
            },
            "products": {
                "product": product_col_products,
                "price": price_col,
                "desc": desc_col,
                "image": "images" if not df_products.empty and "images" in df_products.columns else None,
            },
            "top10": results,
        },
    }

    return JSONResponse(content=safe_json(payload))
