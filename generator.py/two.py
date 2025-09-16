from tqdm import tqdm
"""
Realistic Global E-Commerce Synthetic Data Generator
Outputs:
 - product_catalogue.csv
 - customers.csv
 - transactions.csv
 - sales_daily.csv (aggregated per day per product-region)
 - inventory.csv

Dependencies: pandas, numpy, faker
pip install pandas numpy faker
"""

import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import timedelta, datetime
import os

fake = Faker()
FALLBACK_SEED = 1234
random.seed(FALLBACK_SEED)
np.random.seed(FALLBACK_SEED)
Faker.seed(FALLBACK_SEED)

# ---------------------- PARAMETERS ---------------------- #
NUM_PRODUCTS = 1200        # number of distinct SKUs
NUM_CUSTOMERS = 5000       # number of customers
NUM_TRANSACTIONS = 60000   # number of line-items generated across transactions (approx)
START_DATE = pd.to_datetime("2024-01-01")
END_DATE = pd.to_datetime("2024-12-31")
OUTPUT_DIR = "synthetic_data_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- CONFIG: categories & brands ---------------------- #
CATEGORIES = {
    "Electronics": ["Sony","Samsung","Apple","Bose","LG","Panasonic","Philips","Dell","HP","Lenovo"],
    "Fashion": ["Nike","Adidas","Zara","H&M","Uniqlo","Levi's","Gucci","Prada","Under Armour","Puma"],
    "Home & Kitchen": ["IKEA","Philips","Hamilton Beach","Instant Pot","Kenwood"],
    "Beauty": ["L'Oreal","Maybelline","Sephora","Dove","Nivea","The Body Shop"],
    "Sports": ["Adidas","Nike","Puma","Decathlon","Yonex"],
    "Toys & Games": ["Hasbro","Lego","Mattel","Funko"],
    "Books": ["Penguin","HarperCollins","Random House"],
    "Groceries": ["WholeFoods","KrogerBrand","TraderJoes","NaturesBest"],
    "Health": ["CVSBrand","WalgreensBrand","Johnson&Johnson"],
    "Furniture": ["IKEA","Ashley","West Elm"],
    "Automotive": ["Bosch","3M","Mobil"],
    "Office Supplies": ["Staples","BIC","Pilot"],
    "Jewelry": ["Tiffany","Pandora"],
    "Garden": ["Gardena","Husqvarna"],
    "Pet Supplies": ["Purina","Pedigree"],
    "Baby": ["Pampers","Johnson&Johnson"],
    "Appliances": ["Whirlpool","GE","Samsung"],
    "Outdoor": ["TheNorthFace","Columbia"],
    "Music": ["Yamaha","Fender"],
    "Misc": ["GenericBrandA","GenericBrandB"]
}
CATEGORY_PRODUCT_TYPES = {
    "Electronics": ["Phone","Laptop","Headphones","Camera","Speaker","Monitor","Tablet"],
    "Fashion": ["Shirt","Shoes","Jacket","Dress","Bag","Watch","Sunglasses"],
    "Home & Kitchen": ["Blender","Cookware","Chair","Table","Vacuum","Lamp","Knife Set"],
    "Beauty": ["Lipstick","Perfume","Shampoo","Conditioner","Cream","Mascara","Foundation"],
    "Sports": ["Tennis Racket","Football","Basketball","Yoga Mat","Running Shoes","Helmet"],
    "Toys & Games": ["Puzzle","Board Game","Action Figure","Doll","LEGO Set"],
    "Books": ["Novel","Biography","Science Book","Comic Book","Cookbook"],
    "Groceries": ["Cereal","Chocolate","Milk","Cheese","Juice","Snack Pack"],
    "Health": ["Vitamins","Pain Relief","Bandages","Protein Powder","First Aid Kit"],
    "Furniture": ["Sofa","Bed","Wardrobe","Desk","Chair","Bookshelf"],
    "Automotive": ["Car Cover","Tire","Oil Filter","GPS Device","Headlight"],
    "Office Supplies": ["Notebook","Pen","Stapler","Marker","Printer"],
    "Jewelry": ["Necklace","Ring","Bracelet","Earrings","Watch"],
    "Garden": ["Hose","Lawn Mower","Pruner","Plant Pot","Garden Tool Set"],
    "Pet Supplies": ["Dog Food","Cat Litter","Pet Toy","Leash","Pet Bed"],
    "Baby": ["Diapers","Baby Bottle","Stroller","Pacifier","Baby Monitor"],
    "Appliances": ["Microwave","Refrigerator","Washing Machine","Toaster","Coffee Maker"],
    "Outdoor": ["Tent","Sleeping Bag","Backpack","Camping Stove","Jacket"],
    "Music": ["Guitar","Keyboard","Microphone","Drum Set","Headphones"],
    "Misc": ["Gift Card","Umbrella","Water Bottle","Flashlight","Backpack"]
}


# ---------------------- CONFIG: category image pools ---------------------- #
CATEGORY_IMAGES = {
    "Electronics": [
        "https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5",
        "https://images.unsplash.com/photo-1517336714731-489689fd1ca8",
        "https://images.unsplash.com/photo-1512499617640-c2f999098c01",
        "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f",
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
        "https://cdn-icons-png.flaticon.com/512/4436/4436481.png",
        "https://cdn-icons-png.flaticon.com/512/4202/4202836.png",
        "https://cdn-icons-png.flaticon.com/512/1046/1046784.png"
    ],
    "Fashion": [
        "https://images.unsplash.com/photo-1521335629791-ce4aec67dd47",
        "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab",
        "https://images.unsplash.com/photo-1520975918318-5bfb3c1f6a5d",
        "https://images.unsplash.com/photo-1537010151-1e2469d3d832",
        "https://images.unsplash.com/photo-1503341455253-b2e723bb3dbb",
        "https://cdn-icons-png.flaticon.com/512/836/836642.png",
        "https://cdn-icons-png.flaticon.com/512/836/836600.png",
        "https://cdn-icons-png.flaticon.com/512/3456/3456426.png"
    ],
    "Groceries": [
        "https://images.unsplash.com/photo-1572441710523-4cfa1a2b48b4",
        "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce",
        "https://images.unsplash.com/photo-1542831371-d531d36971e6",
        "https://images.unsplash.com/photo-1580910051073-e6a1b1b7f9e1",
        "https://images.unsplash.com/photo-1510626176961-4b37d0f0b7c8",
        "https://cdn-icons-png.flaticon.com/512/3075/3075977.png",
        "https://cdn-icons-png.flaticon.com/512/590/590836.png",
        "https://cdn-icons-png.flaticon.com/512/1046/1046787.png"
    ],
    "Home & Kitchen": [
        "https://images.unsplash.com/photo-1505691938895-1758d7feb511",
        "https://images.unsplash.com/photo-1606813909028-63e1d6c1f2e4",
        "https://images.unsplash.com/photo-1600585154340-be6161a56a0c",
        "https://images.unsplash.com/photo-1582582420890-bb38a3a36d17",
        "https://images.unsplash.com/photo-1598300054675-87cbbd41bb7b",
        "https://cdn-icons-png.flaticon.com/512/3515/3515265.png",
        "https://cdn-icons-png.flaticon.com/512/2972/2972165.png",
        "https://cdn-icons-png.flaticon.com/512/1046/1046857.png"
    ],
    "Sports": [
        "https://images.unsplash.com/photo-1599058917212-d750089bc07a",
        "https://images.unsplash.com/photo-1505842465776-3d4f1c8a57f0",
        "https://images.unsplash.com/photo-1599058917504-b3d2ef26738f",
        "https://images.unsplash.com/photo-1517649763962-0c623066013b",
        "https://cdn-icons-png.flaticon.com/512/1046/1046947.png",
        "https://cdn-icons-png.flaticon.com/512/1046/1046951.png",
        "https://cdn-icons-png.flaticon.com/512/1046/1046950.png"
    ],
    "Misc": [
        "https://cdn-icons-png.flaticon.com/512/679/679922.png"
    ]
}

# If category missing in images, default to placeholder
DEFAULT_IMAGE = "https://picsum.photos/400/400"

# ---------------------- Helpers ---------------------- #
def choose_category():
    cats = list(category_weights.keys())
    probs = [category_weights[c] for c in cats]
    return np.random.choice(cats, p=probs)

def make_sku(i):
    return f"SKU-{i:06d}"

def make_barcode(i):
    return f"{random.randint(100000000000,999999999999)}"

def random_date_between(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta,0)))

# ---------------------- Price ranges, weights, regions, payments ---------------------- #
CATEGORY_LIST = list(CATEGORIES.keys())
category_weights = {
    "Electronics": 0.12, "Fashion": 0.13, "Home & Kitchen": 0.08, "Beauty":0.06,
    "Sports":0.05, "Toys & Games":0.04, "Books":0.03, "Groceries":0.12,
    "Health":0.04, "Furniture":0.03, "Automotive":0.02, "Office Supplies":0.02,
    "Jewelry":0.01, "Garden":0.01, "Pet Supplies":0.02, "Baby":0.03,
    "Appliances":0.05, "Outdoor":0.01, "Music":0.01, "Misc":0.03
}
category_weights = {k:v/sum(category_weights.values()) for k,v in category_weights.items()}

PRICE_RANGES = {
    "Electronics": (100, 3000),
    "Fashion": (10, 300),
    "Home & Kitchen": (15, 600),
    "Beauty": (5, 200),
    "Sports": (10, 400),
    "Toys & Games": (5, 200),
    "Books": (5, 50),
    "Groceries": (1, 50),
    "Health": (5,200),
    "Furniture": (50, 2000),
    "Automotive": (10, 400),
    "Office Supplies": (1, 80),
    "Jewelry": (50, 5000),
    "Garden": (10, 300),
    "Pet Supplies": (5, 200),
    "Baby": (5, 200),
    "Appliances": (50, 2500),
    "Outdoor": (20, 400),
    "Music": (10, 1500),
    "Misc": (5, 250)
}

REGIONS = {
    "US": ["New York","San Francisco","Chicago","Los Angeles","Austin"],
    "UK": ["London","Manchester","Birmingham"],
    "IN": ["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai"],
    "EU": ["Berlin","Paris","Amsterdam","Madrid"],
    "AU": ["Sydney","Melbourne","Brisbane"]
}

PAYMENT_METHODS = {
    "US": ["Credit Card","Debit Card","PayPal","Apple Pay"],
    "UK": ["Credit Card","Debit Card","PayPal"],
    "IN": ["UPI","Credit Card","Debit Card","Wallet"],
    "EU": ["Credit Card","Debit Card","SEPA","PayPal"],
    "AU": ["Credit Card","Debit Card","PayPal"]
}

# ---------------------- Seasonal multiplier ---------------------- #
def seasonal_multiplier(date):
    m = 1.0
    if date.weekday() in (5,6): m += 0.08
    if date.month == 11 and date.day in range(25, 31): m += 0.6
    if date.month == 12 and date.day in range(20, 27): m += 0.45
    if date.month in (6,7): m += 0.12
    m += np.random.normal(0, 0.02)
    return max(0.3, m)

# ---------------------- Generate Product Catalogue ---------------------- #
products = []
for i in tqdm(range(1, NUM_PRODUCTS+1), desc="Generating Products"):
    category = choose_category()
    brand = random.choice(CATEGORIES[category])
    product_type = random.choice(CATEGORY_PRODUCT_TYPES.get(category, ["Item"]))
    
    # Price
    low, high = PRICE_RANGES[category]
    price = round(float(np.round(np.random.lognormal(mean=np.log((low+high)/2), sigma=0.9),2)),2)
    price = float(np.clip(price, low, high))
    
    # Image
    image_list = CATEGORY_IMAGES.get(category, [DEFAULT_IMAGE])
    image_url = random.choice(image_list)
    
    products.append({
        "product_id": f"P{i:06d}",
        "product_name": f"{brand} {product_type}",
        "product_unit_price": price,
        "product_category": category,
        "product_description": fake.sentence(nb_words=10),
        "product_image": image_url,
        "brand_name": brand,
        "sku": make_sku(i),
        "barcode": make_barcode(i),
        "launch_date": (START_DATE - pd.Timedelta(days=random.randint(0,900))).date(),
        "is_active": random.choices([True, False], weights=[0.95,0.05])[0]
    })

product_catalogue_df = pd.DataFrame(products)
# (The rest of your customer, transaction, sales, inventory generation code stays the same)
# ...




# ---------------------- Generate Customers ---------------------- #
customers = []
region_choices = list(REGIONS.keys())
region_city_cache = []

# region_choices = list(REGIONS.keys())
# customers = []

for c in tqdm(range(1, NUM_CUSTOMERS + 1), desc="Generating Customers"):
    # weighted region distribution: US, IN, EU, UK, AU
    region = random.choices(region_choices, weights=[0.30, 0.25, 0.20, 0.15, 0.10])[0]
    city = random.choice(REGIONS[region])
    fname = fake.first_name()
    lname = fake.last_name()
    signup_date = fake.date_between(start_date=START_DATE - pd.DateOffset(years=2), end_date=START_DATE)
    membership = random.choices(
        ["Standard", "Silver", "Gold", "Platinum"],
        weights=[0.6, 0.25, 0.1, 0.05]
    )[0]

    customers.append({
        "customer_id": f"C{c:06d}",
        "first_name": fname,
        "last_name": lname,
        "city": city,
        "region": region,
        "email": fake.safe_email(),
        "phone": fake.phone_number(),
        "gender": random.choice(["M", "F"]),
        "age": random.randint(18, 80),
        "signup_date": signup_date,
        "membership_tier": membership,
        "loyalty_points": random.randint(0, 5000)
    })

customers_df = pd.DataFrame(customers)

# ---------------------- Build date-index for transactions generation ---------------------- #
dates = pd.date_range(START_DATE, END_DATE, freq="D").to_pydatetime().tolist()

# Build product popularity score (some products sell more)
# base popularity depends on category; then add per-product noise
cat_popularity = {cat: (category_weights[cat]*100) for cat in CATEGORY_LIST}
product_pop_scores = []
for p in product_catalogue_df.itertuples(index=False):
    base = cat_popularity[p.product_category]
    noise = np.random.normal(1.0, 0.4)
    score = max(0.01, base * noise)
    product_pop_scores.append(score)
product_catalogue_df["pop_score"] = product_pop_scores

# Normalize to probabilities
product_catalogue_df["pop_prob"] = product_catalogue_df["pop_score"] / product_catalogue_df["pop_score"].sum()

# ---------------------- Generate Transactions (line items) ---------------------- #
# We'll generate transactions by choosing a date (with seasonal multiplier), a customer, and 1-4 items per transaction

transaction_records = []
transaction_summary = []  # one per transaction (for transaction table)
txn_id_seq = 1
line_items_generated = 0

with tqdm(total=NUM_TRANSACTIONS, desc="Generating Transactions") as pbar:
    while line_items_generated < NUM_TRANSACTIONS:
        # pick a transaction date with seasonal weighting
        date = random.choice(dates)
        mult = seasonal_multiplier(date)
        # pick a customer
        cust = customers_df.sample(1).iloc[0]
        region = cust["region"]
        city = cust["city"]
        payment_method = random.choices(PAYMENT_METHODS[region], k=1)[0]
        channel = random.choices(["Online","In-store","Mobile App"], weights=[0.65,0.25,0.10])[0]
        status = random.choices(["Completed","Completed","Completed","Refunded","Cancelled"],
                                weights=[0.92,0.92,0.92,0.05,0.03])[0]

        num_items = np.random.choice([1,1,1,2,2,3,4], p=[0.25,0.25,0.2,0.15,0.07,0.05,0.03])
        txn_total = 0.0

        for _ in range(num_items):
            # choose a product by popularity prob
            p_row = product_catalogue_df.sample(weights=product_catalogue_df["pop_prob"], n=1).iloc[0]
            # unit qty: groceries & books have higher qty; electronics usually 1
            if p_row["product_category"] in ("Groceries","Books","Home & Kitchen"):
                qty = np.random.choice([1,1,2,3,5,6], p=[0.45,0.25,0.15,0.08,0.05,0.02 ])
            else:
                qty = int(np.random.choice([1,1,1,2,2,3], p=[0.6,0.2,0.1,0.06,0.03,0.01]))

            unit_price = float(p_row["product_unit_price"])
            base_discount_pct = 0.0
            if status == "Completed":
                mem = cust["membership_tier"]
                mem_discount = {"Standard":0, "Silver":0.02, "Gold":0.05, "Platinum":0.1}[mem]
                promo = 0.0
                if date.month in (11,12):
                    promo = random.choices([0,0.05,0.10,0.20], weights=[0.65,0.2,0.1,0.05])[0]
                coupon = random.choices([0,0.05,0.1], weights=[0.85,0.1,0.05])[0]
                base_discount_pct = mem_discount + promo + coupon

            discount_amount = round(unit_price * qty * base_discount_pct, 2)
            tax_rate = 0.18 if region in ("IN","AU") else 0.12 if region in ("EU","UK") else 0.1
            tax_amount = round((unit_price * qty - discount_amount) * tax_rate, 2)
            line_amount = round(unit_price * qty - discount_amount + tax_amount, 2)
            txn_total += line_amount

            transaction_records.append({
                "transaction_id": f"T{txn_id_seq:08d}",
                "product_id": p_row["product_id"],
                "customer_id": cust["customer_id"],
                "product_name": p_row["product_name"],
                "unit_price": unit_price,
                "product_quantity": qty,
                "discount_applied": discount_amount,
                "tax_amount": tax_amount,
                "line_total": line_amount,
                "date_of_transaction": pd.to_datetime(date).date(),
                "region": region,
                "city": city,
                "mode_of_payment": payment_method,
                "transaction_status": status,
                "channel": channel,
                "delivery_type": random.choice(["Home Delivery","Store Pickup"]),
                "refund_flag": 1 if status in ("Refunded","Cancelled") else 0
            })
            line_items_generated += 1
            pbar.update(1)  # update progress bar per line item

        transaction_summary.append({
            "transaction_id": f"T{txn_id_seq:08d}",
            "num_items": num_items,
            "total_amount": round(txn_total,2),
            "date_of_transaction": pd.to_datetime(date).date(),
            "customer_id": cust["customer_id"],
            "region": region,
            "city": city,
            "mode_of_payment": payment_method,
            "transaction_status": status
        })
        txn_id_seq += 1

# Build DataFrames
transactions_lines_df = pd.DataFrame(transaction_records)
transactions_summary_df = pd.DataFrame(transaction_summary)

# ---------------------- Build aggregated sales table (daily per product per region) ---------------------- #
sales_daily = transactions_lines_df.groupby(
    ["date_of_transaction","product_id","product_name","region"]
).agg({
    "product_quantity":"sum",
    "line_total":"sum",
    "discount_applied":"sum",
    "tax_amount":"sum",
    "transaction_id":"nunique"
}).reset_index().rename(columns={
    "date_of_transaction":"date",
    "product_quantity":"product_quantity",
    "line_total":"total_amount",
    "transaction_id":"transaction_count"
})

# ---------------------- Inventory Simulation ---------------------- #
# Initial stock per product (SKU) depends on category and price (higher-value SKUs have lower initial stock)
inventory_rows = []

for p in tqdm(product_catalogue_df.itertuples(index=False), desc="Generating Inventory"):
    # Base stock depends on category
    base_stock = int(np.clip(
        np.random.normal(loc=300 if p.product_category=="Groceries" else 150, scale=80), 
        20, 2000
    ))
    
    reorder_level = int(np.clip(base_stock * np.random.uniform(0.15, 0.35), 5, base_stock//2))
    lead_time_days = random.randint(3, 14)  # supplier lead time
    supplier_id = f"S{int(p.product_id[1:])%5000:05d}"
    supplier_name = f"{random.choice(['GlobalSupply','PrimeSupplies','MetroTrade','InterGoods','FastSource'])} {supplier_id}"
    
    stock = base_stock
    
    # monthly snapshots across the year
    for month_dt in pd.date_range(START_DATE, END_DATE, freq="ME"):
        # Sold units this month
        sold_units = int(sales_daily[
            (sales_daily["product_id"] == p.product_id) & 
            (pd.to_datetime(sales_daily["date"]).dt.to_period("M") == month_dt.to_period("M"))
        ]["product_quantity"].sum())
        
        stock = max(stock - sold_units, 1)  # ensure stock never zero
        
        # Always generate a realistic order
        order_quantity = int(max(reorder_level * 1.8, reorder_level + sold_units))
        order_status = random.choice(["Placed", "Delivered", "In Transit"])
        delivery_date = (month_dt + pd.Timedelta(days=lead_time_days)).date()
        stock += order_quantity  # replenishment
        
        cogs = round(p.product_unit_price * np.random.uniform(0.5,0.75), 2)
        
        inventory_rows.append({
            "product_id": p.product_id,
            "product_name": p.product_name,
            "product_unit_price": p.product_unit_price,
            "product_category": p.product_category,
            "supplier_name": supplier_name,
            "supplier_id": supplier_id,
            "snapshot_date": month_dt.date(),
            "order_quantity": order_quantity,
            "order_status": order_status,
            "delivery_date": delivery_date,
            "cogs": cogs,
            "inventory_level": stock,
            "reorder_level": reorder_level,
            "lead_time_days": lead_time_days,
            "stockout_flag": int(stock==0),
            "inventory_valuation": round(stock * cogs, 2)
        })

inventory_df = pd.DataFrame(inventory_rows)

# ---------------------- Customers "highest freq purchased items" (small counter data) ---------------------- #
# compute top 1 purchased item per customer (if any)
tqdm.pandas()  # enable progress_apply

# ---------------------- Customer top products ---------------------- #
cust_top = (
    transactions_lines_df.groupby(["customer_id", "product_id"])
    .agg({"product_quantity": "sum"})
    .reset_index()
)

# Sort values
cust_top = cust_top.sort_values(["customer_id", "product_quantity"], ascending=[True, False])

# Get the top product per customer using tqdm progress_apply
cust_top = (
    cust_top.groupby("customer_id", group_keys=False)
    .progress_apply(lambda x: x.iloc[0])
    .reset_index(drop=True)
    .rename(columns={"product_id": "top_product_id", "product_quantity": "top_qty"})
)

# Merge into customers_df
customers_df = customers_df.merge(
    cust_top[["customer_id", "top_product_id"]], on="customer_id", how="left"
)
customers_df["highest_freq_purchased_items"] = customers_df["top_product_id"].fillna("")

# ---------------------- Final cleanup & column ordering ---------------------- #
product_catalogue_df = product_catalogue_df[[
    "product_id","product_name","product_unit_price","product_category","product_description",
    "product_image","brand_name","sku","barcode","launch_date","is_active"
]]

customers_df = customers_df[[
    "customer_id","first_name","last_name","city","region","email","phone",
    "gender","age","signup_date","membership_tier","loyalty_points","highest_freq_purchased_items"
]]

transactions_lines_df = transactions_lines_df[[
    "transaction_id","product_id","customer_id","product_name","unit_price",
    "product_quantity","discount_applied","tax_amount","line_total","date_of_transaction",
    "region","city","mode_of_payment","transaction_status","channel","delivery_type","refund_flag"
]]

transactions_summary_df = transactions_summary_df[[
    "transaction_id","num_items","total_amount","date_of_transaction","customer_id","region","city","mode_of_payment","transaction_status"
]]

sales_daily = sales_daily[[
    "date","product_id","product_name","product_quantity","total_amount","discount_applied","tax_amount","transaction_count","region"
]]

inventory_df = inventory_df[[
    "product_id","product_name","product_unit_price","product_category","supplier_name","supplier_id",
    "snapshot_date","order_quantity","order_status","delivery_date","cogs","inventory_level","reorder_level","lead_time_days",
    "stockout_flag","inventory_valuation"
]]

# ---------------------- Save CSVs ---------------------- #
product_catalogue_df.to_csv(os.path.join(OUTPUT_DIR,"product_catalogue.csv"), index=False)
customers_df.to_csv(os.path.join(OUTPUT_DIR,"customers.csv"), index=False)
transactions_merged_df = transactions_lines_df.merge(
    transactions_summary_df,
    on="transaction_id",
    how="left"
)

# Save only one transactions file
transactions_merged_df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
sales_daily.to_csv(os.path.join(OUTPUT_DIR,"sales_daily.csv"), index=False)
inventory_df.to_csv(os.path.join(OUTPUT_DIR,"inventory.csv"), index=False)

print(f"✅ Generated datasets saved to folder: {OUTPUT_DIR}")
print("Files:")
for f in os.listdir(OUTPUT_DIR):
    print(" -", f)

sales_daily.to_csv(os.path.join(OUTPUT_DIR,"sales_daily.csv"), index=False)
inventory_df.to_csv(os.path.join(OUTPUT_DIR,"inventory.csv"), index=False)

print(f"✅ Generated datasets saved to folder: {OUTPUT_DIR}")
print("Files:")
for f in os.listdir(OUTPUT_DIR):
    print(" -", f)
