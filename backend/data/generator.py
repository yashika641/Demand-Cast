import numpy as np
import pandas as pd
import random
from faker import Faker
from datetime import timedelta, datetime

fake = Faker()

# -------------------------------
# CONFIG
# -------------------------------
NUM_PRODUCTS = 50
NUM_CUSTOMERS = 200
NUM_SUPPLIERS = 10
NUM_LOCATIONS = 5
DAYS = 365 * 1  # 1 year

START_DATE = datetime(2024, 1, 1)
DATE_RANGE = pd.date_range(START_DATE, periods=DAYS)

SEASONS = {
    "festival": ["2024-03-20", "2024-10-30", "2024-11-12"],
    "summer": (5, 7),
    "winter": (12, 2)
}

# -------------------------------
# HELPERS
# -------------------------------

def seasonal_multiplier(date):
    """Generate seasonal demand multipliers."""
    m = 1.0
    
    # weekend effect
    if date.weekday() >= 5:
        m *= np.random.uniform(1.1, 1.4)
    
    # summer boost
    if SEASONS["summer"][0] <= date.month <= SEASONS["summer"][1]:
        m *= np.random.uniform(1.05, 1.25)
    
    # winter drop (or rise depending on category)
    if date.month in [12, 1, 2]:
        m *= np.random.uniform(0.9, 1.1)

    # festival spike
    if date.strftime("%Y-%m-%d") in SEASONS["festival"]:
        m *= np.random.uniform(1.3, 2.2)

    return m


def promo_price_effect(price, promo_flag):
    """Simulate promos increasing sales volume."""
    if promo_flag == 1:
        return price * np.random.uniform(0.6, 0.85)
    return price


# -------------------------------
# 1. PRODUCT MASTER
# -------------------------------
product_ids = [f"P{i:04d}" for i in range(NUM_PRODUCTS)]
brands = ["Nestle", "HUL", "PepsiCo", "Parle", "Dabur", "Britannia", "Tata"]
categories = ["Beverages", "Snacks", "Cosmetics", "Grocery", "Household"]
subcats = {
    "Beverages": ["Tea", "Coffee", "Juice"],
    "Snacks": ["Chips", "Cookies", "Namkeen"],
    "Cosmetics": ["Cream", "Shampoo", "Soap"],
    "Grocery": ["Spices", "Flour", "Rice"],
    "Household": ["Cleaner", "Detergent"]
}

products = []
for pid in product_ids:
    cat = random.choice(categories)
    products.append({
        "sku_id": pid,
        "sku_name": fake.word() + " " + cat,
        "category": cat,
        "subcategory": random.choice(subcats[cat]),
        "brand": random.choice(brands),
        "uom": random.choice(["kg", "g", "L", "ml", "pcs"]),
        "variant": random.choice(["Classic", "Premium", "Eco", "Family Pack"])
    })

PRODUCTS = pd.DataFrame(products)

# -------------------------------
# 2. PRODUCT CATALOGUE (ENRICHED)
# -------------------------------
CATALOGUE = PRODUCTS.copy()
CATALOGUE["catalogue_id"] = ["C" + x[1:] for x in PRODUCTS["sku_id"]]
CATALOGUE["title"] = CATALOGUE["sku_name"]
CATALOGUE["description"] = [fake.text(max_nb_chars=120) for _ in range(NUM_PRODUCTS)]
CATALOGUE["images"] = ["image_" + str(i) + ".jpg" for i in range(NUM_PRODUCTS)]
CATALOGUE["tags"] = ["tag1,tag2,tag3"] * NUM_PRODUCTS
CATALOGUE["mrp"] = np.random.randint(50, 500, NUM_PRODUCTS)
CATALOGUE["rating"] = np.round(np.random.uniform(2.5, 4.9, NUM_PRODUCTS), 1)
CATALOGUE["reviews_count"] = np.random.randint(10, 2000, NUM_PRODUCTS)
CATALOGUE["is_active"] = 1

# -------------------------------
# 3. CUSTOMER MASTER
# -------------------------------
customers = []
for cid in range(NUM_CUSTOMERS):
    customers.append({
        "customer_id": f"CUST{cid:04d}",
        "customer_name": fake.name(),
        "email": fake.email(),
        "city": fake.city(),
        "state": fake.state(),
        "country": "India",
        "signup_date": fake.date_between(start_date="-2y", end_date="today"),
        "loyalty_score": np.random.randint(1, 100)
    })

CUSTOMERS = pd.DataFrame(customers)

# -------------------------------
# 4. SUPPLIERS & PURCHASE ORDERS
# -------------------------------
supplier_ids = [f"S{i:03d}" for i in range(NUM_SUPPLIERS)]

po_records = []
for pid in product_ids:
    for _ in range(np.random.randint(5, 20)):
        supp = random.choice(supplier_ids)
        order_date = fake.date_between(start_date="-1y", end_date="today")
        lead_time = np.random.randint(3, 20)
        delay = np.random.choice([0, 1], p=[0.85, 0.15])
        actual_delivery = order_date + timedelta(days=int(lead_time + delay * np.random.randint(1, 5)))

        po_records.append({
            "po_id": f"PO{fake.unique.random_int(10000, 99999)}",
            "sku_id": pid,
            "supplier_id": supp,
            "order_date": order_date,
            "expected_delivery_date": order_date + timedelta(days=lead_time),
            "actual_delivery_date": actual_delivery,
            "quantity_ordered": np.random.randint(50, 500)
        })

SUPPLIERS = pd.DataFrame(po_records)

# -------------------------------
# 5. PRICING + PROMOS
# -------------------------------
pricing_records = []
for date in DATE_RANGE:
    for pid in product_ids:
        base_price = np.random.randint(50, 300)
        promo_flag = np.random.choice([0, 1], p=[0.9, 0.1])
        promo_pct = np.random.uniform(10, 50) if promo_flag else 0

        pricing_records.append({
            "date": date,
            "sku_id": pid,
            "location_id": f"L{np.random.randint(1, NUM_LOCATIONS+1)}",
            "regular_price": base_price,
            "promo_price": base_price * (1 - promo_pct/100),
            "discount_percent": promo_pct,
            "promo_type": random.choice(["None", "BOGO", "Flat Off", "Seasonal"])
        })

PRICING_PROMO = pd.DataFrame(pricing_records)

# -------------------------------
# 6. EXTERNAL EVENTS
# -------------------------------
events = [
    ("2024-01-26", "Republic Day", "holiday"),
    ("2024-03-10", "Holi", "festival"),
    ("2024-04-14", "Heatwave Warning", "weather"),
    ("2024-10-30", "Diwali", "festival")
]
EXTERNAL_EVENTS = pd.DataFrame([{
    "date": e[0],
    "event_name": e[1],
    "event_type": e[2],
    "severity": np.random.uniform(1, 5)
} for e in events])

# -------------------------------
# 7. INVENTORY
# -------------------------------
inventory_records = []
for date in DATE_RANGE:
    for pid in product_ids:
        inventory_records.append({
            "date": date,
            "sku_id": pid,
            "location_id": f"L{np.random.randint(1, NUM_LOCATIONS+1)}",
            "stock_on_hand": np.random.randint(20, 300),
            "stock_in_transit": np.random.randint(0, 50),
            "reserved_stock": np.random.randint(0, 20)
        })

INVENTORY = pd.DataFrame(inventory_records)

# -------------------------------
# 8. SALES (FULLY LINKED)
# -------------------------------
sales_records = []
txn_id = 100000

for date in DATE_RANGE:
    for pid in product_ids:
        # base demand
        base = np.random.randint(0, 25)

        # apply seasonality
        base *= seasonal_multiplier(date)

        # promo uplift
        row = PRICING_PROMO[(PRICING_PROMO['date'] == date) & (PRICING_PROMO['sku_id'] == pid)].iloc[0]
        price = promo_price_effect(row["regular_price"], row["discount_percent"] > 0)

        # randomness
        demand = max(0, int(base + np.random.normal(0, 3)))

        for _ in range(demand):
            sales_records.append({
                "transaction_id": f"TXN{txn_id}",
                "date": date,
                "sku_id": pid,
                "customer_id": random.choice(CUSTOMERS["customer_id"].values),
                "location_id": f"L{np.random.randint(1, NUM_LOCATIONS+1)}",
                "units_sold": 1,
                "price": price,
                "on_promotion": int(row["discount_percent"] > 0),
                "payment_method": random.choice(["Cash", "UPI", "Card"]),
                "channel": random.choice(["Online", "Offline"])
            })
            txn_id += 1

SALES = pd.DataFrame(sales_records)

# ------------------------------------
# EXPORT (optional)
# ------------------------------------
PRODUCTS.to_csv("PRODUCTS.csv", index=False)
CATALOGUE.to_csv("PRODUCT_CATALOGUE.csv", index=False)
CUSTOMERS.to_csv("CUSTOMERS.csv", index=False)
SUPPLIERS.to_csv("SUPPLIERS.csv", index=False)
PRICING_PROMO.to_csv("PRICING_PROMO.csv", index=False)
EXTERNAL_EVENTS.to_csv("EXTERNAL_EVENTS.csv", index=False)
INVENTORY.to_csv("INVENTORY.csv", index=False)
SALES.to_csv("SALES.csv", index=False)

print("Dataset Generated Successfully!")
