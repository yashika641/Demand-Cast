import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import time
import random
import warnings
from tqdm import tqdm  # progress bar

# ------------------------
# 0. Suppress FutureWarnings from pytrends
# ------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="pytrends")

# ------------------------
# 1. Use cleaned product list directly
# ------------------------
unique_products = amul_products_cleaned = [
    'Amul Chocolate', 
    'Amul Chocozoo ',
    'Amul Kool Milk',
    'Amul Kool  ',
    'Amul Pro Milk ', 
    'Amul Orange Juice', 
    'Amul Apple Juice',
    'Amul Yogurt ',
    'Amul Mithai ',
    'Amul Paratha',
    'Amul Paneer Tikka', 
    'Amul Cheese ',
    'Amul Taaza UHT Milk', 
    'Amul Gold UHT Milk', 
    'Amul Slim UHT Milk',
    'Amul Chocolate', 
    'Amul Kool Café Can', 
    'Amul Cheesy',
    'Amul Potato Chips',
    'Amul Peanuts', 
    'Amul Veggie ',
    'Amul Cheese Crackers',
    'Amul Croissant',
    'Amul Muffin', 
    'Amul Cheese',
    'Amul Donut',
    'Amul Protein Shake ', 
    'Amul Protein Shake ', 
    'Amul Dahi',
    'Amul Lassi',
    'Amul Buttermilk', 
]


# ------------------------
# 2. Fetch Google Trends (one product at a time + retry + delay)
# ------------------------
pytrends = TrendReq(hl="en-US", tz=360)
all_trends = pd.DataFrame()

def fetch_trend_single(keyword):
    """Fetch Google Trends for a single keyword with retry logic"""
    for attempt in range(3):  # retry max 3 times
        try:
            pytrends.build_payload([keyword], cat=0, timeframe="today 12-m", geo="IN")
            data = pytrends.interest_over_time()
            if not data.empty:
                data = data.drop(columns=["isPartial"], errors="ignore").reset_index()
                data = data.rename(columns={keyword: keyword})
                return data
            return None
        except Exception as e:
            print(f"⚠️ Error fetching {keyword}, attempt {attempt+1}: {e}")
            time.sleep(60)  # wait before retry
    return None

# Progress bar loop
for product in tqdm(unique_products, desc="Fetching Google Trends"):
    data = fetch_trend_single(product)
    
    if data is not None:
        if all_trends.empty:
            all_trends = data
        else:
            all_trends = pd.merge(all_trends, data, on="date", how="outer")
        print(f"✅ Got data for: {product}")
    else:
        print(f"❌ Failed to fetch data for: {product}")
    
    # Sleep between requests to avoid 429
    time.sleep(random.randint(10, 30))

# ------------------------
# 3. Save & Plot
# ------------------------
all_trends.to_csv("amul_trends_cleaned2.csv", index=False)
print("📊 All trends saved to amul_trends_cleaned.csv")

# Example plot for first 5 products
plt.figure(figsize=(12,6))
for col in all_trends.columns[1:6]:
    plt.plot(all_trends["date"], all_trends[col], label=col)

plt.title("Google Trends (Amul Products - Cleaned Names)")
plt.xlabel("Date")
plt.ylabel("Interest Over Time")
plt.legend()
plt.grid(True, linestyle="dotted", alpha=0.5)
plt.tight_layout()
plt.show()
