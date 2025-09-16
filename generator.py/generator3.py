import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta, datetime

fake = Faker()

# Load reference dataset
ref_df = pd.read_csv(r"C:\Users\palya\Desktop\DemandCast\Demand-Cast\datasets\amul_product_catalogue.csv")  # SKU_ID, Product_Name, Category, Price, Popularity_Score

# Regions
region_temp = {
    "North": (5, 35),
    "South": (20, 40),
    "East": (15, 38),
    "West": (10, 40),
    "Central": (15, 40)
}


# Indian festival dates (approximate)
festivals = [
    # 2022
    { "2022-01-01" : "New Year's Day" },
    { "2022-01-14" : "Makar Sankranti / Pongal" },
    { "2022-01-18" : "Thaipusam" },
    { "2022-01-26" : "Republic Day" },
    { "2022-02-05" : "Vasant Panchami" },
    { "2022-02-14" : "Valentine's Day" },
    { "2022-03-01" : "Maha Shivaratri" },
    { "2022-03-17" : "Holika Dahan" },
    { "2022-03-18" : "Holi" },
    { "2022-04-01" : "April Fool's Day" },
    { "2022-04-02" : "Ugadi / Gudi Padwa / Telugu New Year" },
    { "2022-04-10" : "Ram Navami" },
    { "2022-04-11" : "Chaitra Navratri Parana" },
    { "2022-04-14" : "Baisakhi / Ambedkar Jayanti" },
    { "2022-04-16" : "Hanuman Jayanti" },
    { "2022-05-01" : "Labor Day / May Day" },
    { "2022-05-03" : "Akshaya Tritiya" },
    { "2022-07-01" : "Jagannath Rath Yatra" },
    { "2022-07-10" : "Ashadhi Ekadashi" },
    { "2022-07-13" : "Guru Purnima" },
    { "2022-07-31" : "Hariyali Teej" },
    { "2022-08-11" : "Raksha Bandhan" },
    { "2022-08-15" : "Independence Day" },
    { "2022-08-19" : "Krishna Janmashtami" },
    { "2022-10-05" : "Dussehra" },
    { "2022-10-24" : "Diwali" },
    { "2022-10-26" : "Govardhan Puja" },
    { "2022-10-27" : "Bhai Dooj" },
    { "2022-10-31" : "Halloween" },
    { "2022-11-08" : "Karva Chauth" },
    { "2022-11-23" : "Gurpurab (Guru Nanak Jayanti)" },
    { "2022-12-24" : "Christmas Eve" },
    { "2022-12-25" : "Christmas" },
    { "2022-12-26" : "Boxing Day" },

    # 2023
    { "2023-01-01" : "New Year's Day" },
    { "2023-01-14" : "Makar Sankranti / Pongal" },
    { "2023-01-26" : "Republic Day" },
    { "2023-02-05" : "Guru Ravidas Jayanti / Hazarat Ali's Birthday" },
    { "2023-02-14" : "Valentine's Day"},
    { "2023-03-07" : "Holika Dahan" },
    { "2023-03-08" : "Holi" },
    { "2023-03-22" : "Ugadi / Gudi Padwa / Telugu New Year" },
    { "2023-03-30" : "Ram Navami" },
    { "2023-04-01" : "April Fool's Day" },
    { "2023-04-06" : "Hanuman Jayanti" },
    { "2023-04-14" : "Baisakhi / Ambedkar Jayanti" },
    { "2023-04-21" : "Mahavir Jayanti" },
    { "2023-05-01" : "Labor Day / May Day" },
    { "2023-05-05" : "Eid ul-Fitr" },
    { "2023-07-28" : "Muharram / Ashura" },
    { "2023-08-15" : "Independence Day" },
    { "2023-08-22" : "Raksha Bandhan" },
    { "2023-08-29" : "Janmashtami" },
    { "2023-09-17" : "Ganesh Chaturthi" },
    { "2023-09-25" : "Navratri Begins" },
    { "2023-10-04" : "Dussehra" },
    { "2023-10-23" : "Diwali" },
    { "2023-10-25" : "Govardhan Puja" },
    { "2023-10-26" : "Bhai Dooj" },
    { "2023-10-31" : "Halloween" },
    { "2023-11-12" : "Karva Chauth" },
    { "2023-11-27" : "Gurpurab (Guru Nanak Jayanti)" },
    { "2023-12-24" : "Christmas Eve" },
    { "2023-12-25" : "Christmas"},
    { "2023-12-26" : "Boxing Day" },

    # 2024
    { "2024-01-01" : "New Year's Day" },
    { "2024-01-14" : "Makar Sankranti / Pongal" },
    { "2024-01-26" : "Republic Day" },
    { "2024-02-17" : "Maha Shivaratri" },
    { "2024-02-14" : "Valentine's Day" },
    { "2024-03-24" : "Holika Dahan" },
    { "2024-03-25" : "Holi" },
    { "2024-03-29" : "Ugadi / Gudi Padwa / Telugu New Year" },
    { "2024-04-01" : "April Fool's Day" },
    { "2024-04-09" : "Eid ul-Fitr" },
    { "2024-04-14" : "Baisakhi / Ambedkar Jayanti" },
    { "2024-04-17" : "Ram Navami" },
    { "2024-04-20" : "Mahavir Jayanti" },
    { "2024-04-23" : "Hanuman Jayanti" },
    { "2024-06-27" : "Jagannath Rath Yatra" },
    { "2024-08-15" : "Independence Day" },
    { "2024-08-22" : "Raksha Bandhan" },
    { "2024-08-29" : "Janmashtami" },
    { "2024-09-17" : "Ganesh Chaturthi" },
    { "2024-09-25" : "Navratri Begins" },
    { "2024-10-04" : "Dussehra" },
    { "2024-10-23" : "Diwali" },
    { "2024-10-25" : "Govardhan Puja" },
    { "2024-10-26" : "Bhai Dooj" },
    { "2024-10-31" : "Halloween" },
    { "2024-11-12" : "Karva Chauth" },
    { "2024-11-27" : "Gurpurab (Guru Nanak Jayanti)"},
    { "2024-12-24" : "Christmas Eve" },
    { "2024-12-25" : "Christmas" },
    { "2024-12-26" : "Boxing Day" },

    # 2025
    { "2025-01-01" : "New Year's Day" },
    { "2025-01-14" : "Makar Sankranti / Pongal" },
    { "2025-01-26" : "Republic Day" },
    { "2025-02-05" : "Guru Ravidas Jayanti / Hazarat Ali's Birthday" },
    { "2025-02-14" : "Valentine's Day" },
    { "2025-03-13" : "Holika Dahan" },
    { "2025-03-14" : "Holi" },
    { "2025-03-30" : "Ugadi / Gudi Padwa / Telugu New Year" },
    { "2025-04-01" : "April Fool's Day" },
    { "2025-04-06" : "Ram Navami" },
    { "2025-04-12" : "Hanuman Jayanti" },
    { "2025-04-14" : "Baisakhi / Ambedkar Jayanti" },
    { "2025-04-30" : "Akshaya Tritiya" },
    { "2025-06-27" : "Jagannath Rath Yatra" },
    { "2025-08-15" : "Independence Day" },
    { "2025-08-26" : "Hartalika Teej" },
    { "2025-08-27" : "Ganesh Chaturthi" },
    { "2025-08-29" : "Janmashtami" },
    { "2025-09-17" : "Navratri Begins" },
    { "2025-10-04" : "Dussehra" },
    { "2025-10-23" : "Diwali" },
    { "2025-10-25" : "Govardhan Puja" },
    { "2025-10-26" : "Bhai Dooj" },
    { "2025-10-31" : "Halloween" },
    { "2025-12-24" : "Christmas Eve" },
    { "2025-12-25" : "Christmas" },
    { "2025-12-26" : "Boxing Day"}
]


# Seasonal multipliers by month per category (example)
seasonal_factors = {
    "Ice Creams": {12: 0.8, 1: 0.7, 2: 0.7, 3: 1.0, 4: 1.2, 5: 1.3, 6: 1.4, 7: 1.3, 8: 1.2, 9: 1.0, 10: 0.9, 11: 0.8},
    "Dairy Products": {12: 1.1, 1: 1.1, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.9, 6: 0.9, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.1},
    # default factor = 1 for other categories
}

# def is_festival(date):
#     date_str = date.strftime("%Y-%m-%d")
#     for fest in festivals:
#         if date_str in fest:
#             return True
#     return False


def get_seasonal_factor(category, month):
    return seasonal_factors.get(category, {}).get(month, 1.0)


def generate_sales_data_daily(num_days=5*365):
    data = []
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=num_days)
    # festival_dict = {}
    festivals_dict = {}
    # Create a dictionary mapping festival dates to festival names
    for fest in festivals:
        for date_str, name in fest.items():
            festivals_dict[date_str] = name
            
    print(festivals_dict)

    
    def get_festival(date):
        """
    Accepts a datetime.date or Timestamp.
    Returns festival name or None.
    """
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime("%Y-%m-%d")
        elif isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        elif isinstance(date, str):
            date_str = date.strip()
        else:  # datetime.date
            date_str = date.isoformat()
        return festivals_dict.get(date_str, None)
    # Generate continuous daily range
    d2 = datetime.strptime("2025-12-25", "%Y-%m-%d")
    print(get_festival(d2))
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    print(dates)
    for Date in dates:
        festival_name = get_festival(Date)
        
    # Pick 1-3 random categories to get boosted for the festival
        if festival_name:
            boost_categories = random.sample(list(ref_df['Category'].unique()), 
                                        k=random.randint(1, 3))
        else:
            boost_categories = []

        for _ in range(5):  # simulate multiple products per day
            sku_row = ref_df.sample(n=1).iloc[0]

            SKU_Code = sku_row['SKU_ID']
            Product_Name = sku_row['Product_Name']
            Product_Category = sku_row['Category']
            base_price = pd.to_numeric(sku_row['Price'], errors='coerce') or 0
            popularity = sku_row.get('Popularity_Score', 3)

            Region = random.choice(list(region_temp.keys()))
            Price_per_Unit = round(base_price * random.uniform(0.9, 1.1), 2)
            seasonal_factor = get_seasonal_factor(Product_Category, Date.month)

        # Apply festival boost only if product category is selected
            # Convert sentiment in [-1, 1] -> multiplier in [0.7, 1.6]
            def sentiment_to_multiplier(s):
                return float(np.clip(1.0 + 0.4 * s, 0.7, 1.6))

            # Base demand (popularity-driven)
            base_demand = np.random.poisson(lam=5 + popularity * 3)

# ------------------

# Apply seasonal factor (month-based multiplier)
            seasonal_factor = get_seasonal_factor(Product_Category, Date.month)

            Festival_Season = festival_name if festival_name else "None"
# Apply festival boost
            festival_multiplier = 1.5 if Product_Category in boost_categories else 1.0

            Weather_Temp = round(random.uniform(*region_temp[Region]), 1)
            Promotion_Flag = random.choices(["Yes", "No"], weights=[0.2, 0.8])[0]
# Promotion boost
            promo_multiplier = 1.3 if Promotion_Flag == "Yes" else 1.0

# ------------------
# Temperature effect
            temp_multiplier = 1.0
            if Product_Category.lower().find("ice cream") != -1:
    # Ice creams sell more in hot weather
                if Weather_Temp >= 30:
                    temp_multiplier = 1.5
                elif Weather_Temp <= 10:
                    temp_multiplier = 0.7
            elif Product_Category.lower().find("milk") != -1 or Product_Category.lower().find("dairy") != -1:
    # Dairy products sell steadily, slight winter boost
                if Weather_Temp <= 15:
                    temp_multiplier = 1.2
            elif Weather_Temp >= 35:
                temp_multiplier = 0.9
            elif Product_Category.lower().find("butter") != -1 or Product_Category.lower().find("cheese") != -1:
    # Butter/cheese more in winter
                if Weather_Temp <= 15:
                    temp_multiplier = 1.3
                else:
                    temp_multiplier = 0.95
# ------------------
            Social_Sentiment = round(random.uniform(-1, 1), 2)
            sentiment_multiplier = float(np.exp(0.35 * Social_Sentiment))  # smooth, ~[0.71, 1.42]

# ------------------
# Region bias (example)
            region_bias = {
    "North": 1.1,   # colder -> more dairy
    "South": 1.2,   # hot -> more ice creams
    "East": 1.0,
    "West": 1.05,
    "Central": 1.0
}
            region_multiplier = region_bias.get(Region, 1.0)

# ------------------
# Final Units Sold
            Units_Sold = max(
    1,
    int(
        base_demand
        * seasonal_factor
        * festival_multiplier
        * promo_multiplier
        * temp_multiplier
        * region_multiplier
        * sentiment_multiplier
    ),
)
# Revenue depends on units sold * price
            Revenue = Units_Sold * Price_per_Unit

            Competitor_Price = round(base_price * random.uniform(0.85, 1.15), 2)

            data.append([
            Date.date(), Product_Category, Product_Name, SKU_Code, Region,
            Units_Sold, Price_per_Unit, Revenue, Promotion_Flag,
            Festival_Season, Weather_Temp, Competitor_Price, Social_Sentiment
            ])


    columns = [
        "Date", "Product_Category", "Product_Name", "SKU_Code", "Region",
        "Units_Sold", "Price_per_Unit", "Revenue", "Promotion_Flag",
        "Festival_Season", "Weather_Temp", "Competitor_Price", "Social_Sentiment"
    ]

    return pd.DataFrame(data, columns=columns)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    sales_df = generate_sales_data_daily()
    print(sales_df.head())
    sales_df.to_csv("synthetic_sales_2yrs.csv", index=False)

