import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime
from faker import Faker

fake = Faker()

# Load reference dataset
ref_df = pd.read_csv(r"C:\Users\palya\Desktop\DemandCast\Demand-Cast\amul_product_catalogue.csv")  # SKU_ID, Product_Name, Category, Price, Popularity_Score

# Regions / store locations
store_locations = ["North", "South", "East", "West", "Central"]
payment_methods = ["Cash", "Card", "UPI", "Wallet", "Net Banking"]
customer_types = ["Regular", "New", "Loyal", "Occasional","First-time","Returning","VIP","Seasonal","Business","Family"]
channels = ["Offline Store", "Online Portal", "Mobile App",'franchise store']

# Festival dictionary (for last 2 yrs)
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
# Use your existing festivals list here
festivals_dict = {}
for d in festivals:
    festivals_dict.update(d)  # Flatten list of dicts into single dict

# Seasonal multipliers by month per category (example)
seasonal_factors = {
    "Ice Creams": {12: 0.8, 1: 0.7, 2: 0.7, 3: 1.0, 4: 1.2, 5: 1.3, 6: 1.4, 7: 1.3, 8: 1.2, 9: 1.0, 10: 0.9, 11: 0.8},
    "Dairy Products": {12: 1.1, 1: 1.1, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.9, 6: 0.9, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.1},
}

def get_seasonal_factor(category, month):
    return seasonal_factors.get(category, {}).get(month, 1.0)

def get_festival_name(date):
    date_str = date.strftime("%Y-%m-%d")
    return festivals_dict.get(date_str, None)

def generate_transaction_data_2yrs(num_rows=200000):
    data = []
    end_date = datetime.today()
    start_date = end_date - timedelta(days=2*365)
    date_range = (end_date - start_date).days

    for i in range(num_rows):
        sku_row = ref_df.sample(n=1).iloc[0]

        SKU_ID = sku_row['SKU_ID']
        Product_Name = sku_row['Product_Name']
        Category = sku_row['Category']
        Unit_Price = float(sku_row['Price'])
        popularity = sku_row.get('Popularity_Score', 3)

        # Random date in last 2 years
        Date = start_date + timedelta(days=random.randint(0, date_range))
        month = Date.month

        # Store location
        Store_Location = random.choice(store_locations)

        # Seasonal & festival adjustments on quantity
        seasonal_factor = get_seasonal_factor(Category, month)
        festival_factor = 1.5 if get_festival_name(Date) else 1.0

        # Quantity sold
        Quantity_Sold = max(1, int(np.random.poisson(lam=5 + popularity*3) * seasonal_factor * festival_factor))

        # Total amount
        Total_Amount = round(Quantity_Sold * Unit_Price, 2)

        # Discount applied randomly (0-20%)
        Discount_Applied = round(Total_Amount * random.uniform(0, 0.2), 2)

        # Payment method
        Payment_Method = random.choice(payment_methods)

        # Customer type
        Customer_Type = random.choice(customer_types)

        # Channel
        Channel = random.choice(channels)

        # Transaction ID (unique)
        Transaction_ID = f"T{i+1:06d}"

        data.append([
            Transaction_ID, Date.date(), SKU_ID, Product_Name, Category,
            Quantity_Sold, Unit_Price, Total_Amount, Store_Location,
            Payment_Method, Discount_Applied, Customer_Type, Channel
        ])

    columns = [
        "Transaction_ID", "Date", "SKU_ID", "Product_Name", "Category",
        "Quantity_Sold", "Unit_Price", "Total_Amount", "Store_Location",
        "Payment_Method", "Discount_Applied", "Customer_Type", "Channel"
    ]

    return pd.DataFrame(data, columns=columns)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    transaction_df = generate_transaction_data_2yrs(num_rows=20000)
    print(transaction_df.head())
    transaction_df.to_csv("synthetic_transaction_2yrs.csv", index=False)
