import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

# -----------------------------
# Load your reference dataset
# -----------------------------
# Example: reference dataset has SKU_ID, Product_Name, Supplier_Name, Category, Sub_Category
ref_df = pd.read_csv(r"C:\Users\palya\Desktop\DemandCast\Demand-Cast\datasets\amul_product_catalogue.csv")

# Number of dummy rows you want to generate
num_rows = 2000

# -----------------------------
# Prepare dummy columns
# -----------------------------
dummy_data = []
sub_categories = {
    "Dairy Products": [
        "Full Cream Milk",
        "Toned Milk",
        "Double Toned Milk",
        "Skimmed Milk",
        "Flavored Milk",
        "Lactose-Free Milk",
        "UHT Milk",
        "Organic Milk",
        "Others"
    ],
    "Butter & Spreads": [
        "Salted Butter",
        "Unsalted Butter",
        "Flavored Butter",
        "Cheese Spread",
        "Margarine",
        "Others"
    ],
    "Cheese & Cheese Products": [
        "Processed Cheese",
        "Mozzarella Cheese",
        "Cheddar Cheese",
        "Paneer",
        "Emmental Cheese",
        "Gouda Cheese",
        "Cream Cheese",
        "Cheese Cubes",
        "Cheese Slices",
        "Others"
    ],
    "Ice Creams": [
        "Family Pack",
        "Cone",
        "Cup",
        "Candy Bar",
        "Sugar-Free",
        "Kulfi",
        "Premium Ice Cream",
        "Others"
    ],
    "Beverages": [
        "Flavored Milk Drink",
        "Buttermilk",
        "Lassi",
        "Juice",
        "Cold Coffee",
        "Energy Drink",
        "Probiotic Drink",
        "Others"
    ],
    "Dairy Products": [
        "Fresh Cream",
        "Whipping Cream",
        "Cooking Cream",
        "Others"
    ],
    "Dairy Products": [
        "Paneer Blocks",
        "Paneer Cubes",
        "Frozen Paneer",
        "Others"
    ],
    "Probiotic & Health Range": [
        "Plain Curd",
        "Greek Yogurt",
        "Flavored Yogurt",
        "Probiotic Curd",
        "Misti Doi",
        "Others"
    ],
    "Chocolates": [
        "Dark Chocolate",
        "Milk Chocolate",
        "Fruit & Nut Chocolate",
        "Sugar-Free Chocolate",
        "Cocoa Powder",
        "Chocolate Syrup",
        "Others"
    ],
    "Ghee & Cooking Essentials": [
        "Cow Ghee",
        "Buffalo Ghee",
        "Organic Ghee",
        "Cooking Ghee",
        "Others"
    ],
    "Bakery & Frozen": [
        "Pizza Base",
        "Frozen Snacks",
        "Frozen Paratha",
        "Frozen Sweets",
        "Others"
    ]
}

from rapidfuzz import fuzz

def get_subcategory(product_name, category, sub_categories_dict, threshold=60):
    if category in sub_categories_dict:
        best_match = "Others"
        best_score = 0
        for subcat in sub_categories_dict[category]:
            score = fuzz.partial_ratio(product_name.lower(), subcat.lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_match = subcat
        return best_match
    else:
        return "Others"



for _ in range(num_rows):
    # Randomly pick a SKU from reference dataset
    sku_row = ref_df.sample(n=1).iloc[0]
    
    SKU_ID = sku_row['SKU_ID']
    Product_Name = sku_row['Product_Name']
    Category = sku_row['Category']
    Sub_Category = get_subcategory(Product_Name,Category,sub_categories)
    Supplier_Name = sku_row.get('Supplier_Name', fake.company())
    Storage_Type = sku_row['Storage_Type'] # <- use column from dataset
    
    # Randomly generate other columns
    Pack_Size = sku_row['Packaging_Size']
    Unit_Cost_Price = sku_row['Price']
    Unit_Selling_Price = round(Unit_Cost_Price * random.uniform(1.1, 1.5), 2)
    Stock_Quantity = random.randint(100, 500)
    Reorder_Level = random.randint(100, 1000)
    Supplier_Lead_Time_Days = random.randint(1, 30)
    
    Manufacturing_Date = fake.date_between(start_date='-1y', end_date='today')
    Shelf_Life_Days = random.randint(7, 365)
    Expiry_Date = pd.to_datetime(Manufacturing_Date) + pd.to_timedelta(Shelf_Life_Days, unit='d')
    
    
    Batch_Number = f"B{random.randint(1000,9999)}"
    Location = f"Warehouse-{random.randint(1,10)}"
    
    dummy_data.append([
        SKU_ID, Product_Name, Category, Sub_Category, Pack_Size,
        Unit_Cost_Price, Unit_Selling_Price, Stock_Quantity, Reorder_Level,
        Supplier_Name, Supplier_Lead_Time_Days, Expiry_Date.date(), Storage_Type,
        Manufacturing_Date, Batch_Number, Shelf_Life_Days, Location
    ])

# -----------------------------
# Create DataFrame
# -----------------------------
columns = ['SKU_ID','Product_Name','Category','Sub_Category','Pack_Size',
        'Unit_Cost_Price','Unit_Selling_Price','Stock_Quantity','Reorder_Level',
        'Supplier_Name','Supplier_Lead_Time_Days','Expiry_Date','Storage_Type',
        'Manufacturing_Date','Batch_Number','Shelf_Life_Days','Location']

dummy_df = pd.DataFrame(dummy_data, columns=columns)

# -----------------------------
# Save to CSV
# -----------------------------
dummy_df.to_csv("dummy_inventory_data.csv", index=False)

print("Dummy inventory data generated successfully!")

