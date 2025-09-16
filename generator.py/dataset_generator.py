import random
import pandas as pd
from faker import Faker
import re

fake = Faker()

# Amul SKUs grouped into categories
categories = {
    "Dairy Products": [
        "Amul Gold Milk 1L", "Amul Taaza Milk 1L", "Amul Slim & Trim Milk 500ml",
        "Amul Lactose-Free Milk 1L", "Amul Fresh Cream 250ml", "Amul Masti Buttermilk 500ml",
        "Amul Spiced Buttermilk 500ml", "Amul Paneer 200g", "Amul Malai Paneer 1kg",
        "Amul Mithai Mate Condensed Milk 400g"
    ],
    "Butter & Spreads": [
        "Amul Butter 100g", "Amul Butter 500g", "Amul Lite Spread 250g",
        "Amul Garlic & Herbs Butter 100g", "Amul Peanut Spread Crunchy 200g",
        "Amul Peanut Spread Creamy 200g", "Amul Cheese Spread Yummy Plain 200g",
        "Amul Cheese Spread Red Chilli Flakes 200g", "Amul Cheese Spread Garlic 200g"
    ],
    "Cheese & Cheese Products": [
        "Amul Cheese Slices 200g", "Amul Processed Cheese Cube 200g", "Amul Cheese Block 500g",
        "Amul Mozzarella Cheese 200g", "Amul Pizza Cheese 1kg", "Amul Gouda Cheese 200g",
        "Amul Emmental Cheese 200g", "Amul Cheddar Cheese 200g"
    ],
    "Ice Creams": [
        "Amul Vanilla Ice Cream 1L", "Amul Chocolate Ice Cream 1L", "Amul Butter Scotch Ice Cream 1L",
        "Amul Rajbhog Ice Cream 1L", "Amul Choco Chips Ice Cream 1L",
        "Amul Kulfi 60ml", "Amul Strawberry Magic 750ml", "Amul Cassata Slice 120ml",
        "Amul Dark Chocolate Ice Cream 1L", "Amul Kesar Pista Ice Cream 1L"
    ],
    "Chocolates": [
        "Amul Milk Chocolate 150g", "Amul Dark Chocolate 150g", "Amul Fruit & Nut Chocolate 150g",
        "Amul Mystic Mocha Chocolate 150g", "Amul India Twilight Tryst 150g",
        "Amul Almond Bar 40g", "Amul Chocozoo Kids Pack 250g"
    ],
    "Beverages": [
        "Amul Kool Flavoured Milk Bottle 200ml", "Amul Kool Café 200ml", "Amul Kool Koko 200ml",
        "Amul Kool Rose Milk 200ml", "Amul Kool Elaichi Milk 200ml", "Amul Kool Saffron Milk 200ml",
        "Amul Pro Milk Drink 200g", "Amul Tru Orange Juice 200ml", "Amul Tru Apple Juice 200ml"
    ],
    "Ghee & Cooking Essentials": [
        "Amul Pure Ghee 500ml", "Amul Cow Ghee 1L", "Amul Buffalo Ghee 1L",
        "Amul Cooking Butter 500g", "Amul White Butter 500g"
    ],
    "Probiotic & Health Range": [
        "Amul Probiotic Dahi 400g", "Amul Masti Dahi 400g", "Amul Greek Yogurt Blueberry 100g",
        "Amul Greek Yogurt Strawberry 100g", "Amul Greek Yogurt Mango 100g"
    ],
    "Specialty Products": [
        "Amul Mithai Gulab Jamun Tin 1kg", "Amul Rasmalai 500g", "Amul Basundi 1L",
        "Amul Shrikhand Mango 500g", "Amul Shrikhand Elaichi 500g", "Amul Shrikhand Kesar 500g"
    ],
    "Bakery & Frozen": [
        "Amul Pizza Base 200g", "Amul Frozen Paratha 400g", "Amul Frozen Paneer Tikka 500g",
        "Amul Frozen Cheese Corn Nuggets 400g", "Amul Frozen Veggie Burger Patty 500g"
    ],
    "UHT & Long Life": [
        "Amul Taaza UHT Milk 1L", "Amul Gold UHT Milk 1L", "Amul Slim UHT Milk 1L",
        "Amul Mithai Mate UHT 200g", "Amul Cream Tetra Pack 250ml"
    ],
    "Others": [
        "Amul Camel Milk 500ml", "Amul A2 Cow Milk 1L", "Amul Camel Milk Chocolate 150g",
        "Amul Organic Ghee 500ml", "Amul Butter Cookies 200g",
        "Amul Happy Treats Veggie Stix 400g", "Amul Ice Cream Cake 1kg",
        "Amul Kool Café Can 250ml", "Amul Cheese Tin 500g"
    ],
    # ✅ New categories added
    "Snacks & Savories": [
        "Amul Cheesy Corn Puffs 150g", "Amul Spicy Potato Chips 200g", "Amul Masala Peanuts 250g",
        "Amul Veggie Sticks 100g", "Amul Cheese Crackers 120g"
    ],
    "Bakery Items": [
        "Amul Chocolate Brownie 100g", "Amul Butter Croissant 80g", "Amul Vanilla Muffin 90g",
        "Amul Cheese Danish 120g", "Amul Chocolate Donut 80g"
    ],
    "Health & Nutrition": [
        "Amul Protein Shake Vanilla 200ml", "Amul Protein Shake Chocolate 200ml", "Amul Multigrain Dahi 200g",
        "Amul Low Fat Yogurt 100g", "Amul Fortified Milk 1L"
    ],
    "Beverages (New Flavors)": [
        "Amul Mango Lassi 250ml", "Amul Strawberry Lassi 250ml", "Amul Buttermilk Masala 200ml",
        "Amul Choco Milk 200ml", "Amul Cold Coffee 200ml"
    ]
}


# Storage types
storage_types = ["Cold Storage", "Ambient", "Frozen", "Refrigerated", "Room Temperature", "Chilled", "Hot Storage", "hot & Cold Storage"]

# Suppliers
suppliers = ["Amul Dairy Anand", "GCMMF Ltd.", "Local Distributor",'New Jindal Enterprises', 'Kumar Traders', 'Sharma Suppliers', 'Verma Distributors', 'Agarwal Enterprises', 'Singh Wholesale', 'Patel Suppliers','Jaanvi Foods','Aditya Food Services Company','Spectrum Dairies Pvt Ltd']

# Special Tags
special_tags = ['Best Seller','New Arrival','Limited Edition','Health Friendly','Popular Choice','Classic Favorite','Budget Friendly','Premium'
                ,'Organic','None','Seasonal Special','Kids’ Favorite','Dairy Delight','Creamy Choice','Low Fat','Sugar-Free'
                ,'Fortified','Gift Pack','Party Pack','Bulk Pack','Eco Friendly','Chef’s Choice','Trendy Pick','Quick Snack','Everyday Essential']

# Function to extract packaging size from product name
def extract_packaging_size(product_name):
    match = re.search(r'\d+\s?(g|kg|ml|L)', product_name, re.IGNORECASE)
    if match:
        return match.group()
    else:
        return None  # fallback if no size is found

def get_price(category, packaging):
    """
    Generate realistic price based on category and packaging.
    """
    # Base ranges per category (in Rs)
    price_ranges = {
        "Dairy Products": (30, 100),
        "Butter & Spreads": (40, 200),
        "Cheese & Cheese Products": (100, 500),
        "Ice Creams": (50, 300),
        "Chocolates": (20, 150),
        "Beverages": (20, 120),
        "Ghee & Cooking Essentials": (200, 800),
        "Probiotic & Health Range": (40, 250),
        "Specialty Products": (50, 400),
        "Bakery & Frozen": (30, 250),
        "UHT & Long Life": (30, 150),
        "Others": (20, 500)
    }

    low, high = price_ranges.get(category, (20, 600))
    
    # Optionally adjust price according to packaging size
    if packaging:
        match = re.match(r"(\d+)\s?(g|kg|ml|L)", packaging, re.IGNORECASE)
        if match:
            qty = int(match.group(1))
            unit = match.group(2).lower()
            if unit == "kg" or unit == "l":
                qty *= 1000  # Convert kg/L to grams/ml for calculation
            # scale price roughly with size
            scale_factor = qty / 100  # 100g/ml as base
            price = int(low + (high - low) * random.random() * scale_factor)
            # ensure price is within range
            price = max(low, min(price, high))
            return price

    return random.randint(low, high)

# In your record generation, replace the price line with:

def generate_catalogue():
    data = []
    used_products = set()

    for category, products in categories.items():
        for product in products:
            if product not in used_products:  # ensure uniqueness
                packaging = extract_packaging_size(product)

                record = {
                    "SKU_ID": fake.unique.ean(length=8),
                    "Product_Name": product,
                    "Category": category,
                    "Price": get_price(category, packaging),
                    "Packaging_Size": packaging,
                    "Supplier": random.choice(suppliers),
                    "Storage_Type": random.choice(storage_types),
                    "Shelf_Life_Days": random.randint(5, 365),
                    "Seasonal_Flag": random.choice(["Yes", "No"]),
                    "Special_Tag": random.choice(special_tags),
                    "Launch_Date": fake.date_between(start_date="-5y", end_date="today"),
                    "Discontinued_Flag": random.choice(["Yes", "No"]),
                    "Popularity_Score": round(random.uniform(1, 5), 2)
                }
                data.append(record)
                used_products.add(product)

    df = pd.DataFrame(data)
    return df

# Example usage
if __name__ == "__main__":
    df = generate_catalogue()
    print(df.head())
    df.to_csv("amul_product_catalogue.csv", index=False)
