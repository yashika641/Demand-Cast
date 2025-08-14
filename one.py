import random
import pandas as pd
from datetime import datetime, timedelta

# Category, subcategory, pack size, packaging, storage, shelf life
cat_structure = {
    'Dairy Milk': {
        'Gold': [('1L', 'Tetra Pack'), ('500ml', 'Tetra Pack'), ('200ml', 'Tetra Pack')],
        'Taaza': [('1L', 'Tetra Pack'), ('500ml', 'Tetra Pack')],
        'Shakti': [('500ml', 'Tetra Pack'), ('1L', 'Tetra Pack')],
        'Slim': [('1L', 'Tetra Pack')],
        'Lactose-Free': [('1L', 'Tetra Pack')],
    },
    'Butter': {
        'Classic': [('500g', 'Carton'), ('100g', 'Carton')],
        'Lite': [('250g', 'Carton')],
        'Garlic': [('100g', 'Carton')],
        'Salted': [('500g', 'Carton')],
        'Unsalted': [('500g', 'Carton')],
    },
    'Cheese': {
        'Block': [('200g', 'Box'), ('500g', 'Box')],
        'Pizza': [('100g', 'Box'), ('200g', 'Box')],
        'Mozzarella': [('200g', 'Box')],
        'Vegan': [('200g', 'Box')],
        'Keto Cheese': [('200g', 'Box')],
    },
    'Paneer': {
        'Malai': [('200g', 'Pouch'), ('1kg', 'Pouch')],
        'Low-Fat': [('200g', 'Pouch')],
        'Protein-Enriched': [('200g', 'Pouch')],
    },
    'Ice Creams & Desserts': {
        'Vanilla': [('500ml', 'Tub'), ('1L', 'Tub')],
        'Chocolate': [('500ml', 'Tub')],
        'Kesar Pista': [('500ml', 'Tub')],
        'Vegan Ice Cream': [('500ml', 'Tub')],
        'Kulfi': [('60ml', 'Stick'), ('100ml', 'Stick')],
        'Frozen Yogurt': [('100ml', 'Cup')],
        'Dessert Cups': [('100ml', 'Cup')],
    },
    'Yogurt': {
        'Probiotic': [('100g', 'Cup'), ('500g', 'Cup')],
        'Masti': [('500g', 'Cup')],
        'Greek Yogurt': [('100g', 'Cup')],
        'High Protein Yogurt': [('100g', 'Cup')],
    },
    'Chocolate & Confectionery': {
        'Dark': [('150g', 'Box')],
        'Milk': [('150g', 'Box')],
        'Fruit & Nut': [('150g', 'Box')],
        'Chocolate Syrup': [('200ml', 'Bottle')],
    },
    'Beverages': {
        'Kool Café': [('200ml', 'Bottle')],
        'Kool Koko': [('200ml', 'Bottle')],
        'Kool Badam': [('200ml', 'Bottle')],
        'Turmeric Milk': [('200ml', 'Bottle')],
        'Buttermilk': [('200ml', 'Pouch'), ('1L', 'Pouch')],
        'Lassi': [('200ml', 'Pouch')],
        'Flavored Milk': [('200ml', 'Bottle')],
    },
    'Protein Products': {
        'Whey Protein': [('500g', 'Jar')],
        'High Protein Milk': [('200ml', 'Bottle')],
        'Protein Bar': [('60g', 'Box')],
        'Protein Shakes': [('200ml', 'Bottle')],
    },
    'Vegan & Plant-Based': {
        'Almond Milk': [('1L', 'Tetra Pack')],
        'Soy Milk': [('1L', 'Tetra Pack')],
        'Oat Milk': [('1L', 'Tetra Pack')],
        'Vegan Cheese': [('200g', 'Box')],
    },
    'Functional Products': {
        'Immunity Booster Drink': [('200ml', 'Bottle')],
        'Keto Cheese': [('200g', 'Box')],
        'Haldi Doodh Powder': [('200g', 'Jar')],
        'Probiotic Drink': [('200ml', 'Bottle')],
        'Sugar-Free Products': [('200g', 'Box')],
    },
    'Value-Added Dairy': {
        'Keto-friendly Milk': [('1L', 'Tetra Pack')],
        'Sugar-Free Ghee': [('500ml', 'Jar')],
        'Fortified Milk': [('1L', 'Tetra Pack')],
        'Omega-3 Enriched Milk': [('1L', 'Tetra Pack')],
    },
}

# Price ranges (INR) by category
price_map = {
    'Dairy Milk': (25, 80),
    'Butter': (40, 180),
    'Cheese': (80, 300),
    'Paneer': (60, 400),
    'Ice Creams & Desserts': (30, 250),
    'Yogurt': (20, 100),
    'Chocolate & Confectionery': (40, 200),
    'Beverages': (20, 60),
    'Protein Products': (40, 600),
    'Vegan & Plant-Based': (60, 250),
    'Functional Products': (40, 300),
    'Value-Added Dairy': (50, 200),
}

# Storage and shelf life
storage_map = {
    'Dairy Milk': ('Refrigerated', 7, 15),
    'Butter': ('Refrigerated', 90, 180),
    'Cheese': ('Refrigerated', 90, 180),
    'Paneer': ('Refrigerated', 7, 15),
    'Ice Creams & Desserts': ('Frozen', 180, 365),
    'Yogurt': ('Refrigerated', 10, 30),
    'Chocolate & Confectionery': ('Ambient', 180, 365),
    'Beverages': ('Ambient', 30, 120),
    'Protein Products': ('Ambient', 180, 365),
    'Vegan & Plant-Based': ('Ambient', 30, 120),
    'Functional Products': ('Ambient', 30, 180),
    'Value-Added Dairy': ('Ambient', 30, 180),
}

# Suppliers
suppliers = [
    'Amul Dairy Plant Anand', 'Amul Dairy Plant Mumbai', 'Amul Dairy Plant Kolkata',
    'Amul Dairy Plant Delhi', 'Amul Dairy Plant Bengaluru', 'Amul Dairy Plant Hyderabad'
]

# Special tags
special_tags = ['Limited Edition', 'Sugar-Free', 'Vegan', 'Keto-Friendly', 'High Protein', 'Summer Special', 'Diwali Pack', 'Winter Special', '']

# Helper for launch date
def random_launch():
    days_ago = random.randint(0, 5*365)
    return (datetime.today() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

# Helper for popularity by category
pop_map = {
    'Dairy Milk': (60, 100),
    'Butter': (50, 90),
    'Cheese': (40, 85),
    'Paneer': (40, 80),
    'Ice Creams & Desserts': (60, 100),
    'Yogurt': (40, 80),
    'Chocolate & Confectionery': (60, 100),
    'Beverages': (50, 90),
    'Protein Products': (30, 80),
    'Vegan & Plant-Based': (20, 70),
    'Functional Products': (20, 70),
    'Value-Added Dairy': (20, 70),
}

# Seasonal subcategories
seasonal_subcats = ['Kulfi', 'Dessert Cups', 'Summer Special', 'Diwali Pack', 'Winter Special', 'Mango', 'Holi', 'Christmas']

# Generate SKUs
rows = []
sku_counter = 2000
for cat, subcats in cat_structure.items():
    for subcat, pack_opts in subcats.items():
        n_skus = random.randint(2, 5)
        for i in range(n_skus):
            pack, packaging = random.choice(pack_opts)
            sku = f'AM-{sku_counter}'
            product_name = f'Amul {subcat} {cat} {pack}'
            min_price, max_price = price_map[cat]
            price1 = round(random.uniform(min_price, max_price-10), 2)
            price2 = round(price1 + random.uniform(5, 30), 2)
            supplier = random.choice(suppliers)
            storage, shelf_min, shelf_max = storage_map[cat]
            shelf_life = random.randint(shelf_min, shelf_max)
            # Seasonal
            seasonal_flag = 'Yes' if (subcat in seasonal_subcats or random.random() < 0.07) else 'No'
            # Special tag
            tag = ''
            if seasonal_flag == 'Yes':
                tag = random.choice(['Summer Special', 'Diwali Pack', 'Winter Special'])
            elif 'Sugar-Free' in subcat or 'Keto' in subcat or 'Vegan' in subcat or 'Protein' in subcat:
                tag = subcat
            elif random.random() < 0.08:
                tag = random.choice([t for t in special_tags if t])
            # Launch date
            launch = random_launch()
            # Discontinued
            discontinued = 'Yes' if random.random() < 0.05 else 'No'
            # Popularity
            pop_min, pop_max = pop_map[cat]
            pop_score = random.randint(pop_min, pop_max)
            rows.append([
                sku, product_name, cat, subcat, pack, packaging, price1, price2, supplier, storage, shelf_life,
                seasonal_flag, tag, launch, discontinued, pop_score
            ])
            sku_counter += 1
            if len(rows) >= 300:
                break
        if len(rows) >= 300:
            break
    if len(rows) >= 300:
        break

# DataFrame
columns = [
    'SKU_ID', 'Product_Name', 'Category', 'Sub_Category', 'Pack_Size', 'Packaging_Type',
    'Min_Price', 'Max_Price', 'Supplier', 'Storage_Type', 'Shelf_Life_Days',
    'Seasonal_Flag', 'Special_Tag', 'Launch_Date', 'Discontinued_Flag', 'Popularity_Score'
]
df = pd.DataFrame(rows, columns=columns)

# Save to CSV
cat_path = r'C:\Users\palya\Desktop\DemandCast\Demand-Cast\amul_products_catalogue.csv'
df.to_csv(cat_path, index=False)

# Show sample and stats
print(df.head())
print(df.dtypes)
print(df.shape)
df['Category'].value_counts()