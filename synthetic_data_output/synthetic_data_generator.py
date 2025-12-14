#!/usr/bin/env python3
"""
synthetic_data_batch_generator.py

Generates 20 full e-commerce datasets in one run.
- Uses 11 real product catalogues (provided paths).
- Fakes the rest of the product catalogues.
- Includes edge cases and randomized column names for robust column detection.
- Generates customers, inventory, transactions, returns/cancellations, sales aggregates.
- CLI + optional YAML/JSON config.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta
from math import sin, pi

import numpy as np
import pandas as pd
from faker import Faker

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_config(path):
    if not path:
        return {}
    if path.lower().endswith(('.yml', '.yaml')):
        if not YAML_AVAILABLE:
            raise RuntimeError('pyyaml required for YAML config. Install: pip install pyyaml')
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f) or {}

def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"[INFO] Saved {path}")

def maybe_tqdm(iterable, enable=True, **kwargs):
    return tqdm(iterable, **kwargs) if enable and TQDM_AVAILABLE else iterable

def randomize_colnames(df, keywords):
    """
    Randomize column names but keep keywords intact for robust column finder testing.
    """
    new_cols = []
    for c in df.columns:
        key_hit = next((k for k in keywords if k in c.lower()), None)
        if key_hit:
            # Slightly change the name but keep keyword
            new_cols.append(f"{c}_{random.randint(1,99)}")
        else:
            new_cols.append(f"col_{random.randint(1000,9999)}")
    df.columns = new_cols
    return df

# -----------------------------
# Product Catalogue Generator
# -----------------------------

def load_or_generate_products(fake, n_products, input_paths=None, seed=42, show_progress=True):
    """
    Load multiple catalogues or generate synthetic products.
    Returns a list of product dataframes.
    """
    input_paths = input_paths or []
    product_dfs = []
    rng = np.random.default_rng(seed)
    categories = {
        'electronics': ['mobile', 'laptop', 'accessory'],
        'home': ['kitchen', 'furniture', 'decor'],
        'clothing': ['men', 'women', 'kids'],
        'beauty': ['skincare', 'makeup'],
        'grocery': ['staples', 'snacks', 'beverages']
    }

    keywords = ['product_id','product_name','category','subcategory','unit_price','description']

    for idx, path in enumerate(maybe_tqdm(input_paths, show_progress, desc="Loading product CSVs")):
        if os.path.exists(path):
            df = pd.read_csv(path,on_bad_lines='skip')
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
            # normalize required columns
            for col in ['product_id','product_name','category','unit_price']:
                if col not in df.columns:
                    df[col] = f"dummy_{col}_{idx}"
            df = randomize_colnames(df, keywords)
            product_dfs.append(df)
        else:
            print(f"[WARN] CSV not found: {path}")

    # Generate remaining synthetic catalogues to reach n_products
    total_loaded = len(product_dfs)
    while total_loaded < n_products:
        n = 50  # products per synthetic catalogue
        rows = []
        for i in range(n):
            pid = f'p{100000 + total_loaded*100 + i}'
            cat = rng.choice(list(categories.keys()))
            sub = rng.choice(categories[cat])
            price = round(abs(rng.normal(50, 40))+1, 2)
            name = f"{sub.title()} {fake.word().title()} {i}"
            desc = fake.sentence(nb_words=10)
            rows.append({'product_id': pid, 'product_name': name, 'category': cat,
                         'subcategory': sub, 'unit_price': price, 'description': desc})
        df_syn = pd.DataFrame(rows)
        df_syn = randomize_colnames(df_syn, keywords)
        product_dfs.append(df_syn)
        total_loaded += 1

    return product_dfs[:n_products]

# -----------------------------
# Customer Generator
# -----------------------------

def generate_customers(n_customers, fake, seed=42, show_progress=True):
    rng = np.random.default_rng(seed)
    segments = ['bronze','silver','gold','platinum']
    customer_rows = []
    for i in maybe_tqdm(range(n_customers), show_progress, desc="Customers"):
        cid = f'c{100000+i}'
        name = fake.name()
        email = fake.free_email()
        phone = fake.msisdn()[:12]
        loyalty = float(round(rng.random()*100,2))
        segment = rng.choice(segments, p=[0.5,0.3,0.15,0.05])
        pref_size = int(rng.integers(1,3))
        preferred_categories = rng.choice(['electronics','home','clothing','beauty','grocery'], size=pref_size, replace=False).tolist()
        customer_rows.append({'customer_id':cid,'name':name,'email':email,'phone':phone,
                              'loyalty_score':loyalty,'customer_segment':segment,
                              'preferred_categories':'|'.join(preferred_categories)})
    return pd.DataFrame(customer_rows)

# -----------------------------
# Inventory Generator
# -----------------------------
def find_column(df, keywords, default=None):
    """
    Find a column in df that matches any of the keywords (case-insensitive).
    Returns the column name or default if not found.
    """
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return default

def generate_inventory(product_df, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Find product ID column
    pid_col = find_column(product_df, ['product_id', 'id', 'productid', 'product code', 'sku'])
    if pid_col is None:
        raise ValueError("No product ID column found in product dataframe")
    
    inventory_data = []
    for idx, row in product_df.iterrows():
        pid = row[pid_col]
        qty = np.random.randint(0, 100)
        inventory_data.append({'product_id': pid, 'stock_qty': qty})
    
    inventory_df = pd.DataFrame(inventory_data)
    return inventory_df


# -----------------------------
# Transactions Generator
# -----------------------------

def generate_transactions(customers_df, product_df, start_date, end_date, seed=42, base_tx_per_day=200, show_progress=True):
    rng = np.random.default_rng(seed)
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    days = (end-start).days +1
    product_ids = product_df['product_id'].tolist()
    price_map = dict(zip(product_df['product_id'],product_df.get('unit_price',np.ones(len(product_df)))))
    customer_ids = customers_df['customer_id'].tolist()
    tx_rows=[]
    tx_id_counter = 2000000
    payment_methods = ['credit_card','upi','debit_card','cod','netbanking','wallet']

    for day_offset in maybe_tqdm(range(days), show_progress, desc="Days"):
        date = start + pd.Timedelta(days=int(day_offset))
        expected_tx = max(1,int(base_tx_per_day*rng.normal(1.0,0.1)))
        for _ in range(expected_tx):
            tx_id = f'tx{tx_id_counter}'
            tx_id_counter +=1
            customer = rng.choice(customer_ids)
            pid = rng.choice(product_ids)
            qty = int(max(1,rng.poisson(1.2)))
            unit_price = round(float(price_map.get(pid,10))*(1+rng.normal(0,0.02)),2)
            total = round(qty*unit_price,2)
            pay = rng.choice(payment_methods)
            tx_time = date + pd.Timedelta(seconds=int(rng.integers(0,86400)))
            tx_rows.append({'transaction_id':tx_id,'customer_id':customer,'product_id':pid,
                            'quantity':qty,'unit_price':unit_price,'total_amount':total,
                            'transaction_date':tx_time,'payment_method':pay})
    tx_df = pd.DataFrame(tx_rows)
    tx_df = tx_df.merge(product_df[['product_id','product_name','category']], on='product_id', how='left')
    return tx_df

# -----------------------------
# Returns & Cancellations
# -----------------------------

def generate_returns_and_cancellations(tx_df, seed=42, show_progress=True):
    rng = np.random.default_rng(seed)
    cancel_prob = 0.01
    return_prob = 0.03
    cancel_rows=[]
    return_rows=[]
    for _, row in maybe_tqdm(tx_df.iterrows(), show_progress, desc="Returns/Cancels"):
        if rng.random()<cancel_prob:
            cancel_date = row['transaction_date'] + pd.Timedelta(days=int(rng.integers(0,3)))
            cancel_rows.append({'transaction_id':row['transaction_id'],'cancel_date':cancel_date})
        elif rng.random()<return_prob:
            ret_date = row['transaction_date'] + pd.Timedelta(days=int(rng.integers(2,30)))
            return_amount = round(row['total_amount']*rng.uniform(0.5,1.0),2)
            return_rows.append({'transaction_id':row['transaction_id'],'return_date':ret_date,'refund_amount':return_amount})
    cancels = pd.DataFrame(cancel_rows)
    returns = pd.DataFrame(return_rows)
    return returns, cancels

# -----------------------------
# Main Batch Flow
# -----------------------------
def generate_batch(args):
    # Use CLI args or defaults
    n_datasets = args.n_datasets
    n_customers = args.n_customers
    start_date = args.start
    end_date = args.end
    dataset_dir = args.out_dir or r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output"
    seed = args.seed

    # Default product catalogues if none provided
    DEFAULT_PRODUCT_CATALOGUES = [
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\amazon_200_products_catalog.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\amul_product_catalogue.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\branded_product_catalogue.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\erp_products_catalogue.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\meesho_product_catalogue.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\nykaa_product_catalogue.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\shopsy_product_catalogue.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\streaming_products_catalog.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\tira_product_catalogue.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\walmart_product_catalogue.csv",
        r"C:\Users\palya\Desktop\Demand-Cast\synthetic_data_output\dmart_product_catalogue.csv"
    ]
    product_catalogues = args.product_catalogues if args.product_catalogues else DEFAULT_PRODUCT_CATALOGUES

    # Seed random generators
    Faker.seed(seed)
    fake = Faker()
    random.seed(seed)
    np.random.seed(seed)

    # Ensure output directory exists
    ensure_dir(dataset_dir)

    for d in range(n_datasets):
        print(f"\n=== Generating Dataset {d+1} ===")
        out_path = os.path.join(dataset_dir, f"dataset_{d+1}")
        ensure_dir(out_path)

        # Load or generate products
        product_dfs = load_or_generate_products(fake, n_products=20, input_paths=product_catalogues, seed=seed+d)
        product_df = pd.concat(product_dfs, ignore_index=True)

        # Generate customers
        customers_df = generate_customers(n_customers, fake, seed=seed+d)

        # Generate inventory
        inventory_df = generate_inventory(product_df, seed=seed+d)

        # Generate transactions
        tx_df = generate_transactions(customers_df, product_df, start_date, end_date, seed=seed+d)

        # Generate returns and cancellations
        returns_df, cancels_df = generate_returns_and_cancellations(tx_df, seed=seed+d)

        # Save all CSVs
        save_csv(product_df, os.path.join(out_path, 'product_catalogue.csv'))
        save_csv(customers_df, os.path.join(out_path, 'customers.csv'))
        save_csv(inventory_df, os.path.join(out_path, 'inventory.csv'))
        save_csv(tx_df, os.path.join(out_path, 'transactions.csv'))
        save_csv(returns_df, os.path.join(out_path, 'returns.csv'))
        save_csv(cancels_df, os.path.join(out_path, 'cancellations.csv'))

    print("\nAll datasets generated successfully!")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Generate multiple synthetic e-commerce datasets with edge cases')
    p.add_argument('--n_datasets', type=int, default=20, help='Number of datasets to generate')
    p.add_argument('--n_customers', type=int, default=500, help='Number of customers per dataset')
    p.add_argument('--out_dir', type=str, default='./synthetic_datasets', help='Output folder')
    p.add_argument('--product_catalogues', nargs='*', help='Paths to real product catalogue CSVs')
    p.add_argument('--start', type=str, default='2025-01-01', help='Start date for transactions')
    p.add_argument('--end', type=str, default='2025-10-10', help='End date for transactions')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()

# -----------------------------
# Entry
# -----------------------------

if __name__=="__main__":
    args = parse_args()
    generate_batch(args)
    print("\nAll datasets generated successfully!")
    
