#version 1
import streamlit as st
import pandas as pd
import random
from column_finder import column_finder
import streamlit.components.v1 as components

# Fallback product images
images_df = [
    "https://cdn-icons-png.flaticon.com/512/4436/4436481.png",  # Laptop
    "https://cdn-icons-png.flaticon.com/512/2933/2933245.png",  # Phone
    "https://cdn-icons-png.flaticon.com/512/3209/3209265.png",  # Headphones
    "https://cdn-icons-png.flaticon.com/512/2921/2921222.png",  # Camera
    "https://cdn-icons-png.flaticon.com/512/2920/2920664.png",  # Smartwatch
    "https://cdn-icons-png.flaticon.com/512/891/891462.png",    # Shopping Bag
    "https://cdn-icons-png.flaticon.com/512/3081/3081559.png",  # Gift Box
    "https://cdn-icons-png.flaticon.com/512/2920/2920054.png",  # TV
    "https://cdn-icons-png.flaticon.com/512/1040/1040230.png",  # Tablet
    "https://cdn-icons-png.flaticon.com/512/1055/1055646.png",  # Game Controller
    "https://cdn-icons-png.flaticon.com/512/4727/4727422.png",  # Speaker
    "https://cdn-icons-png.flaticon.com/512/2921/2921822.png",  # Keyboard
    "https://cdn-icons-png.flaticon.com/512/2921/2921800.png",  # Mouse
    "https://cdn-icons-png.flaticon.com/512/2921/2921827.png",  # Printer
    "https://cdn-icons-png.flaticon.com/512/1048/1048946.png",  # Washing Machine
    "https://cdn-icons-png.flaticon.com/512/1048/1048948.png",  # Refrigerator
    "https://cdn-icons-png.flaticon.com/512/1048/1048956.png",  # Microwave
    "https://cdn-icons-png.flaticon.com/512/3122/3122929.png",  # Shoes
    "https://cdn-icons-png.flaticon.com/512/892/892458.png",    # T-shirt
    "https://cdn-icons-png.flaticon.com/512/891/891462.png"     # Cart (fallback)
]

def product_dashboard(products_df):
    df = products_df.copy()
    st.title("🛍️ Product Dashboard")
    st.write("This dashboard provides an overview of the products in the store.")

    # Possible column names
    POSSIBLE_PRODUCT_COLS = [
        "product", "product_name", "productname",
        "product_title", "item", "item_name", "model", "model_name"
    ]
    price_columns = [
        "price", "unit_price", "cost", "cost_price", "purchase_price",
        "selling_price", "mrp", "retail_price", "wholesale_price",
        "list_price", "standard_price", "discounted_price", "final_price",
        "net_price", "gross_price", "sale_price", "current_price",
        "base_price", "original_price", "offer_price"
    ]
    desc_possible_names = [
        'description','desc','product_description','item_description','details',
        'info','information','about','long_description','short_description',
        'product_details','item_details','features','product_info','specs',
        'specifications','product_specifications','overview','summary','text',
        'narrative','body','content'
    ]

    # Find matching columns
    product_col = column_finder(df, POSSIBLE_PRODUCT_COLS)
    price_col = column_finder(df, price_columns)
    desc_col = column_finder(df, desc_possible_names)

    # Handle image column
    image_col = [col for col in df.columns if col.lower() in ["image", "images", "img_url", "picture", "photo"]]
    if image_col:
        df["images"] = df[image_col[0]]
    else:
        df["images"] = [random.choice(images_df) for _ in range(len(df))]

    # Build HTML cards
    cards_html = ""
    for _, p in df.iterrows():
        product_name = p[product_col] if product_col else "Unknown Product"
        product_desc = p[desc_col] if desc_col else "No description available"
        product_price = p[price_col] if price_col else "N/A"
        product_img = p["images"]

        cards_html += f"""
            <div class="card">
                <img src="{product_img}" alt="{product_name}" class="card-img">
                <div class="card-info">
                    <h3>{product_name}</h3>
                    <p>{product_desc}</p>
                    <h4 style="color:green;">₹ {product_price}</h4>
                </div>
            </div>
        """

    # Inject CSS + HTML + JS
    html_code = f"""
    <style>
    .carousel-container {{
        position: relative;
        width: 90%;
        margin: 20px auto;
        overflow: hidden;
    }}
    .carousel {{
        display: flex;
        gap: 20px;
        overflow-x: hidden;
        scroll-behavior: smooth;
        padding: 20px 0;
    }}
    .card {{
        width: 220px;
        height: 320px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        flex-shrink: 0;
        text-align: center;
        padding: 15px;
        transition: transform 0.4s ease, box-shadow 0.4s ease, filter 0.4s ease;
        position: relative;
    }}
    .card-img {{
        width: 100px;
        height: 100px;
        object-fit: contain;
        margin: 10px auto;
        display: block;
    }}
    .card-info {{
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    .card:hover {{
        transform: scale(1.1);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        z-index: 10;
    }}
    .card:hover .card-info {{
        opacity: 1;
    }}
    .carousel:hover .card:not(:hover) {{
        filter: blur(3px) brightness(0.7);
    }}
    .nav-btn {{
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(0,0,0,0.6);
        color: white;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        cursor: pointer;
        font-size: 20px;
        z-index: 20;
    }}
    .nav-btn:hover {{
        background: rgba(0,0,0,0.8);
    }}
    .prev-btn {{ left: -10px; }}
    .next-btn {{ right: -10px; }}
    </style>

    <div class="carousel-container">
        <button class="nav-btn prev-btn" onclick="moveCarousel(-1)">&#8592;</button>
        <div class="carousel" id="carousel">
            {cards_html}
        </div>
        <button class="nav-btn next-btn" onclick="moveCarousel(1)">&#8594;</button>
    </div>

    <script>
    let carousel = document.getElementById("carousel");
    let offset = 0;
    function moveCarousel(direction) {{
        offset += direction * 240; // scroll width
        carousel.scrollTo({{ left: offset, behavior: "smooth" }});
    }}
    </script>
    """

    components.html(html_code, height=550, scrolling=False)

#version 2
import streamlit as st
import pandas as pd
import random
from column_finder import column_finder

# Fallback product images
images_df = [
    "https://cdn-icons-png.flaticon.com/512/4436/4436481.png",  # Laptop
    "https://cdn-icons-png.flaticon.com/512/2933/2933245.png",  # Phone
    "https://cdn-icons-png.flaticon.com/512/3209/3209265.png",  # Headphones
    "https://cdn-icons-png.flaticon.com/512/2921/2921222.png",  # Camera
    "https://cdn-icons-png.flaticon.com/512/2920/2920664.png",  # Smartwatch
    "https://cdn-icons-png.flaticon.com/512/891/891462.png",    # Shopping Bag
    "https://cdn-icons-png.flaticon.com/512/3081/3081559.png",  # Gift Box
    "https://cdn-icons-png.flaticon.com/512/2920/2920054.png",  # TV
    "https://cdn-icons-png.flaticon.com/512/1040/1040230.png",  # Tablet
    "https://cdn-icons-png.flaticon.com/512/1055/1055646.png",  # Game Controller
    "https://cdn-icons-png.flaticon.com/512/4727/4727422.png",  # Speaker
    "https://cdn-icons-png.flaticon.com/512/2921/2921822.png",  # Keyboard
    "https://cdn-icons-png.flaticon.com/512/2921/2921800.png",  # Mouse
    "https://cdn-icons-png.flaticon.com/512/2921/2921827.png",  # Printer
    "https://cdn-icons-png.flaticon.com/512/1048/1048946.png",  # Washing Machine
    "https://cdn-icons-png.flaticon.com/512/1048/1048948.png",  # Refrigerator
    "https://cdn-icons-png.flaticon.com/512/1048/1048956.png",  # Microwave
    "https://cdn-icons-png.flaticon.com/512/3122/3122929.png",  # Shoes
    "https://cdn-icons-png.flaticon.com/512/892/892458.png",    # T-shirt
    "https://cdn-icons-png.flaticon.com/512/891/891462.png"     # Cart (fallback)
]

def product_dashboard(products_df):
    df = products_df.copy()
    st.title("🛍️ Product Dashboard")
    st.write("This dashboard provides an overview of the products in the store.")

    # Possible column names
    POSSIBLE_PRODUCT_COLS = [
        "product", "product_name", "productname",
        "product_title", "item", "item_name", "model", "model_name"
    ]
    price_columns = [
        "price", "unit_price", "cost", "cost_price", "purchase_price",
        "selling_price", "mrp", "retail_price", "wholesale_price",
        "list_price", "standard_price", "discounted_price", "final_price",
        "net_price", "gross_price", "sale_price", "current_price",
        "base_price", "original_price", "offer_price"
    ]
    desc_possible_names = [
        'description','desc','product_description','item_description','details',
        'info','information','about','long_description','short_description',
        'product_details','item_details','features','product_info','specs',
        'specifications','product_specifications','overview','summary','text',
        'narrative','body','content'
    ]

    # Find matching columns
    product_col = column_finder(df, POSSIBLE_PRODUCT_COLS)
    price_col = column_finder(df, price_columns)
    desc_col = column_finder(df, desc_possible_names)

    # Handle image column
    image_col = [col for col in df.columns if col.lower() in ["image", "images", "img_url", "picture", "photo"]]
    if image_col:
        df['images'] = df[image_col[0]]
    else:
        df['images'] = [random.choice(images_df) for _ in range(len(df))]

    # Build HTML cards
    cards_html = ""
    for _, p in df.iterrows():
        product_name = p[product_col] if product_col else "Unknown Product"
        product_desc = p[desc_col] if desc_col else "No description available"
        product_price = p[price_col] if price_col else "N/A"
        product_img = p['images']

        cards_html += f"""
            <div class="card">
                <img src="{product_img}" alt="{product_name}" class="card-img">
                <h3>{product_name}</h3>
                <p>{product_desc}</p>
                <h4 style="color:green;">₹ {product_price}</h4>
            </div>
        """

    # Inject CSS + HTML + JS
    html_code = f"""
    <style>
    .carousel {{
      display: flex;
      align-items: center;
      gap: 20px;
      overflow-x: hidden;
      margin: 30px auto;
      width: 90%;
    }}
    .card {{
      width: 220px;
      height: 320px;
      background: white;
      border-radius: 15px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
      flex-shrink: 0;
      text-align: center;
      padding: 15px;
    }}
    .card-img {{
      width: 80px;
      height: 80px;
      object-fit: contain;
      margin: 10px auto;
      display: block;
    }}
    .controls {{
      text-align: center;
      margin-top: 20px;
    }}
    button {{
      padding: 10px 20px;
      margin: 0 10px;
      border: none;
      border-radius: 8px;
      background: #4CAF50;
      color: white;
      font-size: 16px;
      cursor: pointer;
    }}
    button:hover {{
      background: #45a049;
    }}
    </style>

    <div class="carousel" id="carousel">
        {cards_html}
    </div>

    <div class="controls">
      <button onclick="moveCarousel(-1)">⬅️ Prev</button>
      <button onclick="moveCarousel(1)">Next ➡️</button>
    </div>

    <script>
    let carousel = document.getElementById("carousel");
    let offset = 0;
    function moveCarousel(direction) {{
      offset += direction * 240;
      carousel.scrollTo({{ left: offset, behavior: "smooth" }});
    }}
    </script>
    """

    # ✅ use components.html instead of st.markdown
    st.components.v1.html(html_code, height=500, scrolling=False)