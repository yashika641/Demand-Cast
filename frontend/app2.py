import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from inventory_models_functions import lead_time_dashboard
from dashboards.sales_dashboard import sales_dashboard
from dashboards.inventory_dashboard import inventory_dashboard
from dashboards.product_dashboard import product_dashboard
from dashboards.customer_dashboard import customer_dashboard
from dashboards.promtions_dashboard import promotion_dashboard
from streamlit_cards import card


if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to_next_page(page_name):
    st.session_state.page = page_name
    st.rerun()
    st.markdown("""
        <script>
            window.scrollTo(0, 0);
        </script>
    """, unsafe_allow_html=True)


# ----------------set page config----------------#
st.set_page_config(page_title="DemandCast", page_icon=":bar_chart:", layout="wide")
if st.session_state.page == "home":
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Playfair+Display:wght@700&family=Raleway:wght@400;600&display=swap" rel="stylesheet">

    <style>
    /* Navbar */
    .navbar {
        position: fixed;
        margin-top: 40px;
        top: 0;
        left: 0;
        width: 100%;
        height: 90px;
        backdrop-filter: blur(10px); 
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 30px;
        z-index: 9999;
        transition: top 0.4s ease;
        
    }
    .navbar-logo img {
        height: 60px;
        width: 100px;
        border-radius: 8px;
        margin-right: 10px;
    }
    .navbar-logo span {
        color: #ffffff;
        font-weight: bold;
        font-size: 32px;
        font-family: 'Playfair Display', serif;
    }
    .navbar-menu a {
        color: #ffffff;
        margin-left: 30px;
        font-size: 18px;
        font-family: 'Raleway', sans-serif;
        text-decoration: none;
    }
    .navbar-menu a:hover {
        color: #00ffff;
    }

    .stApp {
        background: transparent !important;
    }
    video.bg-video {
        position: fixed;     /* allow scrolling */
        top: 0;
        left: 0;
        display:flex;
        align-items: center;
        justify-content: center;    
        width: 110%;           /* fill full width */
        height: 100%;           /* scale height proportionally */
        min-height: 100%;       /* ensures it covers vertically */
        object-fit: cover;    /* show entire video (no cropping) */
        z-index: -1;            /* push behind content */
        background: black;  
        background-position:center;
        background-size: cover;
    }


    /* Sections general (half width + left aligned text) */
    .section {
        max-width: 100%;
        margin: 60px auto;
        text-align: center;
    }

    /* First section forced left */
    .section.first {
        max-width: 50%;
        margin: 60px 0 60px 60px;  /* pushed to left */
        text-align: left;
    }

    .section h2 {
        font-family: 'Playfair Display', serif;
        font-size: 48px;
        color: #00ffff;
        margin-bottom: 20px;
        
    }

    .section p {
        font-size: 20px;
        font-family: 'Raleway', sans-serif;
        color: #e0f7fa;
        margin-bottom: 30px;
        
    }

    /* Interactive Cards */
    .pillars {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
    }

    .pillar {
        background: rgba(0,0,0,0.5);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        transition: all 0.4s ease;
        cursor: pointer;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        backdrop-filter: blur(6px);
    }
    
    .pillar img {
        height: 70px;
        width: auto;
        border-radius: 8px;
        margin-bottom: 10px;
        margin-right: 15px;
    }


    .pillar:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 12px 30px rgba(0,255,255,0.6);
        background: rgba(0,0,0,0.7);
    }

    .pillar h3 {
        font-size: 35px;
        color: #80deea;
        margin-bottom: 12px;
        text-decoration: underline;
    }

    .pillar p {
        font-size: 24px;
        color: #ffffff;
        font-family: 'playfair display', sans-serif;
        margin-bottom: 0;   
    }

    /* Buttons */
    .cta-button {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        padding: 14px 32px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px #00c6ff;
    }

    .cta-button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px #00c6ff;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 0px 60px;
        margin-top: 60px;
    }
    </style>

    <script>
    let lastScrollTop = 0;
    window.addEventListener("scroll", function(){
       let st = window.pageYOffset || document.documentElement.scrollTop;
       const navbar = document.querySelector(".navbar");
       if (st > lastScrollTop){
           navbar.style.top = "-100px";
       } else {
           navbar.style.top = "0";
       }
       lastScrollTop = st <= 0 ? 0 : st;
    }, false);
    </script>

    <video autoplay muted loop class="bg-video">
        <source src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/bg-video1.mp4" type="video/mp4" >
    </video>
    <div class="navbar">
        <div class="navbar-logo" style="display: flex; align-items: center;">
            <img src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/logo1.PNG" alt="Logo" style="height: 70px;">
            <h3 style= "margin-top: 15px; font-size: 35px;">DemandCast</h3>
        </div>
        <div class="navbar-menu">
            <a href="#home">Home</a>
            <a href="#solutions">Solutions</a>
            <a href="#docs">Docs</a>
            <a href="#tools">Tools</a>
            <a href="#contact">Contact</a>
        </div>
    </div>

    <div style="padding-top: 100px;"></div>

    <div class="section first">
        <h2>Smarter Forecasting. Stronger Decisions.</h2>
        <p>Welcome to <b>DemandCast</b>, your AI-driven partner for sales forecasting and demand planning. Transform uncertainty into clarity with intelligent analytics.</p>
        <button class="cta-button" style="margin-left: 25px;">Learn More</button>
    </div>

    <div class="section">
        <h2>Key Analytical Pillars</h2>
        <div class="pillars">
            <div class="pillar">
                <h3>Connect</h3>
                <img src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/connect.png" alt=logo >
                <p>Integrate seamlessly with your data sources.</p>
            </div>
            <div class="pillar">
                <h3>Analyze</h3>
                <img src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/analyze.png" alt=logo >
                <p>Extract actionable insights from demand signals.</p>
            </div>
            <div class="pillar">
                <h3>Act</h3>
                <img src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/act.png" alt=logo >
                <p>Drive decisions using DemandCast KPIs.</p>
            </div>
        </div>
    </div>

    <div class="section" style="display:flex; justify-content: space-between; align-items: center;">
        <div>
        <h2 style= "margin: 60px 0 60px 60px;max-width: 50%;text-align: left;">How It Works</h2>
        <p  style= "margin: 60px 0 60px 60px;max-width: 50%;text-align: left;">DemandCast simplifies your analytics workflow from data ingestion to decision-making. Schedule a demo to see it in action.</p>
        </div>
        <div>
        <video src="https://raw.githubusercontent.com/yashika641/Demand-Cast/refs/heads/main/datasets/Canva%202025-07-27%2019-47-17.mp4" alt=logo autoplay muted loop style= height:400px; width:auto></video>
        </div>
    </div>

    <div class="section">
        <h2>Success Stories</h2>
        <p>Our clients report improved forecasting accuracy, increased revenue, and enhanced operational efficiency. Join the transformation.</p>
    </div>

    <div class="footer">
        <h2>Ready to Transform Your Business?</h2>
        <p>Enter your business email to get started.</p>
        <input type="email" placeholder="you@example.com" style="padding:10px; width:300px; border-radius:6px; border:none; margin-right:10px;">
    </div>
    """, unsafe_allow_html=True)

# Final Get Started button

    # Inject custom CSS
    st.markdown("""
        <style>
        .stButton > button {
            background: linear-gradient(135deg, #8B5E3C,#00ffff); /* brown → beige gradient */
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #D2B48C, #8B5E3C); /* reverse gradient */
            transform: scale(1.05);
            box-shadow: 0px 6px 14px rgba(0,0,0,0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    # Centering the button
    col1, col2, col3 = st.columns([3.7, 2, 3])
    with col2:
        if st.button("Let's Get Started 🚀"):
            go_to_next_page("page1")



# -----------------files upload page -----------------#

elif st.session_state.page == "page1":
    # Hero title
    st.markdown(
        """
        <style>
        .stApp {
            background: transparent !important;
        }
        video.bg-video {
            position: fixed;     /* allow scrolling */
            top: 0;
            left: 0;
            display:flex;
            align-items: center;
            justify-content: center;    
            width: 110%;           /* fill full width */
            height: 100%;           /* scale height proportionally */
            min-height: 100%;       /* ensures it covers vertically */
            object-fit: cover;    /* show entire video (no cropping) */
            z-index: -1;            /* push behind content */
            background: black;  
            background-position:center;
            background-size: cover;
        }

        /* Upload box styling */
        section[data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.7);
            border: 2px dashed #8b5e3c;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        }
        section[data-testid="stFileUploader"]:hover {
            background: rgba(245, 222, 179, 0.8);
            border: 2px solid #6f4e37;
            box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
        }

        /* Upload text */
        section[data-testid="stFileUploader"] label {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #4b3832 !important;
        }

        /* Styled button */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #061c49, #28062d);
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 28px;
            border-radius: 12px;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(90deg, #28062d, #061c49);
            transform: scale(1.05);
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        }
        </style>
        <video autoplay muted loop class="bg-video">
            <source src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/bg-video1.mp4" type="video/mp4" >
        </video>
        """,
        unsafe_allow_html=True,
    )

    # Hero Title
    st.markdown(
        "<h1 style='text-align:center; color:#60d2f8;'>📂 Give us your files! We need them 🚀</h1>",
        unsafe_allow_html=True,
    )
    st.subheader("Upload your datasets for smarter business insights")
    st.markdown("<p style='font-size:20px; color:#60d2f8;'>You can upload **CSV or Excel** datasets for sales, transactions, inventory, products, <br> suppliers, customers, and campaigns. Our AI engine will automatically detect and categorize <br> files based on their names.</p>", unsafe_allow_html=True)
    # st.markdown("<p style='font-size:18px;'>This is medium text</p>", unsafe_allow_html=True)

    # File Uploader
    # Custom CSS for uploader
    st.markdown("""
    <style>
    /* Uploader container */
    .stFileUploader {
        border: 2px dashed #8B5E3C;
        border-radius: 15px;
        padding: 25px;
        background: linear-gradient(135deg, #3b58bc, #9f36b7);
        text-align: center;
        transition: all 0.3s ease-in-out;
    }
    .stFileUploader:hover {
        border-color: #5a3825;
        background: linear-gradient(135deg, #061c49, #28062d );
    }

    /* Label text */
    .stFileUploader label {
        font-size: 18px !important;
        font-weight: bold;
        color: #5a3825;
    }

    /* "Browse Files" button */
    .stFileUploader button {
        background-color:#28062d !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease-in-out;
    }
    .stFileUploader button:hover {
        background-color: #061c49 !important;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Actual uploader
    uploaded_files = st.file_uploader(
    "📑 Drag & Drop or Browse your datasets",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

    # Extra info for guidance
    st.markdown("### 📌 Supported File Types")
    st.info("We currently support **CSV (.csv)** and **Excel (.xlsx)** formats.")

    st.markdown("### 🔍 File Detection Guide")
    st.write(
        "- Example names: `sales_data.csv`, `transactions_q1.xlsx`, `customers_list.csv`\n"
        "- Required columns: IDs, dates, quantities, prices, revenues"
    )

    st.markdown("### 💡 Tips for Best Results")
    st.success(
        "✔ Use simple, clear file names\n"
        "✔ Avoid merged cells in Excel files\n"
        "✔ One dataset per file\n"
        "✔ Include headers in the first row"
    )

    # Upload Button
    if st.button("🚀 Upload and Proceed", key="upload_button"):
        if not uploaded_files:
            st.warning("⚠ Please select files to upload before clicking the button.")
        else:
            # Initialize datasets
            sales_df = None
            transactions_df = None
            products_df = None
            inventory_df = None
            customer_df = None
            promotion_df = None
            supplier_df = None

            for uploaded_file in uploaded_files:
                try:
                    # Read CSV or Excel
                    if uploaded_file.name.lower().endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # Detect dataset type by file name
                    name = uploaded_file.name.lower()
                    if "sales" in name:
                        sales_df = df
                    elif (
                        "transactions" in name
                        or "purchases" in name
                        or "txn" in name
                        or "orders" in name
                        or "order" in name
                    ):
                        transactions_df = df
                    elif "product" in name or "catalogue" in name or "item" in name:
                        products_df = df
                    elif "inventory" in name or "stock" in name:
                        inventory_df = df
                    elif "customer" in name:
                        customer_df = df
                    elif "campaign" in name or "promotions" in name:
                        promotion_df = df
                    elif "supplier" in name:
                        supplier_df = df
                    else:
                        st.warning(
                            f"{uploaded_file.name} could not be identified. Please use a valid file name."
                        )

                except Exception as e:
                    st.error(f"Failed to load {uploaded_file.name}: {e}")

            # Save datasets in session_state
            st.session_state.sales_df = sales_df
            st.session_state.transactions_df = transactions_df
            st.session_state.products_df = products_df
            st.session_state.inventory_df = inventory_df
            st.session_state.customer_df = customer_df
            st.session_state.promotion_df = promotion_df
            st.session_state.supplier_df = supplier_df

            # Move to next page
            go_to_next_page("page2")
# ------------------sidebar------------------#
elif st.session_state.page == "page2":
    st.markdown("""
    <style>
    .stApp {
        background: transparent !important;
    }
    video.bg-video {
        position: fixed;     /* allow scrolling */
        top: 0;
        left: 0;
        display:flex;
        align-items: center;
        justify-content: center;    
        width: 110%;           /* fill full width */
        height: 100%;           /* scale height proportionally */
        min-height: 100%;       /* ensures it covers vertically */
        object-fit: cover;    /* show entire video (no cropping) */
        z-index: -1;            /* push behind content */
        background: black;  
        background-position:center;
        background-size: cover;
    }
    </style>
    <video autoplay muted loop class="bg-video">
        <source src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/bg-video1.mp4" type="video/mp4" >
    </video>
""", unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* Hide outer Streamlit containers */
        div.css-18d2u8u, div.css-ndsp4m, div.css-ysd9e6 {
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Optional: remove spacing around the card */
        section.main > div {
            padding-top: 0rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h1 style='text-align:center; color:#60d2f8;'>DemandCast</h1>
        <h3 style='text-align:center; color:#60d2f8;'>Unlock your world of business analytics with DemandCast</h3>

    """, unsafe_allow_html=True)

    # Card styles
    style = {
        "card": {
            "width": "250px",
            "height": "250px",
            "border-radius": "0px",
            "box-shadow": "none",
            "background-color": "rgba(0,0,0,0)",
            "border": "none",
            "display": "flex",
            "flex-direction": "column",
            "justify-content": "center",
            "align-items": "center",
            "padding": "10px",
            "transition": "transform 0.3s ease",
        },
        "card_hover": {"transform": "scale(1.03)", "cursor": "pointer"},
        "filter": {"background-color": "rgba(0, 0, 0, 0)"},
        "div": {
            "background-color": "transparent",
            "display": "flex",
            "flex-direction": "column",
            "justify-content": "center",
            "align-items": "center",
        },
        "title": {"color": "#ffffff", "font-size": "24px", "font-weight": "600", "text-align": "center"},
        "text": {"color": "#dddddd", "font-size": "16px", "text-align": "center", "margin-top": "10px"},
        "image": {"width": "80px", "height": "80px", "margin-bottom": "15px"}
    }

    # First row of cards
    col1, col2, col3, col4, col5 = st.columns([1,3,3,3,1])
    with col2:
        card(
            title="Sales and Revenue Insights",
            text="Custom sales and revenue dashboards specially curated for you.",
            image="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/sales_service_logo.png",
            styles=style,
            key="card1",
            on_click=lambda: go_to_next_page("Sales and Revenue")
        )

    with col3:
        card(
            title="Inventory and Supply Chain",
            text="Provides an in-depth look of your inventory.",
            image="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/inventory_service_logo.png",
            styles=style,
            key="card2",
            on_click=lambda: go_to_next_page("Inventory and Supply chain")
        )

    with col4:
        card(
            title="Customer and Market Insights",
            text="Provides an in-depth look of your customer and market data.",
            image="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/customer_serivce_logo.png",
            styles=style,
            key="card3",
            on_click=lambda: go_to_next_page("Customer and Market insights")
        )

    # Second row of cards
    col1, col2, col3, col4, col5, col6 = st.columns([1,1,3,3,1,1])
    with col3:
        card(
            title="Pricing and Campaign Optimization",
            text="Provides an in-depth look of your pricing and campaign data.",
            image="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/promotions_services_logo.png",
            styles=style,
            key="card4",
            on_click=lambda: go_to_next_page("Pricing and Campaign Optimization")
        )

    with col4:
        card(
            title="Product Catalogue",
            text="Provides an in-depth look of your product and market data.",
            image="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/products_services_logo.png",
            styles=style,
            key="card5",
            on_click=lambda: go_to_next_page("product_catalogue")
        )

    # Social icons at the bottom
    st.markdown("""
        <div style='text-align:center; margin-top:40px;'>
            <a href='https://facebook.com' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/facebook.svg' width='30px' style='margin:0 10px; filter: invert(1);'/>
            </a>
            <a href='https://twitter.com' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/twitter.svg' width='30px' style='margin:0 10px; filter: invert(1);'/>
            </a>
            <a href='https://instagram.com' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/instagram.svg' width='30px' style='margin:0 10px; filter: invert(1);'/>
            </a>
            <a href='https://github.com' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg' width='30px' style='margin:0 10px; filter: invert(1);'/>
            </a>
            <a href='https://linkedin.com' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg' width='30px' style='margin:0 10px; filter: invert(1);'/>
            </a>
            <a href='https://yourportfolio.com' target='_blank'>
                <img src='https://www.flaticon.com/free-icon/domain_7710466?term=website&page=1&position=4&origin=search&related_id=7710466' width='30px' style='margin:0 10px; filter: invert(1);'/>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ------------------sales and revenue------------------#
elif st.session_state.page == "Sales and Revenue":
    sales_df = st.session_state["sales_df"]
    sales_dashboard(sales_df)
# #------------------inventory and supply chain------------------#
elif st.session_state.page == "Inventory and Supply chain":
    sales_df = st.session_state["sales_df"]
    inventory_df = st.session_state["inventory_df"]
    inventory_dashboard(inventory_df, sales_df)
# # ------------------Customer and Market insights ------------------#
elif st.session_state.page == "Customer and Market insights":
    customer_df = st.session_state["customer_df"]
    transactions_df = st.session_state["transactions_df"]
    customer_dashboard(customer_df, transactions_df)
    # lead_time_dashboard()
# # ------------------Pricing and Campaign Optimization------------------#

elif st.session_state.page == "Pricing and Campaign Optimization":
    customer_df = st.session_state["customer_df"]
    promotion_df = st.session_state["promotion_df"]
    products_df = st.session_state["products_df"]
    sales_df = st.session_state["sales_df"]
    inventory_df = st.session_state["inventory_df"]
    transactions_df = st.session_state["transactions_df"]
    promotion_dashboard(promotion_df,sales_df, inventory_df, transactions_df, products_df, customer_df)
# # ------------------Forecasting and Predicting------------------#

elif st.session_state.page == "product_catalogue":
    products_df = st.session_state["products_df"]
    sales_df = st.session_state["sales_df"]
    inventory_df = st.session_state["inventory_df"]
    transactions_df = st.session_state["transactions_df"]
    #
    product_dashboard(products_df, sales_df, inventory_df, transactions_df)
