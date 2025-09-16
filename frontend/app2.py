import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.google_trends_api import fetch_google_trends_best_keyword
from utils.column_finder import column_finder

# from inventory_models_functions import lead_time_dashboard
from dashboards.sales_dashboard import sales_dashboard
from dashboards.inventory_dashboard import inventory_dashboard
from dashboards.product_dashboard import product_dashboard
from dashboards.customer_dashboard import customer_dashboard

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to_next_page(page_name):
    st.session_state.page = page_name
    st.rerun()


# ----------------set page config----------------#
st.set_page_config(page_title="DemandCast", page_icon=":bar_chart:", layout="wide")
# Navbar HTML & CSS
st.markdown("""
<style>
/* Navbar styling */
.navbar {
    position: fixed;
    top: 100px;
    left: 10px;
    width: 100%;
    height: 100px;
    background-color: transparent;  /* Transparent background */
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 30px;  /* reduced side padding */
    z-index: 9999;
}

/* Logo + Brand */
.navbar-logo {
    display: flex;
    align-items: center;
}
.navbar-logo img {
    height: 50px;       /* slightly smaller */
    margin-right: 50px;  /* reduced gap between logo and text */
}
.navbar-logo span {
    color: white;
    font-weight: bold;
    font-size: 35px;
    font-style: italic;
}

/* Menu */
.navbar-menu {
    display: flex;
    align-items: center;
}
.navbar-menu a {
    color: white;
    text-decoration: none;
    margin-left: 30px;  /* reduced gap between menu items */
    font-weight: 500;
    font-size: 25px;
    transition: 0.3s;
}
.navbar-menu a:hover {
    text-decoration: underline;
}
</style>
<div class="navbar">
    <div class="navbar-logo">
        <img src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/Gemini_Generated_Image_r3inpnr3inpnr3in-removebg-preview%20(1).png" 
             alt="Logo" style="height:150px; width:auto;">
        <span>DemandCast</span>
    </div>
    <div class="navbar-menu">
        <a href="#home">Home</a>
        <a href="#solutions">Solutions</a>
        <a href="#docs">Docs</a>
        <a href="#tools">Tools</a>
        <a href="#contact">Contact</a>
    </div>
</div>


<div style="padding-top: 90px;"></div>
""", unsafe_allow_html=True)



if st.session_state.page == "home":
    # ------------------ Navbar HTML & CSS ---------------


    # ------------------ Hero Section ------------------
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://github.com/yashika641/Demand-Cast/blob/02f59593d51aa70b44a417a8afe1522be43cf503/datasets/Gemini_Generated_Image_6uxpod6uxpod6uxp.png");
        background-attachment: fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(
        "<h1 style='text-align:center; color:#1f77b4;'>DemandCast - Your AI Driven Business Partner</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align:center; color:#6e1e24;'>Sales Forecasting and Demand Planning Made Easy</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align:center; color:#fff; font-size:18px; font-family: Georgia; background: rgba(0,0,0,0.5); padding: 15px; border-radius: 8px;'>
        Welcome to <b>DemandCast</b>, your AI-driven business partner for sales forecasting and demand planning. 
        Harness the power of advanced analytics and machine learning to optimize your inventory, reduce costs, and boost your sales performance.
        </p>
        """,
        unsafe_allow_html=True
    )

    # ------------------ Centered Button ------------------
    col1, col2, col3 = st.columns([3.8,3,2])
    with col2:
        if st.button("Let's Get Started"):
            st.session_state.page = "page1"

    st.markdown("</div>", unsafe_allow_html=True)  # Close main-content



# -----------------files upload page -----------------#

# Step 1: File upload page
if st.session_state.page == "page1":
    st.title("Give us your files! We need them 🚀")
    st.subheader(
        "Please upload the needed files. These files are used to analyse your data and provide you with the best insights"
    )

    st.write(
        "Users can upload any of the business datasets required for analytics, such as sales, transactions, inventory, product catalogue, supplier, customer, or campaign data. "
        "The system will automatically detect and categorize each file based on its file name. Uploaded files should contain the key columns necessary for analytics, such as unique identifiers, product IDs, customer IDs, dates, quantities, prices, and revenues."
    )

    uploaded_files = st.file_uploader(
        "Upload all your datasets (sales, transactions, inventory, products, supplier, customers, campaigns)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
    )

    # Upload button
    if st.button("Upload and Proceed", key="upload_button"):
        if not uploaded_files:
            st.warning("Please select files to upload before clicking the button.")
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

            # Save datasets in session_state for next page
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
    st.title("let's get started")

    def switch_page(page_name):
        st.session_state.page = page_name

    # Styling for nice boxes
    st.markdown(
        """
        <style>
        div.stButton > button {
            width: 100%;
            height: 100px;
            border-radius: 12px;
            background-color: #262730;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border: 2px solid #4CAF50;
        }
        div.stButton > button:hover {
            background-color: #4CAF50;
            color: white;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4, col5 = st.columns(5, gap="large")

    with col1:
        st.button(
            "Sales and Revenue", on_click=lambda: switch_page("Sales and Revenue")
        )

    with col2:
        st.button(
            "Inventory and Supply chain",
            on_click=lambda: switch_page("Inventory and Supply chain"),
        )

    with col3:
        st.button(
            "Customer and Market insights",
            on_click=lambda: switch_page("Customer and Market insights"),
        )

    with col4:
        st.button(
            "Pricing and Campaign Optimization",
            on_click=lambda: switch_page("Pricing and Campaign Optimization"),
        )

    with col5:
        st.button(
            "product_catalogue",
            on_click=lambda: switch_page("product_catalogue"),
        )

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
    promotion_df = st.session_state["promotion_df"]
# # ------------------Forecasting and Predicting------------------#

elif st.session_state.page == "product_catalogue":
    products_df = st.session_state["products_df"]
    sales_df = st.session_state["sales_df"]
    inventory_df = st.session_state["inventory_df"]
    transactions_df = st.session_state["transactions_df"]
    #
    product_dashboard(products_df, sales_df, inventory_df, transactions_df)
