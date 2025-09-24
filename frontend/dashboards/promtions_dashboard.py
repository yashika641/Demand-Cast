import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.column_finder import column_finder

def promotion_dashboard(promotion_df, sales_df, transactions_df, inventory_df, product_df, customer_df):
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
            .tab-header {
                font-weight: 700 !important;
                font-size: 1.2rem !important;
                color: #ffffff !important;
            }
            .metric-card {
                background: rgba(255,255,255,0.1);
                border-radius: 12px;
                padding: 14px;
                text-align: center;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                color: #ffffff;
            }
            .metric-card:hover {
                transform: translateY(-6px);
                box-shadow: 0 6px 18px rgba(0,0,0,0.5);
            }
            .dataframe-container {
                background: rgba(255,255,255,0.05);
                border-radius: 10px;
                padding: 8px;
            }
        </style>
        <video autoplay muted loop class="bg-video">
            <source src="https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/bg-video1.mp4" type="video/mp4" >
        </video>
    """, unsafe_allow_html=True)

    st.title("📊 Promotion & Campaign Analytics Dashboard")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Pricing Elasticity",
        "Campaign Performance",
        "A/B Testing",
        "ROI & Conversion",
        "Top Products & Promotions",
        "Customer & Segment Analysis"
    ])

    # -------------------- Pricing Elasticity --------------------
    with tab1:
        st.header("📈 Pricing Elasticity")
        st.info("Dynamic pricing elasticity model integration coming soon.")

    # -------------------- Campaign Performance --------------------
    with tab2:
        st.header("📊 Campaign Performance Analysis")
        # Dynamic detection of campaign columns
        campaign_col = column_finder(promotion_df, ["campaign_name", "campaign", "promo_name"])
        ctr_col = column_finder(promotion_df, ["ctr", "click_through_rate"])
        conversions_col = column_finder(promotion_df, ["conversions", "conversion_count"])
        revenue_col = column_finder(promotion_df, ["revenue", "gain", "sales"])

        if campaign_col and ctr_col and conversions_col and revenue_col:
            st.subheader("Campaign Metrics")
            campaigns = promotion_df[[campaign_col, ctr_col, conversions_col, revenue_col]].copy()
            campaigns.columns = ["Campaign", "CTR (%)", "Conversions", "Revenue"]
            st.dataframe(campaigns, use_container_width=True)
            fig_ctr = px.bar(campaigns, x="Campaign", y="CTR (%)", color="CTR (%)", title="CTR per Campaign", text_auto=True)
            st.plotly_chart(fig_ctr, use_container_width=True)
        else:
            st.info("Campaign metrics not detected in promotion_df.")

    # -------------------- A/B Testing --------------------
    with tab3:
        st.header("🧪 A/B Testing")
        st.info("A/B testing framework coming soon. Detect variants, conversions & significance automatically.")

    # -------------------- ROI & Conversion --------------------
    with tab4:
        st.header("💰 ROI & Conversion Tracking")
        investment_col = column_finder(promotion_df, ["investment", "cost", "spend"])
        gain_col = column_finder(promotion_df, ["gain", "revenue", "profit"])

        if investment_col and gain_col:
            promotion_df["ROI (%)"] = (promotion_df[gain_col] - promotion_df[investment_col]) / promotion_df[investment_col] * 100
            for _, row in promotion_df.iterrows():
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>{row[campaign_col]}</h4>
                        <p>Investment: ${row[investment_col]:,.0f}</p>
                        <p>Gain: ${row[gain_col]:,.0f}</p>
                        <p>ROI: {row['ROI (%)']:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Investment/Gain columns not found in promotion_df.")

    # -------------------- Top Products & Promotions --------------------
    with tab5:
        st.header("🏆 Top Products Impacted by Promotions")
        prod_col = column_finder(product_df, ["product_name", "item_name", "sku"])
        sales_col = column_finder(sales_df, ["quantity_sold", "sales_qty", "units_sold"])
        promo_col = column_finder(sales_df, ["promotion_id", "campaign_id"])

        if prod_col and sales_col:
            top_products = sales_df.groupby(prod_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False).head(10)
            st.dataframe(top_products)
            fig_top = px.bar(top_products, x=prod_col, y=sales_col, color=prod_col, title="Top Products Sold")
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("Product or sales columns not detected.")

    # -------------------- Customer & Segment Analysis --------------------
    with tab6:
        st.header("👥 Customer & Segment Impact")
        age_col = column_finder(customer_df, ["age"])
        gender_col = column_finder(customer_df, ["gender", "sex"])
        region_col = column_finder(customer_df, ["region", "state", "city", "location"])

        if age_col or gender_col or region_col:
            df = customer_df.copy()
            if age_col:
                df["age_group"] = pd.cut(df[age_col], bins=[0,18,25,35,45,60,100], labels=["<18","18-25","26-35","36-45","46-60","60+"])
                age_counts = df["age_group"].value_counts().reset_index()
                age_counts.columns = ["Age Group", "Count"]
                fig_age = px.bar(age_counts, x="Age Group", y="Count", title="Customers by Age Group")
                st.plotly_chart(fig_age, use_container_width=True)
            if gender_col:
                gender_counts = df[gender_col].value_counts().reset_index()
                gender_counts.columns = ["Gender","Count"]
                fig_gender = px.pie(gender_counts, names="Gender", values="Count", title="Gender Distribution")
                st.plotly_chart(fig_gender, use_container_width=True)
            if region_col:
                region_counts = df[region_col].value_counts().reset_index()
                region_counts.columns = ["Region","Count"]
                fig_region = px.bar(region_counts, x="Region", y="Count", title="Region Distribution")
                st.plotly_chart(fig_region, use_container_width=True)
        else:
            st.info("No customer demographic columns detected.")

# -------------------- Back to Home Button --------------------
st.markdown("<div style='margin-top:24px; text-align:center;'>", unsafe_allow_html=True)
if st.button("🏠 Back to Home"):
    st.session_state.page = "page2"
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)
