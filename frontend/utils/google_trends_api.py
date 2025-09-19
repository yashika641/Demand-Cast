import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from tqdm import tqdm
import time
import random
from keybert import KeyBERT
import re
import logging
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Initialize KeyBERT model
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

# Inject CSS styling
st.markdown("""
    <style>
    /* Background */
    .stApp {
            background-image: url("https://raw.githubusercontent.com/yashika641/Demand-Cast/main/datasets/Gemini_Generated_Image_6uxpod6uxpod6uxp.png");
            background-size: cover;
            background-attachment: fixed;
            font-family: 'Montserrat', sans-serif;
            color: #ffffff;
            background-position : center;
        }

    /* Headings */
    h1, h2, h3 {
        color: #e0e7ff;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        animation: fadeInDown 1s ease-in-out;
    }

    /* Card-like containers */
    .block-container {
        padding: 2rem 2rem;
    }
    .reportview-container .main .block-container {
        border-radius: 20px;
        padding: 2rem;
        background: rgba(255,255,255,0.08);
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        animation: fadeIn 1.5s ease;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #2575fc, #6a11cb);
        box-shadow: 0px 6px 15px rgba(0,0,0,0.3);
    }

    /* Animations */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    @keyframes fadeInDown {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)


# --- Keyword generator ---
def generate_trends_keywords_keybert(product_name: str, max_keywords: int = 3) -> list[str]:
    clean_name = re.sub(r"(\d+|gm|g|kg|ml|l|pack|pcs|packets|bottle|promo|sale|free)",
                        "", product_name, flags=re.I)
    clean_name = re.sub(r"[^\w\s]", "", clean_name).strip()

    try:
        keywords = kw_model.extract_keywords(
            clean_name,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=max_keywords
        )
        keyword_list = [kw for kw, score in keywords]
    except Exception as e:
        logging.warning(f"KeyBERT failed for '{product_name}', fallback used. Error: {e}")
        keyword_list = [clean_name]

    if not keyword_list:
        keyword_list = [clean_name]

    return keyword_list[:max_keywords]


# --- Trends fetcher ---
def fetch_google_trends_best_keyword(
    product_name: str,
    timeframe: str = "today 5-y",
    geo: str = "IN",
    max_retries: int = 3,
    min_data_fraction: float = 0.1,
    debug: bool = False
) -> pd.DataFrame:

    pytrends = TrendReq(hl="en-US", tz=360)
    results = {}

    keywords = generate_trends_keywords_keybert(product_name)

    for kw in tqdm(keywords, desc=f"Fetching trends for {product_name}"):
        for attempt in range(max_retries):
            try:
                pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
                data = pytrends.interest_over_time()

                if not data.empty:
                    data = data.drop(columns=["isPartial"], errors="ignore")
                    frac = (data[kw] > 0).sum() / len(data)
                    results[kw] = (data[kw], frac)
                    if debug:
                        logging.info(f"Keyword '{kw}' -> {frac:.2f} non-zero fraction")

                time.sleep(1 + random.random() * 2)
                break

            except TooManyRequestsError:
                wait = random.randint(5, 15) * (attempt + 1)
                logging.warning(f"Rate limit hit for '{kw}', retrying in {wait}s...")
                time.sleep(wait)

            except Exception as e:
                logging.error(f"Error fetching '{kw}': {e}")
                break

    valid_results = {k: v for k, v in results.items() if v[1] >= min_data_fraction}
    if not valid_results:
        logging.warning(f"No valid data for '{product_name}'")
        return pd.DataFrame()

    best_keyword = max(valid_results, key=lambda k: valid_results[k][1])
    best_data = valid_results[best_keyword][0]

    if debug:
        logging.info(f"Best keyword for '{product_name}' -> '{best_keyword}'")

    trends_df = pd.DataFrame(best_data)
    trends_df.columns = [product_name]
    trends_df[product_name] = pd.to_numeric(trends_df[product_name], errors="coerce").fillna(0)

    return trends_df


# --- Streamlit UI ---
# st.title("📊 Google Trends Keyword Analyzer (Blue-Purple Styled)")
# st.write("Generate **Google Trends-friendly keywords** for a product using **KeyBERT** and fetch the best performing trends data.")

# with st.form("trends_form"):
#     product_name = st.text_input("Enter a Product Name:", "Amul Butter 500gm Pack")
#     submitted = st.form_submit_button("🔍 Analyze Trends")

# if submitted:
#     with st.spinner("Fetching trends data... ⏳"):
#         trends_df = fetch_google_trends_best_keyword(product_name, debug=True)

#     if not trends_df.empty:
#         st.success(f"✅ Successfully fetched Google Trends for: {product_name}")
#         st.line_chart(trends_df, use_container_width=True)
#         st.dataframe(trends_df.tail(10).style.set_properties(
#             **{"background-color": "#f3e8ff", "color": "black", "border-color": "purple"}
#         ))
#     else:
#         st.error("❌ No valid trends data found. Try a different product.")
