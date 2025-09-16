import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from tqdm import tqdm
import time
import random
from keybert import KeyBERT
import re

# Initialize KeyBERT model
kw_model = KeyBERT(model='all-MiniLM-L6-v2')  # lightweight and fast

def generate_trends_keywords_keybert(product_name, max_keywords=3):
    """
    Generate Google-Trends-friendly keywords from a product name using KeyBERT.
    Removes sizes, colors, promotional words, and unnecessary details.
    Returns a list of keywords (strings).
    """
    # Preprocess product name: remove numbers, sizes, units, punctuation
    clean_name = re.sub(r"(\d+|gm|g|kg|ml|l|pack|pcs|packets|bottle|promo|sale|free)", "", product_name, flags=re.I)
    clean_name = re.sub(r"[^\w\s]", "", clean_name).strip()

    # Extract keywords
    keywords = kw_model.extract_keywords(clean_name, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=max_keywords)
    # keywords is a list of tuples: (keyword, score)
    keyword_list = [kw for kw, score in keywords]

    if not keyword_list:
        # fallback: original product name
        keyword_list = [clean_name]

    return keyword_list[:max_keywords]


# --- Google Trends Fetch Function ---
def fetch_google_trends_best_keyword(product_name, timeframe='today 5-y', geo='IN', 
                                    max_retries=3, min_data_fraction=0.1):
    """
    Fetch Google Trends data for a product name and return the column
    corresponding to the keyword with the most valid data.
    """
    pytrends = TrendReq(hl="en-US", tz=360)
    trends_df = pd.DataFrame()

    # Generate keyword variants using KeyBERT
    keywords = generate_trends_keywords_keybert(product_name)

    results = {}  # keyword -> (data_series, fraction_non_zero)

    for kw in tqdm(keywords, desc="Fetching Google Trends"):
        for attempt in range(max_retries):
            try:
                pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
                data = pytrends.interest_over_time()
                if not data.empty:
                    data = data.drop(columns=['isPartial'], errors='ignore')
                    frac = (data[kw] > 0).sum() / len(data)
                    results[kw] = (data[kw], frac)
                time.sleep(random.randint(1, 5))
                break
            except TooManyRequestsError:
                wait = random.randint(5, 15) * (attempt + 1)
                print(f"Rate limit hit for '{kw}', retrying in {wait} seconds...")
                time.sleep(wait)
            except Exception as e:
                print(f"Error fetching '{kw}': {e}")
                break

    # Filter keywords that meet minimum fraction
    valid_results = {k: v for k, v in results.items() if v[1] >= min_data_fraction}
    if not valid_results:
        print(f"No valid trends data found for '{product_name}'")
        return pd.DataFrame()

    # Pick the keyword with the highest fraction of non-zero data
    best_keyword = max(valid_results, key=lambda k: valid_results[k][1])
    best_data = valid_results[best_keyword][0]

    trends_df = pd.DataFrame(best_data)
    trends_df.columns = [product_name]

    # Convert to numeric, coerce errors (strings become NaN)
    trends_df[product_name] = pd.to_numeric(trends_df[product_name], errors='coerce')

    # Fill NaN with 0 (optional)
    trends_df[product_name] = trends_df[product_name].fillna(0)
    return trends_df



