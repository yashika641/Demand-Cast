import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from tqdm import tqdm
import time
import random
from keybert import KeyBERT
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Initialize KeyBERT model
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

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

