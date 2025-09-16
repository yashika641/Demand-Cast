import pandas as pd
from pytrends.request import TrendReq
import os

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=330)  # tz=330 for India

# File where data will be stored
FILE_NAME = "google_trends_data.csv"

def fetch_trends(keyword):
    """Fetch Google Trends data for a keyword and append to CSV"""
    try:
        # Build payload
        pytrends.build_payload([keyword], timeframe='today 12-m')  # last 12 months
        
        # Get interest over time
        data = pytrends.interest_over_time()
        
        if not data.empty:
            data = data.reset_index()
            data = data[['date', keyword]]
            
            # Append or create CSV
            if os.path.exists(FILE_NAME):
                data.to_csv(FILE_NAME, mode='a', header=False, index=False)
            else:
                data.to_csv(FILE_NAME, index=False)
            
            print(f"✅ Data for '{keyword}' saved to {FILE_NAME}")
        else:
            print(f"⚠️ No data found for '{keyword}'")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    while True:
        kw = input("\nEnter a keyword (or type 'exit' to quit): ")
        if kw.lower() == "exit":
            break
        fetch_trends(kw)
