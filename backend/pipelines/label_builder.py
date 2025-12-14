import pandas as pd

def create_labels_churn(transactions=None, activity=None, inactivity_days=90):
    # Use inactivity to mark churn
    if activity is not None:
        activity["date"] = pd.to_datetime(activity["date"], errors="coerce")
        last_seen = activity.groupby("customer_id")["date"].max().reset_index(name="last_seen")
        ref = activity["date"].max()
        last_seen["days_inactive"] = (ref - last_seen["last_seen"]).dt.days
        last_seen["churn_flag"] = (last_seen["days_inactive"] >= inactivity_days).astype(int)
    else:
        # fallback using transactions recency
        transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")
        last_txn = transactions.groupby("customer_id")["date"].max().reset_index(name="last_txn")
        ref = transactions["date"].max()
        last_txn["days_since_txn"] = (ref - last_txn["last_txn"]).dt.days
        last_txn["churn_flag"] = (last_txn["days_since_txn"] >= inactivity_days).astype(int)
        last_seen = last_txn[["customer_id", "churn_flag"]]
    return last_seen
