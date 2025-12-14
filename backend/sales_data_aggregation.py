import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact

# ---------------------------------------------
# 1. Column-wise aggregation (categorical/numerical)
# ---------------------------------------------
def sales_aggregation_over_time_dict(df, date_col, sales_column, agg_freq="Initial"):
    """
    Aggregates sales over time and returns result as serializable dict.
    Parameters
    ----------
    df : pd.DataFrame
        Input data containing sales and date.
    date_col : str
        Column name for dates.
    sales_column : str
        Column name for sales values.
    agg_freq : str
        One of ['Initial', 'Daily', 'Monthly', 'Yearly'].
    Returns
    -------
    dict with keys labels (datetime/str) and values (floats/ints)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    freq_map = {
        'Yearly': ('Y', 'Yearly Sales Over Time', 'Year'),
        'Monthly': ('M', 'Monthly Sales Over Time', 'Month'),
        'Daily': ('D', 'Daily Sales Over Time', 'Day'),
        'Initial': (None, 'Raw Sales Over Time', 'Date')
    }
    if agg_freq not in freq_map:
        raise ValueError("agg_freq must be one of: 'Initial', 'Yearly', 'Monthly', 'Daily'")
    freq, title, xlabel = freq_map[agg_freq]
    # Apply resampling if needed
    if freq:
        df_sales_agg = df.resample(freq)[sales_column].sum()
    else:
        df_sales_agg = df[sales_column]

    # Convert index and values to lists/serializable types
    labels = df_sales_agg.index.strftime('%Y-%m-%d').tolist() if freq else df_sales_agg.index.tolist()
    values = df_sales_agg.values.tolist()
    return {
        "labels": labels,
        "values": values,
        "title": title,
        "xlabel": xlabel,
        "ylabel": sales_column
    }
