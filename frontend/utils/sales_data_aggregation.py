import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact

# ---------------------------------------------
# 1. Column-wise aggregation (categorical/numerical)
# ---------------------------------------------
def sales_aggregation_by_column(df, agg_col, sales_column, agg_column_type):
    """
    Aggregates sales data by a given column depending on its type.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing sales.
    agg_col : str
        Column to aggregate on.
    sales_column : str
        Column containing sales values.
    agg_column_type : str
        'categorical' → take first sales value per group
        'numerical'   → take mean sales per group
    """
    df = df.copy()
    if agg_column_type == 'categorical':
        df[sales_column] = df.groupby(agg_col)[sales_column].transform('sum')
    elif agg_column_type == 'numerical':
        df[sales_column] = df.groupby(agg_col)[sales_column].transform('mean')
    else:
        raise ValueError("agg_column_type must be 'categorical' or 'numerical'")
    return df


# ---------------------------------------------
# 2. Time-based aggregation with interactive plotting
# ---------------------------------------------
def sales_aggregation_over_time(df, date_col, sales_column, agg_freq="Initial"):
    """
    Aggregates sales over time and plots results.

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
        df_sales_agg = df

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_sales_agg.index, df_sales_agg.values, marker='o', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(sales_column)
    ax.grid(True)
    return fig


# ---------------------------------------------
# 3. Interactive control (for Jupyter Notebooks)
# ---------------------------------------------
def interactive_sales_plot(df, date_col="date", sales_column="sales"):
    """
    Provides an interactive dropdown to choose aggregation frequency.
    """
    interact(lambda agg_freq: sales_aggregation_over_time(df, date_col, sales_column, agg_freq),
            agg_freq=['Initial', 'Daily', 'Monthly', 'Yearly'])
