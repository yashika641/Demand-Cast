import re
import pandas as pd

def column_finder(df,possible_names):
    """
    Find the best matching column in df.columns given a list of possible_names.
    Returns the actual column name if found, else None.
    """
    cols= [c.lower().replace(" ","").replace("_","").replace("-","") for c in df.columns]
    # possible_names = [name.lower().replace(" ","").replace("_","").replace("-","") for name in possible_names]
    
    for name in possible_names:
        pattern = name.lower().replace(" ","").replace("_","").replace("-","")
        for idx,col in enumerate(cols):
            if pattern in col:
                return df.columns[idx]
    return None

