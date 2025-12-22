def map_column(df, synonyms):
    df_cols = [c.lower().strip() for c in df.columns]

    for syn in synonyms:
        syn = syn.lower().strip()
        for i, col in enumerate(df_cols):
            if syn == col or syn in col:
                return df.columns[i]   # return actual column name

    return None
