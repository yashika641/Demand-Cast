import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def preprocess_data(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
    
    # Handle missing values
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col].fillna(df[col].min(), inplace=True)
        
    # Normalize numerical columns
    for col in numerical_cols:
        scaler = StandardScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
        
    # Encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1) 
    
    # Extract date features
    # for col in date_cols:
    #     df[f'{col}_year'] = df[col].dt.year
    #     df[f'{col}_month'] = df[col].dt.month
    #     df[f'{col}_day'] = df[col].dt.day
    #     df[f'{col}_hour'] = df[col].dt.hour
    #     df.drop(columns=[col], inplace=True)
        
    return df
        
    