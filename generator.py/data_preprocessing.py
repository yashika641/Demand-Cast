from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import category_encoders as ce  # for TargetEncoder
import pandas as pd

# ----------------- Encoders -----------------

def label_encoder(df, cat_cols):
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df

def one_hot_encoder(df, cat_cols):
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def ordinal_encoder(df, cat_cols, categories='auto'):
    oe = OrdinalEncoder(categories=categories)
    df[cat_cols] = oe.fit_transform(df[cat_cols])
    return df

def target_encoder(df, cat_cols, target):
    te = ce.TargetEncoder(cols=cat_cols)
    df[cat_cols] = te.fit_transform(df[cat_cols], target)
    return df

def multi_label_binarizer(df, col):
    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df[col]), 
    columns=mlb.classes_, 
    index=df.index))
    df.drop(columns=[col], inplace=True)
    return df

# ---------- Scalers ------------#

def standard_scaler(df, num_cols):
    ss = StandardScaler()
    df[num_cols] = ss.fit_transform(df[num_cols])
    return df

def robust_scaler(df, num_cols):
    rs = RobustScaler()
    df[num_cols] = rs.fit_transform(df[num_cols])
    return df

def min_max_scaler(df, num_cols):
    mms = MinMaxScaler()
    df[num_cols] = mms.fit_transform(df[num_cols])
    return df

def DROP_ROWS(df):
    df = df.dropna()
    return df

def fill_with_mean(df):
    df = df.fillna(df.mean(numeric_only=True))
    return df

def fill_with_median(df):
    df = df.fillna(df.median(numeric_only=True))
    return df

def fill_with_mode(df):
    df = df.fillna(df.mode().iloc[0])
    return df

def bfill(df):
    df = df.fillna(method='bfill')
    return df

def ffill(df):
    df = df.fillna(method='ffill')
    return df