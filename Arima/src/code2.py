import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm import tqdm
import pickle
import logging
import warnings

warnings.filterwarnings("ignore")

# ----------------- LOGGER -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# ----------------- DATA CLEANING & PREPROCESSING -----------------
def cleaning_preprocessing(csv_path):
    log.info("Starting data cleaning and preprocessing...")
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    if 'Festival_Season' in df.columns:
        df.drop(columns=['Festival_Season'], inplace=True)
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Weekly aggregation
    agg_dict = {
        'Units_Sold': 'sum',
        'Revenue': 'sum',
        'Price_per_Unit': 'mean',
        'Weather_Temp': 'mean',
        'Promotion_Flag': 'first',
        'Competitor_Price': 'mean',
        'Social_Sentiment': 'mean',
    }
    df_weekly = df.groupby(['Product_Category', 'Region']).resample('W', on='Date').agg(agg_dict).reset_index()

    # One-hot encode Product & Region
    df_weekly = pd.get_dummies(df_weekly, columns=['Product_Category', 'Region'])

    # Standardize numerical columns
    num_cols = ['Price_per_Unit', 'Weather_Temp', 'Competitor_Price', 'Social_Sentiment']
    scaler = StandardScaler()
    df_weekly[num_cols] = scaler.fit_transform(df_weekly[num_cols])

    log.info("Data cleaning and preprocessing completed.")
    return df_weekly


# ----------------- TRAIN-TEST SPLIT -----------------
def train_test_split_by_series(df, test_size=0.2, min_length=20):
    product_cols = [col for col in df.columns if col.startswith('Product_Category_')]
    region_cols = [col for col in df.columns if col.startswith('Region_')]

    series_dict = {}
    for prod_col in product_cols:
        for reg_col in region_cols:
            df_series = df[(df[prod_col] == 1) & (df[reg_col] == 1)].copy()
            if len(df_series) < min_length:
                continue
            df_series.set_index('Date', inplace=True)
            split_idx = int(len(df_series) * (1 - test_size))
            train = df_series.iloc[:split_idx]
            test = df_series.iloc[split_idx:]
            series_dict[(prod_col, reg_col)] = (train, test)
    log.info(f"Created {len(series_dict)} product-region series splits.")
    return series_dict


# ----------------- AUTO ARIMA -----------------
def auto_arima_search(train, exog_train=None, seasonal_m=7, min_periods=20):
    if len(train) < min_periods:
        log.warning(f"Series too short for ARIMA: {len(train)}")
        return None
    try:
        stepwise_model = auto_arima(
            train['Units_Sold'],
            exogenous=exog_train,
            start_p=0, start_q=0, max_p=5, max_q=5,
            d=None, max_d=2,
            start_P=0, start_Q=0, max_P=3, max_Q=3,
            D=None, max_D=2,
            seasonal=True, m=seasonal_m,
            information_criterion='aic',
            stepwise=False, trace=False,
            suppress_warnings=True
        )
        return stepwise_model.order, stepwise_model.seasonal_order
    except Exception as e:
        log.error(f"Auto ARIMA failed: {e}")
        return None


# ----------------- BUILD & TRAIN SARIMAX -----------------
def build_and_train_model(df_weekly, order_dict, seasonal_order_dict, min_length=20):
    results_dict = {}
    print(df_weekly.dtypes)
    for col in df_weekly.columns:
        df_weekly[col] = pd.to_numeric(df_weekly[col], errors='coerce').fillna(0)
    exog_cols = [col for col in df_weekly.columns if col not in ['Date','Units_Sold','Revenue']]

    product_cols = [col for col in df_weekly.columns if col.startswith('Product_Category_')]
    region_cols = [col for col in df_weekly.columns if col.startswith('Region_')]
    log.info("Starting SARIMAX model training for all series...")
    for prod_col in tqdm(product_cols, desc="Products"):
        for reg_col in region_cols:
            df_series = df_weekly[(df_weekly[prod_col]==1) & (df_weekly[reg_col]==1)].copy()
            if len(df_series) < min_length:
                continue
            df_series.set_index('Date', inplace=True)
            split_idx = int(len(df_series)*0.8)
            train = df_series.iloc[:split_idx]
            train['Units_Sold'] = pd.to_numeric(train['Units_Sold'], errors='coerce')
            train = train.dropna(subset=['Units_Sold'])

            exog_train = train[exog_cols] if exog_cols else None
            if exog_train is not None:
                exog_train = exog_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            order = order_dict.get((prod_col, reg_col), (1,0,1))
            seasonal_order = seasonal_order_dict.get((prod_col, reg_col), (1,1,1,52))

            model = SARIMAX(
                train['Units_Sold'],
                order=order,
                seasonal_order=seasonal_order,
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            results_dict[(prod_col, reg_col)] = res
            log.info(f"Trained SARIMAX for {prod_col} | {reg_col}")
    log.info("SARIMAX training completed.")
    return results_dict


# ----------------- SAFE MAPE -----------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100


# ----------------- FORECAST & EVALUATE -----------------
def forecast_evaluate(results, train, test, exog_test=None):
    pred = results.get_forecast(steps=len(test), exog=exog_test)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    y_truth = test['Units_Sold']

    mse = mean_squared_error(y_truth, y_forecasted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_truth, y_forecasted)
    mape = safe_mape(y_truth, y_forecasted)

    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "MAPE"],
        "Value": [mse, rmse, mae, mape]
    })
    log.info("\n" + metrics_df.to_string())

    plt.figure(figsize=(14,7))
    plt.plot(train.index, train['Units_Sold'], label="Train")
    plt.plot(test.index, y_truth, label="Test")
    plt.plot(test.index, y_forecasted, color="red", label="Forecast")
    plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='k', alpha=0.2)
    plt.legend()
    plt.show()

    return y_forecasted, metrics_df


# ----------------- MAIN PIPELINE -----------------
def main():
    csv_path = r'C:\Users\palya\Desktop\DemandCast\Demand-Cast\synthetic_sales_2yrs.csv'

    # -------------------- 1. Data Cleaning --------------------
    log.info("Starting data cleaning and preprocessing...")
    df_cleaned = cleaning_preprocessing(csv_path)
    log.info(f"Cleaned data shape: {df_cleaned.shape}")

    # -------------------- 2. Train-Test Split --------------------
    log.info("Splitting data into product-region series...")
    series_dict = train_test_split_by_series(df_cleaned)
    log.info(f"Number of series created: {len(series_dict)}")

    # -------------------- 3. Auto ARIMA Search --------------------
    order_dict = {}
    seasonal_order_dict = {}
    log.info("Starting auto ARIMA search for all series...")
    for key in tqdm(series_dict.keys(), desc="ARIMA Search"):
        train, test = series_dict[key]
        exog_cols = [col for col in train.columns if col not in ['Units_Sold', 'Revenue']]
        exog_train = train[exog_cols] if exog_cols else None

        result = auto_arima_search(train, exog_train=exog_train)
        if result:
            order, seasonal_order = result
            order_dict[key] = order
            seasonal_order_dict[key] = seasonal_order
            log.info(f"ARIMA order for {key}: {order}, Seasonal order: {seasonal_order}")
        else:
            log.warning(f"Skipping ARIMA for series {key} due to insufficient data or error.")

    log.info(f"Generated ARIMA orders for {len(order_dict)} series.")

    # -------------------- 4. Train SARIMAX Models --------------------
    log.info("Starting SARIMAX model training...")
    results = build_and_train_model(df_cleaned, order_dict, seasonal_order_dict)
    log.info(f"Trained SARIMAX models for {len(results)} series.")

    # -------------------- 5. Save Models --------------------
    model_path = 'sarimax_models.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(results, f)
    log.info(f"SARIMAX models saved to {model_path}")

    # Save ARIMA orders for reproducibility
    order_path = 'arima_orders.pkl'
    with open(order_path, 'wb') as f:
        pickle.dump({'order_dict': order_dict, 'seasonal_order_dict': seasonal_order_dict}, f)
    log.info(f"ARIMA orders saved to {order_path}")

    # -------------------- 6. Optional: Save metrics / splits --------------------
    # For future evaluation, you can also save series_dict (train/test splits)
    splits_path = 'train_test_splits.pkl'
    with open(splits_path, 'wb') as f:
        pickle.dump(series_dict, f)
    log.info(f"Train/test splits saved to {splits_path}")

    log.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
