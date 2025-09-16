import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from pmdarima import auto_arima
import pickle
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import MultiLabelBinarizer
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", ValueWarning)

# ----------- LOGGER -----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Arima.src.logger import get_logger
log = get_logger(__name__)


# ----------- DATA CLEANING -----------
def cleaning_preprocessing():
    log.info('Cleaning data...')

    df=pd.read_csv(r'C:\Users\palya\Desktop\DemandCast\Demand-Cast\synthetic_sales_2yrs.csv' )
    print(df.head(10))
    # print(df.info())
    # print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df['Date'].duplicated().sum())
    print(df.index.duplicated().sum())
    print(df.shape)
    print(df.dtypes)
    # Check duplicate rows
    df.drop_duplicates(inplace=True)
    df.drop(columns=['Festival_Season'],inplace=True)
    print(df.shape)
    df.dropna(inplace=True)
    print(df.shape)
    print(df.duplicated().sum())



# Ensure date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

# Optional: set index
    df.set_index('Date', inplace=True)

# Aggregation dictionary
    agg_dict = {
    'Units_Sold': 'sum',
    'Revenue': 'sum',
    'Price_per_Unit': 'mean',
    'Weather_Temp': 'mean',
    'Promotion_Flag': 'first',       # keep one representative value
    'Competitor_Price': 'mean',
    'Social_Sentiment': 'mean',
}

# Group by date and reset index
    df_daily = df.groupby('Date','Product_Name','Region').agg(agg_dict).reset_index()
    
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily.set_index('Date', inplace=True)

# Example: weekly sum for Units_Sold and Revenue, weekly mean for others
    df_weekly = df_daily.groupby(['Product_Name', 'Region']).resample('W').agg({
    'Units_Sold': 'sum',
    'Revenue': 'sum',
    'Price_per_Unit': 'mean',
    'Promotion_Flag': 'first',
    'Weather_Temp': 'mean',
    'Competitor_Price': 'mean',
    'Social_Sentiment': 'mean'
}).reset_index()
    
    df_daily = df.groupby(['Product_Name', 'Region']).resample('W', on='Date').agg(agg_dict).reset_index()

    # One-hot encode Product & Region


    print(df_daily.head())
    print(df_daily.shape)
    print(type(df_daily))
    print(df_daily.dtypes)
    print('index_before',df_daily.index)
    df_daily.set_index('Date', inplace=True)
    print('index_after',df_daily.index)

    
    
    plt.plot(df_daily.index, df_daily['Units_Sold'], label='Original', color='blue', linestyle='-')

# See some duplicate rows
    print(df_daily.head(10))
    print('null value',df_daily.isnull().sum())
    print(df_daily.shape)
    print(df_daily.index.duplicated().sum())
    print(df_daily.dtypes)
    
    print(df_daily.columns)
    num_cols = df_daily.select_dtypes(include=['int64', 'float64']).columns.drop("Units_Sold")
    print("Numerical columns:", num_cols)
    
    
    print(df_daily.head(10))
    print(df_daily.shape)
        
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = cat_cols.drop('Date')  # Exclude 'Date' from categorical columns
    print("Categorical columns:", cat_cols)
    
    # df_daily = pd.get_dummies(df_weekly, columns=['Product_Name', 'Region'])
    
    mlb = MultiLabelBinarizer()

# Columns that contain multiple labels in lists
    for col in cat_cols:
        df_encoded = pd.DataFrame(mlb.fit_transform(df_weekly[col]),
                            columns=[f"{col}_{cls}" for cls in mlb.classes_])
        df_weekly = pd.concat([df_weekly.drop(columns=[col]), df_encoded], axis=1)

# Optional: encode Product_Name and Region for exogenous variables
    for col in ['Product_Name', 'Region']:
        df_encoded = pd.get_dummies(df_weekly[col], prefix=col)
        df_weekly = pd.concat([df_weekly.drop(columns=[col]), df_encoded], axis=1)

    
    scaler=StandardScaler()
    for col in num_cols:
        df_daily[col]=scaler.fit_transform(df_daily[col].values.reshape(-1,1))
        # print(df[col].unique())
        log.info(f'Scaled column: {col}')
        
    # df_daily.drop(columns=['Social_Sentiment','Weather_Temp',],inplace=True)
    # print(df_daily.head(10))
        
    plt.plot(df_daily.index, df_daily['Units_Sold'], label='After Scaling', color='red', linestyle=':')

    plt.title('Units Sold Over Time (All Versions)')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Your series
    y = df_daily['Units_Sold']

# ADF test
    adf_result = adfuller(y)
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    for key, value in adf_result[4].items():
        print(f'Critical Value {key}: {value}')

# Interpretation
    if adf_result[1] <= 0.05:
        print("Series is stationary → no differencing needed (d=0)")
    else:
        print("Series is non-stationary → differencing needed (d=1 or more)")

    y_diff = y.diff().dropna()

    # Plot ACF & PACF for original series or differenced series
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plot_acf(y, lags=50, ax=plt.gca())
    plt.title('ACF Plot')

    plt.subplot(1,2,2)
    plot_pacf(y, lags=50, ax=plt.gca(), method='ywm')
    plt.title('PACF Plot')

    plt.tight_layout()
    plt.show()
    
    # If weekly seasonality (m=7)
    plot_acf(y, lags=28)  # 4 weeks
    plt.show()

    plot_pacf(y, lags=28, method='ywm')
    plt.show()



    return df_daily

# ----------- TRAIN-TEST SPLIT -----------
def train_test_split_by_series(df, test_size=0.2, min_length=20):
    """
    Splits the dataframe into train and test sets for each product-region combination.
    
    Parameters:
        df (pd.DataFrame): The input dataframe containing 'Date', product, and region columns.
        test_size (float): Fraction of data to be used as test set.
        min_length (int): Minimum number of rows required to create a split.
    
    Returns:
        dict: Keys are (product_col, region_col), values are (train_df, test_df).
    """
    df = df.sort_values('Date')  # Ensure time order
    product_cols = [col for col in df.columns if col.startswith('Product_Name_')]
    region_cols = [col for col in df.columns if col.startswith('Region_')]
    
    series_dict = {}
    for prod_col in product_cols:
        for reg_col in region_cols:
            df_series = df[(df[prod_col]==1) & (df[reg_col]==1)].copy()
            if len(df_series) < min_length:
                continue  # Skip very short series
            df_series.set_index('Date', inplace=True)
            split_idx = int(len(df_series) * (1 - test_size))
            train = df_series.iloc[:split_idx]
            test = df_series.iloc[split_idx:]
            series_dict[(prod_col, reg_col)] = (train, test)
    
    print(f"Created {len(series_dict)} product-region series splits.")
    return series_dict



# ----------- AUTO ARIMA SEARCH -----------
def auto_arima_search(train, exog_train=None, seasonal_m=7, min_periods=20):
    """
    Performs auto ARIMA search for a given train series.

    Parameters:
        train (pd.DataFrame): Training data with 'Units_Sold' column.
        exog_train (pd.DataFrame): Optional exogenous variables.
        seasonal_m (int): Seasonality period (e.g., 7 for weekly).
        min_periods (int): Minimum number of rows to perform ARIMA search.

    Returns:
        tuple: (order, seasonal_order) if successful, else None.
    """
    if len(train) < min_periods:
        print("Series too short for ARIMA:", len(train))
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
            stepwise=False, trace=True,
            suppress_warnings=True
        )
        print(stepwise_model.summary())
        return stepwise_model.order, stepwise_model.seasonal_order
    except Exception as e:
        print("Auto ARIMA failed:", e)
        return None
    

# ----------- BUILD & TRAIN MODEL -----------
from statsmodels.tsa.statespace.sarimax import SARIMAX

def build_and_train_model(df_weekly, order_dict=None, seasonal_order_dict=None, min_length=20):
    """
    Trains SARIMAX models for each product-region combination.

    Parameters:
        df_weekly (pd.DataFrame): Weekly data with products, regions, and 'Units_Sold'.
        order_dict (dict): Dictionary {(product, region): order} from auto_arima.
        seasonal_order_dict (dict): Dictionary {(product, region): seasonal_order} from auto_arima.
        min_length (int): Minimum series length to train a model.

    Returns:
        dict: {(product, region): fitted SARIMAX model}
    """
    results_dict = {}
    exog_cols = [col for col in df_weekly.columns if col not in ['Date','Units_Sold','Revenue']]

    for product in df_weekly['Product_Name'].unique():
        for region in df_weekly['Region'].unique():
            df_prod = df_weekly[(df_weekly[f'Product_Name_{product}']==1) &
                                (df_weekly[f'Region_{region}']==1)].copy()
            
            if len(df_prod) < min_length:
                continue  # skip short series

            df_prod.set_index('Date', inplace=True)

            # Train/test split
            split_index = int(len(df_prod)*0.8)
            train = df_prod.iloc[:split_index]
            test = df_prod.iloc[split_index:]

            exog_train = train[exog_cols] if exog_cols else None
            exog_test = test[exog_cols] if exog_cols else None

            # Get ARIMA orders from provided dicts
            order = order_dict.get((product, region), (1,0,1))
            seasonal_order = seasonal_order_dict.get((product, region), (1,1,1,52))

            # Fit SARIMAX
            model = SARIMAX(
                train['Units_Sold'],
                order=order,
                seasonal_order=seasonal_order,
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            results_dict[(product, region)] = res
            print(f"Trained SARIMAX for {product} | {region}")

    return results_dict



# ----------- SAFE MAPE -----------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100


# ----------- FORECAST & EVALUATE -----------
def forecast_evaluate(results, train, test, exog_test=None):
    # Forecast
    pred = results.get_forecast(steps=len(test), exog=exog_test)
    pred_ci = pred.conf_int()

    # Extract forecasted and actual values
    y_forecasted = pred.predicted_mean
    y_truth = test['Units_Sold']

    # Metrics
    mse = mean_squared_error(y_truth, y_forecasted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_truth, y_forecasted)
    mape = safe_mape(y_truth, y_forecasted)

    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "MAPE"],
        "Value": [mse, rmse, mae, mape]
    })
    print(metrics_df)

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['Units_Sold'], label="Train")
    plt.plot(test.index, y_truth, label="Test")
    plt.plot(test.index, y_forecasted, color="red", label="Forecast")
    plt.fill_between(
        pred_ci.index,
        pred_ci.iloc[:, 0],
        pred_ci.iloc[:, 1],
        color='k', alpha=.2
    )
    plt.legend()
    plt.show()

    return y_forecasted, metrics_df



# ----------- SAVE MODEL -----------
def save_model(results, model_path='sarimax_model.pkl'):
    with open(model_path, 'wb') as pkl:
        pickle.dump(results, pkl)
    print(f'Model saved to {model_path}')
    return model_path


# ----------- MAIN PIPELINE -----------
def main():

    # Clean + preprocess
    df_cleaned = cleaning_preprocessing()
    
    df_cleaned.to_csv('cleaned_data.csv', index=False)
    
    
    # Split
    series_dict = train_test_split_by_series(df_cleaned, test_size=0.2)

    # # Exogenous variables
    exog_cols = [ 'Weather_Temp', 'Competitor_Price', 'Social_Sentiment']
    # After encoding list columns
    # exog_cols = [col for col in df_cleaned.columns if col != 'Units_Sold']

# Convert all to numeric (float)
    df_cleaned[exog_cols] = df_cleaned[exog_cols].apply(pd.to_numeric, errors='coerce')
    df_cleaned.fillna(method='ffill', inplace=True)


# Fill any remaining NaNs

    exog_train = train[exog_cols]
    
    exog_test = test[exog_cols]

    print(train.dtypes)
    print(exog_train.dtypes)

    exog_train = train[exog_cols].astype(float)
    exog_test = test[exog_cols].astype(float)

    print(train.dtypes)
    print(exog_train.dtypes)
    
    print(df_cleaned.head())
    print(df_cleaned.shape)
    print(exog_test.head())
    print(exog_test.shape)
    print(exog_test.isnull().sum())
    print(exog_train.head())
    print(exog_train.isnull().sum())
    print(exog_train.shape)
    print(test.head())
    print(test.shape)
    print(train.head())
    print(train.shape)

    # # Find best orders
    # Initialize dictionaries
    order_dict = {}
    seasonal_order_dict = {}

    for key, (train, test) in series_dict.items():
        # Get exogenous variables if needed
        exog_cols = [col for col in train.columns if col not in ['Units_Sold', 'Revenue']]
        exog_train = train[exog_cols] if exog_cols else None

        result = auto_arima_search(train, exog_train=exog_train, seasonal_m=7, min_periods=20)
    
        if result is not None:
            order, seasonal_order = result
            order_dict[key] = order
            seasonal_order_dict[key] = seasonal_order
        else:
            print(f"Skipping ARIMA for series {key} due to insufficient data or error.")

    print(f"Generated orders for {len(order_dict)} series.")
    # # Train
    results = build_and_train_model(train, order_dict=order_dict,seasonal_order_dict=seasonal_order_dict exog_train)

    # # Evaluate
    y_forecasted, metrics_df = forecast_evaluate(results, train, test, exog_test)

    # # Save model
    # model_path = save_model(results)


if __name__ == "__main__":
    main()
