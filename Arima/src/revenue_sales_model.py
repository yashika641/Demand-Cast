import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import tqdm
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # optional: silence sklearn deprecation noise

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from tqdm import tqdm
import time

df= pd.read_csv(r'C:\Users\palya\Desktop\DemandCast\Demand-Cast\datasets\cleaned_data.csv')

# ----------------------------
print("Initial columns:", df.columns.tolist())
# strip whitespace from column names
df.columns = df.columns.str.strip()

# find the date column robustly (case-insensitive match for 'date')
date_col_candidates = [c for c in df.columns if c.lower() == "date"]
if not date_col_candidates:
    raise KeyError("⚠️ Could not find a 'Date' column (case-insensitive). Check your CSV columns.")
DATE_COL = date_col_candidates[0]

# parse datetime
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")    
if df[DATE_COL].isna().all():
    raise ValueError("⚠️ All parsed dates are NaT. Check your date format in the CSV.")
# drop rows with NaT dates
df = df.dropna(subset=[DATE_COL])

# set index & sort
df = df.set_index(DATE_COL).sort_index()
print("Datetime index set. Example index range:", df.index.min(), "→", df.index.max())
# ----------------------------
# 2) Optional resampling (faster for huge series)
# If you have ~1.5 lakh rows, weekly aggregation will speed up ARIMA massively.
RESAMPLE_TO_WEEKLY = True

if RESAMPLE_TO_WEEKLY:
    # we’ll compute numeric means; for non-numeric exog we forward-fill later
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    df_numeric_weekly = numeric_df.resample("W").mean()

    # bring back non-numeric columns via forward-fill on weekly index (optional)
    non_num = df.drop(columns=numeric_df.columns, errors="ignore")
    if not non_num.empty:
        non_num_weekly = non_num.resample("W").last().ffill()
        df = pd.concat([df_numeric_weekly, non_num_weekly], axis=1)
    else:
        df = df_numeric_weekly

    df = df.ffill()  # fill any gaps after resample
    season_m = 52     # weekly seasonality ~ yearly cycle
else:
    df = df.sort_index().ffill()
    season_m = 7      # example: daily data with weekly seasonality

print("Post-resample shape:", df.shape)
print("Columns after resample/clean:", df.columns.tolist())

# ----------------------------
# 3) Ensure required columns exist & are numeric
# ----------------------------
TARGET_COL = "Revenue"
EXOG_COLS = ["Promotion_Flag", "Weather_Temp", "Competitor_Price", "Festival_Season"]

for col in [TARGET_COL] + EXOG_COLS:
    if col not in df.columns:
        raise KeyError(f"⚠️ Required column '{col}' not found in DataFrame.")

# coerce exog columns to numeric if needed (e.g., 'Yes'/'No' → 1/0)
for col in EXOG_COLS:
    if not np.issubdtype(df[col].dtype, np.number):
        # simple binary mapping if strings; otherwise factorize
        df[col] = df[col].astype(str).str.strip().str.lower()
        if set(df[col].unique()) <= {"yes", "no", "1", "0", "true", "false"}:
            df[col] = df[col].map({"yes":1, "no":0, "1":1, "0":0, "true":1, "false":0}).astype(float)
        else:
            df[col] = pd.factorize(df[col])[0].astype(float)

# final NA fill for safety
df[[TARGET_COL] + EXOG_COLS] = df[[TARGET_COL] + EXOG_COLS].ffill().bfill()

# ----------------------------
# 4) Train/Test split (aligned target + exog)
# ----------------------------
ts = df[TARGET_COL].astype(float)
exog = df[EXOG_COLS].astype(float)

train_size = int(len(ts) * 0.8)
train_y, test_y = ts.iloc[:train_size], ts.iloc[train_size:]
train_X, test_X = exog.iloc[:train_size], exog.iloc[train_size:]

print("Train size:", train_y.shape, "Test size:", test_y.shape)
print("Train NA check (target):", int(train_y.isna().sum()), "dtype:", train_y.dtype)
# ----------------------------

# 5) Progress bar for auto_arima via callback
# ----------------------------
# estimate total combos for progress (rough; stepwise reduces it)
def estimate_total_models(max_p, max_q, max_P, max_Q, seasonal=True):
    total = (max_p + 1) * (max_q + 1)
    if seasonal:
        total *= (max_P + 1) * (max_Q + 1)
    return max(total, 1)

max_p, max_q, max_P, max_Q = 2, 2, 2, 2
total_models = estimate_total_models(max_p, max_q, max_P, max_Q, seasonal=True)
pbar = tqdm(total=total_models, desc="auto_arima search", unit="model")

def progress_cb(res):
    # res is a dict of fit results (pmdarima internal); we just tick the bar
    # guard against over-run
    if pbar.n < pbar.total:
        pbar.update(1)

# IMPORTANT:
# - stepwise=True ignores n_jobs (pmdarima constraint); we set n_jobs=1 to avoid warnings
# - we DO NOT pass exog to auto_arima here to keep search faster; we add exog in final SARIMAX

stepwise_model=auto_arima(
    train_y,
    start_p=0,start_q=0,
    max_p=max_p, max_q=max_q,
    start_P=0,start_Q=0,
    max_P=max_P, max_Q=max_Q,
    m=season_m,
    seasonal=True,
    stepwise=False,
    d=None, D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    n_jobs=1,
    callback=progress_cb,
    # see https://github.com/alkaline-ml/pmdarima/issues/220
    # for why we need to set n_jobs=1 here
    # n_jobs=1,
    # n_fits
)

pbar.close()

print("\nBest order found:", stepwise_model.order, "seasonal:", stepwise_model.seasonal_order)
print(stepwise_model.summary())

order = stepwise_model.order
seasonal_order = stepwise_model.seasonal_order

# ----------------------------
# 6) SARIMAX training with progress bar
# ----------------------------      
print("\nFitting SARIMAX model...")
print("\nFitting SARIMAX model...")
with tqdm(total=1, desc="Training SARIMAX", bar_format="{l_bar}{bar} [ elapsed: {elapsed} ]") as fitbar:
    model = SARIMAX(
        train_y,
        order=order,
        seasonal_order=seasonal_order,
        exog=train_X,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    fitbar.update(1)

print(results.summary())
# ----------------------------
# 7) In-sample prediction for test window
# ----------------------------
pred = results.predict(
    start=train_y.index[-1],
    end=test_y.index[-1],
    exog=test_X,
    dynamic=False
)
pred = pred.reindex(test_y.index)

# ----------------------------
# 7b) Error Metrics on Test Set
# ----------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(test_y, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_y, pred)
mape = np.mean(np.abs((test_y - pred) / test_y)) * 100

print("\n📊 Test Set Evaluation Metrics:")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%")

# --- (plot goes here if you want) ---
plt.figure(figsize=(12, 6))
plt.plot(train_y.index, train_y, label="Train")
plt.plot(test_y.index, test_y, label="Test")
plt.plot(pred.index, pred, label="Predictions")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 8) Future 30-step forecast
# ----------------------------
future_steps = 30
freq = pd.infer_freq(ts.index)
if freq is None:
    freq = "W" if RESAMPLE_TO_WEEKLY else "D"

future_index = pd.date_range(ts.index[-1] + pd.tseries.frequencies.to_offset(freq),
                            periods=future_steps, freq=freq)

last_exog = exog.iloc[[-1]].copy()
future_exog = pd.DataFrame(
    np.repeat(last_exog.values, future_steps, axis=0),
    columns=EXOG_COLS, index=future_index
)

future_forecast = results.predict(
    start=len(ts),
    end=len(ts) + future_steps - 1,
    exog=future_exog
)

print("\nFuture Forecast (next 30 steps):")
print(future_forecast)

# ----------------------------
# 1. Naïve Baseline
# ----------------------------
# Predict each test point as the last known value
naive_forecast = test_y.shift(1).fillna(train_y.iloc[-1])

mse_naive = mean_squared_error(test_y, naive_forecast)
rmse_naive = np.sqrt(mse_naive)
mae_naive = mean_absolute_error(test_y, naive_forecast)
mape_naive = np.mean(np.abs((test_y - naive_forecast) / test_y)) * 100

print("\n📊 Naïve Baseline Metrics:")
print(f"MSE  : {mse_naive:.4f}")
print(f"RMSE : {rmse_naive:.4f}")
print(f"MAE  : {mae_naive:.4f}")
print(f"MAPE : {mape_naive:.2f}%")

# ----------------------------
# 2. Moving Average Baseline (window=3)
# ----------------------------
moving_avg_forecast = test_y.rolling(window=3).mean().shift(1).fillna(train_y.iloc[-1])

mse_ma = mean_squared_error(test_y, moving_avg_forecast)
rmse_ma = np.sqrt(mse_ma)
mae_ma = mean_absolute_error(test_y, moving_avg_forecast)
mape_ma = np.mean(np.abs((test_y - moving_avg_forecast) / test_y)) * 100

print("\n📊 Moving Average Baseline Metrics (window=3):")
print(f"MSE  : {mse_ma:.4f}")
print(f"RMSE : {rmse_ma:.4f}")
print(f"MAE  : {mae_ma:.4f}")
print(f"MAPE : {mape_ma:.2f}%")