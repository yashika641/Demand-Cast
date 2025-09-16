import warnings
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.special import boxcox1p, inv_boxcox1p
from tqdm import tqdm
import sys, os

# Logger import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Arima.src.logger import get_logger
log = get_logger(__name__)


# ------------------------------
# Walk-forward RMSE with optional step size
# ------------------------------
def walk_forward_rmse(y, X, order, sorder, start_frac=0.7, step=2, maxiter=150):
    """ Walk-forward validation with downsampling (step>1 to speed up) """
    n0 = int(len(y) * start_frac)
    preds, actuals = [], []
    for t in range(n0, len(y), step):
        y_train, X_train = y.iloc[:t], X.iloc[:t]
        y_test_pt, X_test_pt = y.iloc[t:t+1], X.iloc[t:t+1]
        if len(y_train) < (sorder[3] * 2):  # skip if not enough data for seasonal period
            return np.inf, np.inf
        try:
            res = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=sorder,
                        enforce_stationarity=False, enforce_invertibility=False
                        ).fit(disp=False, method='lbfgs', maxiter=maxiter)
            pred = res.predict(start=y_test_pt.index[0], end=y_test_pt.index[0], exog=X_test_pt)
            preds.append(float(pred.iloc[0]))
            actuals.append(float(y_test_pt.iloc[0]))
        except Exception:
            return np.inf, np.inf
    if not preds:
        return np.inf, np.inf
    preds, actuals = np.array(preds), np.array(actuals)
    rmse = np.sqrt(np.mean((preds - actuals)**2))
    mape = np.mean(np.abs((actuals - preds) / np.maximum(1e-8, np.abs(actuals)))) * 100
    return rmse, mape


# ------------------------------
# Grid search for best SARIMA orders
# ------------------------------
def optimize_sarima(ts, exog, season_m,
                    p_values=[0,1,2],
                    d_values=[0,1],
                    q_values=[0,1,2],
                    P_values=[0,1],
                    D_values=[0,1],
                    Q_values=[0,1],
                    start_frac=0.7,
                    step=2):
    best_score = (np.inf, np.inf)
    best_order, best_seasonal_order = None, None
    total_combinations = len(p_values)*len(d_values)*len(q_values)*len(P_values)*len(D_values)*len(Q_values)
    print(f"Searching SARIMA orders ({total_combinations} combinations)...")

    with tqdm(total=total_combinations, desc="SARIMA Grid Search") as pbar:
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                order = (p,d,q)
                                sorder = (P,D,Q,season_m)
                                rmse, mape = walk_forward_rmse(ts, exog, order, sorder,
                                                            start_frac=start_frac, step=step)
                                if rmse < best_score[0]:
                                    best_score = (rmse, mape)
                                    best_order, best_seasonal_order = order, sorder
                                    print(f"✅ New best: order={order}, seasonal={sorder}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
                                pbar.update(1)

    print(f"\n🏆 Best SARIMA: order={best_order}, seasonal={best_seasonal_order}, RMSE={best_score[0]:.4f}")
    return best_order, best_seasonal_order


# ------------------------------
# Main SARIMA pipeline
# ------------------------------
def sarima_forecast_pipeline(
    csv_path,
    target_col="Units_Sold",
    exog_cols=["Promotion_Flag", "Weather_Temp", "Competitor_Price", "Festival_Season"],
    seasonal_m=7,
    boxcox_lambda=0.0,
    train_frac=0.8,
    forecast_steps=30,
    plot=True
):
    # Load
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    date_col_candidates = [c for c in df.columns if c.lower() == "date"]
    if not date_col_candidates:
        raise KeyError("⚠️ No 'Date' column found.")
    date_col = date_col_candidates[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Resample weekly
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    df_numeric_weekly = numeric_df.resample("W").mean()
    non_num = df.drop(columns=numeric_df.columns, errors="ignore")
    if not non_num.empty:
        non_num_weekly = non_num.resample("W").last().ffill()
        df = pd.concat([df_numeric_weekly, non_num_weekly], axis=1)
    else:
        df = df_numeric_weekly
    df = df.ffill()

    # Encode categorical exog
    for col in exog_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.factorize(df[col].astype(str).str.strip().str.lower())[0].astype(float)
    df[[target_col] + exog_cols] = df[[target_col] + exog_cols].ffill().bfill()

    # ---- Feature engineering ----
    for L in [1, 2, 3, 7, 14]:
        df[f"{target_col}_lag{L}"] = df[target_col].shift(L)
    for c in exog_cols:
        for L in [1, 2, 3, 7]:
            df[f"{c}_lag{L}"] = df[c].shift(L)
    for w in [3, 7, 14]:
        df[f"{target_col}_rollmean{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"{target_col}_rollstd{w}"] = df[target_col].shift(1).rolling(w).std()
    df = df.dropna()

    # Outlier clipping
    q1, q99 = df[target_col].quantile([0.01, 0.99])
    df[target_col] = df[target_col].clip(q1, q99)

    # Box-Cox transform
    df[target_col] = boxcox1p(df[target_col], boxcox_lambda)

    # Train/Test split
    ts = df[target_col].astype(float)
    feature_cols =  ["Product_Category", "Promotion_Flag", "Festival_Season", 
                "Weather_Temp", "Competitor_Price", "Social_Sentiment",'Product_Name','Region']  # all exog including engineered
    exog = df[feature_cols].astype(float)
    train_size = int(len(ts) * train_frac)
    train_y, test_y = ts.iloc[:train_size], ts.iloc[train_size:]
    train_X, test_X = exog.iloc[:train_size], exog.iloc[train_size:]

    # Scale exogenous
    scaler = StandardScaler()
    train_X = pd.DataFrame(scaler.fit_transform(train_X), index=train_X.index, columns=train_X.columns)
    test_X = pd.DataFrame(scaler.transform(test_X), index=test_X.index, columns=test_X.columns)

    # Optimize SARIMA orders
    best_order, best_seasonal_order = optimize_sarima(train_y, train_X, seasonal_m)

    # Fit final SARIMA
    model = SARIMAX(train_y,
                    order=best_order,
                    seasonal_order=best_seasonal_order,
                    exog=train_X,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=100)

    # Predictions
    pred = results.predict(start=test_y.index[0], end=test_y.index[-1], exog=test_X)
    pred = inv_boxcox1p(pred, boxcox_lambda)
    test_y_inv = inv_boxcox1p(test_y, boxcox_lambda)

    # Metrics
    mse = mean_squared_error(test_y_inv, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_y_inv, pred)
    mape = np.mean(np.abs((test_y_inv - pred) / test_y_inv)) * 100
    print("\n📊 Test Set Metrics:")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # Plot
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(train_y.index, inv_boxcox1p(train_y, boxcox_lambda), label="Train")
        plt.plot(test_y.index, test_y_inv, label="Test")
        plt.plot(pred.index, pred, label="SARIMA Predictions")
        plt.legend()
        plt.show()

    # Future forecast
    future_index = pd.date_range(ts.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq="W")
    future_exog = pd.DataFrame(np.repeat(exog.iloc[[-1]].values, forecast_steps, axis=0),
                            columns=exog.columns, index=future_index)
    future_exog = pd.DataFrame(scaler.transform(future_exog), index=future_index, columns=exog.columns)
    future_forecast = results.get_forecast(steps=forecast_steps, exog=future_exog).predicted_mean
    future_forecast = inv_boxcox1p(future_forecast, boxcox_lambda)

    print("\nFuture Forecast (next {} steps):".format(forecast_steps))
    print(future_forecast)

    return results, pred, test_y_inv, future_forecast

if __name__ == "__main__":
    # Load data
    # Just pass the file path string, not the DataFrame
    csv_path = r"C:\Users\palya\Desktop\DemandCast\Demand-Cast\synthetic_sales_2yrs.csv"
    results, pred, actuals, future_forecast = sarima_forecast_pipeline(csv_path, plot=True)
