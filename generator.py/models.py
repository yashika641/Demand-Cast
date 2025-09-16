from keras.optimizers import Adam, SGD
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
import pandas as pd

def xgboost_regression( x_train,y_train, x_test,y_test, 
    n_estimators,
    max_depth,
    learning_rate,
    min_child_weight,
    subsample,
    colsample_bytree,
    reg_alpha,
    reg_lambda,
    objective,
    eval_metric,
    early_stopping_rounds,
    num_boosting_rounds,
    max_leaves,
    max_bin,
    scale_pos_weight,
    gamma,
    lambda_l1,
    lambda_l2,
    importance_type):
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective=objective,
        eval_metric=eval_metric,
        max_leaves=max_leaves,
        max_bin=max_bin,
        scale_pos_weight=scale_pos_weight,
        gamma=gamma,
        importance_type=importance_type,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        verbosity=1
    )

    # Fit model
    if x_test is not None and y_test is not None:
        model.fit(
            x_train, y_train,
            eval_set=[(x_test, y_test)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=True
        )
    else:
        model.fit(x_train, y_train, verbose=True)

    return model

def prophet_forecast_train_test(
    x_train, 
    y_train,
    x_test,
    y_test,
    changepoint_prior_scale,
    seasonality_mode ,
    seasonality_prior_scale ,
    holidays_prior_scale ,
    daily_seasonality ,
    weekly_seasonality ,
    yearly_seasonality 
):
    """
    Train and forecast using Prophet with explicit train and test sets.

    Args:
        x_train, x_test: datetime series (pandas Series or list)
        y_train, y_test: target series (pandas Series or list)
        changepoint_prior_scale: flexibility of trend changes
        seasonality_mode: 'additive' or 'multiplicative'
        seasonality_prior_scale: weight of seasonality prior
        holidays_prior_scale: weight of holidays prior
        daily_seasonality, weekly_seasonality, yearly_seasonality: booleans

    Returns:
        model: fitted Prophet model
        forecast: forecast DataFrame including test period
    """

    # Create DataFrame for training
    train_df = pd.DataFrame({'ds': x_train, 'y': y_train})

    # Initialize Prophet model
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality
    )

    # Fit model
    model.fit(train_df)

    # Create future DataFrame including test period
    future_df = pd.DataFrame({'ds': list(x_train) + list(x_test)})

    # Forecast
    forecast = model.predict(future_df)

    return model, forecast


def arima_sarima_train_test(
    y_train, 
    y_test=None,
    p: int = 1, 
    d: int = 1, 
    q: int = 1, 
    P: int = 1, 
    D: int = 1, 
    Q: int = 1, 
    m: int = 1,
    seasonal: bool = True
):
    """
    Train ARIMA or SARIMA model on training data, optionally forecast on test data.

    Args:
        y_train: training target series (pd.Series or list)
        y_test: optional test series
        p,d,q: ARIMA orders
        P,D,Q,m: seasonal orders
        seasonal: whether to use SARIMA

    Returns:
        model: fitted ARIMA/SARIMA model
        forecast: forecasted values (for test period if provided, else in-sample)
    """
    if seasonal:
        model = SARIMAX(y_train, order=(p,d,q), seasonal_order=(P,D,Q,m), enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = ARIMA(y_train, order=(p,d,q))
    
    fitted_model = model.fit()
    
    if y_test is not None:
        steps = len(y_test)
        forecast = fitted_model.forecast(steps=steps)
    else:
        forecast = fitted_model.fittedvalues

    return fitted_model, forecast

def train_lstm(
    x_train,
    y_train,
    x_test,
    y_test,
    timesteps: int = 10,
    hidden_size: int = 32,
    dropout: float = 0.1,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    optimizer: str = 'Adam',
    loss_function: str = 'mean_squared_error',
    metrics_list: list = ['mean_squared_error']
):
    """
    Build and train LSTM model.

    Args:
        x_train, y_train, x_test, y_test: training and testing datasets (numpy arrays)
        timesteps: input sequence length
        hidden_size: number of LSTM units
        dropout: dropout rate
        epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
        optimizer: 'Adam' or 'SGD'
        loss_function: loss function for training
        metrics_list: list of metrics

    Returns:
        model: trained Keras LSTM model
        history: training history
    """
    # Define model
    x_train = x_train.values.reshape((x_train.shape[0], timesteps, x_train.shape[1]//timesteps))
    x_test = x_test.values.reshape((x_test.shape[0], timesteps, x_test.shape[1]//timesteps))

    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))  # Assuming regression task

    # Select optimizer
    if optimizer.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer. Choose 'Adam' or 'SGD'.")

    model.compile(optimizer=opt, loss=loss_function, metrics=metrics_list)

    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history
