import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima


df=pd.read_csv(r'C:\Users\palya\Desktop\DemandCast\Demand-Cast\datasets\cleaned_data.csv')

print(df.head())
# Make sure your data is sorted
df = df.sort_values('Date')

# Set Date as index
df = df.set_index('Date')

# We'll model on Units_Sold
ts = df['Units_Sold']


train_size = int(len(ts) * 0.8)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

stepwise_model =auto_arima(train,
                        start_p=1,start_q=1,
                        max_p=5,max_q=5,
                        m=7,
                        start_P=0,seasonal=True,
                        d=None,D=1,trace=True,
                        error_action='ignore',suppress_warnings=True,
                        stepwise=True)

print(stepwise_model.summary())

order = stepwise_model.order
seasonal_order = stepwise_model.seasonal_order
exog_features = df[['Promotion_Flag', 'Weather_Temp', 'Competitor_Price','Festival_Season']].iloc[:train_size]
model=SARIMAX(train,
            order=order,
            exog=exog_features.iloc[:train_size],
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
            )

results=model.fit()
print(results.summary())

predictions =results.predict(start=len(train),end=len(train)+len(test)-1,exog=exog_features.iloc[train_size:],dynamic=False).rename('Predictions')
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Predictions', color='red')
plt.legend()
plt.show()

future_steps = 30  # predict next 30 days
future_forecast = results.predict(start=len(ts), end=len(ts)+future_steps-1, dynamic=False)
print(future_forecast)
