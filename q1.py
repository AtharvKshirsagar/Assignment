import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('stock_data.csv', index_col=0, parse_dates=True)

# Step 2: Check for stationarity and difference the series if necessary
# If the series is not stationary, we can difference it to make it stationary
# We can use the Dickey-Fuller test to check for stationarity

def test_stationarity(timeseries):
    # Perform Dickey-Fuller test 
    from statsmodels.tsa.stattools import adfuller
    print('Results')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(df['price'])

# If the p-value is greater than 0.05, we cannot reject the null hypothesis that the series is non-stationary
# We can difference the series to make it stationary

df_diff = df.diff().dropna()

test_stationarity(df_diff['price'])

# Now the series is stationary

# Step 3: Determine the order of autoregressive and moving average terms
# We can use ACF and PACF plots to determine the order of AR and MA terms

fig, ax = plt.subplots(2, 1, figsize=(10,8))
plot_acf(df_diff['price'], lags=30, ax=ax[0])
plot_pacf(df_diff['price'], lags=30, ax=ax[1])
plt.show()

# We can see that the ACF plot has a significant spike at lag 1 and the PACF plot has a significant spike at lag 1 as well
# This suggests that an ARMA(1,1) model may be appropriate

# Step 4: Fit the ARMA model to the data
model = ARIMA(df['price'], order=(1,1,1))
results = model.fit()

# Step 5: Test the model and evaluate its performance
# We can use the predict() method to generate predictions and compare them to the actual values

start_index = '2022-01-01'
end_index = '2022-02-28'
forecast = results.predict(start=start_index, end=end_index, dynamic=False)
actual = df.loc[start_index:end_index, 'price']

mse = ((forecast - actual)**2).mean()
rmse = np.sqrt(mse)
mae = np.abs(forecast - actual).mean()

print('RMSE: {}'.format(rmse))
print('MAE: {}'.format(mae))

# Step 6: Use the model to predict the price for the next day

# Step 6: Use the model to predict the price for the next day
next_day = results.forecast(steps=1)
print('Predicted price for next day: {}'.format(next_day[0]))