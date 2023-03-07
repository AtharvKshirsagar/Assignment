# Importing the necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Loading the data into a Pandas DataFrame
df = pd.read_csv('stock_data')

# Preparing the data for training and testing
train = df[:252] #stock working day
test = df[252:]

# Fitting an ARMA model on the training dataset
model = ARIMA(train['Close'], order=(1, 1, 1))
model_fit = model.fit()

# Making predictions on the testing dataset
predictions = model_fit.forecast(steps=len(test))[0]

# Calculating the accuracy of the predictions
rmse = np.sqrt(mean_squared_error(test['Close'], predictions))
print('RMSE:', rmse)

# Calculating the profit/loss based on the ARMA predictions
test['Predictions'] = predictions
test['Signal'] = np.where(test['Predictions'] > test['Close'], 1, -1)
test['PL'] = test['Signal'] * (test['Open'] - test['Close'])
profit_loss = test['PL'].sum()
print('Profit/Loss:', profit_loss)
