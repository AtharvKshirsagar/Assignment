import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the data from CSV file
data = pd.read_csv('stock_data.csv')

# Convert the 'Date' column to a datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date' column as the index of the DataFrame
data.set_index('Date', inplace=True)

# Extract the 'Close' column as a numpy array
close_prices = data['Close'].to_numpy()

# Fit an ARMA model to the entire data
model = sm.tsa.ARIMA(close_prices, order=(1, 0, 1))
result = model.fit()

# Predict the next day's price using the fitted model
next_day_price = result.forecast()[0]

print(f"Predicted price for the next day: {next_day_price}")
