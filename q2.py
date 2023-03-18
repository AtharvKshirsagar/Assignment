import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the data from CSV file
data = pd.read_csv('stock_data.csv')

# Convert the 'Date' column to a datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date' column as the index of the DataFrame
data.set_index('Date', inplace=True)

# Extract the 'Close' column as a numpy array
close_prices = data['Close'].to_numpy()

# Split the data into training and testing sets
train_data = close_prices[:-90]
test_data = close_prices[-90:]

# Fit an ARMA model to the training data
model = sm.tsa.ARIMA(train_data, order=(1, 0, 1))
result = model.fit()

# Make predictions on the testing data using the fitted model
predictions = result.predict(start=len(train_data), end=len(close_prices)-1)

# Calculate the mean absolute error (MAE) of the predictions
mae = np.mean(np.abs(predictions - test_data))

# Plot the actual and predicted values
plt.plot(close_prices[-90:], label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

# Calculate the profit/loss based on the ARMA predictions
buy_price = close_prices[-91]  # The price at the start of the test period
sell_price = close_prices[-1]  # The price at the end of the test period
predicted_prices = np.concatenate(([buy_price], predictions))
returns = np.diff(predicted_prices) / predicted_prices[:-1]
profit_loss = np.sum(returns) * buy_price

print(f"ARMA model MAE: {mae}")
print(f"ARMA model profit/loss: {profit_loss}")
