import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import datetime
from pandas.plotting import lag_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

df = pd.read_csv("stock_data.csv")
df.head(5)

plt.figure()
lag_plot(df['Open'], lag=3)
plt.title('stock prediction graph')
plt.show()

plt.plot(df["Date"], df["Close"])
plt.xticks(np.arange(0,1259, 200), df['Date'][0:1259:200])
plt.title("Q3")
plt.xlabel("x-axis_time")
plt.ylabel("y-axis_price")
plt.show()


train, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
training = train['Close'].values
test_data = test_data['Close'].values
history = [x for x in training]
model_predictions = []
N_test_observations = len(test_data)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(4,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)
MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))