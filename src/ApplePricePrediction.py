
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout


np.random.seed(7)

def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset)-step_size-1):
        a = dataset[i:(i+step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)


def load_data():
    data = pd.read_csv("combined.csv", usecols=[0,1,3,4,5,6])
    return data
'''
0-> date
1-> Ticker
3-> First Trade price -> Open
4-> High Trade price -> High
5-> Low Trade price -> Low
6-> Last Trade price -> Close
'''
# df = load_data()


df = pd.read_csv("../dataset/data/2000-2020/AAPL_2000-01-01_to_2020-10-31.csv")
# D:\4_SEMESTER\CMPE-295B\Github\Stock-Prediction-using-Neural-Networks\dataset\all_stocks_2000-01-01_to_2020-10-31.csv
dataset = df.loc[df['Ticker'] == 'AAPL']
obs = np.arange(1, len(dataset) + 1, 1)
# plt.plot(obs,dataset.iloc[:, 3:4])
# plt.xlabel('Apple Open price 2007')
# plt.show()

# TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset[['Open','High', 'Low', 'Close']].mean(axis = 1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
close_val = dataset[['Close']]

# PLOTTING ALL INDICATORS IN ONE PLOT
# plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
# plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
# plt.plot(obs, close_val, 'g', label = 'Closing price')
# plt.legend(loc = 'upper right')
# plt.show()

training_size = int(len(dataset) * 0.75)
test_size = len(dataset) - training_size

training_data, test_data = dataset.iloc[0:training_size,:], dataset.iloc[training_size:len(dataset),:]

training_set = training_data.iloc[:, 3:4].values

test_set = test_data.iloc[:, 3:4].values
# plt.plot(training_set)
# plt.xlabel('training set')
# plt.show()
# plt.plot(test_set)
# plt.xlabel('test set')
# plt.show()

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []   #Initialize training data
y_train = []   #Initialize testing data

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# plt.plot(X_train, color = 'red', label = 'Open price every 60 readings (X-value)')
# plt.plot(y_train, color = 'blue', label = "Open price after 60 readings window (Y-value)")
# plt.xlabel('normalized training set')
# plt.show()

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

regressor.save('lstm_apple.h5')

# real values
test_set = test_set.reshape(-1,1)
test_set_scaled = sc.transform(test_set)
X_test = []
for i in range(60, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)




fig = plt.figure(1)

plt.plot(test_set , color = 'red', label = 'Real Apple Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.savefig("./static/apple.png")
