
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

def concat_data():
    script_dir = os.path.dirname(__file__)  # Script directory

    full_path = os.path.join(script_dir, './2007')

    all_folders = glob(full_path+"/*")

    frames = []
    for folder in sorted(all_folders):
        all_files = glob(folder + "/*.csv")
        li = []
        for filename in all_files:
            print(filename)
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)
        frames.append(frame)
    result = pd.concat(frames)

    result.to_csv(r'combined.csv', index=False)

def load_data():
    data = pd.read_csv("combined.csv", usecols=[0,1,3,4,5,6])
    return data

df = load_data()
dataset = df.loc[df['Ticker'] == 'AAPL']

plt.plot(dataset.iloc[:, 2:3])
plt.show()

training_size = int(len(dataset) * 0.75)
test_size = len(dataset) - training_size

training_data, test_data = dataset.iloc[0:training_size,:], dataset.iloc[training_size:len(dataset),:]

training_set = training_data.iloc[:, 2:3].values

test_set = test_data.iloc[:, 2:3].values
plt.plot(training_set)
plt.show()
plt.plot(test_set)
plt.show()

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []   #Initialize training data
y_train = []   #Initialize testing data

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

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

regressor.save('lstm.h5')

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

plt.plot(test_set, color = 'red', label = 'Real Apple Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()

# dataset = df.loc[df['Ticker'] == 'AAPL']
# del dataset['Ticker']

# # dataset = dataset.reindex(index = dataset.index[::-1])

# obs = np.arange(1, len(dataset) + 1, 1)


# OHLC_avg = dataset.mean(axis = 1)


# OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) 
# scaler = MinMaxScaler(feature_range=(0, 1))
# OHLC_avg = scaler.fit_transform(OHLC_avg)

# train_OHLC = int(len(OHLC_avg) * 0.75)
# test_OHLC = len(OHLC_avg) - train_OHLC
# train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# # making time series dataset (FOR TIME T, VALUES FOR TIME T+1)
# trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
# testX, testY = preprocessing.new_dataset(test_OHLC, 1)

# # reshaping train and test data
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# step_size = 1

# model = Sequential()
# model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
# model.add(LSTM(16))
# model.add(Dense(1))
# model.add(Activation('linear'))

# model.compile(loss='mean_squared_error', optimizer='adagrad') # compare with other optimizers
# model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

# # predict
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)

# # de nomrmalize for plotting
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])


# # rmse
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train RMSE: %.2f' % (trainScore))

# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test RMSE: %.2f' % (testScore))


# trainPredictPlot = np.empty_like(OHLC_avg)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict


# testPredictPlot = np.empty_like(OHLC_avg)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict

# # DE-NORMALIZING MAIN DATASET 
# OHLC_avg = scaler.inverse_transform(OHLC_avg)


# plt.plot(OHLC_avg, 'g', label = 'original dataset')
# plt.plot(trainPredictPlot, 'r', label = 'training set')
# plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
# plt.legend(loc = 'upper right')
# plt.xlabel('Time in Days')
# plt.ylabel('OHLC Value of Apple Stocks')
# plt.show()


# last_val = testPredict[-1]
# last_val_scaled = last_val/last_val
# next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
# print ("Last Day Value:", np.asscalar(last_val))
# print ("Next Day Value:", np.asscalar(last_val*next_val))
