import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
file_name = "AAPL"
dataset=pd.read_csv(file_name+".csv")
print(dataset.head(5))

training_size = int(len(dataset) * 0.75)
test_size = len(dataset) - training_size

training_set, test_set = dataset.iloc[0:training_size,1:2].values, dataset.iloc[training_size:,1:2].values


sc = MinMaxScaler(feature_range = (0, 1))
trainingset_scaled = sc.fit_transform(training_set)
print(trainingset_scaled)

xtrain = []
ytrain = []
for i in range(60, len(training_set)):
    xtrain.append(trainingset_scaled[i-60:i, 0])
    ytrain.append(trainingset_scaled[i, 0])
xtrain, ytrain = np.array(xtrain), np.array(ytrain)
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (xtrain.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

lstm_model.fit(xtrain, ytrain, epochs = 50, batch_size = 32)

dataset_train = dataset.iloc[:training_size, 1:2]
dataset_test = dataset.iloc[training_size:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
print(inputs)
xtest = []
for i in range(60, len(test_set)+60):
    xtest.append(inputs[i-60:i, 0])
xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
print(xtest.shape)

predicted_values = lstm_model.predict(xtest)
predicted_values = sc.inverse_transform(predicted_values)
print(len(predicted_values))
print(len(dataset_test.values))
plt.plot(dataset.loc[training_size:, "Date"],dataset_test.values, color = "red", label = "Real {} Stock Price".format(file_name))
plt.plot(dataset.loc[training_size:, "Date"],predicted_values, color = "blue", label = "Predicted {} Stock Price".format(file_name))
plt.xticks(np.arange(0,1251,60),rotation='vertical')
plt.title('{} Stock Price Prediction'.format(file_name))
plt.xlabel('Time')
plt.ylabel('{} Stock Price'.format(file_name))
plt.legend()
plt.show()