import flask
from flask import Flask, render_template,request, redirect, url_for, Response
from numpy.lib.shape_base import apply_along_axis
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt, mpld3
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import numpy as np
import io
# import StockPricePrediction


SECRET_KEY = '28b83e567ca2d6e8963563f612df55ff'


app = Flask(__name__)

app.config['SECRET_KEY'] = SECRET_KEY
# model_apple=load_model("lstm_apple.h5")
model_google = load_model("google_model.h5")
##########################################
#APPLE
##########################################

# df = pd.read_csv("../dataset/data/2000-2020/AAPL_2000-01-01_to_2020-10-31.csv")

# dataset = df.loc[df['Ticker'] == 'AAPL']

# training_size = int(len(dataset) * 0.75)
# test_size = len(dataset) - training_size

# training_data, test_data = dataset.iloc[0:training_size,:], dataset.iloc[training_size:len(dataset),:]

# training_set = training_data.iloc[:, 3:4].values
# test_set = test_data.iloc[:, 3:4].values

# sc = MinMaxScaler(feature_range = (0, 1))

# test_set = test_set.reshape(-1,1)
# test_set_scaled = sc.fit_transform(test_set)

# X_test = []

# for i in range(60, len(test_set_scaled)):
#     X_test.append(test_set_scaled[i-60:i, 0])

# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predicted_stock_price = model_apple.predict(X_test)
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# ###########################################
# #GOOGLE
# ###########################################
# dataset_g = pd.read_csv("../dataset/GOOGL.csv")

# training_size_g = int(len(dataset_g) * 0.75)
# test_size_g = len(dataset_g) - training_size_g

# training_set_g, test_set_g = dataset.iloc[0:training_size_g,1:2].values, dataset.iloc[training_size_g:,1:2].values


# dataset_train_g = dataset_g.iloc[:training_size_g, 1:2]
# dataset_test_g = dataset_g.iloc[training_size_g:, 1:2]
# dataset_total_g = pd.concat((dataset_train_g, dataset_test_g), axis = 0)
# inputs = dataset_total_g[len(dataset_total_g) - len(dataset_test_g) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs = sc.transform(inputs)
# xtest_g = []
# for i in range(60, len(test_set_g)+60):
#     xtest_g.append(inputs[i-60:i, 0])
# xtest_g = np.array(xtest_g)
# xtest_g = np.reshape(xtest_g, (xtest_g.shape[0], xtest_g.shape[1], 1))

# predicted_values = model_google.predict(xtest_g)
# predicted_values = sc.inverse_transform(predicted_values)



# @app.route("/plot_apple.png", methods = ["GET"])
# def plot_figure_apple():
#     figure = create_figure_apple()
#     output = io.BytesIO()
#     FigureCanvas(figure).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')
    
# def create_figure_apple():
#     fig = plt.figure(1)
#     plt.plot(test_set , color = 'red', label = 'Real Apple Stock Price')
#     plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
#     plt.title('Apple Stock Price Prediction')
#     plt.xlabel('Time')
#     plt.ylabel('Apple Stock Price')
#     plt.legend()
#     # plt.show()
#     return fig 

@app.route("/", methods = ['GET'])
def home():
    return render_template('main.html')

@app.route("/predictions", methods=["GET"])
def predictions():
    return render_template('prediction.html')

@app.route("/analysis", methods=["GET"])
def analysis():
    return render_template('analysis.html')



app.run(port=3000, debug=True)

