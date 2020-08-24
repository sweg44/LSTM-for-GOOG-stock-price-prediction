import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jupyterthemes import jtplot
jtplot.style()
%matplotlib inline

# First you need to input the test set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# Then we need to convert the dataframe to a numpy array as tensorflow only accepts that
training_set = dataset_train.iloc[:, 1:2].values

training_set

# Now we must feature scale, usually when there is a sigmoid function output we use normalisation
from sklearn.preprocessing import MinMaxScaler
# This is the same as import as sc
sc = MinMaxScaler(feature_range=(0,1))
# Now apply the scaler into the data
training_set_scaled = sc.fit_transform(training_set)
# Now view the scaled set, all values are scaled between 0,1
training_set_scaled

# Now we must build the RNN
# We are going to create a data structure that uses 60 timesteps, this is the lookback period.
# There will also be only one output layer
x_train = []
y_train = []

# We have to start at 60 as we need 60 stock prices to look at
for i in range(60, 1258):
    # Here we take i (initial)-60 as it would cancel to 0 (first index)
    x_train.append(training_set_scaled[i-60:i, 0])
    # As indexing starts at 0 we do not need to do t+1 as everything is shifted by one
    y_train.append(training_set_scaled[i, 0])

# We then convert the lists into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Now we must add more indicators (dimensions) to improve the accuracy of the model
# When adding dimensions to a numpy array we use .reshape()
# We only have one indicator, the open stock price
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

----

# Building the RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Add the first layers
# Units are the number of LSTM cells, we have chosen 50 as we want to have a high dimensionality
# We set return_sequences to true as we want to stack the LSTM'S
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))

# We now add a dropout layer to prevent overfitting
regressor.add(Dropout(0.2))

# Second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))

# We now add a dropout layer to prevent overfitting
regressor.add(Dropout(0.2))

# Third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))

# We now add a dropout layer to prevent overfitting
regressor.add(Dropout(0.2))

# Fourth LSTM layer
regressor.add(LSTM(units = 50))

# We now add a dropout layer to prevent overfitting
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units=1))

---

# Now we must compile the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Now we fit the RNN
regressor.fit(x=x_train, y=y_train, epochs=10, batch_size=32)

---

# Now we compare test-train
# First you need to input the train set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# Then we need to convert the dataframe to a numpy array as tensorflow only accepts that
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

x_test = []
# We have to start at 60 as we need 60 stock prices to look at
for i in range(60, 80):
    # Here we take i (initial)-60 as it would cancel to 0 (first index)
    x_test.append(inputs[i-60:i, 0])
    
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

---

# Making predictions
predicted_stock_price = regressor.predict(x_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#from sklearn.metrics import mean_squared_error
mse = mean_squared_error(predicted_stock_price, real_stock_price[-20:])
rmse = np.sqrt(mse)*100
print(rmse)

plt.figsize = (15,18)
plt.plot(real_stock_price, color = 'red', label = 'Real price')
plt.plot(predicted_stock_price, label='Predicted price')
plt.show()

