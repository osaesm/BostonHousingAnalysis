import numpy as np
import keras
import csv
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.backend import eval

# If under y = x, 0; else 1
x_train = []
y_train = []

with open('all/train.csv', newline='') as housing_data_file:
    housing_data_file.readline()
    housing_data = csv.reader(housing_data_file)
    for house in housing_data:
        for i in range(1, len(house)):
            house[i] = float(house[i])
        x_train.append(house[1:-1])
        y_train.append(house[-1])

# converting python lists to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# normalizing inputs and output to 1
for i in range(13):
    x_train[:,i] /= max(abs(x_train[:,i]))

y_train /= max(abs(y_train))

# define our model
model = Sequential()
model.add(Dense(units=1, activation='sigmoid', input_dim=13))
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10000, validation_split=0.2)

# 2 arrays: first one has weights (m in y = mx+b), second has biases (b in y = mx+b) 
print(model.get_weights())
