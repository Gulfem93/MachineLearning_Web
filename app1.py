from ast import Num
from pandas.io.formats.format import DataFrameFormatter
from scipy.sparse import data
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics
from tensorflow.python.keras.activations import relu

#DataFrame
dataframe = pd.read_csv(r"E:\Gülfem\ML_Coding\auto-mpg.csv", na_values=['NA', '?'])

#print(dataframe.head())
#print(dataframe.isnull().sum())

dataframe["horsepower"] = dataframe["horsepower"].fillna(dataframe["horsepower"].median())
#print(dataframe.isnull().sum())

x = dataframe[["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]].values
y = dataframe[["mpg"]].values

#Train-test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

model = Sequential()
model.add(Dense(25, input_dim = x.shape[1] ,activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1))

model.compile(loss = "mean_squared_error", optimizer = "adam")
monitor = EarlyStopping(monitor = "val_loss", min_delta = 1e-3, patience = 5, verbose = 2, mode = "auto", restore_best_weights = True)
model.fit(x_train, y_train, validation_data = (x_test, y_test), callbacks = monitor, verbose = 2, epochs = 250)

prediction = model.predict(x_test)

score = np.sqrt(metrics.mean_squared_error(prediction, y_test))

#model.save(os.path.join(os.getcwd(), "mpg_model.h5"))

cols = [x for x in dataframe.columns if x not in ("mpg", "name")]
print(cols)

print("{")

for i, name in enumerate(cols):
    print(f' "{name}" :{{ "min": { dataframe[name].min() }, "max": { dataframe[name].max() } }} { "," if i < ( len(cols) -1 ) else "" } ')
    
print("}")











