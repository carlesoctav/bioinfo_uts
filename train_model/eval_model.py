import tensorflow as tf
from tensorflow import keras
from model.CNNBiGRU import CNNBiGRU
from keras import metrics
import numpy as np
from scipy.io import loadmat
from pyprojroot import here

file_path = here("data/BLCAcitsmote.mat")
data = loadmat(file_path)

x_mean = np.load("data/x_mean.npy")
x_std = np.load("data/x_std.npy")

# read and process data
x_test = data["data"]
y_test = data["targets"]

x_test = (x_test - x_mean) / x_std

y_test= keras.utils.to_categorical(y_test, num_classes=10)


model = CNNBiGRU()
model.load_weights("model_weight/best.keras")



#metrics
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1Score()

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', precision, recall, f1])

model.evaluate(x_test, y_test, batch_size=64, verbose=1)
