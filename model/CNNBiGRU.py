import tensorflow as tf
import tensorflow.keras as keras
from scipy.io import loadmat
from pyprojroot import here
import numpy as np


def CNNBiGRU():
    inputs = tf.keras.Input(shape=(20000,))
    x = tf.keras.layers.Dense(1024, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2())(inputs)
    x = tf.keras.layers.Reshape((32, 32, 1))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='gelu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Reshape((128, 64))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, bias_regularizer=tf.keras.regularizers.l2(1e-4), activity_regularizer=tf.keras.regularizers.l2(1e-5)))(x)
    x = tf.keras.layers.Reshape((128, 128, 1))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=2, activation='gelu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Dense(64, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Dropout(rate=0.6)(x)
    x = tf.keras.layers.Dense(32, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Dropout(rate=0.7)(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
    
if __name__ == "__main__":
    data = here("data/BRCA1View20000.mat")
    data  = loadmat(data)
    print(f"==>> data: {data}")
    x_input = data["data"]
    y_input = np.unique(data["targets"])
    print(f"==>> y_inpuy: {y_input}")
    x_input = x_input.T[1:10,:]
    print(f"==>> x_input: {x_input.shape}")
    model = CNNBiGRU()
    print(model(x_input))