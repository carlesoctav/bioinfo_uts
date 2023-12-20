from model.CNNBiGRU import CNNBiGRU
from pyprojroot import here
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from keras import metrics
import numpy as np
import tensorflow_addons as tfa
from sklearn.metrics import classification_report

np.random.seed(7)
tf.random.set_seed(7)



file_path = here("data/BRCA1View20000.mat")

data = loadmat(file_path)


# read and process data
x_input = data["data"].T
y_input = data["targets"].reshape(-1,1)
y_input = y_input - 1


x_feat = (x_input - x_input.mean(axis=0, keepdims=True)) / x_input.std(
    axis=0, keepdims=True
)


y_input = keras.utils.to_categorical(y_input, num_classes=6)

# model arguments
model = CNNBiGRU()
losses = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()


# metrics
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1Score()

# callbacks
csv_logger = tf.keras.callbacks.CSVLogger(here("logs/training_smote.csv"))
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    patience=20,
    mode="max",
    restore_best_weights=True,
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=here("model_weight/best_smote.keras"),
    monitor="val_acc",
    save_best_only=True,
    mode="max",
    save_weights_only=True,
)


# Train arg
epochs = 100
batch_size = 256
val_split = 0.2

# train
model.compile(
    loss=losses,
    optimizer=optimizer,
    metrics=["acc", precision, recall, f1],
)

model.fit(
    x_feat,
    y_input,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=val_split,
    callbacks=[csv_logger, model_checkpoint,early_stopping],
)

y_pred = model.predict(x_feat)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(np.argmax(y_input, axis=1), y_pred))