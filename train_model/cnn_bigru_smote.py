from model.CNNBiGRU import CNNBiGRU
from pyprojroot import here
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from keras import metrics
import numpy as np
import tensorflow_addons as tfa
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(7)
tf.random.set_seed(7)

file_path = here("data/BRCA1View20000_smote.mat")

data = loadmat(file_path)


# read and process data
x_input = data["data"].T
y_input = data["targets"].reshape(-1,1)
y_input = y_input - 1
y_input = keras.utils.to_categorical(y_input, num_classes=6)

x_train, x_test, y_train, y_test = train_test_split( x_input, y_input, test_size=0.2, random_state=42)

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)


# model arguments
model = CNNBiGRU()
losses = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

# metrics
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1Score()

# callbacks
csv_logger = tf.keras.callbacks.CSVLogger(here("logs/training_cnn_bigru_smote.csv"))
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

# train
model.compile(
    loss=losses,
    optimizer=optimizer,
    metrics=["acc", precision, recall, f1],
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[csv_logger, early_stopping, model_checkpoint],
)

# plot 
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.legend(["acc", "val_acc"])
plt.savefig(here("stdout_logs/cnn_bigru_smote_acc.png"))
plt.clf()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.savefig(here("stdout_logs/cnn_bigru_smote_loss.png"))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(np.argmax(y_test, axis=1), y_pred))